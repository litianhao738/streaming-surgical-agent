"""Frozen 16-target, three-arm H0 prompt study; no retries or production edits.

Prepare first with --prepare-only (no credentials or inference). A subsequent
default invocation restores and verifies that plan before any paid dispatch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import sys
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from PIL import Image

from scripts import run_h0_frame_strategy_study as study
from scripts.h0_prompt_refinement import ARMS, variant_request
from scripts.resume_h0_frame_strategy_study import DurableAuditSender, call_accounted
from scripts.run_five_expert_ablation import closure
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    canonical_json_bytes,
    thaw_json,
)
from surgical_agent.api.schema import schema_for

POSITIONS = (9, 12, 15, 18)
CALL_LIMIT, CONCURRENCY, CAP, RESERVE = 48, 3, 0.85, 0.05


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_digest(plan):
    unsigned = dict(plan)
    expected = unsigned.pop("plan_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected:
        raise ValueError("frozen plan digest mismatch")


def configured(args, reference):
    config = study.load_api_config(args.config)
    if (config.provider != "openrouter" or not config.data_upload_authorized
            or config.requested_model_identifier != "qwen/qwen3.8-max-0902"
            or config.requested_model_identifier != reference["model"]
            or config.provider_options.get("initial_prompt_profile") != "fixed_visual_only"
            or study.sha256_file(args.config) != reference["config_sha256"]
            or study.sha256_file(args.dataset / "repair_manifest.json") != reference["repair_manifest_sha256"]):
        raise ValueError("authorized frozen config or dataset differs")
    return config


def encode_images(source_paths):
    def encode(pair):
        name, expected = pair
        path = Path(name)
        if study.sha256_file(path) != expected:
            raise ValueError("source image changed")
        encoded = io.BytesIO()
        with Image.open(path) as image:
            image.convert("RGB").save(encoded, format="PNG", compress_level=9)
        identifier = f"cholectrack20:{path.parent.parent.name}:frame:{int(path.stem)}"
        return identifier, ApiImageInput(identifier, "image/png", encoded.getvalue())
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(encode, source_paths.items()))


def reconstruct(saved, images):
    meta = saved["metadata"]
    request = ApiRequest(
        provider=meta["provider"], model_identifier=meta["requested_model_identifier"],
        endpoint_identifier=meta["endpoint_identifier"], prompt_version=meta["prompt_version"],
        response_schema_version=meta["response_schema_version"], payload=saved["payload"],
        images=tuple(images[item["identifier"]] for item in meta["images"]),
        generation_parameters=meta["generation_parameters"],
    )
    if study.canonical_request_metadata(request).to_mapping() != meta:
        raise ValueError("reconstructed request differs from frozen metadata")
    return request


def mask_precheck(args, samples):
    """The adapter parses annotations; only masks and provenance are inspected here."""
    adapter = study.CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    masks, sources = {}, {}
    for video in study.VIDEOS:
        wanted = [s["frame_id"] for s in samples if s["video_id"] == video]
        seen = []
        for record in adapter.iter_video(video, frame_ids=wanted):
            frame_id = record.inference.target_frame_id
            if record.inference.source_split is not study.DatasetSplit.TRAINING:
                raise ValueError("Training-only study required")
            seen.append(frame_id)
            target = record.frame_supervision
            evaluation = record.evaluation
            if evaluation is not None and evaluation.instance_supervision_available:
                mask = {task: bool(evaluation.instances) and all(
                    getattr(instance.mask, task) for instance in evaluation.instances
                ) for task in study.TASKS}
            else:
                mask = {task: target is not None and bool(getattr(target.mask, task)) for task in study.TASKS}
            if not all(mask.values()):
                raise ValueError(f"fixed target lacks five-head GT: {video}/{frame_id}; do not substitute")
            masks[f"{video}_{frame_id}"] = mask
            for attr in ("annotation_source", "phase_source", "frame_action_source", "manifest_source"):
                name = getattr(record.provenance, attr)
                if name:
                    path = Path(name)
                    if not path.is_absolute():
                        path = args.dataset / path
                    if path.is_file():
                        sources[str(path.resolve())] = study.sha256_file(path)
        if set(seen) != set(wanted):
            raise ValueError("fixed targets missing from dataset")
    return masks, sources


def prepare(args):
    if (args.output / "plan.json").exists():
        config, plan, requests = restore(args)
        if (args.output / "execution.lock").exists():
            raise ValueError("execution already started; preserve frozen experiment")
        return config, plan, requests
    source = read_json(args.source / "plan.json")
    verify_digest(source)
    config = configured(args, source)
    for name, expected in source["source_sha256"].items():
        if study.sha256_file(ROOT / name) != expected:
            raise ValueError("original study dependency changed")
    samples = [{"video_id": video, "frame_id": source["selection"][video]["selected_targets"][position],
                "source_position_zero_based": position}
               for position in POSITIONS for video in study.VIDEOS]
    masks, annotation_sources = mask_precheck(args, samples)
    source_index = {f"cholectrack20:{Path(p).parent.parent.name}:frame:{int(Path(p).stem)}": (p, sha)
                    for p, sha in source["source_png_sha256"].items()}
    saved_bases, selected_paths = {}, {}
    for sample in samples:
        key = f"{sample['video_id']}_{sample['frame_id']}_B"
        saved = read_json(args.source / "requests" / f"{key}.json")
        if saved["metadata"] != source["requests"][key]:
            raise ValueError("original baseline metadata changed")
        saved_bases[key] = saved
        for image in saved["metadata"]["images"]:
            path, expected = source_index[image["identifier"]]
            selected_paths[path] = expected
    images = encode_images(selected_paths)
    requests, order = {}, []
    for index, sample in enumerate(samples):
        base = reconstruct(saved_bases[f"{sample['video_id']}_{sample['frame_id']}_B"], images)
        generation = thaw_json(base.generation_parameters)
        if generation.get("reasoning") != {"effort": "low"} or generation.get("temperature") != 0:
            raise ValueError("unexpected baseline generation parameters")
        rotation = index % len(ARMS)
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            item = {**sample, "arm": arm, "group": "B", "key": f"{sample['video_id']}_{sample['frame_id']}_{arm}"}
            request = variant_request(base, arm)
            if (request.images != base.images or request.generation_parameters != base.generation_parameters
                    or request.response_schema_version != base.response_schema_version):
                raise ValueError("candidate changes images, generation, or schema")
            requests[item["key"]] = request
            order.append(item)
    dependencies = dict(source["source_sha256"])
    for name in ("scripts/run_h0_prompt_refinement_smoke.py", "scripts/h0_prompt_refinement.py",
                 "scripts/run_h0_conservative_input_smoke.py", "scripts/prepare_h0_cost_optimization.py",
                 "scripts/resume_h0_frame_strategy_study.py", "src/surgical_agent/perception/final_only.py",
                 "src/surgical_agent/api/schema.py"):
        dependencies[name] = study.sha256_file(ROOT / name)
    preregistration_path = ROOT / "docs/H0_PROMPT_REFINEMENT_SMOKE_2026-09-06.md"
    preregistration = preregistration_path.read_text(encoding="utf-8")
    plan = {
        "schema_version": "h0_prompt_refinement_smoke_v1", "status": "FROZEN_BEFORE_INFERENCE",
        "model": config.requested_model_identifier, "source_split": "Training", "samples": samples,
        "source_plan_sha256": source["plan_sha256"], "source_plan_path": str((args.source / "plan.json").resolve()),
        "selection_policy": "Fixed positions 9,12,15,18 per video in old 20-target plan; masks only; abort without substitution if incomplete",
        "prior_exposure": "Development targets: old study parsed/scored original 80-target plan offline. These positions were not paid predictions in prior 32-target window or eight-target conservative runs. Not sealed/unseen test data.",
        "label_access_policy": "Adapter parses annotation values before inference; selection and mask precheck inspect masks only, never rank on labels. Freeze all prompts/requests before current scoring. No GT in model inputs.",
        "all_five_head_valid_masks": masks, "annotation_sources_sha256": annotation_sources,
        "arms": list(ARMS), "arm_changes": {
            "baseline": "Exact retained three-frame low request",
            "schema_only": "Only replace complete system schema suffix with short response_format reference; original input retained",
            "tuned": "Schema removal plus conservative empty-field cleanup and generic evidence-use instruction refinement; ontology and boundaries retained",
        }, "call_order": order, "order_policy": "Per-sample rotate baseline/schema_only/tuned; three concurrent requests; completion order uncontrolled",
        "paid_call_limit": CALL_LIMIT, "concurrency": CONCURRENCY, "retries": 0,
        "planning_budget_usd": CAP, "inflight_reserve_usd": RESERVE,
        "spend_policy": "Known spend plus 0.05 USD per outstanding new call <= min(0.85, balance minus 0.10); not provider-enforced cap; stop on unknown cost",
        "adoption_rule": "Candidate must have all five per-task micro-F1 and exact accuracy, all-five joint exact accuracy, and complete valid response rate no lower than same-round baseline; actual cost and uncached price-equivalent mean must both decrease. Must pass all-target and shared-success analyses. A pilot pass is not statistical noninferiority or proof of no effect.",
        "preregistration_snapshot": {"path": "preregistration.md", "sha256": hashlib.sha256(preregistration.encode()).hexdigest()},
        "tracker": False, "repair": False, "gate": False, "memory": False,
        "config_sha256": source["config_sha256"], "repair_manifest_sha256": source["repair_manifest_sha256"],
        "source_sha256": dependencies, "source_png_sha256": selected_paths,
        "requests": {key: study.canonical_request_metadata(request).to_mapping() for key, request in requests.items()},
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("execution.lock", "calls", "predictions.json")):
        raise ValueError("nonfresh experiment output")
    snapshot = args.output / "preregistration.md"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != preregistration:
        raise ValueError("preexisting preregistration differs")
    if not snapshot.exists():
        snapshot.write_text(preregistration, encoding="utf-8", newline="")
    for key, request in requests.items():
        path = args.output / "requests" / f"{key}.json"
        record = {"payload": thaw_json(request.payload), "metadata": plan["requests"][key]}
        if path.exists() and read_json(path) != record:
            raise ValueError("preexisting request differs")
        if not path.exists():
            study.atomic_write_json(path, record)
    study.atomic_write_json(args.output / "plan.json", plan)
    return config, plan, requests


def restore(args):
    plan = read_json(args.output / "plan.json")
    verify_digest(plan)
    snapshot = args.output / plan["preregistration_snapshot"]["path"]
    if study.sha256_file(snapshot) != plan["preregistration_snapshot"]["sha256"]:
        raise ValueError("frozen preregistration snapshot changed")
    config = configured(args, plan)
    source = read_json(Path(plan["source_plan_path"]))
    verify_digest(source)
    if source["plan_sha256"] != plan["source_plan_sha256"]:
        raise ValueError("source plan differs")
    for name, expected in plan["source_sha256"].items():
        if study.sha256_file(ROOT / name) != expected:
            raise ValueError(f"frozen dependency changed: {name}")
    for name, expected in plan["annotation_sources_sha256"].items():
        if study.sha256_file(name) != expected:
            raise ValueError("annotation source changed since mask precheck")
    images = encode_images(plan["source_png_sha256"])
    requests = {}
    for key, metadata in plan["requests"].items():
        saved = read_json(args.output / "requests" / f"{key}.json")
        if saved["metadata"] != metadata:
            raise ValueError("saved request differs from plan")
        requests[key] = reconstruct(saved, images)
    return config, plan, requests


def enrich_evaluation(evaluation):
    result = study.augment_metrics(evaluation)
    frames = result["frames"]
    if any(not all(frame["mask"].values()) for frame in frames):
        raise ValueError("five-head GT availability changed during scoring")
    result["all_five_exact"] = {
        "correct": sum(all(frame["exact"].get(task) is True for task in study.TASKS) for frame in frames),
        "denominator": len(frames),
    }
    result["all_five_exact"]["accuracy"] = (
        result["all_five_exact"]["correct"] / len(frames) if frames else None
    )
    return result


def evaluate_rows(adapter, rows):
    pooled = {"frames": [], "tasks": {task: {"valid_gt": 0, "scored": 0, "correct": 0, "failed_with_gt": 0}
                                      for task in study.TASKS}}
    per_video = {}
    for video in study.VIDEOS:
        chosen = [row for row in rows if row["video_id"] == video]
        if not chosen:
            continue
        result = enrich_evaluation(study.evaluate_offline(adapter, video, chosen))
        per_video[video] = result
        pooled["frames"].extend({"video_id": video, **frame} for frame in result["frames"])
        for task in study.TASKS:
            for key in ("valid_gt", "scored", "correct", "failed_with_gt"):
                pooled["tasks"][task][key] += result["tasks"][task][key]
    return enrich_evaluation(pooled), per_video


def native_usage(directory):
    path = directory / "http_response.json"
    if not path.exists():
        return None
    found = None
    for line in read_json(path)["body"].splitlines():
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("usage"):
                found = event["usage"]
    return found


def usage_means(native):
    if not native:
        return {}
    fields = {
        "prompt_tokens": lambda u: u.get("prompt_tokens"),
        "completion_tokens": lambda u: u.get("completion_tokens"),
        "reasoning_tokens": lambda u: u.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "cached_tokens": lambda u: u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "cache_write_tokens": lambda u: u.get("prompt_tokens_details", {}).get("cache_write_tokens", 0),
        "provider_cost": lambda u: u.get("cost"),
    }
    means = {}
    for key, getter in fields.items():
        values = [getter(row) for row in native]
        means[key] = statistics.mean(values) if all(isinstance(v, (int, float)) for v in values) else None
    if means["completion_tokens"] is not None and means["reasoning_tokens"] is not None:
        means["visible_tokens"] = means["completion_tokens"] - means["reasoning_tokens"]
    if means["prompt_tokens"] is not None and means["completion_tokens"] is not None:
        means["uncached_price_equivalent_not_actual_cost"] = (
            means["prompt_tokens"] * 0.000002 + means["completion_tokens"] * 0.000006
        )
    return means


def wire_audit(args, plan, requests):
    errors, checked = [], 0
    for item in plan["call_order"]:
        path = args.output / "calls" / item["key"] / "wire_request.json"
        if not path.exists():
            continue
        req, wire = requests[item["key"]], read_json(path)
        expected_images = [hashlib.sha256(("data:image/png;base64," + base64.b64encode(i.content).decode()).encode()).hexdigest()
                           for i in req.images]
        content = wire["messages"][1]["content"]
        if (wire["messages"][0]["content"] != req.payload["system_text"]
                or content[0]["text"] != req.payload["input_text"]
                or [part["image_url"]["data_url_sha256"] for part in content[1:]] != expected_images
                or [part["image_url"]["detail"] for part in content[1:]] != ["low", "low", "high"]
                or wire["response_format"]["type"] != "json_schema"
                or wire["response_format"]["json_schema"]["strict"] is not True
                or wire["response_format"]["json_schema"]["schema"] != schema_for(req.response_schema_version)
                or wire["reasoning"] != {"effort": "low"} or wire["temperature"] != 0
                or wire["max_tokens"] != 4096 or wire["model"] != plan["model"]):
            errors.append(item["key"])
        checked += 1
    return {"status": "PASS" if not errors else "FAIL", "checked": checked, "errors": errors}


def analyze(args, plan, rows, requests):
    before = study.sha256_file(args.output / "predictions.json")
    adapter = study.CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    shared = set.intersection(*[{(row["video_id"], row["frame_id"]) for row in rows
                                if row["arm"] == arm and row["status"] == "OK"} for arm in ARMS])
    groups = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        evaluation, per_video = evaluate_rows(adapter, selected)
        paired, paired_video = evaluate_rows(adapter, [row for row in selected
                                             if (row["video_id"], row["frame_id"]) in shared])
        records, native, structural = [], [], []
        raw_count, schema_valid = 0, 0
        for row in selected:
            directory = args.output / "calls" / row["key"]
            records.extend(study.read_call_usage(directory))
            usage = native_usage(directory)
            if usage is not None:
                native.append(usage)
            for path in (directory / "raw_responses").glob("*.json"):
                raw_count += 1
                try:
                    study.validate_final_only(read_json(path)["parsed_payload"])
                except Exception:  # noqa: BLE001, S112 - count failures, never repair outputs
                    continue
                schema_valid += 1
            if row["status"] == "OK":
                structural.append({"key": row["key"], **closure(row["selected_ids"])})
        groups[arm] = {
            "evaluation": evaluation, "per_video": per_video, "paired_same_success": paired,
            "paired_per_video": paired_video, "usage": study.usage_summary(records),
            "native_usage_count": len(native), "native_means": usage_means(native),
            "service_cache_hit_calls": sum(u.get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0 for u in native),
            "complete_schema": {"valid": schema_valid, "raw_parsed_responses": raw_count, "scheduled": len(selected),
                                "rate_over_scheduled": schema_valid / len(selected) if selected else None},
            "structural_only_not_semantic": {
                "denominator_successful": len(structural),
                "ivt_components_in_selected": sum(r["ivt_components_in_selected"] for r in structural),
                "exact_projection": sum(r["exact_projection"] for r in structural), "frames": structural,
            },
        }
    baseline = groups["baseline"]
    decisions = {}
    for arm in ARMS[1:]:
        candidate, failures = groups[arm], []
        for cohort in ("evaluation", "paired_same_success"):
            for task in study.TASKS:
                for metric in ("micro_f1", "exact_accuracy_all_valid_gt"):
                    left = candidate[cohort]["tasks"][task].get(metric)
                    right = baseline[cohort]["tasks"][task].get(metric)
                    if left is None or right is None or left < right:
                        failures.append(f"{cohort}.{task}.{metric}")
            left = candidate[cohort]["all_five_exact"]["accuracy"]
            right = baseline[cohort]["all_five_exact"]["accuracy"]
            if left is None or right is None or left < right:
                failures.append(f"{cohort}.all_five_exact")
        if candidate["complete_schema"]["valid"] < baseline["complete_schema"]["valid"]:
            failures.append("complete_schema_rate")
        if (candidate["usage"]["unpriced_calls"] or baseline["usage"]["unpriced_calls"]
                or candidate["usage"]["reported_cost_usd"] >= baseline["usage"]["reported_cost_usd"]):
            failures.append("lower_fully_accounted_cost")
        left = candidate["native_means"].get("uncached_price_equivalent_not_actual_cost")
        right = baseline["native_means"].get("uncached_price_equivalent_not_actual_cost")
        if left is None or right is None or left >= right:
            failures.append("lower_uncached_price_equivalent")
        if len(rows) != CALL_LIMIT or any(row["status"] == "NOT_ATTEMPTED" for row in rows):
            failures.append("incomplete_experiment")
        decisions[arm] = {"pilot_rule_pass": not failures, "failed_requirements": failures,
                          "caveat": "Passing is not statistical noninferiority; no production change made"}
    after = study.sha256_file(args.output / "predictions.json")
    if before != after:
        raise ValueError("predictions changed during scoring")
    summary = {
        "schema_version": "h0_prompt_refinement_result_v1", "plan_sha256": plan["plan_sha256"],
        "predictions_sha256_before_scoring": before, "predictions_sha256_after_scoring": after,
        "groups": groups, "paired_same_success_targets": sorted(shared), "paired_same_success_count": len(shared),
        "successful_predictions": sum(row["status"] == "OK" for row in rows), "scheduled_predictions": len(rows),
        "usage": study.usage_summary([record for item in plan["call_order"]
                    for record in study.read_call_usage(args.output / "calls" / item["key"])]),
        "wire_audit": wire_audit(args, plan, requests), "adoption_assessment": decisions,
    }
    study.atomic_write_json(args.output / "summary.json", summary)
    return summary


def run(args):
    if args.prepare_only:
        _, plan, _ = prepare(args)
        print(json.dumps({"status": "PREPARED", "calls": len(plan["call_order"]),
                          "plan_sha256": plan["plan_sha256"], "inference_calls": 0}), flush=True)
        return
    config, plan, requests = restore(args)
    if args.analyze_only:
        analyze(args, plan, read_json(args.output / "predictions.json"), requests)
        return
    with (args.output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write("single bounded experiment; never repeat dispatches\n")
    if (args.output / "calls").exists() or (args.output / "cache").exists():
        raise ValueError("fresh calls and cache directories required")
    secret = study.resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    request = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + secret.reveal()})
    with urllib.request.urlopen(request, timeout=20) as response:
        credits = json.load(response)
    study.atomic_write_json(args.output / "credits_before.json", credits)
    available = credits["data"]["total_credits"] - credits["data"]["total_usage"]
    cap = min(CAP, available - 0.10)
    if cap < CAP:
        raise ValueError("insufficient balance for complete bounded three-arm experiment")
    study.AuditSender = DurableAuditSender
    args.cache = args.output / "cache"
    budget = study.ProviderCallBudget(CALL_LIMIT)
    completed, spending, cursor, stop = {}, 0.0, 0, None
    order = plan["call_order"]

    def checkpoint(status):
        study.atomic_write_json(args.output / "predictions.json", [completed[i["key"]] for i in order if i["key"] in completed])
        study.atomic_write_json(args.output / "run_status.json", {
            "status": status, "completed": len(completed), "successful": sum(r["status"] == "OK" for r in completed.values()),
            "known_cost_usd": round(spending, 8), "provider_attempts": budget.used,
            "reason": stop, "plan_sha256": plan["plan_sha256"],
        })

    checkpoint("RUNNING")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        active = {}
        while cursor < len(order) or active:
            while not stop and cursor < len(order) and len(active) < CONCURRENCY:
                if spending + (len(active) + 1) * RESERVE > cap:
                    if not active:
                        stop = "PLANNING_SPEND_LIMIT"
                    break
                item = order[cursor]
                active[executor.submit(call_accounted, item, requests[item["key"]], config, args, secret, budget)] = item
                cursor += 1
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                row = future.result()
                completed[item["key"]] = row
                spending += row["usage"]["reported_cost_usd"]
                if row.get("accounting_error") or row["usage"]["unpriced_calls"]:
                    stop = "UNRECONCILED_ACCOUNTING"
                if row.get("http_status") in {400, 401, 402, 403, 404} or row["status"] == "INTERNAL_FAILURE":
                    stop = row.get("error", "INTERNAL_FAILURE")
                if row["status"] == "OK" and (row["returned_model"] != plan["model"] or row["cache_hit"]):
                    stop = "MODEL_OR_LOCAL_CACHE_MISMATCH"
                checkpoint("RUNNING")
                print(json.dumps({"key": item["key"], "status": row["status"], "completed": len(completed),
                                  "known_cost_usd": round(spending, 8), "stop": stop}), flush=True)
    checkpoint("SCORING")
    rows = [completed.get(item["key"], {**item, "status": "NOT_ATTEMPTED", "error": stop}) for item in order]
    study.atomic_write_json(args.output / "predictions.json", rows)
    summary = analyze(args, plan, rows, requests)
    study.atomic_write_json(args.output / "run_status.json", {
        "status": "STOPPED" if stop else "COMPLETE", "reason": stop, "completed": len(completed),
        "successful": summary["successful_predictions"], "known_cost_usd": round(spending, 8),
        "provider_attempts": budget.used, "plan_sha256": plan["plan_sha256"],
    })
    print(json.dumps({"status": "FINISHED", "successful": summary["successful_predictions"],
                      "known_cost_usd": round(spending, 8)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/h0_frame_strategy_qwen0902_training80_20260905")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/h0_prompt_refinement_qwen0902_20260906")
    parser.add_argument("--dataset", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "docs/API.txt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - never print credentials or provider bodies
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "detail": str(exc) if isinstance(exc, (ValueError, FileExistsError)) else "See preserved artifacts; no inference retry"}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
