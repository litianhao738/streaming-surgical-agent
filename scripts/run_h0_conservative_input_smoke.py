"""Paired original versus conservative input cleanup; preserve low and both schemas."""

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
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from PIL import Image

from scripts import run_h0_frame_strategy_study as study
from scripts.prepare_h0_cost_optimization import TRACK_TEXT
from scripts.resume_h0_frame_strategy_study import DurableAuditSender, call_accounted
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    canonical_json_bytes,
    thaw_json,
)
from surgical_agent.api.schema import schema_for

ARMS = ("baseline", "conservative")
OLD_FRAME_TEXT = ("The causal_frame_ids describe the full window consumed locally. The uploaded\n"
                  "images are only selected_image_frame_ids,")


def conservative_request(base):
    data = json.loads(base.payload["input_text"])
    study.validate_request(base, "B", data["target_frame_id"])
    if (thaw_json(base.generation_parameters)["reasoning"] != {"effort": "low"}
            or data["track_summary"] != {"frames": [], "source_max_frame_id": None, "status": "UNAVAILABLE"}
            or data["workflow_summary"] != {"observed_transitions": [], "phase_stability": None,
                "recent_finalized_phases": [], "source_max_frame_id": None}):
        raise ValueError("requires empty state and low baseline")
    payload = thaw_json(base.payload)
    system = payload["system_text"]
    if system.count(TRACK_TEXT) != 1 or system.count(OLD_FRAME_TEXT) != 1:
        raise ValueError("unexpected source instructions")
    system = system.replace(TRACK_TEXT, "").replace(OLD_FRAME_TEXT, "The uploaded images are selected_image_frame_ids,")
    for key in ("causal_frame_ids", "prior_finalized_prediction", "track_summary", "workflow_summary"):
        del data[key]
    payload.update(system_text=system, input_text=json.dumps(data, sort_keys=True, separators=(",", ":")))
    return replace(base, payload=payload, prompt_version="joint_final_only_conservative_input_v1")


def prepare(args):
    source = json.loads((args.source / "plan.json").read_text(encoding="utf-8"))
    unsigned = dict(source)
    digest = unsigned.pop("plan_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("original plan digest mismatch")
    config = study.load_api_config(args.config)
    if (config.requested_model_identifier != source["model"] or not config.data_upload_authorized
            or study.sha256_file(args.config) != source["config_sha256"]
            or study.sha256_file(args.dataset / "repair_manifest.json") != source["repair_manifest_sha256"]):
        raise ValueError("config or dataset differs from original baseline")
    for path, expected in source["source_sha256"].items():
        if study.sha256_file(ROOT / path) != expected:
            raise ValueError("original source changed")
    proposal = json.loads((args.proposal / "plan.json").read_text(encoding="utf-8"))
    samples = proposal["samples"]
    if len(samples) != 8 or proposal["source_plan_sha256"] != digest:
        raise ValueError("unexpected eight-target proposal")
    source_paths = {f"cholectrack20:{Path(p).parent.parent.name}:frame:{int(Path(p).stem)}": (p, sha)
                    for p, sha in source["source_png_sha256"].items()}
    images, requests, order = {}, {}, []
    for index, sample in enumerate(samples):
        key = f"{sample['video_id']}_{sample['frame_id']}_B"
        saved = json.loads((args.source / "requests" / f"{key}.json").read_text(encoding="utf-8"))
        meta = source["requests"][key]
        if saved["metadata"] != meta:
            raise ValueError("source request metadata changed")
        for image in meta["images"]:
            identifier = image["identifier"]
            if identifier not in images:
                path, expected = source_paths[identifier]
                if study.sha256_file(path) != expected:
                    raise ValueError("source image changed")
                encoded = io.BytesIO()
                with Image.open(path) as source_image:
                    source_image.convert("RGB").save(encoded, format="PNG", compress_level=9)
                images[identifier] = ApiImageInput(identifier, "image/png", encoded.getvalue())
        base = ApiRequest(provider=meta["provider"], model_identifier=meta["requested_model_identifier"],
                          endpoint_identifier=meta["endpoint_identifier"], prompt_version=meta["prompt_version"],
                          response_schema_version=meta["response_schema_version"], payload=saved["payload"],
                          images=tuple(images[i["identifier"]] for i in meta["images"]),
                          generation_parameters=meta["generation_parameters"])
        if study.canonical_request_metadata(base).to_mapping() != meta:
            raise ValueError("reconstructed baseline differs")
        variants = {"baseline": base, "conservative": conservative_request(base)}
        for arm in ARMS if index % 2 == 0 else tuple(reversed(ARMS)):
            item = {**sample, "arm": arm, "group": "B", "key": f"{sample['video_id']}_{sample['frame_id']}_{arm}"}
            order.append(item)
            requests[item["key"]] = variants[arm]
    plan = {"status": "FROZEN_BEFORE_INFERENCE", "model": config.requested_model_identifier,
            "source_plan_sha256": digest, "proposal_file_sha256": study.sha256_file(args.proposal / "plan.json"),
            "samples": samples, "call_order": order, "arms": ARMS, "paid_call_limit": 16,
            "planning_budget_usd": 0.35, "inflight_reserve_usd": 0.05, "concurrency": 2, "retries": 0,
            "spend_policy": "known cost plus 0.05 USD per outstanding new call <= min(0.35, balance minus 0.10); not provider-enforced cap",
            "retained": ["all images", "low reasoning", "4096 max_tokens", "temperature 0", "full ontology", "label boundaries", "system JSON schema", "strict response_format"],
            "removed": ["duplicate causal_frame_ids", "empty prior prediction", "empty track/workflow states", "obsolete explanation of removed fields"],
            "prior_exposure": proposal["prior_exposure"], "selection_uses_gt_values": False,
            "source_sha256": {str(Path(p).relative_to(ROOT)): study.sha256_file(p) for p in
                              (Path(__file__), ROOT / "scripts/resume_h0_frame_strategy_study.py", ROOT / "scripts/prepare_h0_cost_optimization.py")},
            "requests": {key: study.canonical_request_metadata(r).to_mapping() for key, r in requests.items()}}
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    study.atomic_write_json(args.output / "plan.json", plan)
    for key, request in requests.items():
        study.atomic_write_json(args.output / "requests" / f"{key}.json", {
            "payload": thaw_json(request.payload), "metadata": plan["requests"][key]})
    return config, plan, requests


def analyze(args, plan, rows, requests):
    prediction_hash = study.sha256_file(args.output / "predictions.json")
    adapter = study.CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    groups = {}
    for arm in ARMS:
        pooled = {"frames": [], "tasks": {task: {"valid_gt": 0, "scored": 0, "correct": 0, "failed_with_gt": 0} for task in study.TASKS}}
        per_video = {}
        records, native = [], []
        for video in study.VIDEOS:
            chosen = [r for r in rows if r["arm"] == arm and r["video_id"] == video]
            result = study.augment_metrics(study.evaluate_offline(adapter, video, chosen))
            per_video[video] = result
            pooled["frames"].extend({"video_id": video, **r} for r in result["frames"])
            for task in study.TASKS:
                for key in ("valid_gt", "scored", "correct", "failed_with_gt"):
                    pooled["tasks"][task][key] += result["tasks"][task][key]
        for item in plan["call_order"]:
            if item["arm"] != arm:
                continue
            directory = args.output / "calls" / item["key"]
            records.extend(study.read_call_usage(directory))
            path = directory / "http_response.json"
            if path.exists():
                body = json.loads(path.read_text(encoding="utf-8"))["body"]
                for line in body.splitlines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    if event.get("usage"):
                        native.append(event["usage"])
        usage = study.usage_summary(records)
        count = len(native)
        means = {}
        if count:
            means = {"prompt_tokens": statistics.mean(u["prompt_tokens"] for u in native),
                     "completion_tokens": statistics.mean(u["completion_tokens"] for u in native),
                     "reasoning_tokens": statistics.mean(u["completion_tokens_details"]["reasoning_tokens"] for u in native),
                     "visible_tokens": statistics.mean(u["completion_tokens"] - u["completion_tokens_details"]["reasoning_tokens"] for u in native),
                     "cached_tokens": statistics.mean(u.get("prompt_tokens_details", {}).get("cached_tokens", 0) for u in native),
                     "cache_write_tokens": statistics.mean(u.get("prompt_tokens_details", {}).get("cache_write_tokens", 0) for u in native),
                     "provider_cost": statistics.mean(u["cost"] for u in native)}
            means["uncached_price_equivalent_not_actual_cost"] = means["prompt_tokens"] * 0.000002 + means["completion_tokens"] * 0.000006
        groups[arm] = {"evaluation": study.augment_metrics(pooled), "per_video": per_video, "usage": usage,
                       "native_usage_count": count, "native_means": means}
    wire_errors, checked = [], 0
    for item in plan["call_order"]:
        path = args.output / "calls" / item["key"] / "wire_request.json"
        if not path.exists():
            continue
        wire = json.loads(path.read_text(encoding="utf-8"))
        req = requests[item["key"]]
        content = wire["messages"][1]["content"]
        expected = [hashlib.sha256(("data:image/png;base64," + base64.b64encode(i.content).decode()).encode()).hexdigest() for i in req.images]
        if ([part["image_url"]["data_url_sha256"] for part in content[1:]] != expected
                or [part["image_url"]["detail"] for part in content[1:]] != ["low", "low", "high"]
                or wire["response_format"]["json_schema"]["schema"] != schema_for(req.response_schema_version)
                or wire["reasoning"] != {"effort": "low"}
                or wire["model"] != plan["model"]):
            wire_errors.append(item["key"])
        checked += 1
    summary = {"plan_sha256": plan["plan_sha256"], "predictions_sha256_before_scoring": prediction_hash,
               "groups": groups, "successful_predictions": sum(r["status"] == "OK" for r in rows),
               "usage": study.usage_summary([u for item in plan["call_order"] for u in study.read_call_usage(args.output / "calls" / item["key"])]),
               "wire_audit": {"status": "PASS" if not wire_errors else "FAIL", "checked": checked, "errors": wire_errors}}
    if study.sha256_file(args.output / "predictions.json") != prediction_hash:
        raise ValueError("predictions changed during scoring")
    study.atomic_write_json(args.output / "summary.json", summary)
    return summary


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write("single bounded experiment; never repeat dispatches\n")
    secret = study.resolve_api_key(api_key=None, api_key_file=ROOT / "docs/API.txt")
    credit_request = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + secret.reveal()})
    with urllib.request.urlopen(credit_request, timeout=20) as response:
        credits = json.load(response)
    study.atomic_write_json(args.output / "credits_before.json", credits)
    available = credits["data"]["total_credits"] - credits["data"]["total_usage"]
    cap = min(0.35, available - 0.10)
    if cap < 0.25:
        raise ValueError("insufficient balance for bounded paired experiment")
    config, plan, requests = prepare(args)
    study.AuditSender = DurableAuditSender
    args.cache = args.output / "cache"
    budget = study.ProviderCallBudget(16)
    completed, spending, cursor, stop = {}, 0.0, 0, None
    order = plan["call_order"]

    def checkpoint(status):
        study.atomic_write_json(args.output / "predictions.json", [completed[i["key"]] for i in order if i["key"] in completed])
        study.atomic_write_json(args.output / "run_status.json", {"status": status, "completed": len(completed),
            "successful": sum(r["status"] == "OK" for r in completed.values()), "known_cost_usd": round(spending, 8),
            "provider_attempts": budget.used, "reason": stop, "plan_sha256": plan["plan_sha256"]})

    checkpoint("RUNNING")
    with ThreadPoolExecutor(max_workers=2) as executor:
        active = {}
        while cursor < len(order) or active:
            while not stop and cursor < len(order) and len(active) < 2:
                if spending + (len(active) + 1) * 0.05 > cap:
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
                print(json.dumps({"key": item["key"], "status": row["status"], "completed": len(completed), "known_cost_usd": spending, "stop": stop}), flush=True)
    checkpoint("SCORING")
    rows = [completed.get(i["key"], {**i, "status": "NOT_ATTEMPTED", "error": stop}) for i in order]
    study.atomic_write_json(args.output / "predictions.json", rows)
    summary = analyze(args, plan, rows, requests)
    study.atomic_write_json(args.output / "run_status.json", {"status": "STOPPED" if stop else "COMPLETE", "reason": stop,
        "completed": len(completed), "successful": summary["successful_predictions"], "known_cost_usd": spending,
        "provider_attempts": budget.used, "plan_sha256": plan["plan_sha256"]})
    print(json.dumps({"status": "FINISHED", "successful": summary["successful_predictions"], "known_cost_usd": spending}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/h0_frame_strategy_qwen0902_training80_20260905")
    parser.add_argument("--proposal", type=Path, default=ROOT / "artifacts/preflight/h0_cost_two_proposals_20260905")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/h0_conservative_input_qwen0902_20260905")
    parser.add_argument("--dataset", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    run(parser.parse_args())
