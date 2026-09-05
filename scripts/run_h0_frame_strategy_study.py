"""Paired Training-only single/three/six-image H0 study, with frozen requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_five_expert_ablation import BOUNDARY
from scripts.run_pure_h0_smoke import (
    TASKS,
    FinalOnlyRequestBuilder,
    RawResponseAuditTransport,
    evaluate_offline,
)
from scripts.run_raw_adjacent_h0_smoke import AuditSender
from scripts.run_window_density_h0_smoke import augment_metrics, usage_summary
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import schema_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import validate_final_only

VIDEOS = ("VID103", "VID23", "VID31", "VID96")
OFFSETS = {"A": (0,), "B": (-2, -1, 0), "C": (-5, -2, 0), "D": (-5, -4, -3, -2, -1, 0)}
GROUP_NAMES = {"A": "single_current", "B": "three_span2s", "C": "three_span5s", "D": "six_span5s"}


def spread_targets(samples, count):
    """Nearest available target to time-bin centers; never rank by labels."""
    ordered = sorted(samples, key=lambda item: item.target_frame_id)
    if len(ordered) < count or count <= 0:
        raise ValueError("insufficient eligible observations")
    first, last = ordered[0].target_frame_id, ordered[-1].target_frame_id
    chosen = []
    for index in range(count):
        ideal = first + (last - first) * (index + 0.5) / count
        candidates = [s for s in ordered if all(abs(s.target_frame_id - p.target_frame_id) >= 250 for p in chosen)]
        if not candidates:
            raise ValueError("cannot select time-spread targets at least ten seconds apart")
        chosen.append(min(candidates, key=lambda s: (abs(s.target_frame_id - ideal), s.target_frame_id)))
    return sorted(chosen, key=lambda s: s.target_frame_id)


def select_samples(adapter, count):
    selected, metadata = {}, {}
    for video in VIDEOS:
        eligible = []
        for row in adapter.iter_video(video):
            sample = row.inference
            if sample.source_split is not DatasetSplit.TRAINING:
                raise ValueError("study must be Training-only")
            if sample.causal_frame_ids != tuple(sample.target_frame_id + k * 25 for k in OFFSETS["D"]):
                continue
            target = row.frame_supervision
            if row.evaluation is not None and row.evaluation.instance_supervision_available:
                target = aggregate_evaluation_target(row.evaluation, source="h0_frame_study_mask_only")
            if target is not None and target.mask.ivt:
                eligible.append(sample)
        selected[video] = spread_targets(eligible, count)
        metadata[video] = {"eligible_count": len(eligible), "selected_targets": [s.target_frame_id for s in selected[video]]}
    return selected, metadata


def variant_request(base, group, target):
    data = json.loads(base.payload["input_text"])
    original = data["selected_image_frame_ids"]
    ids = tuple(target + 25 * offset for offset in OFFSETS[group])
    by_id = dict(zip(original, base.images, strict=True))
    payload = thaw_json(base.payload)
    payload["openrouter_image_detail_mode"] = "explicit_v1"
    payload["image_details"] = ["low"] * (len(ids) - 1) + ["high"]
    payload["system_text"] += BOUNDARY + (
        "\nImages are ordered from past to present; their true relative seconds are supplied. "
        "Predict ONLY the final target image. Earlier images supply context, not additional output labels. "
        "Do not output the union of actions across the window. All six root keys are required. "
        "Return exactly this JSON schema:\n" + json.dumps(schema_for(base.response_schema_version), sort_keys=True)
    )
    data.update(causal_frame_ids=list(ids), selected_image_frame_ids=list(ids),
                relative_seconds=list(OFFSETS[group]), source_fps=25,
                target_frame_id=target, predict_target_only=True)
    data["temporal_evidence"] = {"schema_version": "fixed_causal_window_v1", "selection_strategy": "fixed_all"}
    payload["input_text"] = json.dumps(data, sort_keys=True, separators=(",", ":"))
    request = replace(base, payload=payload, images=tuple(by_id[f] for f in ids),
                      prompt_version="joint_final_only_frame_strategy_v1",
                      generation_parameters={**thaw_json(base.generation_parameters), "temperature": 0})
    validate_request(request, group, target)
    return request


def validate_request(request, group, target):
    data = json.loads(request.payload["input_text"])
    expected = [target + 25 * value for value in OFFSETS[group]]
    actual = [int(image.identifier.rsplit(":", 1)[1]) for image in request.images]
    if not (actual == data["causal_frame_ids"] == data["selected_image_frame_ids"] == expected):
        raise ValueError("frame/request identities differ")
    if data["relative_seconds"] != list(OFFSETS[group]) or actual[-1] != target:
        raise ValueError("timestamps or target differ")
    if (data["prior_finalized_prediction"] is not None or data["track_summary"]["status"] != "UNAVAILABLE"
            or data["workflow_summary"]["source_max_frame_id"] is not None):
        raise ValueError("H0 study forbids prediction or tracker context")
    if list(request.payload["image_details"]) != ["low"] * (len(actual) - 1) + ["high"]:
        raise ValueError("image detail rules differ")


def prepare(args):
    import torch

    torch.set_num_threads(2)
    config = load_api_config(args.config)
    if config.provider != "openrouter" or not config.data_upload_authorized:
        raise ValueError("requires authorized OpenRouter config")
    if config.requested_model_identifier != "qwen/qwen3.8-max-0902":
        raise ValueError("this frozen study uses one Qwen model")
    model_snapshot = json.loads(args.models_snapshot.read_text(encoding="utf-8"))
    model = next(row for row in model_snapshot["selected_models"] if row["id"] == config.requested_model_identifier)
    if "image" not in model["architecture"]["input_modalities"]:
        raise ValueError("selected model must support images")
    config = replace(config, max_causal_frames=6, max_api_images=6)
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=6)
    samples, selection = select_samples(adapter, args.per_video)
    loader = CausalApiMediaLoader()
    builder = CausalPerceptionContextBuilder(max_frames=6, max_images=6, selection_strategy="fixed_all",
                                            history_image_detail="low", target_image_detail="high")
    requests, order, image_sources = {}, [], {}
    for index in range(args.per_video):
        for video_index, video in enumerate(VIDEOS):
            sample = samples[video][index]
            loaded = loader.load(sample)
            context = builder.build(loaded.runtime_sample, loaded.frames, workflow_snapshot={},
                                    memory_snapshot={}, prior_finalized_prediction=None)
            base = FinalOnlyRequestBuilder(config=config).build(context)
            for path in sample.media_refs:
                image_sources[path] = sha256_file(path)
            rotation = (index + video_index) % 4
            groups = list(OFFSETS)
            for group in groups[rotation:] + groups[:rotation]:
                key = f"{video}_{sample.target_frame_id}_{group}"
                requests[key] = variant_request(base, group, sample.target_frame_id)
                order.append({"key": key, "video_id": video, "frame_id": sample.target_frame_id, "group": group})
        print(json.dumps({"status": "PREPARING", "targets_prepared": (index + 1) * len(VIDEOS)}), flush=True)
    sources = [Path(__file__), ROOT / "scripts/run_pure_h0_smoke.py", ROOT / "scripts/run_raw_adjacent_h0_smoke.py",
               ROOT / "scripts/run_window_density_h0_smoke.py", ROOT / "scripts/run_five_expert_ablation.py",
               ROOT / "src/surgical_agent/api/providers/openrouter.py", ROOT / "src/surgical_agent/data/dataset.py",
               ROOT / "src/surgical_agent/evaluation/frame_ground_truth.py"]
    plan = {"schema_version": "h0_frame_strategy_study_v1", "model": config.requested_model_identifier,
            "source_split": "Training", "selection": selection, "offsets_seconds": OFFSETS,
            "selection_policy": "nearest time-bin centers, >=10s apart; six real contiguous images; target IVT-valid mask",
            "label_access_policy": "annotation values are parsed to derive validity masks, never used to rank samples or in requests; scoring after all predictions",
            "inference_protocol_target_step_raw_frames": 25, "pilot_targets_time_spread": True,
            "tracker": False, "gate": False, "repair": False, "memory": False,
            "provider_call_limit": len(order), "retries": 0, "planning_budget_usd": args.budget,
            "per_inflight_call_reserve_usd": 0.15, "concurrency": args.concurrency,
            "spend_policy": "reported spend plus 0.15 reserve per outstanding call; stop on unpriced/unaccounted calls; not provider-enforced dollar cap",
            "generation": thaw_json(next(iter(requests.values())).generation_parameters),
            "model_pricing_snapshot": model["pricing"], "model_snapshot_sha256": sha256_file(args.models_snapshot),
            "config_sha256": sha256_file(args.config), "repair_manifest_sha256": sha256_file(args.dataset / "repair_manifest.json"),
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            "source_png_sha256": image_sources, "call_order": order,
            "requests": {key: canonical_request_metadata(req).to_mapping() for key, req in requests.items()}}
    plan = json.loads(json.dumps(plan))
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    plan_path = args.output / "plan.json"
    if args.output.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("output differs from frozen plan; use a new directory")
        if (args.output / "run_status.json").exists():
            raise ValueError("execution already started; preserve it, do not repeat calls")
    else:
        args.output.mkdir(parents=True)
        atomic_write_json(plan_path, plan)
    for key, request in requests.items():
        path = args.output / "requests" / f"{key}.json"
        if not path.exists():
            atomic_write_json(path, {"payload": thaw_json(request.payload), "metadata": plan["requests"][key]})
    print(json.dumps({"status": "PREPARED", "calls": len(order), "plan_sha256": plan["plan_sha256"]}), flush=True)
    return config, plan, requests


def read_call_usage(directory):
    path = directory / "api_usage.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def call_one(item, request, config, args, secret, budget):
    directory = args.output / "calls" / item["key"]
    transport = OpenRouterTransport(api_key=secret, endpoint_identifier=config.endpoint_identifier,
                                   timeout_seconds=180, sender=AuditSender(directory, secret))
    client = CachedMultimodalApiClient(
        transport=RawResponseAuditTransport(CompleteAccountingTransport(transport), directory),
        cache=FileApiCache(args.cache), usage=UsageLedger(directory / "api_usage.jsonl"),
        validator=validate_final_only, retry_policy=RetryPolicy(max_attempts=1), provider_call_budget=budget)
    row = {**item, "request_hash": canonical_request_metadata(request).request_hash,
           "causal_frame_ids": [item["frame_id"] + 25 * offset for offset in OFFSETS[item["group"]]]}
    try:
        response = client.call(request)
        value = thaw_json(response.parsed_payload)
        row.update(status="OK", selected_ids={task: [value[task]["selected_id"]] if task == "phase"
                   else value[task]["selected_ids"] for task in TASKS}, returned_model=response.returned_model_identifier,
                   cache_hit=response.cache_hit)
    except ApiError as exc:
        cause = getattr(exc, "cause", exc)
        row.update(status="API_FAILURE", error=cause.code, http_status=getattr(cause, "status_code", None))
    except Exception as exc:  # noqa: BLE001 - persist failure and stop future paid calls
        row.update(status="INTERNAL_FAILURE", error=type(exc).__name__)
    usage = read_call_usage(directory)
    row["usage"] = usage_summary(usage)
    if not usage:
        row["accounting_error"] = "UNACCOUNTED_CALL"
    return row


def score(args, plan, rows):
    digest = sha256_file(args.output / "predictions.json")
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=6)
    groups = {}
    for group in OFFSETS:
        per_video = {}
        pooled = {"frames": [], "tasks": {task: {"valid_gt": 0, "scored": 0, "correct": 0, "failed_with_gt": 0} for task in TASKS}}
        for video in VIDEOS:
            predictions = [r for r in rows if r["video_id"] == video and r["group"] == group]
            result = augment_metrics(evaluate_offline(adapter, video, predictions))
            per_video[video] = result
            for frame in result["frames"]:
                pooled["frames"].append({"video_id": video, **frame})
            for task in TASKS:
                for key in ("valid_gt", "scored", "correct", "failed_with_gt"):
                    pooled["tasks"][task][key] += result["tasks"][task][key]
        pooled = augment_metrics(pooled)
        groups[group] = {"name": GROUP_NAMES[group], "evaluation": pooled, "per_video": per_video,
                         "video_mean_f1": {task: statistics.mean(per_video[v]["tasks"][task]["micro_f1"] for v in VIDEOS) for task in TASKS}}
    assert sha256_file(args.output / "predictions.json") == digest
    usage = [record for item in plan["call_order"] for record in read_call_usage(args.output / "calls" / item["key"])]
    for group, value in groups.items():
        group_usage = [record for item in plan["call_order"] if item["group"] == group
                       for record in read_call_usage(args.output / "calls" / item["key"])]
        value["usage"] = usage_summary(group_usage)
    summary = {"schema_version": "h0_frame_strategy_result_v1", "plan_sha256": plan["plan_sha256"],
               "predictions_sha256_before_scoring": digest, "groups": groups, "usage": usage_summary(usage),
               "successful_predictions": sum(r["status"] == "OK" for r in rows), "scheduled_predictions": len(rows)}
    atomic_write_json(args.output / "summary.json", summary)
    return summary


def run(args):
    if not 1 <= args.per_video <= 20 or not 1 <= args.concurrency <= 8 or not 0 < args.budget <= 10:
        raise ValueError("exceeds bounded study size or budget")
    config, plan, requests = prepare(args)
    if not args.execute:
        return
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    budget = ProviderCallBudget(plan["provider_call_limit"])
    atomic_write_json(args.output / "run_status.json", {"status": "RUNNING", "plan_sha256": plan["plan_sha256"]})
    completed, spending, cursor, stop = {}, 0.0, 0, None
    order = plan["call_order"]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        active = {}
        while cursor < len(order) or active:
            # First four requests validate one complete paired target before concurrency grows.
            capacity = 1 if len(completed) < 4 else args.concurrency
            while not stop and cursor < len(order) and len(active) < capacity:
                if spending + (len(active) + 1) * 0.15 > args.budget:
                    if not active:
                        stop = "PLANNING_SPEND_LIMIT"
                    break
                item = order[cursor]
                active[executor.submit(call_one, item, requests[item["key"]], config, args, secret, budget)] = item
                cursor += 1
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                print(json.dumps({"status": "WAITING", "completed": len(completed), "inflight": len(active), "cost_usd": round(spending, 6)}), flush=True)
            for future in done:
                item = active.pop(future)
                row = future.result()
                completed[item["key"]] = row
                spending += row["usage"]["reported_cost_usd"]
                if row.get("accounting_error") or row["usage"]["unpriced_calls"]:
                    stop = row.get("accounting_error", "UNPRICED_CALL")
                if row.get("http_status") in {400, 401, 402, 403, 404} or row["status"] == "INTERNAL_FAILURE":
                    stop = row.get("error", "INTERNAL_FAILURE")
                if row["status"] == "OK" and row["returned_model"] != config.requested_model_identifier:
                    stop = "RETURNED_MODEL_MISMATCH"
                atomic_write_json(args.output / "predictions.json", [completed[x["key"]] for x in order if x["key"] in completed])
                print(json.dumps({"status": row["status"], "key": item["key"], "completed": len(completed),
                                  "cost_usd": round(spending, 6), "stop": stop}), flush=True)
    rows = [completed.get(item["key"], {**item, "status": "NOT_ATTEMPTED", "error": stop}) for item in order]
    atomic_write_json(args.output / "predictions.json", rows)
    summary = score(args, plan, rows)
    atomic_write_json(args.output / "run_status.json", {"status": "COMPLETE" if not stop else "STOPPED",
                      "reason": stop, "provider_attempts": budget.used, "plan_sha256": plan["plan_sha256"]})
    print(json.dumps({"status": "FINISHED", "successful": summary["successful_predictions"], "usage": summary["usage"]}), flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, default=Path("D:/cholec_dataset"))
    value.add_argument("--config", type=Path, default=ROOT / "configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    value.add_argument("--models-snapshot", type=Path, default=ROOT / "artifacts/preflight/h0_frame_openrouter_models_20260905.json")
    value.add_argument("--api-key-file", type=Path, default=ROOT / "docs/API.txt")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--cache", type=Path, required=True)
    value.add_argument("--per-video", type=int, default=20)
    value.add_argument("--concurrency", type=int, default=8)
    value.add_argument("--budget", type=float, default=10.0)
    value.add_argument("--execute", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
