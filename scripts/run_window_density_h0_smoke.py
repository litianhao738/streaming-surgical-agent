"""Frozen 3-adjacent / 3-spread / 25-dense pure joint H0 diagnostic.

Default prepares inputs only. --execute uses the identical prepared plan, with
at most 18 provider attempts and no retries, tracking, repair, or memory. Only
annotation frame keys are inspected before inference; labels are evaluated after
all prediction rows, including failures, have been persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_five_expert_ablation import BOUNDARY
from scripts.run_pure_h0_smoke import (
    FinalOnlyRequestBuilder,
    RawResponseAuditTransport,
    TASKS,
    evaluate_offline,
)
from scripts.run_raw_adjacent_h0_smoke import AuditSender
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiImageInput, canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiContractError, ApiError
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import schema_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter, MP4_ALIGNMENT_VERSION
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import (
    CausalPerceptionContextBuilder,
    encode_rgb_png,
    require_gold_free,
)
from surgical_agent.perception.final_only import validate_final_only

GROUPS = ("A", "B", "C")
ORDERS = (("C", "A", "B"), ("A", "B", "C"), ("B", "C", "A"))
PROMPT_VERSION = "window_density_joint_h0_v1"


class WindowDensityTransport(OpenRouterTransport):
    """Experimental 25-image allowance; the production transport is unchanged."""

    def _validate_request(self, request):
        if request.prompt_version != PROMPT_VERSION or len(request.images) not in {3, 25}:
            raise ApiContractError("This transport accepts only frozen window-density requests")
        data = json.loads(request.payload["input_text"])
        assert_request_contract(request, tuple(data["causal_frame_ids"]), data["target_frame_id"])
        # Reuse all ordinary provider/endpoint/text/routing checks on a legal
        # validation view. send() still serializes the ORIGINAL complete request.
        super()._validate_request(replace(request, images=request.images[-3:]))


def frame_ids(group, target):
    if target < 25:
        raise ValueError("A complete 25-frame causal history is required")
    if group == "A":
        return (target - 2, target - 1, target)
    if group == "B":
        return (target - 24, target - 12, target)
    if group == "C":
        return tuple(range(target - 24, target + 1))
    raise ValueError("Unknown experimental group")


def read_annotation_index(annotation, targets):
    """Inspect keys only; never inspect, select on, or return label values."""
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    keys = sorted(int(key) for key in raw["annotations"])
    del raw
    if not set(targets).issubset(keys):
        raise ValueError("All frozen targets must have exact annotation keys")
    selected = [key for key in keys if targets[0] <= key <= targets[-1]]
    if selected != targets:
        raise ValueError("Frozen targets must be consecutive annotation timestamps")
    return {
        "annotation_sha256": sha256_file(annotation),
        "inspected_content": "annotations object keys only; no label values used",
        "exact_target_keys_present": targets,
        "annotation_key_count": len(keys),
    }


def decode_contiguous_images(video, video_id, first, last):
    """One seek followed by consecutive reads; one-based ID maps to index ID-1."""
    import cv2
    import numpy as np
    import torch

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Raw video unavailable")
        media = {
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        if media["fps"] != 25 or not 1 <= first <= last <= media["frame_count"]:
            raise ValueError("Expected 25 fps video and valid raw frame bounds")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first - 1):
            raise ValueError("Decoder seek failed")
        images, base_tensors = {}, {}
        for fid in range(first, last + 1):
            before = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not math.isfinite(before) or abs(before - (fid - 1)) > 0.5:
                raise ValueError("Decoder position before read is not exact")
            ok, bgr = capture.read()
            after = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not ok or bgr is None or not math.isfinite(after) or abs(after - fid) > 0.5:
                raise ValueError("Decoder read or position after read is not exact")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            images[fid] = ApiImageInput(
                f"cholectrack20:{video_id}:frame:{fid}", "image/png", encode_rgb_png(tensor)
            )
            # The first target's legal three-image context supplies the unchanged
            # production ontology/prompt; all experimental image contracts below
            # are independently rebuilt and checked for each group and target.
            if fid in (first + 22, first + 23, first + 24):
                base_tensors[fid] = tensor
        return images, base_tensors, media
    finally:
        capture.release()


def build_request(base, *, images, group, target, fps):
    ids = frame_ids(group, target)
    payload = thaw_json(base.payload)
    payload["openrouter_image_detail_mode"] = "explicit_v1"
    payload["image_details"] = ["low"] * (len(ids) - 1) + ["high"]
    payload["system_text"] += BOUNDARY + (
        "\nImages are ordered from oldest to newest. Predict only the last, target frame. "
        "Use earlier images only as causal motion evidence ending at that target; "
        "do not union labels across the window. "
        "All six root keys are mandatory: schema_version, instrument, verb, target, ivt, phase. "
        "Do not stop after instrument. The exact output schema is:\n"
        + json.dumps(schema_for(base.response_schema_version), separators=(",", ":"))
    )
    data = json.loads(payload["input_text"])
    data["target_frame_id"] = target
    data["causal_frame_ids"] = list(ids)
    data["selected_image_frame_ids"] = list(ids)
    data["temporal_evidence"] = {
        "schema_version": "fixed_causal_window_v1", "selection_strategy": "fixed_all",
        "candidate_frame_ids": list(ids), "selected_image_frame_ids": list(ids),
        "omitted_image_frame_ids": [], "window_frame_count": len(ids),
        "uploaded_image_count": len(ids),
    }
    data["raw_video_sampling"] = {
        "fps": fps, "frame_id_step": ids[1] - ids[0],
        "window_span_seconds": (ids[-1] - ids[0]) / fps,
        "relative_to_target_seconds": [(fid - target) / fps for fid in ids],
        "source_frame_numbering": "one_based",
    }
    require_gold_free(data)
    payload["input_text"] = json.dumps(data, sort_keys=True, separators=(",", ":"))
    request = replace(base, payload=payload, images=tuple(images[fid] for fid in ids),
                      prompt_version=PROMPT_VERSION)
    assert_request_contract(request, ids, target)
    return request


def assert_request_contract(request, ids, target):
    data = json.loads(request.payload["input_text"])
    actual_ids = tuple(int(image.identifier.rsplit(":", 1)[1]) for image in request.images)
    if not (actual_ids == tuple(ids) == tuple(data["causal_frame_ids"])
            == tuple(data["selected_image_frame_ids"])):
        raise ValueError("Images and all temporal frame IDs must agree")
    if tuple(sorted(set(ids))) != tuple(ids) or ids[-1] != target or min(ids) < 1:
        raise ValueError("Experimental images must be unique, ordered, and causal")
    if data["target_frame_id"] != target or len(request.images) not in {3, 25}:
        raise ValueError("Experimental target or image count mismatch")
    details = tuple(request.payload["image_details"])
    if details != ("low",) * (len(ids) - 1) + ("high",):
        raise ValueError("All historical images must be low detail and target high")
    evidence = data["temporal_evidence"]
    if (evidence["candidate_frame_ids"] != list(ids)
            or evidence["selected_image_frame_ids"] != list(ids)
            or evidence["window_frame_count"] != len(ids)
            or evidence["uploaded_image_count"] != len(ids)):
        raise ValueError("Temporal evidence does not match uploaded images")
    if (data["prior_finalized_prediction"] is not None
            or data["workflow_summary"]["source_max_frame_id"] is not None
            or data["track_summary"]["status"] != "UNAVAILABLE"):
        raise ValueError("This experiment has no prediction history or tracker")


def augment_metrics(evaluation):
    """Keep every valid GT row; failures are misses, never denominator removal."""
    for task, metrics in evaluation["tasks"].items():
        tp = fp = fn = 0
        for row in evaluation["frames"]:
            if not row["mask"][task]:
                continue
            truth = set(row["gt"][task])
            predicted = set(row["h0"][task]) if row["status"] == "OK" else set()
            tp += len(truth & predicted)
            fp += len(predicted - truth)
            fn += len(truth - predicted)
        valid = metrics["valid_gt"]
        metrics.update(
            exact_accuracy_all_valid_gt=metrics["correct"] / valid if valid else None,
            response_coverage=metrics["scored"] / valid if valid else None,
            tp=tp, fp=fp, fn=fn,
            micro_precision=tp / (tp + fp) if tp + fp else (0.0 if valid else None),
            micro_recall=tp / (tp + fn) if tp + fn else (0.0 if valid else None),
            micro_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0.0 if valid else None),
        )
    evaluation["metric"] = "masked_set_exact_and_micro_PRF_NOT_mAP"
    evaluation["failure_policy"] = (
        "Every valid GT remains in exact denominator; failed/not-attempted rows receive "
        "no exact credit and empty predictions for TP/FP/FN. Missing task GT is masked."
    )
    evaluation["ivt_policy"] = "All selected/annotated IVT IDs, including null 94-99; not official non-null mAP"
    return evaluation


def read_usage(directory):
    return [json.loads(line) for path in directory.glob("*/api_usage.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usage_summary(records):
    latencies = [row["total_latency_ms"] for row in records
                 if row.get("total_latency_ms") is not None and row["provider_call_count"]]
    return {
        "provider_calls": sum(row["provider_call_count"] for row in records),
        "cache_hits": sum(row["cache_hit"] for row in records),
        "reported_cost_usd": round(sum(row.get("provider_cost") or 0 for row in records), 6),
        "unpriced_calls": sum(row["provider_call_count"] for row in records if row.get("provider_cost") is None),
        "latency_ms": {"mean": statistics.mean(latencies) if latencies else None,
                       "min": min(latencies) if latencies else None,
                       "max": max(latencies) if latencies else None},
    }


def prepare(args):
    import torch

    if args.start < 25 or not 1 <= args.count <= 6:
        raise ValueError("Require start >= 25 and 1..6 target timestamps")
    config = load_api_config(args.config)
    if (config.provider != "openrouter"
            or config.requested_model_identifier != "qwen/qwen3.8-max-0902"
            or not config.data_upload_authorized
            or config.provider_options.get("initial_prompt_profile") != "fixed_visual_only"):
        raise ValueError("Requires the authorized fixed-visual OpenRouter Qwen baseline")
    video = args.dataset / "Testing" / args.video / (args.video.lower() + ".mp4")
    targets = [args.start + 25 * index for index in range(args.count)]
    annotation_index = read_annotation_index(video.with_suffix(".json"), targets)
    images, tensors, media = decode_contiguous_images(video, args.video, args.start - 24, targets[-1])
    initial_ids = frame_ids("A", args.start)
    sample = InferenceSample(args.video, args.start, initial_ids,
        tuple(images[fid].identifier for fid in initial_ids), DatasetSplit.TESTING, MP4_ALIGNMENT_VERSION)
    context = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="high").build(
        sample, torch.stack([tensors[fid] for fid in initial_ids]), workflow_snapshot={},
        memory_snapshot={}, prior_finalized_prediction=None)
    base = FinalOnlyRequestBuilder(config=config).build(context)
    requests = {}
    order = []
    for index, target in enumerate(targets):
        for group in ORDERS[index % len(ORDERS)]:
            key = f"{target}_{group}"
            requests[key] = build_request(base, images=images, group=group, target=target, fps=media["fps"])
            order.append({"key": key, "group": group, "frame_id": target})
    dependencies = [Path(__file__), ROOT / "scripts/run_pure_h0_smoke.py",
        ROOT / "scripts/run_raw_adjacent_h0_smoke.py", ROOT / "scripts/run_five_expert_ablation.py",
        ROOT / "src/surgical_agent/perception/context_builder.py",
        ROOT / "src/surgical_agent/perception/joint_api_vlm.py",
        ROOT / "src/surgical_agent/api/providers/openrouter.py"]
    plan = {
        "profile": "pure_joint_h0_window_density_ABC", "video_id": args.video,
        "source_split": "Testing", "purpose": "diagnostic_not_sealed_test_result",
        "targets": targets, "target_frame_step": 25, "media": media,
        "video_path": str(video.resolve()), "video_sha256": sha256_file(video),
        "source_frame_numbering": "one_based; decoder_index=frame_id-1",
        "decoder": "single seek followed by consecutive raw reads; checked before/after positions",
        "annotation_index_precheck": annotation_index,
        "model": config.requested_model_identifier, "config_sha256": sha256_file(args.config),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in dependencies},
        "groups": {"A": "3 adjacent: [t-2,t-1,t]", "B": "3 spread: [t-24,t-12,t]",
                   "C": "25 dense: every raw frame [t-24,...,t]"},
        "call_order": order, "order_policy": "C,A,B / A,B,C / B,C,A repeated per target",
        "provider_call_limit": len(order), "retries": 0, "planning_budget_usd": 1.50,
        "per_call_spend_reserve_usd": 0.15,
        "spend_policy": "Stop before next call if reported spend + reserve exceeds budget; reserve is not a provider-enforced price cap. Stop on unpriced or unaccounted call.",
        "tracker": False, "gate": False, "verifier": False, "repair": False, "memory": False,
        "experimental_contract": "Extend only this script's ApiRequest to 25 images; formal max6/fixed3 unchanged",
        "evaluation": "Labels read only after all predictions persist; exact target GT with per-task masks; no interpolation; failures retained",
        "requests": {key: canonical_request_metadata(req).to_mapping() for key, req in requests.items()},
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    plan_path = args.output / "plan.json"
    if args.output.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing output differs from this frozen plan")
        if (args.output / "run_status.json").exists():
            raise ValueError("Execution already started; preserve outputs and use a new directory")
    else:
        args.output.mkdir(parents=True)
        atomic_write_json(plan_path, plan)
    for key, request in requests.items():
        record = {"payload": thaw_json(request.payload), "metadata": plan["requests"][key]}
        path = args.output / "requests" / f"{key}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Prepared request changed")
        if not path.exists():
            atomic_write_json(path, record)
    (args.output / "images").mkdir(exist_ok=True)
    for fid, image in images.items():
        path = args.output / "images" / f"{fid}.png"
        if path.exists() and path.read_bytes() != image.content:
            raise ValueError("Prepared image changed")
        if not path.exists():
            path.write_bytes(image.content)
    print(json.dumps({"status": "PREPARED", "plan_sha256": plan["plan_sha256"],
        "targets": targets, "calls": len(order), "unique_raw_images": len(images), "media": media}), flush=True)
    return config, plan, requests


def run(args):
    config, plan, requests = prepare(args)
    if not args.execute:
        return
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    atomic_write_json(args.output / "run_status.json", {"status": "RUNNING", "plan_sha256": plan["plan_sha256"]})
    budget = ProviderCallBudget(plan["provider_call_limit"])
    predictions, spending, stopped = [], 0.0, None
    for item in plan["call_order"]:
        key, group, fid = item["key"], item["group"], item["frame_id"]
        request = requests[key]
        row = {"video_id": args.video, "frame_id": fid, "group": group,
               "causal_frame_ids": list(frame_ids(group, fid)),
               "request_hash": plan["requests"][key]["request_hash"]}
        if spending + plan["per_call_spend_reserve_usd"] > plan["planning_budget_usd"]:
            stopped = "PLANNING_SPEND_LIMIT"
        if stopped:
            row.update(status="NOT_ATTEMPTED", error=stopped)
        else:
            directory = args.output / "calls" / key
            transport = WindowDensityTransport(api_key=secret, endpoint_identifier=config.endpoint_identifier,
                timeout_seconds=180, sender=AuditSender(directory, secret))
            client = CachedMultimodalApiClient(
                transport=RawResponseAuditTransport(CompleteAccountingTransport(transport), directory),
                cache=FileApiCache(args.cache), usage=UsageLedger(directory / "api_usage.jsonl"),
                validator=validate_final_only, retry_policy=RetryPolicy(max_attempts=1), provider_call_budget=budget)
            previous_used = budget.used
            try:
                response = client.call(request)
                value = thaw_json(response.parsed_payload)
                row.update(status="OK", selected_ids={task: [value[task]["selected_id"]]
                    if task == "phase" else value[task]["selected_ids"] for task in TASKS},
                    returned_model=response.returned_model_identifier, cache_hit=response.cache_hit)
            except ApiError as exc:
                cause = getattr(exc, "cause", exc)
                row.update(status="API_FAILURE", error=cause.code,
                           http_status=getattr(cause, "status_code", None))
                if row["http_status"] in {400, 401, 402, 403}:
                    stopped = cause.code
            except Exception as exc:
                row.update(status="INTERNAL_FAILURE", error=type(exc).__name__)
                stopped = "INTERNAL_FAILURE"
            records = read_usage(args.output / "calls")
            spending = sum(record.get("provider_cost") or 0 for record in records)
            if any(record["provider_call_count"] and record.get("provider_cost") is None for record in records):
                stopped = "UNPRICED_CALL"
            if budget.used > previous_used and not (directory / "api_usage.jsonl").exists():
                stopped = "UNACCOUNTED_CALL"
        predictions.append(row)
        atomic_write_json(args.output / "predictions.json", predictions)
        print(json.dumps(row), flush=True)
    # The annotation index was inspected during preparation, but no label values
    # entered sample selection, requests, or model calls. Labels are read here.
    predictions_sha = sha256_file(args.output / "predictions.json")
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    evaluations = {}
    for group in GROUPS:
        group_rows = [row for row in predictions if row["group"] == group]
        evaluation = augment_metrics(evaluate_offline(adapter, args.video, group_rows))
        evaluations[group] = evaluation
        by_frame = {row["frame_id"]: row for row in evaluation["frames"]}
        for row in group_rows:
            row["evaluation"] = by_frame[row["frame_id"]]
    atomic_write_json(args.output / "predictions_with_evaluation.json", predictions)
    usage = read_usage(args.output / "calls")
    summary = {
        "profile": plan["profile"], "plan_sha256": plan["plan_sha256"],
        "targets": plan["targets"], "predictions_sha256_before_gt": predictions_sha,
        "successful_predictions": sum(row["status"] == "OK" for row in predictions),
        "scheduled_predictions": len(predictions), "stopped_reason": stopped,
        "actual_provider_attempts": budget.used, **usage_summary(usage),
        "groups": {group: {
            "successful_predictions": sum(row["status"] == "OK" and row["group"] == group for row in predictions),
            **usage_summary([record for item in plan["call_order"] if item["group"] == group
                for record in usage
                if record.get("request_hash") == plan["requests"][item["key"]]["request_hash"]]),
            "evaluation": evaluations[group],
        } for group in GROUPS},
    }
    atomic_write_json(args.output / "summary.json", summary)
    atomic_write_json(args.output / "run_status.json", {
        "status": "COMPLETE" if summary["successful_predictions"] == len(predictions) else "COMPLETE_WITH_FAILURES",
        "successful_predictions": summary["successful_predictions"], "plan_sha256": plan["plan_sha256"],
        "stopped_reason": stopped,
    })
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "config", "api-key-file", "cache", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--video", default="VID06")
    parser.add_argument("--start", type=int, default=30001)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
