"""Synchronous Gemini H0 -> grounded Verify/Repair with paired offline scoring.

The existing Qwen H0 and historical runners are unchanged. Preflight freezes a
small Training-only selection; --execute authorizes its bounded provider calls.
"""

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.providers.openrouter import (
    HttpResponse,
    OpenRouterTransport,
    _decode_sse_response,
    _sse_line_has_content,
)
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import validator_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.main_h0 import load_main_h0_prompt
from surgical_agent.research.verification.grounded_pipeline import run_grounded_target

MODEL = "google/gemini-3.8-flash"
VIDEOS = ("VID02", "VID04", "VID11", "VID17")
TASK_ATTRS = {"instrument": "instrument_ids", "verb": "verb_ids", "target": "target_ids",
              "ivt": "triplet_ids", "phase": "phase_id"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _videos(args=None, videos=None):
    selected = tuple(videos if videos is not None else getattr(args, "videos", VIDEOS))
    if (not selected or len(set(selected)) != len(selected)
            or any(not isinstance(video, str) or not video for video in selected)):
        raise ValueError("provide one or more distinct video identifiers")
    return selected


def _evaluation_target(resolved):
    target = resolved.frame_supervision
    if (resolved.evaluation is not None and
            resolved.evaluation.instance_supervision_available):
        target = aggregate_evaluation_target(resolved.evaluation, source="grounded_e2e_offline")
    return target


def _task_masks(target):
    return {task: target is not None and getattr(target.mask, task) for task in TASK_ATTRS}


def prepare(args, *, videos=None):
    """Freeze timestamp-selected windows, then inspect their task availability."""
    import torch

    torch.set_num_threads(2)
    videos = _videos(args, videos)
    require_all_task_gt = getattr(args, "require_all_task_gt", False)
    if (not math.isfinite(args.budget_usd) or not math.isfinite(args.reserve_usd)
            or not 0 < args.reserve_usd <= args.budget_usd):
        raise ValueError("positive finite budget and per-call reserve required")
    if (args.output / "execution.lock").exists():
        raise ValueError("already dispatched; use saved results, never repeat this run")
    config_path = ROOT / "configs/perception/joint_openrouter_h0.yaml"
    config = load_api_config(config_path)
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    context_builder = CausalPerceptionContextBuilder(
        max_frames=3, max_images=3, selection_strategy="fixed_all",
        history_image_detail="low", target_image_detail="high",
    )
    loader, builder = CausalApiMediaLoader(), JointPerceptionRequestBuilder(config=config)
    requests, selection = {}, []
    for video in videos:
        available = list(adapter.iter_inference_video(video))
        if len(available) < 3:
            raise ValueError(f"insufficient Training timeline: {video}")
        for index in (len(available) // 3, 2 * len(available) // 3):
            sample = available[index]
            if sample.source_split is not DatasetSplit.TRAINING:
                raise ValueError("all selected videos must belong to Training")
            loaded = loader.load(sample)
            context = context_builder.build(
                loaded.runtime_sample, loaded.frames, workflow_snapshot={},
                memory_snapshot={}, prior_finalized_prediction=None,
            )
            base = builder.build(context)
            payload = thaw_json(base.payload)
            payload["openrouter_routing_profile"] = "strict_google_ai_studio"
            if payload["system_text"] != load_main_h0_prompt():
                raise ValueError("frozen initial prompt drift")
            request = replace(
                base, model_identifier=MODEL,
                prompt_version="joint_perception_main_h0_gemini_grounded_v1", payload=payload,
            )
            key = f"{video}_{sample.target_frame_id}"
            requests[key] = request
            selection.append({
                "key": key, "video_id": video, "frame_id": sample.target_frame_id,
                "source_split": "Training", "causal_frame_ids": list(sample.causal_frame_ids),
                "proposal_slot": "FIRST" if len(selection) % 2 == 0 else "SECOND",
                "request_metadata": canonical_request_metadata(request).to_mapping(),
                "source_images": {str(p): _hash(p) for p in sample.media_refs},
            })
    # Timestamp selection is complete before annotation availability is read.
    # Missing masks can fail this run, never move its targets or affect a request.
    masks_by_key = {}
    for video in videos:
        frame_ids = [item["frame_id"] for item in selection if item["video_id"] == video]
        for resolved in adapter.iter_video(video, frame_ids=frame_ids):
            key = f"{video}_{resolved.inference.target_frame_id}"
            if key in masks_by_key:
                raise ValueError("duplicate target in task-availability check")
            masks_by_key[key] = _task_masks(_evaluation_target(resolved))
    if set(masks_by_key) != set(requests):
        raise ValueError("GT availability adapter did not return every frozen target")
    for item in selection:
        item["task_masks"] = masks_by_key[item["key"]]
    missing = {key: [task for task, valid in masks.items() if not valid]
               for key, masks in masks_by_key.items() if not all(masks.values())}
    if require_all_task_gt and missing:
        raise ValueError("fixed targets lack required task GT; no calls dispatched: "
                         + json.dumps(missing, sort_keys=True))
    pricing_snapshot = _read(args.pricing_snapshot)
    if pricing_snapshot["data"]["id"] != MODEL:
        raise ValueError("pricing snapshot model mismatch")
    endpoint = next(e for e in pricing_snapshot["data"]["endpoints"]
                    if e["tag"] == "google-ai-studio")
    sources = [Path(__file__), config_path,
               ROOT / "src/surgical_agent/perception/main_h0.py",
               ROOT / "src/surgical_agent/perception/prompts/perception_prompt_main_h0.txt",
               ROOT / "src/surgical_agent/research/verification/grounded_pipeline.py",
               ROOT / "src/surgical_agent/research/verification/grounded_repair.py",
               ROOT / "src/surgical_agent/research/verification/final_only_grounded.py",
               ROOT / "src/surgical_agent/evaluation/repair_comparison.py"]
    plan = {
        "schema_version": "gemini_grounded_e2e_plan_v1", "model": MODEL,
        "routing_profile": "strict_google_ai_studio", "source_split": "Training",
        "videos": list(videos),
        "selection": selection, "target_count": len(selection),
        "max_provider_calls": 4 * len(selection), "max_calls_per_target": 4,
        "budget_usd": args.budget_usd, "per_call_reserve_usd": args.reserve_usd,
        "budget_is_provider_enforced": False, "retries": 0,
        "protocol": "one frozen H0, blind locator, IVT proposal, optional single contrast review",
        "selection_rule": "available timeline indices floor(n/3), floor(2n/3); fixed before task-mask validation",
        "gt_availability_check": {
            "performed": True, "require_all_tasks": require_all_task_gt,
            "label_values_used_for_selection_or_requests": False,
            "resampling_after_mask_check": False,
            "fully_annotated_targets": len(selection) - len(missing),
            "valid_targets_by_task": {task: sum(masks[task] for masks in masks_by_key.values())
                                      for task in TASK_ATTRS},
        },
        "phase_policy": "keep H0", "tracker": False, "gate": False, "cross_window_memory": False,
        "system_prompt_sha256": hashlib.sha256(load_main_h0_prompt().encode()).hexdigest(),
        "pricing": endpoint["pricing"], "pricing_snapshot_sha256": _hash(args.pricing_snapshot),
        "source_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in sources},
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    if (args.output / "plan.json").exists():
        if _read(args.output / "plan.json") != plan:
            raise ValueError("preflight plan changed; use a new run directory")
    elif args.output.exists():
        raise ValueError("fresh run directory required")
    else:
        args.output.mkdir(parents=True)
        atomic_write_json(args.output / "plan.json", plan)
        atomic_write_json(args.output / "endpoints.json", pricing_snapshot)
        for key, request in requests.items():
            atomic_write_json(args.output / "requests" / f"{key}.json", {
                "metadata": canonical_request_metadata(request).to_mapping(),
                "payload": thaw_json(request.payload),
            })
    print(json.dumps({"status": "PREFLIGHT_PASSED", "model": MODEL,
                      "targets": len(selection), "max_provider_calls": plan["max_provider_calls"],
                      "budget_usd": args.budget_usd, "provider_calls": 0}), flush=True)
    return adapter, plan, requests


class AuditSender:
    """Persist safe wire metadata and raw responses before provider parsing."""

    def __init__(self, directory, secret, sender=None):
        self.directory, self.secret, self.sender = directory, secret, sender
        self.native = None

    def _durable_send(self, url, headers, body, timeout):
        started = perf_counter()
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                chunks, first_content, generation = [], None, None
                with (self.directory / "response_stream.sse").open("xb") as stream:
                    for line in response:
                        chunks.append(line)
                        stream.write(line.replace(self.secret.reveal().encode(), b"[REDACTED]"))
                        stream.flush()
                        if first_content is None and _sse_line_has_content(line):
                            first_content = (perf_counter() - started) * 1000
                        if generation is None and line.startswith(b"data:"):
                            try:
                                event = json.loads(line[5:])
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if isinstance(event, dict) and event.get("id"):
                                generation = event["id"]
                                atomic_write_json(self.directory / "generation.json", {"id": generation})
                return HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()},
                                    b"".join(chunks), first_content,
                                    (perf_counter() - started) * 1000)
        except urllib.error.HTTPError as exc:
            return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read(),
                                total_latency_ms=(perf_counter() - started) * 1000)

    def __call__(self, url, headers, body, timeout):
        with (self.directory / "dispatch.lock").open("x", encoding="utf-8") as marker:
            marker.write("one authorized synchronous attempt; never redispatch\n")
        wire = json.loads(body)
        for message in wire["messages"]:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        value = item["image_url"]["url"]
                        item["image_url"]["url"] = "[image omitted; bound by request metadata]"
                        item["image_url"]["data_url_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        atomic_write_json(self.directory / "wire_request.json", wire)
        response = (self.sender or self._durable_send)(url, headers, body, timeout)
        text = response.body.decode("utf-8", errors="replace").replace(self.secret.reveal(), "[REDACTED]")
        atomic_write_json(self.directory / "http_response.json", {
            "status_code": response.status_code, "body": text,
        })
        try:
            self.native = (_decode_sse_response(response.body)
                           if any(line.startswith("data:") for line in text.splitlines())
                           else json.loads(text))
        except (ValueError, TypeError, ApiError):
            self.native = None
        return response


class AccountedCalls:
    def __init__(self, output, plan, secret):
        self.output, self.plan, self.secret = output, plan, secret
        self.budget = ProviderCallBudget(plan["max_provider_calls"])
        self.records = []
        self.stopped = None

    @property
    def spent(self):
        return sum((Decimal(str(r["cost_usd"])) for r in self.records
                    if r.get("cost_usd") is not None), Decimal(0))

    def call(self, key, stage, request):
        if self.stopped:
            return None
        if (self.budget.remaining == 0 or
                self.spent + Decimal(str(self.plan["per_call_reserve_usd"])) >
                Decimal(str(self.plan["budget_usd"]))):
            self.stopped = "BUDGET_STOP"
            return None
        directory = self.output / "calls" / key / stage
        if directory.exists():
            raise ValueError("stage already recorded; refusing redispatch")
        directory.mkdir(parents=True)
        metadata = canonical_request_metadata(request)
        atomic_write_json(directory / "request.json", {
            "metadata": metadata.to_mapping(), "payload": thaw_json(request.payload),
        })
        audit = AuditSender(directory, self.secret)
        transport = OpenRouterTransport(
            api_key=self.secret, endpoint_identifier=request.endpoint_identifier,
            timeout_seconds=180, sender=audit,
        )
        client = CachedMultimodalApiClient(
            transport=CompleteAccountingTransport(transport),
            cache=FileApiCache(self.output / "cache"), usage=UsageLedger(directory / "api_usage.jsonl"),
            validator=validator_for(request.response_schema_version),
            retry_policy=RetryPolicy(max_attempts=1), provider_call_budget=self.budget,
        )
        used_before = self.budget.used
        result = {"key": key, "stage": stage, "request_hash": metadata.request_hash,
                  "status": "FAILURE", "cost_usd": None}
        try:
            response = client.call(request)
            result.update(status="OK", payload=thaw_json(response.parsed_payload),
                          returned_model=response.returned_model_identifier,
                          cost_usd=response.provider_cost, cache_hit=response.cache_hit,
                          input_tokens=response.input_tokens, output_tokens=response.output_tokens)
            atomic_write_json(directory / "response_record.json", response.to_persisted_mapping())
        except ApiError as exc:
            result["error"] = getattr(getattr(exc, "cause", exc), "code", "api_error")
        result["provider_calls"] = self.budget.used - used_before
        native = audit.native if isinstance(audit.native, dict) else {}
        native_usage = native.get("usage")
        if not isinstance(native_usage, dict):
            native_usage = {}
        result["native_usage"] = native_usage
        result["provider_request_id"] = native.get("id")
        if result["cost_usd"] is None:
            cost = native_usage.get("cost")
            if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
                result["cost_usd"] = cost
                result["cost_source"] = "raw_response_usage_after_parse_failure"
        returned_model = result.get("returned_model") or native.get("model")
        if returned_model is not None and not (
            isinstance(returned_model, str)
            and (returned_model == MODEL or returned_model.startswith(MODEL + "-"))
        ):
            result.update(status="FAILURE", error="model_identity_mismatch")
            result.pop("payload", None)
            self.stopped = "MODEL_IDENTITY_MISMATCH"
        if result["provider_calls"] and result["cost_usd"] is None:
            self.stopped = "UNPRICED_CALL_STOP"
        self.records.append(result)
        atomic_write_json(directory / "result.json", result)
        atomic_write_json(self.output / "calls_summary.json", self.records)
        print(json.dumps({k: result.get(k) for k in
                          ("key", "stage", "status", "error", "cost_usd")}), flush=True)
        return result.get("payload") if result["status"] == "OK" else None


def score_saved(adapter, rows, *, videos=None):
    """Use GT values for scoring only after all model outputs are persisted."""
    videos = _videos(videos=videos if videos is not None else dict.fromkeys(r["video_id"] for r in rows))
    if {row["video_id"] for row in rows} - set(videos):
        raise ValueError("scoring videos omit saved prediction rows")
    scored = []
    for video in videos:
        video_rows = {r["frame_id"]: r for r in rows if r["video_id"] == video}
        if not video_rows:
            continue
        targets = {}
        for resolved in adapter.iter_video(video, frame_ids=video_rows):
            targets[resolved.inference.target_frame_id] = _evaluation_target(resolved)
        if set(targets) != set(video_rows):
            raise ValueError("GT adapter did not return every frozen target")
        for frame_id, row in video_rows.items():
            target = targets[frame_id]
            mask = _task_masks(target)
            gt = {}
            for task, attr in TASK_ATTRS.items():
                value = getattr(target, attr) if mask[task] else None
                gt[task] = ([value] if task == "phase" else list(value)) if mask[task] else None
            scored.append({"video_id": video, "frame_id": frame_id, "gt": gt, "mask": mask,
                           **{arm: row.get(arm) for arm in ("h0", "h1", "final")}})
    return compute_repair_comparison(scored), scored


def run(args):
    videos = _videos(args)
    adapter, plan, requests = prepare(args, videos=videos)
    if not args.execute:
        return
    if args.api_key_file is None:
        raise ValueError("--execute requires an external --api-key-file")
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    with (args.output / "execution.lock").open("x", encoding="utf-8") as marker:
        marker.write(plan["plan_sha256"] + "\n")
    calls = AccountedCalls(args.output, plan, secret)
    rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "status": "NOT_ATTEMPTED",
             "h0": None, "h1": None, "final": None} for s in plan["selection"]]
    atomic_write_json(args.output / "predictions.json", rows)
    for index, selected in enumerate(plan["selection"]):
        if calls.stopped:
            break
        key = selected["key"]
        result = run_grounded_target(
            requests[key], lambda stage, request, key=key: calls.call(key, stage, request),
            proposal_slot=selected["proposal_slot"],
        )
        rows[index] = {"video_id": selected["video_id"], "frame_id": selected["frame_id"], **result}
        atomic_write_json(args.output / "predictions.json", rows)
    predictions_hash = _hash(args.output / "predictions.json")
    # No further model call is permitted once GT evaluation begins.
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    comparison, scored = score_saved(adapter, rows, videos=videos)
    atomic_write_json(args.output / "scored_predictions.json", scored)
    costs = {}
    for record in calls.records:
        stage = costs.setdefault(record["stage"], {"provider_calls": 0, "cost_usd": Decimal(0),
                                                   "unpriced_calls": 0})
        stage["provider_calls"] += record["provider_calls"]
        if record["cost_usd"] is None:
            stage["unpriced_calls"] += record["provider_calls"]
        else:
            stage["cost_usd"] += Decimal(str(record["cost_usd"]))
    stage_costs = {k: {**v, "cost_usd": float(v["cost_usd"])} for k, v in costs.items()}
    report = {
        "status": "COMPLETE" if all(r["status"] != "NOT_ATTEMPTED" for r in rows) else "STOPPED",
        "stop_reason": calls.stopped, "model": MODEL, "targets": len(rows),
        "h0_successes": sum(r["status"] == "OK" for r in rows),
        "provider_calls": calls.budget.used, "cost_usd": float(calls.spent),
        "repair_added_cost_usd": float(sum(v["cost_usd"] for k, v in costs.items() if k != "h0")),
        "unpriced_calls": sum(v["unpriced_calls"] for v in costs.values()),
        "stage_costs": stage_costs,
        "accepted_repairs": sum(r.get("admission", {}).get("decision") == "ACCEPT" for r in rows),
        "plan_sha256": plan["plan_sha256"], "predictions_sha256": predictions_hash,
        "comparison": comparison,
    }
    if _hash(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    if not all(_hash(ROOT / path) == digest for path, digest in plan["source_sha256"].items()):
        raise ValueError("source implementation changed after preflight")
    atomic_write_json(args.output / "summary.json", report)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps({k: v for k, v in report.items() if k != "comparison"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pricing-snapshot", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--reserve-usd", type=float, default=.04)
    parser.add_argument("--videos", nargs="+", default=list(VIDEOS))
    parser.add_argument("--require-all-task-gt", action="store_true")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
