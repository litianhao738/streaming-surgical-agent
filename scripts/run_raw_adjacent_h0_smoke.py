"""Six raw-adjacent targets, pure joint API H0, no tracking or repair.

Default prepares a frozen plan without credentials or API calls. Re-run with
--execute and identical arguments to execute it. GT is read only after predictions.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_pure_h0_smoke import FinalOnlyRequestBuilder, RawResponseAuditTransport, evaluate_offline
from scripts.run_five_expert_ablation import BOUNDARY
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.providers.openrouter import OpenRouterTransport, urllib_send_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import schema_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter, MP4_ALIGNMENT_VERSION
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import validate_final_only


class AuditSender:
    def __init__(self, directory, secret):
        self.directory, self.secret = directory, secret

    def __call__(self, url, headers, body, timeout):
        wire = json.loads(body)
        for message in wire["messages"]:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        uri = item["image_url"]["url"]
                        item["image_url"]["url"] = "[image omitted; SHA-bound]"
                        item["image_url"]["data_url_sha256"] = hashlib.sha256(uri.encode()).hexdigest()
        atomic_write_json(self.directory / "wire_request.json", wire)
        response = urllib_send_json(url, headers, body, timeout)
        atomic_write_json(self.directory / "http_response.json", {
            "status_code": response.status_code,
            "body": response.body.decode("utf-8", errors="replace").replace(self.secret.reveal(), "[REDACTED]"),
        })
        return response


def run(args):
    import cv2

    if args.start < 3:
        raise ValueError("Start must have two preceding raw frames")
    config = load_api_config(args.config)
    if (config.requested_model_identifier != "qwen/qwen3.8-max-0902"
            or not config.data_upload_authorized
            or config.provider_options.get("initial_prompt_profile") != "fixed_visual_only"):
        raise ValueError("This run requires the authorized fixed-visual Qwen baseline")
    video = args.dataset / "Testing" / args.video / (args.video.lower() + ".mp4")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Raw video unavailable")
        media = {"fps": capture.get(cv2.CAP_PROP_FPS), "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                 "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()
    if args.start + 5 > media["frame_count"] or media["fps"] != 25:
        raise ValueError("Raw frame bounds or expected FPS mismatch")
    loader = CausalApiMediaLoader()
    builder = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="high")
    requests = {}
    for fid in range(args.start, args.start + 6):
        ids = (fid - 2, fid - 1, fid)
        sample = InferenceSample(args.video, fid, ids, (str(video),) * 3,
                                 DatasetSplit.TESTING, MP4_ALIGNMENT_VERSION)
        loaded = loader.load(sample)
        context = builder.build(loaded.runtime_sample, loaded.frames, workflow_snapshot={},
                                memory_snapshot={}, prior_finalized_prediction=None)
        base = FinalOnlyRequestBuilder(config=config).build(context)
        payload = thaw_json(base.payload)
        payload["openrouter_image_detail_mode"] = "explicit_v1"
        payload["image_details"] = ["low", "low", "high"]
        payload["system_text"] += BOUNDARY + (
            "\nAll six root keys are mandatory: schema_version, instrument, verb, target, ivt, phase. "
            "Do not stop after instrument. The exact output schema is:\n"
            + json.dumps(schema_for(base.response_schema_version), separators=(",", ":")))
        data = json.loads(payload["input_text"])
        data["raw_video_sampling"] = {"fps": media["fps"], "frame_id_step": 1,
                                      "window_span_seconds": 2 / media["fps"]}
        payload["input_text"] = json.dumps(data, separators=(",", ":"))
        requests[fid] = replace(base, payload=payload, prompt_version="raw_adjacent_joint_h0_v1")
    plan = {"profile": "pure_joint_h0_raw_adjacent", "video_id": args.video,
            "source_split": "Testing", "purpose": "diagnostic_not_sealed_test_result",
            "video_path": str(video.resolve()), "video_sha256": sha256_file(video), "media": media,
            "targets": list(requests), "source_frame_numbering": "one_based; decoder_index=frame_id-1",
            "input_frame_step": 1, "target_frame_step": 1, "model": config.requested_model_identifier,
            "config_sha256": sha256_file(args.config), "script_sha256": sha256_file(Path(__file__)),
            "provider_call_limit": 6, "retries": 0,
            "planning_budget_usd": 0.15, "per_call_spend_reserve_usd": 0.04,
            "spend_policy": "Stop before next call if reported spend + reserve exceeds planning budget; reserve is not a provider-enforced price cap. Stop on unpriced call.",
            "tracker": False, "gate": False, "verifier": False, "repair": False, "memory": False,
            "evaluation": "Exact frame GT only, read after predictions; no carry-forward or interpolation",
            "requests": {str(f): canonical_request_metadata(r).to_mapping() for f, r in requests.items()}}
    plan_path = args.output / "plan.json"
    if args.output.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing output does not match the prepared plan")
        if (args.output / "run_status.json").exists():
            raise ValueError("Execution already started; preserve outputs")
    else:
        args.output.mkdir(parents=True)
        atomic_write_json(plan_path, plan)
    for fid, req in requests.items():
        record = {"payload": thaw_json(req.payload), "metadata": canonical_request_metadata(req).to_mapping()}
        path = args.output / "requests" / f"{fid}.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("Prepared request differs")
        if not path.exists():
            atomic_write_json(path, record)
        for im in req.images:
            path = args.output / "images" / f"{im.identifier.rsplit(':', 1)[1]}.png"
            path.parent.mkdir(exist_ok=True)
            if path.exists() and path.read_bytes() != im.content:
                raise ValueError("Prepared frame bytes differ")
            if not path.exists():
                path.write_bytes(im.content)
    print(json.dumps({"status": "PREPARED", "targets": plan["targets"], "media": media}), flush=True)
    if not args.execute:
        return
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    atomic_write_json(args.output / "run_status.json", {"status": "RUNNING"})
    budget = ProviderCallBudget(6)
    rows, spending, stopped = [], 0.0, None
    for fid, req in requests.items():
        row = {"video_id": args.video, "frame_id": fid, "causal_frame_ids": [fid-2, fid-1, fid]}
        if spending + plan["per_call_spend_reserve_usd"] > plan["planning_budget_usd"]:
            stopped = "PLANNING_SPEND_LIMIT"
        if stopped:
            row.update(status="NOT_ATTEMPTED", error=stopped)
        else:
            directory = args.output / "calls" / str(fid)
            transport = OpenRouterTransport(api_key=secret, endpoint_identifier=config.endpoint_identifier,
                timeout_seconds=180, sender=AuditSender(directory, secret))
            client = CachedMultimodalApiClient(
                transport=RawResponseAuditTransport(CompleteAccountingTransport(transport), directory),
                cache=FileApiCache(args.cache), usage=UsageLedger(directory / "api_usage.jsonl"),
                validator=validate_final_only, retry_policy=RetryPolicy(max_attempts=1), provider_call_budget=budget)
            try:
                response = client.call(req)
                value = thaw_json(response.parsed_payload)
                row.update(status="OK", selected_ids={t: [value[t]["selected_id"]] if t == "phase"
                    else value[t]["selected_ids"] for t in ("instrument", "verb", "target", "ivt", "phase")},
                    returned_model=response.returned_model_identifier, cache_hit=response.cache_hit)
            except ApiError as exc:
                cause = getattr(exc, "cause", exc)
                row.update(status="API_FAILURE", error=cause.code)
                if getattr(cause, "status_code", None) in {401, 402}:
                    stopped = cause.code
            records = [json.loads(line) for line in (directory / "api_usage.jsonl").read_text().splitlines()]
            spending += sum(r.get("provider_cost") or 0 for r in records)
            if any(r["provider_call_count"] and r.get("provider_cost") is None for r in records):
                stopped = "UNPRICED_CALL"
        rows.append(row)
        atomic_write_json(args.output / "predictions.json", rows)
        print(json.dumps(row), flush=True)
    # No annotation file was opened before all inference outputs were persisted.
    annotation = video.with_suffix(".json")
    raw = json.loads(annotation.read_text())
    available = {int(fid) for fid in raw["annotations"]}
    annotated_rows = [r for r in rows if r["frame_id"] in available]
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    evaluation = evaluate_offline(adapter, args.video, annotated_rows)
    by_frame = {r["frame_id"]: r for r in evaluation["frames"]}
    for row in rows:
        row["evaluation"] = by_frame.get(row["frame_id"], {"status": "NO_EXACT_FRAME_GT",
            "mask": {t: False for t in ("instrument", "verb", "target", "ivt", "phase")}})
    atomic_write_json(args.output / "predictions_with_evaluation.json", rows)
    usage = [json.loads(line) for p in (args.output / "calls").glob("*/api_usage.jsonl")
             for line in p.read_text().splitlines()]
    summary = {"profile": plan["profile"], "targets": plan["targets"],
        "successful_predictions": sum(r["status"] == "OK" for r in rows),
        "provider_calls": sum(r["provider_call_count"] for r in usage),
        "cache_hits": sum(r["cache_hit"] for r in usage),
        "reported_cost_usd": round(spending, 6),
        "unpriced_calls": sum(r["provider_call_count"] for r in usage if r.get("provider_cost") is None),
        "evaluation": evaluation, "no_exact_frame_gt": [r["frame_id"] for r in rows if r["frame_id"] not in available]}
    atomic_write_json(args.output / "summary.json", summary)
    atomic_write_json(args.output / "run_status.json", {"status": "COMPLETE" if summary["successful_predictions"] == 6
        else "COMPLETE_WITH_FAILURES", "successful_predictions": summary["successful_predictions"]})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset", "config", "api-key-file", "cache", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
