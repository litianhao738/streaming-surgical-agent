"""Frozen Training H0 -> shared candidate -> old and neutral-presence reviews.

Offline preflight is the default. --execute authorizes at most five direct
OpenRouter POSTs per target, without retries. All predictions are persisted
before GT labels are used for paired scoring. No Tracker or workflow memory.
"""

import argparse
import base64
import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_grounded_api_pipeline import TASK_ATTRS, score_saved
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY, load_main_h0_prompt
from surgical_agent.perception.ontology_prompt import (
    ACADEMIC_MEDICAL_CONTEXT,
    load_prompt_ontology_text,
)
from surgical_agent.research.verification.diff_review import build_change_claims
from surgical_agent.research.verification.final_only_grounded import final_labels
from surgical_agent.research.verification.grounded_pipeline import (
    _visual_input,
    run_grounded_target,
)
from surgical_agent.research.verification.grounded_repair import (
    REVIEW_1000_VERSION,
    REVIEW_VERSION,
    make_contact_crops,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    PRESENCE_REVIEW_INSTRUCTION,
    PRESENCE_REVIEW_VERSION,
    build_presence_review_input,
    evaluate_presence_review,
    validate_presence_review,
)

MODEL = "qwen/qwen3.8-max-0902"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ARMS = ("old_final", "final_a", "final_b")
REVIEW_LENGTH_VERSIONS = {
    300: (REVIEW_VERSION, PRESENCE_REVIEW_VERSION),
    1000: (REVIEW_1000_VERSION, PRESENCE_REVIEW_1000_VERSION),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mask_only(resolved):
    """Read availability, never aggregate or access annotation label values."""
    evaluation = resolved.evaluation
    if evaluation is not None and evaluation.instance_supervision_available:
        return {task: bool(evaluation.instances) and all(
            getattr(instance.mask, task) for instance in evaluation.instances
        ) for task in TASK_ATTRS}
    target = resolved.frame_supervision
    return {task: target is not None and getattr(target.mask, task) for task in TASK_ATTRS}


def strip_unique_items(value):
    """Alibaba wire adaptation only; canonical prompts/local schemas stay full."""
    if isinstance(value, dict):
        return {k: strip_unique_items(v) for k, v in value.items() if k != "uniqueItems"}
    if isinstance(value, list):
        return [strip_unique_items(v) for v in value]
    return value


def wire_body(request):
    if (request.provider != "openrouter" or request.model_identifier != MODEL
            or request.endpoint_identifier != ENDPOINT
            or request.payload.get("openrouter_routing_profile") != "strict_alibaba"):
        raise ValueError("frozen Qwen / strict Alibaba identity required")
    generation = thaw_json(request.generation_parameters)
    if generation != {"temperature": 0, "max_output_tokens": 4096,
                       "reasoning": {"effort": "low"}}:
        raise ValueError("frozen H0 generation parameters changed")
    schema = schema_for(request.response_schema_version)
    content = [{"type": "text", "text": request.payload["input_text"]}]
    for image, detail in zip(request.images, request.payload["image_details"], strict=True):
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{image.mime_type};base64," + base64.b64encode(image.content).decode(),
            "detail": detail,
        }})
    return {"model": MODEL, "messages": [
        {"role": "system", "content": request.payload["system_text"]},
        {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": request.response_schema_version, "strict": True,
            "schema": strip_unique_items(schema)}},
        "provider": {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True},
        "temperature": 0, "reasoning": {"effort": "low"}, "max_tokens": 4096, "stream": False}


def safe_wire(body):
    result = deepcopy(body)
    for item in result["messages"][1]["content"][1:]:
        url = item["image_url"]["url"]
        item["image_url"]["url"] = "[image bytes omitted; SHA-bound]"
        item["image_url"]["data_url_sha256"] = hashlib.sha256(url.encode()).hexdigest()
    return result


def build_review_request(base, row, *, schema_version=PRESENCE_REVIEW_1000_VERSION):
    crops, manifest = make_contact_crops(base.images[-1], row["locator"],
                                         target_frame_id=row["frame_id"])
    if manifest != row["crop_manifest"]:
        raise ValueError("presence review crops differ from shared candidate crops")
    visual = _visual_input(base)
    evidence = [{"ref": f"frame:{frame}", "image_index": index,
                 "image_identifier": image.identifier, "sha256": image.sha256,
                 "kind": "target_full_frame" if frame == row["frame_id"] else "history_full_frame"}
                for index, (frame, image) in enumerate(zip(
                    visual["causal_frame_ids"], base.images, strict=True))]
    evidence += [{"ref": f"crop:{item['instance_id']}", "image_index": len(base.images) + index,
                  "image_identifier": crop.identifier, "sha256": crop.sha256,
                  "kind": "target_crop", "target_frame_id": row["frame_id"]}
                 for index, (item, crop) in enumerate(zip(manifest, crops, strict=True))]
    refs, full_ref = [e["ref"] for e in evidence], f"frame:{row['frame_id']}"
    neutral = build_presence_review_input(row["h0"], row["h1"],
                                          allowed_evidence_refs=refs, full_frame_ref=full_ref)
    payload = thaw_json(base.payload)
    payload["input_text"] = json.dumps({**visual, **neutral,
                                        "evidence_manifest": evidence,
                                        "crop_manifest": manifest}, sort_keys=True, separators=(",", ":"))
    instruction = PRESENCE_REVIEW_INSTRUCTION.replace(PRESENCE_REVIEW_VERSION, schema_version)
    payload["system_text"] = (ACADEMIC_MEDICAL_CONTEXT + "\n" + instruction
                              + "\n" + load_prompt_ontology_text() + _LABEL_BOUNDARY
                              + "\nReturn JSON only. Required output schema:\n"
                              + json.dumps(schema_for(schema_version), separators=(",", ":")))
    payload["image_details"] = ["low"] * (len(base.images) - 1) + ["high"] * (1 + len(crops))
    payload["openrouter_image_detail_mode"] = "explicit_v1"
    return replace(base, payload=payload, images=base.images + crops,
                   response_schema_version=schema_version,
                   prompt_version=schema_version + "_causal_v1"), evidence, full_ref


def run_target(base, call, *, proposal_slot, review_observation_max_chars=1000):
    contrast_version, presence_version = REVIEW_LENGTH_VERSIONS[review_observation_max_chars]
    row = run_grounded_target(base, call, proposal_slot=proposal_slot,
                              review_schema_version=contrast_version)
    row.update(old_final=deepcopy(row["final"]), final_a=deepcopy(row["h0"]),
               final_b=deepcopy(row["h0"]), presence_review=None,
               presence_status="NO_CHANGED_CANDIDATE", decision_a="KEEP", decision_b="KEEP")
    if row["h0"] is None or row["h1"] is None or not build_change_claims(row["h0"], row["h1"]):
        return row
    request, evidence, full_ref = build_review_request(base, row, schema_version=presence_version)
    row["presence_binding"] = {"request_metadata": canonical_request_metadata(request).to_mapping(),
                               "evidence_manifest": evidence, "full_frame_ref": full_ref}
    review = call("presence_review", request)
    try:
        if review is not None:
            validate_presence_review(review, h0=row["h0"], h1=row["h1"],
                                     allowed_evidence_refs=[e["ref"] for e in evidence],
                                     full_frame_ref=full_ref, schema_version=presence_version)
    except (ApiError, TypeError, ValueError, OverflowError) as exc:
        row["stage_errors"]["presence_review"] = type(exc).__name__
        review = None
    result = evaluate_presence_review(row["h0"], row["h1"], review,
                                      allowed_evidence_refs=[e["ref"] for e in evidence],
                                      full_frame_ref=full_ref, schema_version=presence_version)
    row.update(result)
    row.update(presence_review=review, presence_status="OK" if review is not None else "REVIEW_FAILURE_KEEP")
    return row


class DirectCalls:
    """Experiment-local direct HTTP and durable audit; no cache or retry layer."""

    def __init__(self, output, cap, secret, *, timeout=180, sender=None):
        self.output, self.cap, self.secret, self.timeout = output, cap, secret, timeout
        self.sender = sender or requests.post
        self.lock = threading.Lock()
        self.used, self.records, self.stopped = 0, [], None

    def call(self, key, stage, request):
        body = wire_body(request)
        wire = canonical_json_bytes(body)
        directory = self.output / "calls" / key / stage
        with self.lock:
            if self.stopped or self.used >= self.cap:
                return None
            directory.mkdir(parents=True, exist_ok=False)
            atomic_write_json(directory / "request.json", {
                "metadata": canonical_request_metadata(request).to_mapping(),
                "payload": thaw_json(request.payload), "wire": safe_wire(body),
                "wire_sha256": hashlib.sha256(wire).hexdigest(),
            })
            (directory / "dispatch.lock").write_text("One direct POST. No redispatch.\n", encoding="utf-8")
            self.used += 1
        record = {"key": key, "stage": stage, "status": "FAILURE", "provider_calls": 1,
                  "cost_usd": None, "request_hash": canonical_request_metadata(request).request_hash}
        payload, started = None, perf_counter()
        try:
            response = self.sender(ENDPOINT, data=wire, headers={
                "Authorization": "Bearer " + self.secret.reveal(), "Content-Type": "application/json"},
                timeout=(15, self.timeout), allow_redirects=False)
            atomic_write_json(directory / "http_response.json", {
                "status_code": response.status_code,
                "body": response.text.replace(self.secret.reveal(), "[REDACTED]")})
            record["http_status"] = response.status_code
            if response.status_code in (401, 402, 403):
                with self.lock:
                    self.stopped = f"FATAL_HTTP_{response.status_code}"
            native = response.json()
            if not isinstance(native, dict):
                raise TypeError("provider response root must be an object")
            if response.status_code == 400:
                error_text = json.dumps(native.get("error", {})).casefold()
                if any(word in error_text for word in ("schema", "unsupported parameter", "invalid_request")):
                    with self.lock:
                        self.stopped = "FATAL_SHARED_REQUEST_FORMAT"
            record.update(provider_request_id=native.get("id"), returned_model=native.get("model"),
                          provider=native.get("provider"), usage=native.get("usage"))
            usage = native.get("usage")
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if type(cost) in (float, int) and math.isfinite(cost) and cost >= 0:
                record["cost_usd"] = cost
            if response.status_code != 200:
                record["error"] = "http_error"
            elif (native.get("model") != MODEL or native.get("provider") != "Alibaba"
                  or not isinstance(native.get("id"), str) or not native["id"]):
                record["error"] = "model_or_provider_identity_mismatch"
            else:
                candidate = json.loads(native["choices"][0]["message"]["content"])
                validator = validator_for(request.response_schema_version)
                validator(candidate)
                payload = candidate
                record.update(status="OK", payload=candidate)
        except (requests.RequestException, OSError, ApiError, ValueError, TypeError,
                KeyError, IndexError, OverflowError) as exc:
            # Exception text can contain credentials or request headers.
            record["error"] = type(exc).__name__
        record["latency_seconds"] = round(perf_counter() - started, 3)
        with self.lock:
            self.records.append(record)
            atomic_write_json(directory / "result.json", record)
            atomic_write_json(self.output / "calls_summary.json", self.records)
            print(json.dumps({k: record.get(k) for k in
                              ("key", "stage", "status", "error", "cost_usd", "latency_seconds")}), flush=True)
        return payload


def prepare(args):
    import torch

    torch.set_num_threads(2)
    if not 1 <= args.workers <= 4 or not 1 <= args.timeout_seconds <= 600:
        raise ValueError("workers must be 1..4; timeout must be 1..600 seconds")
    review_limit = getattr(args, "review_observation_max_chars", 1000)
    if review_limit not in REVIEW_LENGTH_VERSIONS:
        raise ValueError("review observation limit must be 300 or 1000 characters")
    targets = read(args.targets_json)
    selection = deepcopy(targets["selection"])
    keys = [f"{s['video_id']}_{s['frame_id']}" for s in selection]
    if not 1 <= len(selection) <= 24 or len(set(keys)) != len(keys):
        raise ValueError("one to 24 distinct frozen Training targets required")
    if any(s.get("source_split") != "Training" for s in selection):
        raise ValueError("Testing/Validation targets are forbidden")
    if (args.output / "execution.lock").exists():
        raise ValueError("already dispatched; never repeat this run")
    config_path = ROOT / "configs/perception/joint_openrouter_h0.yaml"
    config = load_api_config(config_path)
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    context_builder = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="high")
    loader, builder = CausalApiMediaLoader(), JointPerceptionRequestBuilder(config=config)
    bases, artifact_hashes = {}, {str(args.targets_json.resolve()): sha(args.targets_json),
                                  str(args.pricing_snapshot.resolve()): sha(args.pricing_snapshot)}
    for video in dict.fromkeys(s["video_id"] for s in selection):
        needed = {s["frame_id"] for s in selection if s["video_id"] == video}
        available = {s.target_frame_id: s for s in adapter.iter_inference_video(video)
                     if s.target_frame_id in needed}
        if set(available) != needed:
            raise ValueError("selected target is unavailable; no resampling")
        for selected in (s for s in selection if s["video_id"] == video):
            sample = available[selected["frame_id"]]
            if sample.source_split is not DatasetSplit.TRAINING:
                raise ValueError("dataset disagrees with Training manifest")
            if list(sample.causal_frame_ids) != selected["causal_frame_ids"]:
                raise ValueError("frozen causal frame availability changed")
            loaded = loader.load(sample)
            context = context_builder.build(loaded.runtime_sample, loaded.frames,
                workflow_snapshot={}, memory_snapshot={}, prior_finalized_prediction=None)
            base = builder.build(context)
            if base.payload["system_text"] != load_main_h0_prompt():
                raise ValueError("frozen H0 prompt drift")
            wire_body(base)
            key = f"{video}_{sample.target_frame_id}"
            bases[key] = base
            selected.update(key=key, request_metadata=canonical_request_metadata(base).to_mapping(),
                            source_images={str(p): sha(p) for p in sample.media_refs})
            artifact_hashes.update(selected["source_images"])
        for resolved in adapter.iter_video(video, frame_ids=sorted(needed)):
            selected = next(s for s in selection if s["video_id"] == video
                            and s["frame_id"] == resolved.inference.target_frame_id)
            if _mask_only(resolved) != selected["task_masks"]:
                raise ValueError("task availability changed after fixed selection")
            for name in ("annotation_source", "phase_source", "frame_action_source", "manifest_source"):
                source = getattr(resolved.provenance, name)
                if source is not None:
                    path = Path(source)
                    if not path.is_absolute():
                        path = args.dataset_root / path
                    if path.is_file():
                        artifact_hashes[str(path.resolve())] = sha(path)
    for index, selected in enumerate(selection):
        selected["proposal_slot"] = "FIRST" if index % 2 == 0 else "SECOND"
    pricing = read(args.pricing_snapshot)
    if pricing["data"]["id"] != MODEL:
        raise ValueError("pricing model identity mismatch")
    endpoint = next(e for e in pricing["data"]["endpoints"] if e["tag"] == "alibaba")
    if endpoint["status"] != 0 or "structured_outputs" not in endpoint["supported_parameters"]:
        raise ValueError("Alibaba endpoint is not ready")
    paths = [Path(__file__), ROOT / "scripts/run_grounded_api_pipeline.py", config_path]
    paths += [ROOT / "src/surgical_agent" / p for p in (
        "api/schema.py", "perception/main_h0.py", "perception/final_only.py",
        "perception/joint_api_vlm.py", "perception/context_builder.py", "perception/ontology_prompt.py",
        "perception/prompts/perception_prompt_main_h0.txt", "research/signals/resources/ivt_components_v1.csv",
        "research/verification/grounded_pipeline.py", "research/verification/grounded_repair.py",
        "research/verification/final_only_grounded.py", "research/verification/diff_review.py",
        "research/verification/presence_review.py", "evaluation/repair_comparison.py")]
    plan = {"schema_version": "presence_review_fresh_trial_plan_v1", "model": MODEL,
            "routing_profile": "strict_alibaba", "source_split": "Training", "selection": selection,
            "target_count": len(selection), "max_provider_calls": 5 * len(selection), "retries": 0,
            "workers": args.workers, "timeout_seconds": [15, args.timeout_seconds],
            "review_observation_max_chars": review_limit,
            "contrast_review_schema_version": REVIEW_LENGTH_VERSIONS[review_limit][0],
            "presence_review_schema_version": REVIEW_LENGTH_VERSIONS[review_limit][1],
            "proposal_observation_max_chars": 300,
            "dispatch_order": "first frozen target sequential sentinel, then remaining independent targets",
            "protocol": "fixed H0 -> blind locator -> proposal -> old contrast / neutral presence; A and B share presence",
            "cost_policy": "no dollar stop; bounded provider-call count; unknown costs remain null",
            "phase_policy": "keep H0", "tracker": False, "gate": False, "cross_window_memory": False,
            "wire_schema_adaptation": "remove uniqueItems only; full schema prompt/local validation unchanged",
            "pricing": endpoint["pricing"], "source_artifact_sha256": artifact_hashes,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "selection_manifest_sha256": sha(args.targets_json),
            "label_values_used_for_selection_or_requests": False}
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    if (args.output / "plan.json").exists():
        if read(args.output / "plan.json") != plan:
            raise ValueError("preflight changed; use a fresh directory")
    elif args.output.exists():
        raise ValueError("fresh output directory required")
    else:
        args.output.mkdir(parents=True)
        atomic_write_json(args.output / "plan.json", plan)
        atomic_write_json(args.output / "endpoints.json", pricing)
        for key, base in bases.items():
            atomic_write_json(args.output / "requests" / f"{key}.json", {
                "metadata": canonical_request_metadata(base).to_mapping(), "payload": thaw_json(base.payload)})
        for path in paths:
            destination = args.output / "frozen_source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    print(json.dumps({"status": "PREFLIGHT_PASSED", "targets": len(selection), "model": MODEL,
                      "max_provider_calls": plan["max_provider_calls"], "provider_calls": 0}), flush=True)
    return adapter, plan, bases


def assert_frozen(plan):
    for path, digest in plan["source_artifact_sha256"].items():
        if sha(path) != digest:
            raise ValueError("source artifact changed after preflight")
    for path, digest in plan["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("source code changed after preflight")


def score_all(adapter, rows):
    _, truth_rows = score_saved(adapter, [{**r, "final": r["old_final"]} for r in rows])
    truth = {(r["video_id"], r["frame_id"]): r for r in truth_rows}
    scored = [{**r, "gt": truth[(r["video_id"], r["frame_id"])]["gt"],
               "mask": truth[(r["video_id"], r["frame_id"])]["mask"]} for r in rows]
    return {arm: compute_repair_comparison([{**r, "final": r[arm]} for r in scored])
            for arm in ARMS}, scored


def accounting(records):
    stages = {}
    for r in records:
        stage = stages.setdefault(r["stage"], {"provider_calls": 0, "successful_responses": 0,
                                               "known_cost_usd": 0.0, "unpriced_calls": 0})
        stage["provider_calls"] += 1
        stage["successful_responses"] += r["status"] == "OK"
        stage["unpriced_calls"] += r["cost_usd"] is None
        stage["known_cost_usd"] += r["cost_usd"] or 0.0
    for stage in stages.values():
        stage["cost_usd"] = None if stage["unpriced_calls"] else stage["known_cost_usd"]
    known = sum(s["known_cost_usd"] for s in stages.values())
    unknown = sum(s["unpriced_calls"] for s in stages.values())
    return {"provider_calls": len(records), "known_cost_usd": known,
            "cost_usd": None if unknown else known, "unpriced_calls": unknown,
            "accounting_complete": unknown == 0, "stage_costs": stages}


def run_work(index, selected, base, calls, *, review_observation_max_chars=1000):
    """Preserve known H0 and halt dispatch on an unexpected implementation error."""
    key = selected["key"]
    try:
        row = run_target(base, lambda stage, request: calls.call(key, stage, request),
                         proposal_slot=selected["proposal_slot"],
                         review_observation_max_chars=review_observation_max_chars)
    except Exception as exc:  # noqa: BLE001 - terminal audit boundary, never resume after a code error
        with calls.lock:
            calls.stopped = calls.stopped or "UNEXPECTED_IMPLEMENTATION_ERROR"
            h0_record = next((r for r in calls.records if r["key"] == key
                              and r["stage"] == "h0" and r["status"] == "OK"), None)
        h0 = final_labels(h0_record["payload"]) if h0_record else None
        row = {"video_id": selected["video_id"], "frame_id": selected["frame_id"],
               "status": "IMPLEMENTATION_ERROR", "error": type(exc).__name__,
               "h0": h0, "h1": None, **{arm: deepcopy(h0) for arm in ARMS},
               "presence_status": "IMPLEMENTATION_ERROR_KEEP", "decision_a": "KEEP", "decision_b": "KEEP"}
    return index, row


def run(args):
    adapter, plan, bases = prepare(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("--execute requires --api-key-file")
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    assert_frozen(plan)
    with (args.output / "execution.lock").open("x", encoding="utf-8") as marker:
        marker.write(plan["plan_sha256"] + "\nPAID_DIRECT_POSTS_NO_RETRIES\n")
    calls = DirectCalls(args.output, plan["max_provider_calls"], secret, timeout=args.timeout_seconds)
    rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "status": "NOT_ATTEMPTED",
             "h0": None, "h1": None, **{arm: None for arm in ARMS}} for s in plan["selection"]]
    atomic_write_json(args.output / "predictions.json", rows)

    def work(index, selected):
        return run_work(index, selected, bases[selected["key"]], calls,
                        review_observation_max_chars=plan["review_observation_max_chars"])

    index, result = work(0, plan["selection"][0])
    rows[index] = result
    atomic_write_json(args.output / "predictions.json", rows)
    remaining = [] if calls.stopped else list(enumerate(plan["selection"]))[1:]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, index, selected) for index, selected in remaining]
        for future in as_completed(futures):
            index, result = future.result()
            rows[index] = result
            atomic_write_json(args.output / "predictions.json", rows)
            print(json.dumps({"target_finished": f"{result['video_id']}_{result['frame_id']}",
                              "h0_status": result["status"], "presence_status": result["presence_status"]}), flush=True)
    # No further model call is allowed once independent GT scoring starts.
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = sha(args.output / "predictions.json")
    assert_frozen(plan)
    comparisons, scored = score_all(adapter, rows)
    atomic_write_json(args.output / "scored_predictions.json", scored)
    report = {"schema_version": "presence_review_fresh_trial_result_v1", "model": MODEL,
              "mode": "PAID_FRESH_TRAINING_PAIRED_DIAGNOSTIC", "targets": len(rows),
              "stop_reason": calls.stopped, "plan_sha256": plan["plan_sha256"],
              "predictions_sha256": predictions_hash, "comparisons": comparisons,
              "h0_successes": sum(r["h0"] is not None for r in rows),
              "presence_successes": sum(r.get("presence_status") == "OK" for r in rows),
              **accounting(calls.records)}
    if sha(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", report)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--pricing-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--review-observation-max-chars", type=int, choices=(300, 1000), default=1000)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
