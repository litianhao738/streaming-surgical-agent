"""Replay frozen grounded candidates with one auditable label-diff review.

Default is offline preflight. --mock uses synthetic INSUFFICIENT responses and
never calls a provider; --execute makes at most one synchronous call per frozen
candidate. Historical images, prompts, candidates and responses remain intact.
These already-inspected Training examples are mechanism diagnostics, not a new
generalization benchmark or evidence for choosing prompts on Testing.
"""

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_grounded_api_pipeline import MODEL, AccountedCalls
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    canonical_json_bytes,
    thaw_json,
)
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.context_builder import encode_rgb_png
from surgical_agent.research.verification.diff_review import (
    DIFF_REVIEW_INSTRUCTION,
    DIFF_REVIEW_VERSION,
    build_change_claims,
    build_diff_review_input,
    evaluate_diff_review,
)
from surgical_agent.research.verification.final_only_grounded import (
    final_labels,
    finalize_grounded_repair,
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_pipeline import (
    _request,
    _visual_input,
)
from surgical_agent.research.verification.grounded_repair import make_contact_crops


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _remember(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = _hash(path)
    return path


def _restore_request(saved, images):
    metadata = saved["metadata"]
    request = ApiRequest(
        provider=metadata["provider"],
        model_identifier=metadata["requested_model_identifier"],
        endpoint_identifier=metadata["endpoint_identifier"],
        prompt_version=metadata["prompt_version"],
        response_schema_version=metadata["response_schema_version"],
        generation_parameters=metadata["generation_parameters"],
        payload=saved["payload"], images=tuple(images),
    )
    if canonical_request_metadata(request).to_mapping() != metadata:
        raise ValueError("saved request/image SHA or canonical metadata mismatch")
    if (request.provider != "openrouter" or request.model_identifier != MODEL
            or request.endpoint_identifier != "https://openrouter.ai/api/v1/chat/completions"
            or request.payload.get("openrouter_routing_profile") != "strict_google_ai_studio"):
        raise ValueError("replay requires the frozen strict Google AI Studio Gemini route")
    return request


def _restore_images(selected, metadata, hashes):
    """Re-encode PNGs through the same float-tensor path as the original H0."""
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(2)
    paths = selected["source_images"]
    frame_ids = selected["causal_frame_ids"]
    if len(paths) != len(frame_ids) or len(metadata["images"]) != len(frame_ids):
        raise ValueError("source image count does not match causal window")
    images = []
    for frame_id, recorded in zip(frame_ids, metadata["images"], strict=True):
        matches = [Path(path) for path in paths if Path(path).stem == f"{frame_id:06d}"]
        if len(matches) != 1:
            raise ValueError("source PNG does not uniquely identify its causal frame")
        path = _remember(matches[0], hashes)
        if hashes[str(path)] != paths[str(matches[0])]:
            raise ValueError("original source PNG SHA mismatch")
        expected_identifier = f"cholectrack20:{selected['video_id']}:frame:{frame_id}"
        if recorded["identifier"] != expected_identifier:
            raise ValueError("source image video/frame identity mismatch")
        with Image.open(path) as source:
            array = np.array(source.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        images.append(ApiImageInput(expected_identifier, "image/png", encode_rgb_png(tensor)))
    return tuple(images)


def _verify_stage(source, key, stage, images, expected_payload, expected_hash, hashes):
    directory = source / "calls" / key / stage
    saved = _read(_remember(directory / "request.json", hashes))
    request = _restore_request(saved, images)
    result = _read(_remember(directory / "result.json", hashes))
    response = _read(_remember(directory / "response_record.json", hashes))
    if (saved["metadata"]["request_hash"] != expected_hash
            or result.get("request_hash") != expected_hash
            or response.get("request_hash") != expected_hash
            or result.get("status") != "OK"
            or result.get("payload") != expected_payload
            or response.get("parsed_payload") != expected_payload
            or response.get("requested_model_identifier") != MODEL
            or not response.get("provider_request_id")
            or result.get("provider_request_id") != response.get("provider_request_id")):
        raise ValueError(f"historical {stage} response is not bound to its saved request")
    return request


def restore_target(source, selected, row, hashes):
    """Verify the old H0 -> locator -> proposal -> review chain without GT."""
    if selected.get("source_split") != "Training":
        raise ValueError("only frozen Training diagnostics are supported")
    key = selected["key"]
    if key != f"{row['video_id']}_{row['frame_id']}":
        raise ValueError("prediction/selection target mismatch")
    original = _read(_remember(source / "requests" / f"{key}.json", hashes))
    if original["metadata"] != selected["request_metadata"]:
        raise ValueError("source plan/request metadata mismatch")
    images = _restore_images(selected, original["metadata"], hashes)
    base = _restore_request(original, images)
    visual = _visual_input(base)
    if (visual["video_id"] != row["video_id"]
            or visual["target_frame_id"] != row["frame_id"]
            or list(visual["causal_frame_ids"]) != selected["causal_frame_ids"]):
        raise ValueError("historical request target is not the selected target")
    for stage, payload in (("h0", row["h0_payload"]), ("locator", row["locator"])):
        _verify_stage(source, key, stage, images, payload, row["request_hashes"][stage], hashes)
    if final_labels(row["h0_payload"]) != row["h0"]:
        raise ValueError("saved H0 labels do not match the provider payload")
    crops, manifest = make_contact_crops(images[-1], row["locator"], target_frame_id=row["frame_id"])
    if manifest != row["crop_manifest"]:
        raise ValueError("reconstructed crop SHA/geometry differs from historical crops")
    proposal_request = _verify_stage(
        source, key, "proposal", images + crops, row["proposal"],
        row["request_hashes"]["proposal"], hashes,
    )
    proposal_input = json.loads(proposal_request.payload["input_text"])
    if (proposal_input.get("predicted_instance_regions") != row["locator"]
            or proposal_input.get("crop_manifest") != manifest):
        raise ValueError("historical proposal did not receive this locator/crop manifest")
    prepared = prepare_grounded_review(
        row["h0"], row["locator"], row["proposal"], proposal_slot=row["proposal_slot"],
    )
    if prepared["h1"] != row["h1"]:
        raise ValueError("candidate cannot be derived from the frozen proposal")
    if not prepared["review_required"]:
        if row["review"] is not None or row["final"] != row["h0"]:
            raise ValueError("unavailable/unchanged candidate must preserve historical H0")
        return None, [], None
    old_review_request = _verify_stage(
        source, key, "review", images + crops, row["review"],
        row["request_hashes"]["review"], hashes,
    )
    review_input = json.loads(old_review_request.payload["input_text"])
    if review_input.get("hypotheses") != prepared["hypotheses"]:
        raise ValueError("historical review hypotheses are not bound to H0/H1")
    old = finalize_grounded_repair(
        row["h0"], row["locator"], row["proposal"], row["review"],
        proposal_slot=row["proposal_slot"],
    )
    if old["final"] != row["final"]:
        raise ValueError("historical final cannot be reproduced from its review")
    evidence = [
        {"ref": f"frame:{frame}", "image_index": index,
         "image_identifier": image.identifier, "sha256": image.sha256,
         "kind": "target_full_frame" if frame == row["frame_id"] else "history_full_frame"}
        for index, (frame, image) in enumerate(zip(selected["causal_frame_ids"], images, strict=True))
    ]
    evidence += [
        {"ref": f"crop:{item['instance_id']}", "image_index": len(images) + index,
         "image_identifier": crop.identifier, "sha256": crop.sha256, "kind": "target_crop",
         "instance_id": item["instance_id"], "target_frame_id": row["frame_id"]}
        for index, (item, crop) in enumerate(zip(manifest, crops, strict=True))
    ]
    allowed = [item["ref"] for item in evidence]
    full_ref = f"frame:{row['frame_id']}"
    extra = build_diff_review_input(
        row["h0"], row["h1"], allowed_evidence_refs=allowed, full_frame_ref=full_ref,
    )
    extra.update(
        evidence_manifest=evidence, crop_manifest=manifest,
        predicted_instance_regions=row["locator"],
        proposal_instance_labels=[
            {name: instance[name] for name in ("instance_id", "instrument_id", "ivt_ids")}
            for instance in row["proposal"]["instances"]
        ],
        instance_mapping_policy=("Only the candidate has proposed instance assignments; "
                                 "H0 labels are frame-level sets with no instance assignments. "
                                 "Regions and assignments are unverified model predictions."),
    )
    request = _request(base, visual, DIFF_REVIEW_VERSION, DIFF_REVIEW_INSTRUCTION, extra, crops)
    return request, evidence, full_ref


def prepare(args):
    if (not math.isfinite(args.budget_usd) or not math.isfinite(args.reserve_usd)
            or not 0 < args.reserve_usd <= args.budget_usd):
        raise ValueError("positive finite budget and reserve required")
    if type(args.max_targets) is not int or not 1 <= args.max_targets <= 24:
        raise ValueError("max_targets must lie in 1..24")
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be separate from historical source artifacts")
    if (output / "execution.lock").exists():
        raise ValueError("already dispatched; never repeat this run")
    hashes = {}
    old_plan = _read(_remember(source / "plan.json", hashes))
    rows = _read(_remember(source / "predictions.json", hashes))
    _remember(source / "scored_predictions.json", hashes)  # bytes only; GT is read after inference
    if old_plan.get("source_split") != "Training" or old_plan.get("model") != MODEL:
        raise ValueError("source must be a frozen Gemini Training experiment")
    selected_by_key = {item["key"]: item for item in old_plan["selection"]}
    if len(selected_by_key) != len(old_plan["selection"]):
        raise ValueError("duplicate source selection")
    candidates = [row for row in rows if row.get("h1") is not None
                  and build_change_claims(row["h0"], row["h1"])]
    if not candidates or len(candidates) > args.max_targets:
        raise ValueError("candidate count exceeds cap or is zero; no automatic resampling")
    requests, selection, frozen_rows = {}, [], []
    row_keys = [f"{row['video_id']}_{row['frame_id']}" for row in rows]
    if len(set(row_keys)) != len(rows) or set(row_keys) != set(selected_by_key):
        raise ValueError("source predictions must retain every frozen target exactly once")
    for row in rows:
        key = f"{row['video_id']}_{row['frame_id']}"
        request, evidence, full_ref = restore_target(source, selected_by_key[key], row, hashes)
        if request is not None:
            requests[key] = request
            selection.append({
                "key": key, "video_id": row["video_id"], "frame_id": row["frame_id"],
                "source_split": "Training", "evidence_manifest": evidence,
                "full_frame_ref": full_ref, "claims": build_change_claims(row["h0"], row["h1"]),
                "request_metadata": canonical_request_metadata(request).to_mapping(),
            })
        frozen_rows.append({"video_id": row["video_id"], "frame_id": row["frame_id"],
                            "h0": row["h0"], "h1": row["h1"], "old_final": row["final"],
                            "review_required": request is not None})
    pricing = _read(_remember(args.pricing_snapshot, hashes))
    if pricing["data"]["id"] != MODEL:
        raise ValueError("pricing snapshot model mismatch")
    endpoint = next(item for item in pricing["data"]["endpoints"] if item["tag"] == "google-ai-studio")
    source_files = [Path(__file__), ROOT / "scripts/run_grounded_api_pipeline.py",
                    ROOT / "src/surgical_agent/api/__init__.py",
                    ROOT / "src/surgical_agent/api/schema.py",
                    ROOT / "src/surgical_agent/perception/final_only.py",
                    ROOT / "src/surgical_agent/perception/ontology_prompt.py",
                    ROOT / "src/surgical_agent/research/signals/resources/ivt_components_v1.csv",
                    ROOT / "src/surgical_agent/research/verification/diff_review.py",
                    ROOT / "src/surgical_agent/research/verification/grounded_pipeline.py",
                    ROOT / "src/surgical_agent/research/verification/grounded_repair.py",
                    ROOT / "src/surgical_agent/research/verification/final_only_grounded.py",
                    ROOT / "src/surgical_agent/evaluation/repair_comparison.py"]
    plan = {
        "schema_version": "diff_review_replay_plan_v1", "model": MODEL,
        "routing_profile": "strict_google_ai_studio", "source": str(source),
        "selection": selection, "target_count": len(rows), "review_target_count": len(selection),
        "source_target_count": len(rows),
        "selection_rule": "all already-inspected Training rows with a valid changed frozen H1",
        "interpretation": "mechanism diagnostic only; not an independent generalization result",
        "max_provider_calls": len(selection), "max_calls_per_target": 1, "retries": 0,
        "budget_usd": args.budget_usd, "per_call_reserve_usd": args.reserve_usd,
        "budget_is_provider_enforced": False, "pricing": endpoint["pricing"],
        "gt_used_in_requests_or_selection": False, "phase_policy": "keep H0",
        "source_artifact_sha256": hashes,
        "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in source_files},
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    if (output / "plan.json").exists():
        if _read(output / "plan.json") != plan:
            raise ValueError("preflight changed; use a fresh output directory")
    elif output.exists():
        raise ValueError("fresh output directory required")
    else:
        output.mkdir(parents=True)
        atomic_write_json(output / "plan.json", plan)
        atomic_write_json(output / "frozen_predictions.json", frozen_rows)
        atomic_write_json(output / "endpoints.json", pricing)
        for key, request in requests.items():
            atomic_write_json(output / "requests" / f"{key}.json", {
                "metadata": canonical_request_metadata(request).to_mapping(),
                "payload": thaw_json(request.payload),
            })
        for path in source_files:
            destination = output / "frozen_source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    print(json.dumps({"status": "PREFLIGHT_PASSED", "target_count": len(rows),
                      "review_target_count": len(selection),
                      "max_provider_calls": len(selection), "budget_usd": args.budget_usd,
                      "provider_calls": 0, "interpretation": plan["interpretation"]}), flush=True)
    return plan, requests, frozen_rows


def _assert_frozen(plan):
    for path, digest in plan["source_artifact_sha256"].items():
        if _hash(path) != digest:
            raise ValueError("source artifact changed after preflight")
    for path, digest in plan["source_sha256"].items():
        if _hash(ROOT / path) != digest:
            raise ValueError("implementation changed after preflight")


def mock_response(claims):
    return {"schema_version": DIFF_REVIEW_VERSION, "assessments": [
        {"change_id": claim["change_id"], "verdict": "INSUFFICIENT",
         "observation": "Synthetic offline response; no visual finding is asserted.",
         "evidence_refs": [], "scope": "FRAME", "full_frame_reviewed": False}
        for claim in claims
    ]}


def score_saved(source, rows):
    """Read independent persisted GT only after every review/final is saved."""
    labels = _read(source / "scored_predictions.json")
    by_key = {(row["video_id"], row["frame_id"]): row for row in labels}
    if len(by_key) != len(labels):
        raise ValueError("duplicate saved GT target")
    scored = []
    for row in rows:
        truth = by_key[(row["video_id"], row["frame_id"])]
        if (truth["h0"] != row["h0"] or truth["h1"] != row["h1"]
                or truth["final"] != row["old_final"]):
            raise ValueError("saved independent scoring does not match frozen predictions")
        scored.append({**row, "gt": truth["gt"], "mask": truth["mask"]})
    comparisons = {}
    for arm in ("old_final", "final_a", "final_b"):
        comparisons[arm] = compute_repair_comparison([
            {**row, "final": row[arm]} for row in scored
        ])
    return comparisons, scored


def run(args):
    if args.execute and args.mock:
        raise ValueError("--execute and --mock are mutually exclusive")
    plan, requests, rows = prepare(args)
    if not args.execute and not args.mock:
        return plan
    if args.execute and args.api_key_file is None:
        raise ValueError("--execute requires an external --api-key-file")
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file) if args.execute else None
    _assert_frozen(plan)
    with (args.output / "execution.lock").open("x", encoding="utf-8") as marker:
        marker.write(plan["plan_sha256"] + ("\nPAID\n" if args.execute else "\nOFFLINE_MOCK\n"))
    calls = AccountedCalls(args.output, plan, secret) if args.execute else None
    for row in rows:
        row.update(status="NOT_ATTEMPTED" if row["review_required"] else "NO_CANDIDATE_KEEP",
                   final_a=deepcopy(row["h0"]), final_b=deepcopy(row["h0"]),
                   decision_a="KEEP", decision_b="KEEP",
                   reason_a="REVIEW_NOT_ATTEMPTED" if row["review_required"] else "NO_CHANGED_CANDIDATE",
                   reason_b="REVIEW_NOT_ATTEMPTED" if row["review_required"] else "NO_CHANGED_CANDIDATE")
    atomic_write_json(args.output / "predictions.json", rows)
    rows_by_key = {f"{row['video_id']}_{row['frame_id']}": row for row in rows}
    for selected in plan["selection"]:
        key = selected["key"]
        row = rows_by_key[key]
        review = (calls.call(key, "diff_review", requests[key]) if calls
                  else mock_response(selected["claims"]))
        result = evaluate_diff_review(
            row["h0"], row["h1"], review,
            allowed_evidence_refs=[item["ref"] for item in selected["evidence_manifest"]],
            full_frame_ref=selected["full_frame_ref"],
        )
        row.update(result)
        row.update(review=review, status="OK" if review is not None else "REVIEW_FAILURE_KEEP",
                   request_hash=selected["request_metadata"]["request_hash"])
        atomic_write_json(args.output / "predictions.json", rows)
        if calls and calls.stopped:
            break
    if calls:
        calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = _hash(args.output / "predictions.json")
    _assert_frozen(plan)
    comparisons, scored = score_saved(Path(plan["source"]), rows)
    atomic_write_json(args.output / "scored_predictions.json", scored)
    unpriced_calls = sum(record["provider_calls"] for record in calls.records
                         if record.get("cost_usd") is None) if calls else 0
    known_added_cost = float(calls.spent) if calls else 0.0
    report = {
        "schema_version": "diff_review_replay_result_v1",
        "status": "COMPLETE" if all(row["status"] != "NOT_ATTEMPTED" for row in rows) else "STOPPED",
        "mode": "PAID_REPLAY_DIAGNOSTIC" if calls else "OFFLINE_SYNTHETIC_MOCK_NOT_MODEL_RESULTS",
        "interpretation": plan["interpretation"], "targets": len(rows),
        "review_targets": len(plan["selection"]), "source_target_count": plan["source_target_count"],
        "provider_calls": calls.budget.used if calls else 0,
        "known_added_cost_usd": known_added_cost,
        "added_cost_usd": None if unpriced_calls else known_added_cost,
        "unpriced_calls": unpriced_calls,
        "accounting_complete": unpriced_calls == 0,
        "stop_reason": calls.stopped if calls else "OFFLINE_MOCK_FINISHED",
        "plan_sha256": plan["plan_sha256"], "predictions_sha256": predictions_hash,
        "comparisons": comparisons,
    }
    if _hash(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", report)
    if secret:
        assert_secret_absent(secret, (path for path in args.output.rglob("*") if path.is_file()))
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pricing-snapshot", type=Path, required=True)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--budget-usd", type=float, default=.20)
    parser.add_argument("--reserve-usd", type=float, default=.05)
    parser.add_argument("--api-key-file", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--mock", action="store_true")
    run(parser.parse_args())
