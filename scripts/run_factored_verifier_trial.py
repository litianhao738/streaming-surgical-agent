"""Optional Training-only, one-request-per-proposition names1000 diagnostic.

No H0 or candidate is regenerated. More requests also supply more total output
tokens, so differences from the names1000 batch arm cannot isolate context
separation as their sole cause. Default is offline preflight; no automatic retry.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.research.verification.factored_presence import (
    FACTORED_PRESENCE_PROMPT_VERSION,
    merge_factored_presence_reviews,
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
)

VARIANT = "factored_names1000"
INTERPRETATION = (
    "Training development only. H0, candidate, images, full ontology, review schema, "
    "and per-request generation settings match names1000. One call per proposition "
    "changes both contextual isolation and total calls/output-token allowance; "
    "semantic differences cannot be attributed solely to isolation."
)


def checked_saved_names_wire(saved, request):
    """Bind metadata, text, image hashes and the full reconstructed wire body."""
    wire = shared.wire_body(request)
    if (saved["metadata"] != canonical_request_metadata(request).to_mapping()
            or saved["payload"] != thaw_json(request.payload)
            or saved["wire"] != shared.safe_wire(wire)):
        raise ValueError("saved names request metadata/payload/wire binding mismatch")
    return hashlib.sha256(canonical_json_bytes(wire)).hexdigest()


def verify_names_completion(plan, names_requests, rows):
    """Execution gate: inspect completion/transport evidence, never scored GT.

    Offline preparation can run while the original names batch is in flight.
    Before any factored execution, every reference names call must have a bound
    actual HTTP 200 response. A schema-invalid response may still serve as a
    failed baseline; it must not be promoted to a successful review here.
    """
    source = Path(plan["source_names_batch"])
    if not (source / "summary.json").is_file():
        raise ValueError("names batch must complete before factored execution")
    batch_plan, completion = shared.read(source / "plan.json"), shared.read(source / "summary.json")
    unsigned = {key: value for key, value in batch_plan.items() if key != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != batch_plan["plan_sha256"]
            or batch_plan["plan_sha256"] != plan["source_names_batch_plan_sha256"]
            or completion.get("plan_sha256") != batch_plan["plan_sha256"]
            or completion.get("predictions_sha256") != shared.sha(source / "predictions.json")
            or batch_plan["keys"] != plan["keys"]):
        raise ValueError("names batch completion/plan/predictions hash mismatch")
    completed_rows = shared.read(source / "predictions.json")
    completed = {f"{row['video_id']}_{row['frame_id']}": row for row in completed_rows}
    if len(completed) != len(completed_rows) or set(completed) != set(plan["keys"]):
        raise ValueError("completed names batch omits or duplicates frozen targets")
    for row in rows:
        previous = completed[f"{row['video_id']}_{row['frame_id']}"]
        if previous["h0"] != row["h0"] or previous["h1"] != row["h1"]:
            raise ValueError("completed names batch changed frozen H0 or H1")
    hashes = {str(source / name): shared.sha(source / name)
              for name in ("plan.json", "summary.json", "predictions.json")}
    details = []
    for selected in plan["selection"]:
        key = selected["key"]
        directory = source / "calls" / key / "names1000"
        files = ("request.json", "result.json", "http_response.json", "dispatch.lock")
        if not all((directory / name).is_file() for name in files):
            raise ValueError("reference names call lacks actual dispatch/HTTP evidence")
        request, result, http = (shared.read(directory / name)
                                 for name in ("request.json", "result.json", "http_response.json"))
        expected = names_requests[key]
        wire_hash = checked_saved_names_wire(request, expected)
        if (wire_hash != selected["source_names_wire_sha256"]
                or request["wire_sha256"] != wire_hash
                or request["metadata"] != selected["source_names_request_metadata"]
                or result["request_hash"] != request["metadata"]["request_hash"]
                or result.get("key") != key or result.get("stage") != "names1000"
                or result.get("provider_calls") != 1):
            raise ValueError("actual names dispatch wire/request/result binding mismatch")
        if http["status_code"] != 200 or result.get("http_status") != 200:
            raise ValueError("reference names response must have auditable native HTTP 200 evidence")
        native = json.loads(http["body"])
        if (native.get("model") != shared.MODEL or native.get("provider") != "Alibaba"
                or not native.get("id") or native["id"] != result.get("provider_request_id")
                or result.get("returned_model") != native["model"]
                or result.get("provider") != native["provider"]):
            raise ValueError("actual names native model/provider/id binding mismatch")
        saved_review = completed[key]["variants"]["names1000"]["review"]
        if saved_review != result.get("payload"):
            raise ValueError("completed names review differs from its actual local response")
        if result["status"] == "OK" and json.loads(native["choices"][0]["message"]["content"]) != saved_review:
            raise ValueError("successful names review differs from native response content")
        hashes.update({str(directory / name): shared.sha(directory / name) for name in files})
        details.append({"key": key, "wire_sha256": wire_hash,
                        "request_hash": result["request_hash"],
                        "provider_request_id": native["id"], "baseline_local_status": result["status"]})
    return {"schema_version": "factored_names_execution_binding_v1",
            "names_batch_plan_sha256": batch_plan["plan_sha256"],
            "reference_calls": details, "gt_values_read": False,
            "source_artifact_sha256": hashes}


def prepare(args):
    if not 1 <= args.max_calls <= 72 or not 1 <= args.timeout_seconds <= 600:
        raise ValueError("explicit call cap 1..72 and timeout 1..600 required")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("factored output must be separate from its frozen source")
    if (output / "execution.lock").exists():
        raise ValueError("already dispatched; never repeat this run")
    source_plan = shared.read(source / "plan.json")
    if source_plan.get("source_split") != "Training":
        raise ValueError("factored runner supports Training only; Validation/Testing are forbidden")
    if source_plan.get("model") != shared.MODEL:
        raise ValueError("frozen H0 model identity mismatch")
    completion = shared.read(source / "summary.json")
    unsigned = {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != source_plan["plan_sha256"]
            or completion.get("plan_sha256") != source_plan["plan_sha256"]
            or completion.get("predictions_sha256") != shared.sha(source / "predictions.json")):
        raise ValueError("source collection is incomplete or completion hashes mismatch")
    selected = {row["key"]: row for row in source_plan["selection"]}
    keys = shared.read(args.targets_json)["keys"]
    if (not 1 <= len(keys) <= 16 or len(keys) != len(set(keys))
            or keys != [row["key"] for row in source_plan["selection"]]):
        raise ValueError("retain every frozen collection target in its original order; no resampling")
    source_rows = shared.read(source / "predictions.json")
    by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in source_rows}
    if len(by_key) != len(source_rows) or set(by_key) != set(keys):
        raise ValueError("source predictions do not cover every selected target exactly once")
    hashes = {str(source / name): shared.sha(source / name)
              for name in ("plan.json", "predictions.json", "summary.json")}
    hashes[str(args.targets_json.resolve())] = shared.sha(args.targets_json)
    hashes[str(args.pricing_snapshot.resolve())] = shared.sha(args.pricing_snapshot)
    batch_path = args.names_batch.resolve()
    batch_plan = shared.read(batch_path / "plan.json")
    unsigned_batch = {key: value for key, value in batch_plan.items() if key != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned_batch)).hexdigest() != batch_plan["plan_sha256"]
            or Path(batch_plan["source"]).resolve() != source or batch_plan["source_split"] != "Training"
            or batch_plan["keys"] != keys or "names1000" not in batch_plan["variants"]):
        raise ValueError("original names batch does not match this frozen Training collection")
    hashes[str(batch_path / "plan.json")] = shared.sha(batch_path / "plan.json")
    endpoint, reservation = shared.checked_pricing(shared.read(args.pricing_snapshot))
    names_requests, split_requests, rows, selection = {}, {}, [], []
    for key in keys:
        item, old = selected[key], by_key[key]
        if item["source_split"] != "Training":
            raise ValueError("target split is not Training")
        _, restored = shared.restore_target(source, item, old, hashes)
        rows.append({"video_id": old["video_id"], "frame_id": old["frame_id"],
                     "h0": old["h0"], "h1": old["h1"], "variants": {}})
        if restored is None:
            continue
        base_review, evidence, full_ref = restored
        names = shared.variant_request(base_review, "names1000")
        baseline_path = batch_path / "requests" / key / "names1000.json"
        original_names = shared.read(baseline_path)
        names_wire_sha256 = checked_saved_names_wire(original_names, names)
        hashes[str(baseline_path)] = shared.sha(baseline_path)
        parts = split_factored_presence_request(names)
        names_requests[key] = names
        part_metadata = []
        for request in parts:
            shared.check_request_envelope(request)
            proposition_id = json.loads(request.payload["input_text"])["propositions"][0]["proposition_id"]
            split_requests[(key, proposition_id)] = request
            part_metadata.append({"proposition_id": proposition_id,
                                  "wire_sha256": hashlib.sha256(canonical_json_bytes(shared.wire_body(request))).hexdigest(),
                                  "request_metadata": canonical_request_metadata(request).to_mapping()})
        selection.append({"key": key, "full_frame_ref": full_ref, "evidence_manifest": evidence,
                          "source_names_request_metadata": canonical_request_metadata(names).to_mapping(),
                          "source_names_wire_sha256": names_wire_sha256,
                          "wire_identical_to_names_batch": len(parts) == 1 and part_metadata[0]["wire_sha256"] == names_wire_sha256,
                          "parts": part_metadata})
    count = len(split_requests)
    if count > args.max_calls:
        raise ValueError(f"frozen proposition count {count} exceeds explicit call cap {args.max_calls}")
    sources = shared._source_files() + [Path(__file__),
        ROOT / "src/surgical_agent/research/verification/factored_presence.py",
        ROOT / "docs/VERIFIER_FACTORED_CONDITIONAL_PROTOCOL_2026-09-07.md"]
    plan = {"schema_version": "factored_verifier_trial_plan_v1", "source": str(source),
            "source_split": "Training", "confirmation": False, "model": shared.MODEL,
            "variants": [VARIANT], "keys": keys, "target_count": len(keys), "selection": selection,
            "review_target_count": len(selection), "review_call_count": count,
            "max_provider_calls": args.max_calls, "retries": 0,
            "source_names_batch_call_count": len(selection),
            "source_names_batch": str(batch_path),
            "source_names_batch_plan_sha256": batch_plan["plan_sha256"],
            "execution_gate": "completed names batch plus actual saved wire/native-response binding; no scored GT read",
            "wire_identical_repeat_keys": [s["key"] for s in selection if s["wire_identical_to_names_batch"]],
            "singleton_interpretation": "one-proposition targets repeat identical wire input; differences measure repeat variability, not isolation",
            "additional_calls_vs_names_batch": count - len(selection),
            "maximum_output_tokens_per_request": 4096,
            "maximum_total_output_tokens_factored": 4096 * count,
            "maximum_total_output_tokens_names_batch": 4096 * len(selection),
            "factored_prompt_version": FACTORED_PRESENCE_PROMPT_VERSION,
            "goal_id": args.goal_id, "goal_budget_usd": str(shared.GOAL_CAP_USD),
            "development_budget_usd": str(shared.DEVELOPMENT_CAP_USD),
            "budget_ledger": str(args.budget_ledger.resolve()), "reservation": reservation,
            "timeout_seconds": args.timeout_seconds, "pricing": endpoint["pricing"],
            "source_artifact_sha256": hashes,
            "source_sha256": {str(path.relative_to(ROOT)): shared.sha(path) for path in sources},
            "gt_values_used_in_requests_or_selection": False,
            "merge_policy": "every original proposition exactly once; any missing/invalid part keeps whole H0",
            "interpretation": INTERPRETATION}
    if shared.save_plan(output, plan, sources):
        atomic_write_json(output / "frozen_predictions.json", rows)
        for selected_row in selection:
            key = selected_row["key"]
            names = names_requests[key]
            atomic_write_json(output / "source_names_requests" / f"{key}.json", {
                "metadata": canonical_request_metadata(names).to_mapping(), "payload": thaw_json(names.payload)})
            for part in selected_row["parts"]:
                identity = part["proposition_id"]
                request = split_requests[(key, identity)]
                atomic_write_json(output / "requests" / key / f"{identity}.json", {
                    "metadata": canonical_request_metadata(request).to_mapping(),
                    "payload": thaw_json(request.payload), "wire": shared.safe_wire(shared.wire_body(request))})
    print(json.dumps({"status": "PREFLIGHT_PASSED", "targets": len(keys), "factored_calls": count,
                      "names_batch_calls": len(selection), "provider_calls": 0,
                      "interpretation": INTERPRETATION}), flush=True)
    return plan, names_requests, split_requests, rows


def merge_and_evaluate(row, names_request, reviews):
    body = json.loads(names_request.payload["input_text"])
    merged = merge_factored_presence_reviews(names_request, reviews)
    result = evaluate_presence_review(row["h0"], row["h1"], merged,
        allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"],
        schema_version=PRESENCE_REVIEW_1000_VERSION)
    return {**result, "review": merged, "status": "OK" if merged is not None else "REVIEW_FAILURE_KEEP"}


def run(args):
    plan, names_requests, requests_by_part, rows = prepare(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("--execute requires an external API key file")
    # Completion and actual-wire guards run before public price lookup, key
    # resolution, budget reservation or any paid transport is entered.
    completion_binding = verify_names_completion(plan, names_requests, rows)
    atomic_write_json(args.output / "names_completion_binding.json", completion_binding)
    secret, ledger, calls = shared.start_execution(args, plan)
    by_key = {f"{row['video_id']}_{row['frame_id']}": row for row in rows}
    for key, row in by_key.items():
        row["variants"][VARIANT] = {"status": "NOT_ATTEMPTED" if key in names_requests else "NO_CHANGED_CANDIDATE",
            "final_a": deepcopy(row["h0"]), "final_b": deepcopy(row["h0"]),
            "decision_a": "KEEP", "decision_b": "KEEP", "review": None}
        row["factored_parts"] = []
    atomic_write_json(args.output / "predictions.json", rows)
    for selected in plan["selection"]:
        if calls.stopped:
            break
        key, reviews = selected["key"], []
        row = by_key[key]
        for part in selected["parts"]:
            if calls.stopped:
                break
            identity = part["proposition_id"]
            review = calls.call(key, identity, requests_by_part[(key, identity)])
            reviews.append(review)
            row["factored_parts"].append({"proposition_id": identity,
                "request_hash": part["request_metadata"]["request_hash"], "review": review,
                "status": "SHAPE_VALID_PENDING_MERGE" if review is not None else "REQUEST_FAILURE"})
            # Save every returned response immediately; GT remains unavailable.
            atomic_write_json(args.output / "predictions.json", rows)
        row["variants"][VARIANT] = merge_and_evaluate(row, names_requests[key], reviews)
        atomic_write_json(args.output / "predictions.json", rows)
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = shared.sha(args.output / "predictions.json")
    shared.assert_frozen(plan)
    for path, expected in completion_binding["source_artifact_sha256"].items():
        if shared.sha(path) != expected:
            raise ValueError("reference names completion evidence changed during factored inference")
    scored, comparisons = shared.score_saved_rows(args, rows, [VARIANT])
    atomic_write_json(args.output / "scored_predictions.json", scored)
    report = {"schema_version": "factored_verifier_trial_result_v1", "source_split": "Training",
              "model": shared.MODEL, "targets": len(rows), "variants": [VARIANT],
              "stop_reason": calls.stopped, "plan_sha256": plan["plan_sha256"],
              "predictions_sha256": predictions_hash, "comparisons": comparisons,
              "goal_budget": ledger.snapshot(), "interpretation": INTERPRETATION,
              "expected_factored_calls": plan["review_call_count"],
              "source_names_batch_call_count": plan["source_names_batch_call_count"],
              "successful_merged_reviews": sum(r["variants"][VARIANT]["status"] == "OK" for r in rows),
              **shared.accounting(calls.records)}
    if shared.sha(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", report)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--names-batch", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pricing-snapshot", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
