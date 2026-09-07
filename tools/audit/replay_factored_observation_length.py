"""Completed Training-only factored sensitivity replay. No API or label-guided edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.factored_presence import (
    merge_factored_presence_reviews,
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_observation_length import (
    OBSERVATION_LENGTH_NORMALIZATION_VERSION,
    normalize_presence_observation_length,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
)
from tools.audit.audit_presence_trial import _edit_admission

VARIANT = "factored_names1000"
DEFAULT_PROTOCOL = ROOT / "docs/FACTORED_LENGTH_SENSITIVITY_PROTOCOL_2026-09-07.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def key_of(row):
    return f"{row['video_id']}_{row['frame_id']}"


def indexed(rows, keys):
    by_key = {key_of(row): row for row in rows}
    if len(by_key) != len(rows) or list(by_key) != keys:
        raise ValueError("saved rows must cover the complete original target order exactly once")
    return by_key


def completed(run_dir):
    """Check completion and GT-free input hashes before any scored file is read."""
    if not (run_dir / "summary.json").is_file():
        raise ValueError("factored run must complete before replay; summary.json required")
    plan, summary = read(run_dir / "plan.json"), read(run_dir / "summary.json")
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema_version") != "factored_verifier_trial_plan_v1"
            or plan.get("source_split") != "Training" or summary.get("source_split") != "Training"
            or plan.get("variants") != [VARIANT] or summary.get("variants") != [VARIANT]
            or plan.get("model") != shared.MODEL or summary.get("model") != shared.MODEL
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != plan["plan_sha256"]
            or summary.get("plan_sha256") != plan["plan_sha256"]
            or summary.get("predictions_sha256") != sha(run_dir / "predictions.json")
            or summary.get("stop_reason") != "INFERENCE_FINISHED"
            or summary.get("provider_calls") != plan["review_call_count"]
            or plan.get("target_count") != 16 or summary.get("targets") != 16
            or len(plan["keys"]) != 16 or len(set(plan["keys"])) != 16):
        raise ValueError("completed Training factored plan/prediction/dispatch binding mismatch")
    rows = read(run_dir / "predictions.json")
    indexed(rows, plan["keys"])
    if any("gt" in row or "mask" in row for row in rows):
        raise ValueError("replay prediction inputs must not contain GT or masks")
    for relative, expected in plan["source_sha256"].items():
        if sha(ROOT / relative) != expected or sha(run_dir / "frozen_source" / relative) != expected:
            raise ValueError(f"frozen implementation changed: {relative}")
    for path, expected in plan["source_artifact_sha256"].items():
        if sha(path) != expected:
            raise ValueError("a frozen source artifact changed")
    return plan, summary, rows


def check_request(saved, expected, *, expected_wire_hash=None):
    """Bind actual image bytes, canonical metadata, input text and transport body."""
    wire = shared.wire_body(expected)
    wire_hash = hashlib.sha256(canonical_json_bytes(wire)).hexdigest()
    if (saved["metadata"] != canonical_request_metadata(expected).to_mapping()
            or saved["payload"] != thaw_json(expected.payload)
            or ("wire" in saved and saved["wire"] != shared.safe_wire(wire))
            or ("wire_sha256" in saved and saved["wire_sha256"] != wire_hash)
            or (expected_wire_hash is not None and expected_wire_hash != wire_hash)):
        raise ValueError("saved request is not bound to original text/images/parameters/wire")
    return wire_hash


def reconstruct_names(run_dir, plan, rows):
    """Use the frozen H0/locator/proposal and real bytes, never scored labels."""
    source = Path(plan["source"])
    source_plan, original_rows = read(source / "plan.json"), read(source / "predictions.json")
    if source_plan.get("source_split") != "Training":
        raise ValueError("original collection is not Training")
    originals, current = indexed(original_rows, plan["keys"]), indexed(rows, plan["keys"])
    selected = {item["key"]: item for item in source_plan["selection"]}
    names, hashes = {}, {}
    for item in plan["selection"]:
        key = item["key"]
        if selected[key]["source_split"] != "Training":
            raise ValueError("individual target split is not Training")
        old = originals[key]
        if current[key]["h0"] != old["h0"] or current[key]["h1"] != old["h1"]:
            raise ValueError("factored H0/candidate differs from frozen collection")
        _, restored = shared.restore_target(source, selected[key], old, hashes)
        if restored is None:
            raise ValueError("a selected factored target has no changed valid candidate")
        expected = shared.variant_request(restored[0], "names1000")
        saved_path = run_dir / "source_names_requests" / f"{key}.json"
        check_request(read(saved_path), expected, expected_wire_hash=item["source_names_wire_sha256"])
        if canonical_request_metadata(expected).to_mapping() != item["source_names_request_metadata"]:
            raise ValueError("names source metadata differs from the frozen plan")
        hashes[str(saved_path)] = sha(saved_path)
        names[key] = expected
    if len(names) != len(plan["selection"]):
        raise ValueError("duplicate factored target selection")
    return names, hashes


def replay_part(run_dir, key, part, request, original_part):
    """Recover only a bound native response; malformed evidence aborts the replay."""
    identity = part["proposition_id"]
    directory = run_dir / "calls" / key / identity
    required = ("request.json", "result.json", "http_response.json", "dispatch.lock")
    if not all((directory / name).is_file() for name in required):
        raise ValueError("completed call lacks native response/dispatch evidence")
    request_path = run_dir / "requests" / key / f"{identity}.json"
    check_request(read(request_path), request, expected_wire_hash=part["wire_sha256"])
    saved, result, http = (read(directory / name) for name in required[:3])
    wire_hash = check_request(saved, request, expected_wire_hash=part["wire_sha256"])
    metadata = canonical_request_metadata(request).to_mapping()
    body = json.loads(request.payload["input_text"])
    if (metadata != part["request_metadata"] or saved.get("wire_sha256") != wire_hash or "wire" not in saved
            or len(body["propositions"]) != 1 or body["propositions"][0]["proposition_id"] != identity
            or original_part["proposition_id"] != identity
            or original_part["request_hash"] != metadata["request_hash"]
            or result.get("request_hash") != metadata["request_hash"]
            or result.get("key") != key or result.get("stage") != identity
            or result.get("provider_calls") != 1 or result.get("http_status") != http["status_code"]
            or original_part["review"] != result.get("payload")):
        raise ValueError("factored part/request/result identity mismatch")
    audit = {"key": key, "proposition_id": identity, "request_hash": metadata["request_hash"],
             "wire_sha256": wire_hash, "source_local_status": result["status"],
             "source_sha256": {name: sha(directory / name) for name in required},
             "changes": [], "normalization_status": "INELIGIBLE"}
    strict = result.get("payload")
    if http["status_code"] != 200:
        if strict is not None:
            raise ValueError("non-200 source cannot contain a valid local review")
        audit["reason"] = "NOT_HTTP_200"
        return strict, None, audit
    native = json.loads(http["body"])
    if (native.get("model") != shared.MODEL or native.get("provider") != "Alibaba"
            or not isinstance(native.get("id"), str) or not native["id"]
            or native["id"] != result.get("provider_request_id")
            or result.get("returned_model") != native["model"]
            or result.get("provider") != native["provider"]):
        raise ValueError("native HTTP response model/provider/request ID mismatch")
    audit["provider_request_id"] = native["id"]
    try:
        raw = json.loads(native["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError):
        if strict is not None:
            raise ValueError("valid local review contradicts unparsable native content") from None
        audit["reason"] = "NATIVE_CONTENT_NOT_JSON"
        return strict, None, audit
    if result["status"] == "OK":
        if strict != raw:
            raise ValueError("successful local payload differs from native response")
    elif strict is not None or result.get("error") != "ApiSchemaError":
        audit["reason"] = "NOT_A_SCHEMA_FAILURE"
        return strict, None, audit
    try:
        normalized, length_audit = normalize_presence_observation_length(raw)
    except (ApiSchemaError, TypeError, ValueError, OverflowError):
        if strict is not None:
            raise ValueError("originally valid local review fails unchanged v3 contract") from None
        audit["reason"] = "OTHER_SCHEMA_FAILURE"
        return strict, None, audit
    audit.update(length_audit)
    assessment = normalized["assessments"]
    if (len(assessment) != 1 or assessment[0]["proposition_id"] != identity
            or not set(assessment[0]["evidence_refs"]) <= set(body["allowed_evidence_refs"])):
        audit["reason"] = "INVALID_PART_OR_EVIDENCE_BINDING"
        return strict, None, audit
    if strict is None and not audit["changes"]:
        raise ValueError("schema-failed response needs no length normalization; inconsistent source")
    audit.update(normalization_status="LENGTH_RECOVERED" if audit["changes"] else "UNCHANGED_VALID",
                 reason="VALID_ORIGINAL_V3_AND_PART_BINDING")
    return strict, normalized, audit


def merge_and_evaluate(row, request, parts):
    body = json.loads(request.payload["input_text"])
    merged = merge_factored_presence_reviews(request, parts)
    result = evaluate_presence_review(row["h0"], row["h1"], merged,
        allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"],
        schema_version=PRESENCE_REVIEW_1000_VERSION)
    return {**result, "review": merged, "status": "OK" if merged is not None else "REVIEW_FAILURE_KEEP"}


def score_saved_only(run_dir, output_dir, source_rows, derived):
    """First GT read. All derived predictions and audits must already be saved."""
    predictions_path = output_dir / "predictions.json"
    if not predictions_path.is_file() or not (output_dir / "replay_changes.json").is_file():
        raise ValueError("derived predictions and audit must be persisted before GT association")
    before = sha(predictions_path)
    truth = indexed(read(run_dir / "scored_predictions.json"), [key_of(row) for row in source_rows])
    scored = []
    for source, row in zip(source_rows, derived, strict=True):
        original_scored = truth[key_of(row)]
        if (original_scored["h0"] != source["h0"] or original_scored["h1"] != source["h1"]
                or original_scored["variants"] != source["variants"]):
            raise ValueError("saved GT association does not belong to the completed predictions")
        if set(original_scored["mask"]) != {"instrument", "verb", "target", "ivt", "phase"}:
            raise ValueError("saved task mask is incomplete")
        scored.append({**row, "gt": original_scored["gt"], "mask": original_scored["mask"]})
    write_new(output_dir / "scored_predictions.json", scored)
    arms = ("strict_a", "strict_b", "final_a", "final_b")
    comparisons = {arm: compute_repair_comparison([{**row, "final": row[arm]} for row in scored]) for arm in arms}
    admission = {arm: _edit_admission(scored, arm) for arm in arms}
    if sha(predictions_path) != before:
        raise ValueError("derived predictions changed while associating GT")
    return comparisons, admission


def run_replay(run_dir, output_dir, protocol=DEFAULT_PROTOCOL):
    run_dir, output_dir, protocol = Path(run_dir).resolve(), Path(output_dir).resolve(), Path(protocol).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents or output_dir in run_dir.parents:
        raise ValueError("supplemental output must be separate from the original run")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plan, summary, source_rows = completed(run_dir)
    supplemental_sources = [Path(__file__), protocol,
        ROOT / "src/surgical_agent/research/verification/presence_observation_length.py",
        ROOT / "tools/audit/audit_presence_trial.py"]
    supplemental_hashes = {str(path): sha(path) for path in supplemental_sources}
    names, binding_hashes = reconstruct_names(run_dir, plan, source_rows)
    by_key = indexed(source_rows, plan["keys"])
    selected_keys = [item["key"] for item in plan["selection"]]
    if len(set(selected_keys)) != len(selected_keys) or not set(selected_keys) <= set(plan["keys"]):
        raise ValueError("invalid factored target selection")
    output_dir.mkdir(parents=True)
    (output_dir / "supplemental_protocol.md").write_bytes(protocol.read_bytes())
    derived, audits = [], []
    processed = {}
    for selected in plan["selection"]:
        key, row = selected["key"], by_key[selected["key"]]
        split = split_factored_presence_request(names[key])
        expected_ids = [part["proposition_id"] for part in selected["parts"]]
        if ([part["proposition_id"] for part in row["factored_parts"]] != expected_ids
                or len(set(expected_ids)) != len(expected_ids) or len(split) != len(expected_ids)):
            raise ValueError("completed factored row lacks its exact ordered part coverage")
        strict_parts, normalized_parts = [], []
        for part, request, original_part in zip(selected["parts"], split, row["factored_parts"], strict=True):
            strict, normalized, audit = replay_part(run_dir, key, part, request, original_part)
            strict_parts.append(strict)
            normalized_parts.append(normalized)
            audits.append(audit)
            if normalized is not None:
                path = output_dir / "normalized_parts" / key / f"{part['proposition_id']}.json"
                write_new(path, normalized)
                audit.update(normalized_path=str(path.relative_to(output_dir)), normalized_sha256=sha(path))
        strict = merge_and_evaluate(row, names[key], strict_parts)
        if strict != row["variants"][VARIANT]:
            raise ValueError("original strict decisions cannot be reproduced from saved part results")
        processed[key] = merge_and_evaluate(row, names[key], normalized_parts)
    if len(audits) != plan["review_call_count"]:
        raise ValueError("replay does not cover every original dispatch")
    for row in source_rows:
        key, strict = key_of(row), row["variants"][VARIANT]
        normalized = processed.get(key, strict)
        if key not in processed and (row["factored_parts"] or strict["status"] != "NO_CHANGED_CANDIDATE"
                                     or strict["final_a"] != row["h0"] or strict["final_b"] != row["h0"]):
            raise ValueError("unreviewed target is not a genuine original fallback")
        derived.append({"video_id": row["video_id"], "frame_id": row["frame_id"],
                        "h0": deepcopy(row["h0"]), "h1": deepcopy(row["h1"]),
                        "strict_original": deepcopy(strict), "normalized_review_result": normalized,
                        "strict_a": deepcopy(strict["final_a"]), "strict_b": deepcopy(strict["final_b"]),
                        "final_a": deepcopy(normalized["final_a"]), "final_b": deepcopy(normalized["final_b"])})
    write_new(output_dir / "predictions.json", derived)
    write_new(output_dir / "replay_changes.json", audits)
    write_new(output_dir / "request_binding_audit.json", binding_hashes)
    predictions_hash = sha(output_dir / "predictions.json")
    comparisons, admission = score_saved_only(run_dir, output_dir, source_rows, derived)
    report = {"schema_version": "factored_v3_length_sensitivity_v1", "source_split": "Training",
              "analysis_type": "supplemental_engineering_sensitivity_after_paired_training_gt_not_primary",
              "targets": len(derived), "source_run": str(run_dir),
              "source_plan_sha256": plan["plan_sha256"], "source_predictions_sha256": summary["predictions_sha256"],
              "source_file_sha256": {name: sha(run_dir / name) for name in (
                  "plan.json", "summary.json", "predictions.json", "scored_predictions.json")},
              "normalization_version": OBSERVATION_LENGTH_NORMALIZATION_VERSION,
              "protocol_sha256": sha(protocol), "script_sha256": sha(__file__),
              "supplemental_source_sha256": supplemental_hashes,
              "predictions_sha256": predictions_hash, "transform_uses_gt": False,
              "predictions_saved_before_gt_read": True, "new_provider_calls": 0, "new_cost_usd": 0,
              "original_known_cost_usd_not_reallocated": summary.get("known_cost_usd"),
              "original_cost_usd_not_reallocated": summary.get("cost_usd"),
              "status_counts": dict(Counter(item["normalization_status"] for item in audits)),
              "comparisons": comparisons, "edit_admission": admission}
    if sha(run_dir / "predictions.json") != summary["predictions_sha256"]:
        raise ValueError("original predictions changed during replay")
    if any(sha(path) != expected for path, expected in supplemental_hashes.items()):
        raise ValueError("supplemental implementation or protocol changed during replay")
    write_new(output_dir / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    report = run_replay(args.run_dir, args.output_dir, args.protocol)
    print(json.dumps({key: value for key, value in report.items() if key not in {"comparisons", "edit_admission"}}, ensure_ascii=False))
