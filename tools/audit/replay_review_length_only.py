"""Post-run length-only sensitivity replay. No network, GT-guided edits or retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.api.contracts import canonical_json_bytes
from surgical_agent.api.errors import ApiError
from surgical_agent.research.verification.final_only_grounded import (
    finalize_grounded_repair,
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_repair import (
    REVIEW_SCHEMA,
    validate_grounded,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_SCHEMA,
    build_presence_review_input,
    evaluate_presence_review,
    validate_presence_review,
)

DEFAULT_PROTOCOL = ROOT / "artifacts/preflight/presence_review_readiness_20260907/length_only_replay_protocol.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def trim_only(review, stage):
    """Only clip existing strings at their original schema's maximum length."""
    normalized, changes = deepcopy(review), []
    array, field, schema = (
        ("instances", "distinguishing_observation", REVIEW_SCHEMA)
        if stage == "review" else ("assessments", "observation", PRESENCE_REVIEW_SCHEMA)
    )
    maximum = schema["properties"][array]["items"]["properties"][field]["maxLength"]
    if not isinstance(normalized, dict) or not isinstance(normalized.get(array), list):
        return normalized, changes
    for index, item in enumerate(normalized[array]):
        if isinstance(item, dict) and isinstance(item.get(field), str) and len(item[field]) > maximum:
            original = item[field]
            item[field] = original[:maximum]
            changes.append({"path": f"/{array}/{index}/{field}",
                            "original_length": len(original), "retained_length": maximum,
                            "original_text_sha256": hashlib.sha256(original.encode()).hexdigest()})
    return normalized, changes


def _request_binding(request):
    metadata, payload = request["metadata"], request["payload"]
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("request payload hash mismatch")
    body = {key: metadata[key] for key in (
        "provider", "endpoint_identifier", "requested_model_identifier", "prompt_version",
        "response_schema_version", "generation_parameters", "images")}
    body["payload"] = payload
    if hashlib.sha256(canonical_json_bytes(body)).hexdigest() != metadata["request_hash"]:
        raise ValueError("canonical request hash mismatch")
    return json.loads(payload["input_text"])


def replay_stage(row, stage, run_dir, output_dir):
    key = f"{row['video_id']}_{row['frame_id']}"
    source = run_dir / "calls" / key / stage
    audit = {"key": key, "stage": stage, "status": "UNCHANGED", "changes": []}
    if row["h0"] is None or row["h1"] is None:
        audit["reason"] = "NO_VALID_CANDIDATE"
        return None, audit
    if not all((source / filename).is_file() for filename in (
        "request.json", "result.json", "http_response.json", "dispatch.lock")):
        audit["reason"] = "NO_SAVED_HTTP_RESPONSE"
        return None, audit
    audit["source_sha256"] = {
        filename: sha(source / filename)
        for filename in ("request.json", "result.json", "http_response.json", "dispatch.lock")
    }
    try:
        response, request, result = (read(source / filename) for filename in (
            "http_response.json", "request.json", "result.json"))
        if response["status_code"] != 200:
            audit["reason"] = "NOT_HTTP_200"
            return None, audit
        native = json.loads(response["body"])
        if (native.get("model") != request["metadata"]["requested_model_identifier"]
                or native.get("provider") != "Alibaba" or not native.get("id")
                or result.get("provider_calls") != 1
                or result.get("provider_request_id") != native["id"]
                or result["request_hash"] != request["metadata"]["request_hash"]):
            raise ValueError("saved response identity mismatch")
        raw_review = json.loads(native["choices"][0]["message"]["content"])
        normalized, changes = trim_only(raw_review, stage)
        audit["changes"] = changes
        if not changes:
            audit["reason"] = "NO_LENGTH_CHANGE"
            return None, audit
        body = _request_binding(request)
        if body["video_id"] != row["video_id"] or body["target_frame_id"] != row["frame_id"]:
            raise ValueError("request target mismatch")
        prepared = prepare_grounded_review(
            row["h0"], row["locator"], row["proposal"], proposal_slot=row["proposal_slot"])
        if not prepared["review_required"] or prepared["h1"] != row["h1"]:
            raise ValueError("candidate binding mismatch")
        if stage == "review":
            binding = row["review_binding"]
            if (binding["request_hash"] != request["metadata"]["request_hash"]
                    or binding["hypotheses"] != prepared["hypotheses"]
                    or body["hypotheses"] != prepared["hypotheses"]
                    or binding["proposal_slot"] != row["proposal_slot"]
                    or body["predicted_instance_regions"] != row["locator"]
                    or body["crop_manifest"] != row["crop_manifest"]):
                raise ValueError("old review binding mismatch")
            validate_grounded(normalized, schema=REVIEW_SCHEMA)
            if {item["instance_id"] for item in normalized["instances"]} != {
                    item["instance_id"] for item in row["locator"]["instances"]}:
                raise ValueError("old review instance binding mismatch")
            decision = finalize_grounded_repair(
                row["h0"], row["locator"], row["proposal"], normalized,
                proposal_slot=row["proposal_slot"])
            updates = {"old_final": decision["final"], "review": normalized,
                       "old_length_replay_decision": decision["decision"],
                       "old_length_replay_reason": decision["reason"]}
        else:
            binding = row["presence_binding"]
            refs = [item["ref"] for item in binding["evidence_manifest"]]
            full = binding["full_frame_ref"]
            neutral = build_presence_review_input(
                row["h0"], row["h1"], allowed_evidence_refs=refs, full_frame_ref=full)
            if (binding["request_metadata"] != request["metadata"]
                    or any(body[name] != value for name, value in neutral.items())
                    or body["evidence_manifest"] != binding["evidence_manifest"]
                    or body["crop_manifest"] != row["crop_manifest"]):
                raise ValueError("presence review binding mismatch")
            validate_presence_review(normalized, h0=row["h0"], h1=row["h1"],
                                     allowed_evidence_refs=refs, full_frame_ref=full)
            decision = evaluate_presence_review(row["h0"], row["h1"], normalized,
                                                 allowed_evidence_refs=refs, full_frame_ref=full)
            updates = {name: decision[name] for name in (
                "final_a", "final_b", "decision_a", "decision_b", "reason_a", "reason_b",
                "claims", "claim_decisions")}
            updates.update(presence_review=normalized, presence_status="LENGTH_ONLY_REPLAY_VALID")
        normalized_path = output_dir / f"{key}_{stage}_normalized.json"
        write_new(normalized_path, normalized)
        audit.update(status="LENGTH_ONLY_REPLAY_VALID", reason="ORIGINAL_SCHEMA_AND_BINDING_VALID",
                     normalized_file=normalized_path.name, normalized_sha256=sha(normalized_path))
        return updates, audit
    except (ApiError, KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        audit.update(reason="OTHER_INVALID_OR_BINDING_FAILURE", error=type(exc).__name__)
        return None, audit


def run_replay(run_dir, output_dir, protocol=DEFAULT_PROTOCOL):
    run_dir, output_dir, protocol = Path(run_dir), Path(output_dir), Path(protocol)
    # A completion marker is required before even reading the saved predictions.
    if not (run_dir / "summary.json").is_file():
        raise ValueError("main run must be complete (summary.json required)")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary, plan = read(run_dir / "summary.json"), read(run_dir / "plan.json")
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned_plan)).hexdigest() != plan["plan_sha256"]
            or summary["plan_sha256"] != plan["plan_sha256"]):
        raise ValueError("completed plan binding mismatch")
    if sha(run_dir / "predictions.json") != summary["predictions_sha256"]:
        raise ValueError("completed predictions hash mismatch")
    for relative, expected in plan["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen code changed: {relative}")
    protocol_text = protocol.read_text(encoding="utf-8")
    rows = read(run_dir / "predictions.json")
    if any("gt" in row or "mask" in row for row in rows):
        raise ValueError("predictions input must be GT-free")
    output_dir.mkdir(parents=True)
    (output_dir / "length_only_replay_protocol.md").write_text(protocol_text, encoding="utf-8")
    derived, audit_rows = deepcopy(rows), []
    for source_row, derived_row in zip(rows, derived, strict=True):
        derived_row["strict_original_review_metadata"] = {
            key: deepcopy(source_row[key]) for key in (
                "old_final", "final_a", "final_b", "review", "presence_review",
                "presence_status", "decision_a", "decision_b", "reason_a", "reason_b",
                "claims", "claim_decisions", "admission", "stage_errors") if key in source_row
        }
        derived_row["length_replay_metadata"] = {
            "scope": "derived decisions; strict original metadata preserved separately",
            "original_stage_errors_scope": "strict_original_not_replay_status",
            "stages": {},
        }
        for stage in ("review", "presence_review"):
            updates, audit = replay_stage(source_row, stage, run_dir, output_dir)
            audit_rows.append(audit)
            derived_row["length_replay_metadata"]["stages"][stage] = deepcopy(audit)
            if updates is not None:
                derived_row.update(updates)
    write_new(output_dir / "predictions.json", derived)
    write_new(output_dir / "replay_changes.json", audit_rows)
    predictions_hash = sha(output_dir / "predictions.json")
    # All transformed predictions exist on disk before this first GT-value read.
    scored_source = read(run_dir / "scored_predictions.json")
    by_identity = {(row["video_id"], row["frame_id"]): row for row in scored_source}
    if len(by_identity) != len(rows) or len(scored_source) != len(rows):
        raise ValueError("scored identity coverage mismatch")
    scored = []
    for row in derived:
        truth = by_identity[(row["video_id"], row["frame_id"])]
        scored.append({**row, "gt": truth["gt"], "mask": truth["mask"]})
    write_new(output_dir / "scored_predictions.json", scored)
    calls = {"new_provider_calls": 0, "new_cost_usd": 0,
             "original_run_cost_not_reallocated": True,
             "original_run_cost_usd": summary.get("cost_usd"),
             "original_known_cost_usd": summary.get("known_cost_usd")}
    write_new(output_dir / "calls_summary.json", calls)
    report = {
        "schema_version": "length_only_review_sensitivity_v1", "targets": len(derived),
        "analysis_type": "post_protocol_engineering_sensitivity_not_primary_result",
        "transform_uses_gt": False, "predictions_saved_before_gt_read": True,
        "predictions_sha256": predictions_hash, "new_provider_calls": 0,
        "stage_status_counts": dict(Counter(f"{item['stage']}:{item['status']}" for item in audit_rows)),
        "source_run": str(run_dir.resolve()),
        "source_sha256": {name: sha(run_dir / name) for name in (
            "summary.json", "plan.json", "predictions.json", "scored_predictions.json")},
        "protocol_sha256": sha(protocol), "replay_script_sha256": sha(__file__),
    }
    if sha(output_dir / "predictions.json") != predictions_hash:
        raise ValueError("derived predictions changed during GT association")
    write_new(output_dir / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(run_replay(args.run_dir, args.output_dir, args.protocol), ensure_ascii=False))


if __name__ == "__main__":
    main()
