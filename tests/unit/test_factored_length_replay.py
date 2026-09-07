"""Offline native binding and prediction-before-GT tests for length replay."""

import json
from copy import deepcopy

import pytest

from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.research.verification.factored_presence import (
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
)
from tests.integration.test_presence_review_trial import base, stage_response
from tests.unit.test_grounded_pipeline import _responses
from tools.audit import replay_factored_observation_length as replay


def source_request():
    responses = _responses()

    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)

    original = base()
    row = shared.collect_target(original, Calls(), "offline")
    numeric, _, _ = shared.build_review_request(original, row)
    return shared.variant_request(numeric, "names1000"), row


def save_part(run_dir, key, request, *, length=1001, invalid_ref=False):
    body = json.loads(request.payload["input_text"])
    identity = body["propositions"][0]["proposition_id"]
    metadata = canonical_request_metadata(request).to_mapping()
    wire = shared.wire_body(request)
    wire_hash = replay.hashlib.sha256(canonical_json_bytes(wire)).hexdigest()
    saved = {"metadata": metadata, "payload": thaw_json(request.payload),
             "wire": shared.safe_wire(wire), "wire_sha256": wire_hash}
    raw = {"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": [{
        "proposition_id": identity, "presence": "PRESENT", "observation": "x" * length,
        "evidence_refs": ["unprovided:frame" if invalid_ref else body["full_frame_ref"]],
        "scope": "FRAME", "full_frame_reviewed": True,
    }]}
    strict = None if length > 1000 else raw
    result = {"request_hash": metadata["request_hash"], "key": key, "stage": identity,
              "provider_calls": 1, "http_status": 200, "provider_request_id": "offline-native-id",
              "returned_model": shared.MODEL, "provider": "Alibaba",
              "status": "OK" if strict is not None else "FAILURE"}
    if strict is None:
        result["error"] = "ApiSchemaError"
    else:
        result["payload"] = strict
    native = {"model": shared.MODEL, "provider": "Alibaba", "id": "offline-native-id",
              "choices": [{"message": {"content": json.dumps(raw)}}]}
    directory = run_dir / "calls" / key / identity
    for name, value in (("request.json", saved), ("result.json", result),
                        ("http_response.json", {"status_code": 200, "body": json.dumps(native)})):
        replay.write_new(directory / name, value)
    (directory / "dispatch.lock").write_text("Synthetic offline evidence", encoding="utf-8")
    replay.write_new(run_dir / "requests" / key / f"{identity}.json", saved)
    part = {"proposition_id": identity, "wire_sha256": wire_hash, "request_metadata": metadata}
    original = {"proposition_id": identity, "request_hash": metadata["request_hash"],
                "review": strict, "status": "SHAPE_VALID_PENDING_MERGE" if strict else "REQUEST_FAILURE"}
    return part, original, raw


def overwrite_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bound_native_length_failure_is_recovered_without_other_value_changes(tmp_path):
    names, _ = source_request()
    request = split_factored_presence_request(names)[0]
    part, original, raw = save_part(tmp_path, "VID103_101", request, length=1296)
    strict, normalized, audit = replay.replay_part(tmp_path, "VID103_101", part, request, original)
    assert strict is None and audit["normalization_status"] == "LENGTH_RECOVERED"
    expected = deepcopy(raw)
    expected["assessments"][0]["observation"] = "x" * 1000
    assert normalized == expected and raw["assessments"][0]["observation"] == "x" * 1296


@pytest.mark.parametrize("tamper", ["model", "provider", "native_id", "request_payload", "wire_hash",
                                     "result_hash", "part_id", "missing_dispatch", "saved_review"])
def test_inconsistent_native_or_request_binding_aborts_instead_of_salvaging(tmp_path, tamper):
    names, _ = source_request()
    request = split_factored_presence_request(names)[0]
    key = "VID103_101"
    part, original, _ = save_part(tmp_path, key, request)
    directory = tmp_path / "calls" / key / part["proposition_id"]
    if tamper in {"model", "provider", "native_id"}:
        path = directory / "http_response.json"
        http = replay.read(path)
        native = json.loads(http["body"])
        native[{"native_id": "id"}.get(tamper, tamper)] = "mismatch"
        http["body"] = json.dumps(native)
        overwrite_json(path, http)
    elif tamper in {"request_payload", "wire_hash"}:
        path = directory / "request.json"
        saved = replay.read(path)
        if tamper == "request_payload":
            saved["payload"]["system_text"] += " changed"
        else:
            saved["wire_sha256"] = "0" * 64
        overwrite_json(path, saved)
    elif tamper == "result_hash":
        path = directory / "result.json"
        result = replay.read(path)
        result["request_hash"] = "0" * 64
        overwrite_json(path, result)
    elif tamper == "part_id":
        original["proposition_id"] = "p040"
    elif tamper == "missing_dispatch":
        (directory / "dispatch.lock").unlink()
    elif tamper == "saved_review":
        original["review"] = {"forged": True}
    with pytest.raises(ValueError):
        replay.replay_part(tmp_path, key, part, request, original)


def test_long_text_with_unprovided_reference_is_not_recovered(tmp_path):
    names, _ = source_request()
    request = split_factored_presence_request(names)[0]
    part, original, _ = save_part(tmp_path, "VID103_101", request, invalid_ref=True)
    strict, normalized, audit = replay.replay_part(tmp_path, "VID103_101", part, request, original)
    assert strict is normalized is None and audit["reason"] == "INVALID_PART_OR_EVIDENCE_BINDING"


def test_incomplete_run_is_rejected_before_reading_predictions_or_gt(tmp_path, monkeypatch):
    def forbidden_read(path):
        pytest.fail(f"incomplete run must not read {path}")

    monkeypatch.setattr(replay, "read", forbidden_read)
    with pytest.raises(ValueError, match="complete"):
        replay.completed(tmp_path)


def test_full_16_target_replay_persists_every_prediction_before_saved_masked_gt(tmp_path, monkeypatch):
    run_dir, output = tmp_path / "source", tmp_path / "supplement"
    run_dir.mkdir()
    names, old = source_request()
    key, parts, original_parts, strict_reviews = "VID103_101", [], [], []
    for index, request in enumerate(split_factored_presence_request(names)):
        part, original, _ = save_part(run_dir, key, request, length=1001 if index == 0 else 100)
        parts.append(part)
        original_parts.append(original)
        strict_reviews.append(original["review"])
    first = {"video_id": "VID103", "frame_id": 101, "h0": old["h0"], "h1": old["h1"],
             "factored_parts": original_parts, "variants": {}}
    first["variants"][replay.VARIANT] = replay.merge_and_evaluate(first, names, strict_reviews)
    rows = [first]
    for index in range(1, 16):
        unchanged = {"status": "NO_CHANGED_CANDIDATE", "final_a": old["h0"], "final_b": old["h0"],
                     "decision_a": "KEEP", "decision_b": "KEEP", "review": None}
        rows.append({"video_id": "VID103", "frame_id": 101 + index * 25,
                     "h0": old["h0"], "h1": old["h0"], "factored_parts": [],
                     "variants": {replay.VARIANT: unchanged}})
    plan = {"schema_version": "factored_verifier_trial_plan_v1", "source_split": "Training",
            "variants": [replay.VARIANT], "model": shared.MODEL, "target_count": 16,
            "keys": [replay.key_of(row) for row in rows], "source_sha256": {}, "source_artifact_sha256": {},
            "review_call_count": len(parts), "selection": [{"key": key, "parts": parts}]}
    plan["plan_sha256"] = replay.hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    replay.write_new(run_dir / "plan.json", plan)
    replay.write_new(run_dir / "predictions.json", rows)
    replay.write_new(run_dir / "summary.json", {"source_split": "Training", "variants": [replay.VARIANT],
        "model": shared.MODEL, "plan_sha256": plan["plan_sha256"], "targets": 16,
        "predictions_sha256": replay.sha(run_dir / "predictions.json"),
        "stop_reason": "INFERENCE_FINISHED", "provider_calls": len(parts)})
    scored = [{**row, "gt": old["h1"], "mask": dict.fromkeys(old["h1"], True)} for row in rows]
    scored[-1]["mask"]["ivt"] = False
    replay.write_new(run_dir / "scored_predictions.json", scored)
    before = {str(path): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic offline supplemental protocol", encoding="utf-8")
    original_read, gt_reads = replay.read, []

    def guarded_read(path):
        if path == run_dir / "scored_predictions.json":
            assert (output / "predictions.json").is_file()
            assert (output / "replay_changes.json").is_file()
            assert len(original_read(output / "predictions.json")) == 16
            gt_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(replay, "read", guarded_read)
    monkeypatch.setattr(replay, "reconstruct_names", lambda *args: ({key: names}, {}))
    monkeypatch.setattr(shared, "start_execution", lambda *a, **k: pytest.fail("no execution allowed"))
    report = replay.run_replay(run_dir, output, protocol)
    assert len(gt_reads) == 1 and report["new_provider_calls"] == report["new_cost_usd"] == 0
    assert report["status_counts"]["LENGTH_RECOVERED"] == 1
    assert report["comparisons"]["final_b"]["arms"]["final"]["tasks"]["ivt"]["valid_targets"] == 15
    assert all(path.read_bytes() == before[str(path)] for path in run_dir.rglob("*") if path.is_file())
    with pytest.raises(FileExistsError):
        replay.run_replay(run_dir, output, protocol)
