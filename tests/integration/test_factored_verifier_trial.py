"""Offline orchestration checks for the optional per-proposition runner."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import run_factored_verifier_trial as runner
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.research.verification.factored_presence import (
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
)
from tests.integration.test_presence_review_trial import base, stage_response
from tests.unit.test_grounded_pipeline import _responses


def source_request():
    responses = _responses()
    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)
    original = base()
    row = runner.shared.collect_target(original, Calls(), "VID103_101")
    numeric, _, _ = runner.shared.build_review_request(original, row)
    return runner.shared.variant_request(numeric, "names1000"), row


def part_review(request, row):
    body = json.loads(request.payload["input_text"])
    proposition = body["propositions"][0]
    return {"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": [{
        "proposition_id": proposition["proposition_id"],
        "presence": "PRESENT" if proposition["label_id"] in row["h1"][proposition["task"]] else "ABSENT",
        "observation": "Synthetic offline fixture, not a model result.", "evidence_refs": [body["full_frame_ref"]],
        "scope": "FRAME", "full_frame_reviewed": True,
    }]}


def test_full_merge_preserves_original_admission_and_wrong_binding_keeps_whole_frame():
    names, row = source_request()
    parts = split_factored_presence_request(names)
    reviews = [part_review(request, row) for request in parts]
    accepted = runner.merge_and_evaluate(row, names, reviews)
    assert accepted["decision_a"] == "ACCEPT" and accepted["final_a"] == row["h1"]
    wrong = deepcopy(reviews)
    wrong[0]["assessments"][0]["proposition_id"] = "p040"
    refused = runner.merge_and_evaluate(row, names, wrong)
    assert refused["status"] == "REVIEW_FAILURE_KEEP"
    assert refused["final_a"] == refused["final_b"] == row["h0"]
    missing = runner.merge_and_evaluate(row, names, reviews[:-1])
    assert missing["final_a"] == missing["final_b"] == row["h0"]


def test_singleton_wire_is_identical_and_only_local_metadata_version_changes():
    names, _ = source_request()
    payload = thaw_json(names.payload)
    body = json.loads(payload["input_text"])
    body["propositions"] = body["propositions"][:1]
    payload["input_text"] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    singleton = replace(names, payload=payload)
    (part,) = split_factored_presence_request(singleton)
    assert canonical_json_bytes(runner.shared.wire_body(singleton)) == canonical_json_bytes(runner.shared.wire_body(part))
    assert canonical_request_metadata(singleton).request_hash != canonical_request_metadata(part).request_hash


@pytest.mark.parametrize("split", ["Validation", "Testing"])
def test_nontraining_sources_are_rejected_before_any_other_data_or_calls(tmp_path, split):
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text(json.dumps({"source_split": split}), encoding="utf-8")
    with pytest.raises(ValueError, match="Training only"):
        runner.prepare(SimpleNamespace(source=source, output=tmp_path / "run", max_calls=45, timeout_seconds=180))


@pytest.mark.parametrize("stop_after_first", [False, True])
def test_outputs_saved_before_scoring_and_budget_stop_retains_all_targets(tmp_path, monkeypatch, stop_after_first):
    names, first = source_request()
    parts = split_factored_presence_request(names)
    output = tmp_path / "run"
    output.mkdir()
    key = "VID103_101"
    second = {"video_id": "VID103", "frame_id": 126, "h0": deepcopy(first["h0"]), "h1": None, "variants": {}}
    rows = [{name: deepcopy(first[name]) for name in ("video_id", "frame_id", "h0", "h1")}, second]
    rows[0]["variants"] = {}
    metadata = [{"proposition_id": json.loads(request.payload["input_text"])["propositions"][0]["proposition_id"],
                 "request_metadata": canonical_request_metadata(request).to_mapping()} for request in parts]
    plan = {"selection": [{"key": key, "parts": metadata}], "plan_sha256": "offline",
            "review_call_count": len(parts), "source_names_batch_call_count": 1}
    by_part = {(key, item["proposition_id"]): request for item, request in zip(metadata, parts, strict=True)}
    events = []
    class Calls:
        def __init__(self):
            self.stopped, self.records = None, []
        def call(self, target_key, stage, request):
            events.append((target_key, stage))
            self.records.append({"stage": stage, "status": "OK", "cost_usd": .01})
            if stop_after_first:
                self.stopped = "SHARED_GOAL_BUDGET_STOP"
            return part_review(request, first)
    calls = Calls()
    monkeypatch.setattr(runner, "prepare", lambda args: (plan, {key: names}, by_part, rows))
    monkeypatch.setattr(runner, "verify_names_completion", lambda *args: {"source_artifact_sha256": {}})
    monkeypatch.setattr(runner.shared, "start_execution", lambda *args: (
        SecretValue("offline-test-only"), SimpleNamespace(snapshot=dict), calls))
    monkeypatch.setattr(runner.shared, "assert_frozen", lambda plan: None)
    def score(args, scored_rows, variants):
        saved = json.loads((output / "predictions.json").read_text())
        assert saved == scored_rows and len(saved) == 2
        assert calls.stopped
        assert variants == [runner.VARIANT]
        assert saved[1]["variants"][runner.VARIANT]["status"] == "NO_CHANGED_CANDIDATE"
        if stop_after_first:
            assert saved[0]["variants"][runner.VARIANT]["final_b"] == saved[0]["h0"]
        else:
            assert len(saved[0]["factored_parts"]) == len(parts)
            assert saved[0]["variants"][runner.VARIANT]["status"] == "OK"
        return saved, {"offline": True}
    monkeypatch.setattr(runner.shared, "score_saved_rows", score)
    report = runner.run(SimpleNamespace(output=output, execute=True, api_key_file=tmp_path / "unused"))
    assert len(events) == (1 if stop_after_first else len(parts))
    assert report["targets"] == 2
    assert report["successful_merged_reviews"] == (0 if stop_after_first else 1)
    assert "total calls/output-token" in report["interpretation"]


def test_saved_names_safe_wire_is_checked_in_addition_to_metadata_and_payload():
    names, _ = source_request()
    saved = {"metadata": canonical_request_metadata(names).to_mapping(),
             "payload": thaw_json(names.payload),
             "wire": runner.shared.safe_wire(runner.shared.wire_body(names))}
    digest = runner.checked_saved_names_wire(saved, names)
    assert digest == hashlib.sha256(canonical_json_bytes(runner.shared.wire_body(names))).hexdigest()
    saved["wire"]["provider"]["allow_fallbacks"] = True
    with pytest.raises(ValueError, match="wire binding"):
        runner.checked_saved_names_wire(saved, names)


def completion_fixture(tmp_path):
    names, original = source_request()
    key = f"{original['video_id']}_{original['frame_id']}"
    source = tmp_path / "names_batch"
    directory = source / "calls" / key / "names1000"
    directory.mkdir(parents=True)
    parts = split_factored_presence_request(names)
    merged = runner.merge_and_evaluate(original, names, [part_review(part, original) for part in parts])["review"]
    row = {name: deepcopy(original[name]) for name in ("video_id", "frame_id", "h0", "h1")}
    completed = [{**row, "variants": {"names1000": {"review": merged}}}]

    def write(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    batch_plan = {"keys": [key]}
    batch_plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(batch_plan)).hexdigest()
    write(source / "plan.json", batch_plan)
    write(source / "predictions.json", completed)
    write(source / "summary.json", {"plan_sha256": batch_plan["plan_sha256"],
                                     "predictions_sha256": runner.shared.sha(source / "predictions.json")})
    metadata = canonical_request_metadata(names).to_mapping()
    wire = runner.shared.wire_body(names)
    wire_hash = hashlib.sha256(canonical_json_bytes(wire)).hexdigest()
    write(directory / "request.json", {"metadata": metadata, "payload": thaw_json(names.payload),
                                        "wire": runner.shared.safe_wire(wire), "wire_sha256": wire_hash})
    native = {"model": runner.shared.MODEL, "provider": "Alibaba", "id": "fixture-id",
              "choices": [{"message": {"content": json.dumps(merged)}}]}
    write(directory / "http_response.json", {"status_code": 200, "body": json.dumps(native)})
    write(directory / "result.json", {"key": key, "stage": "names1000", "provider_calls": 1,
                                       "http_status": 200, "status": "OK", "payload": merged,
                                       "request_hash": metadata["request_hash"], "provider_request_id": "fixture-id",
                                       "returned_model": runner.shared.MODEL, "provider": "Alibaba"})
    (directory / "dispatch.lock").write_text("Synthetic dispatch marker, no API called.", encoding="utf-8")
    plan = {"source_names_batch": str(source), "source_names_batch_plan_sha256": batch_plan["plan_sha256"],
            "keys": [key], "selection": [{"key": key, "source_names_wire_sha256": wire_hash,
                                          "source_names_request_metadata": metadata}]}
    return plan, {key: names}, [row], directory


def test_execution_gate_binds_completed_prediction_and_actual_wire_native_response(tmp_path):
    plan, names, rows, directory = completion_fixture(tmp_path)
    result = runner.verify_names_completion(plan, names, rows)
    assert result["gt_values_read"] is False
    assert len(result["reference_calls"]) == 1
    assert str(directory / "http_response.json") in result["source_artifact_sha256"]


@pytest.mark.parametrize("tamper", ["completion_hash", "wire_hash", "native_id", "native_payload"])
def test_execution_gate_rejects_tampered_reference_evidence(tmp_path, tamper):
    plan, names, rows, directory = completion_fixture(tmp_path)
    if tamper == "completion_hash":
        path = directory.parents[2] / "summary.json"
        payload = json.loads(path.read_text())
        payload["predictions_sha256"] = "wrong"
    elif tamper == "wire_hash":
        path = directory / "request.json"
        payload = json.loads(path.read_text())
        payload["wire_sha256"] = "wrong"
    else:
        path = directory / "http_response.json"
        payload = json.loads(path.read_text())
        native = json.loads(payload["body"])
        if tamper == "native_id":
            native["id"] = "different"
        else:
            native["choices"][0]["message"]["content"] = "{}"
        payload["body"] = json.dumps(native)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="binding|hash mismatch|differs"):
        runner.verify_names_completion(plan, names, rows)


def test_missing_names_completion_stops_before_budget_or_any_network_boundary(tmp_path, monkeypatch):
    names, row = source_request()
    plan = {"source_names_batch": str(tmp_path / "not_complete")}
    monkeypatch.setattr(runner, "prepare", lambda args: (plan, {"one": names}, {}, [row]))
    def forbidden(*args):
        pytest.fail("completion gate must precede price/key/budget/network work")
    monkeypatch.setattr(runner.shared, "start_execution", forbidden)
    with pytest.raises(ValueError, match="must complete"):
        runner.run(SimpleNamespace(execute=True, api_key_file=tmp_path / "unused"))
