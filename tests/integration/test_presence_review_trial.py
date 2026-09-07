"""Offline safeguards for the bounded paired trial; never contact a provider."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

from scripts import run_presence_review_trial as runner
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY
from tests.unit.test_grounded_pipeline import _base, _responses, _wire


def base():
    request = _base()
    payload = thaw_json(request.payload)
    payload["openrouter_routing_profile"] = "strict_alibaba"
    return replace(request, model_identifier=runner.MODEL, payload=payload)


def neutral_response(request, *, presence="UNCLEAR"):
    body = json.loads(request.payload["input_text"])
    return {"schema_version": request.response_schema_version, "assessments": [
        {"proposition_id": p["proposition_id"], "presence": presence,
         "observation": "Synthetic offline evidence; not a model result.",
         "evidence_refs": [body["full_frame_ref"]], "scope": "FRAME", "full_frame_reviewed": True}
        for p in body["propositions"]]}


def stage_response(responses, stage, request):
    payload = deepcopy(responses[stage])
    payload["schema_version"] = request.response_schema_version
    return payload


@pytest.mark.parametrize("slot", ["FIRST", "SECOND"])
def test_five_stage_chain_shares_frozen_h0_candidate_and_actual_crops(slot):
    original, responses, calls = base(), _responses(slot=slot), []
    digest = canonical_request_metadata(original).request_hash

    def call(stage, request):
        calls.append((stage, request))
        return neutral_response(request) if stage == "presence_review" else stage_response(responses, stage, request)

    row = runner.run_target(original, call, proposal_slot=slot)
    assert [s for s, _ in calls] == ["h0", "locator", "proposal", "review", "presence_review"]
    assert calls[0][1] is original
    assert canonical_request_metadata(original).request_hash == digest
    assert row["h1"]["ivt"] == [59] and row["h0"]["ivt"] == [60]
    assert row["old_final"] == row["h1"]
    assert row["final_a"] == row["final_b"] == row["h0"]
    old, new = calls[-2][1], calls[-1][1]
    assert [i.sha256 for i in old.images] == [i.sha256 for i in new.images]
    body = json.loads(new.payload["input_text"])
    assert _LABEL_BOUNDARY in new.payload["system_text"]
    assert body["crop_manifest"] == row["crop_manifest"]
    assert body["full_frame_ref"] == "frame:101"
    assert len(body["evidence_manifest"]) == len(new.images) == 4
    for forbidden in ("h0", "h1", "hypotheses", "operation", "change_id", "ADD", "REMOVE",
                      "proposal_instance_labels", "predicted_instance_regions"):
        assert forbidden not in new.payload["input_text"]


@pytest.mark.parametrize("failed_stage, expected", [("h0", 1), ("locator", 2), ("proposal", 3)])
def test_optional_failures_keep_original_and_do_not_invent_candidates(failed_stage, expected):
    responses, calls = _responses(), []

    def call(stage, request):
        calls.append(stage)
        return None if stage == failed_stage else stage_response(responses, stage, request)

    row = runner.run_target(base(), call, proposal_slot="FIRST")
    assert len(calls) == expected
    assert row["h1"] is None
    assert row["final_a"] == row["final_b"] == row["h0"]


def test_old_review_failure_does_not_prevent_independent_new_review():
    responses, calls = _responses(), []

    def call(stage, request):
        calls.append(stage)
        if stage == "review":
            return None
        return neutral_response(request) if stage == "presence_review" else stage_response(responses, stage, request)

    row = runner.run_target(base(), call, proposal_slot="FIRST")
    assert len(calls) == 5 and row["presence_status"] == "OK"
    assert row["old_final"] == row["h0"]


def test_wrong_neutral_proposition_binding_falls_back_to_h0():
    responses = _responses()

    def call(stage, request):
        if stage == "presence_review":
            response = neutral_response(request)
            response["assessments"].pop()
            return response
        return stage_response(responses, stage, request)

    row = runner.run_target(base(), call, proposal_slot="FIRST")
    assert row["presence_status"] == "REVIEW_FAILURE_KEEP"
    assert row["final_a"] == row["final_b"] == row["h0"]


def response(*, status=200, cost=.01, payload=None, model=runner.MODEL, provider="Alibaba", error=None):
    native = {"id": "gen-offline", "model": model, "provider": provider,
              "usage": {"cost": cost}, "choices": [{"message": {
                  "content": json.dumps(_wire() if payload is None else payload)}}]}
    if error:
        native["error"] = error
    return SimpleNamespace(status_code=status, text=json.dumps(native), json=lambda: native)


def calls(tmp_path, sender, cap=5):
    return runner.DirectCalls(tmp_path, cap, SecretValue("offline-test-secret"), sender=sender)


def test_direct_transport_freezes_parameters_redacts_image_bytes_and_enforces_cap(tmp_path):
    sent = []

    def sender(url, **kwargs):
        sent.append(json.loads(kwargs["data"]))
        assert url == runner.ENDPOINT
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == (15, 180)
        return response()

    client = calls(tmp_path, sender, cap=1)
    assert client.call("one", "h0", base()) is not None
    assert client.call("two", "h0", base()) is None
    assert len(sent) == client.used == 1
    body = sent[0]
    assert body["provider"] == {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
    assert body["reasoning"] == {"effort": "low"}
    assert body["temperature"] == 0 and body["max_tokens"] == 4096 and not body["stream"]
    saved = (tmp_path / "calls/one/h0/request.json").read_text()
    assert "offline-test-secret" not in saved and "data:image" not in saved
    assert "wire_sha256" in saved


def test_timeout_is_not_retried_unknown_cost_preserved_other_targets_continue(tmp_path):
    invoked = []

    def sender(*args, **kwargs):
        invoked.append(True)
        if len(invoked) == 1:
            raise requests.Timeout("offline-test-secret")
        return response()

    client = calls(tmp_path, sender)
    assert client.call("one", "h0", base()) is None
    assert client.call("two", "h0", base()) is not None
    report = runner.accounting(client.records)
    assert len(invoked) == 2 and client.stopped is None
    assert report["cost_usd"] is None and report["known_cost_usd"] == .01
    assert report["unpriced_calls"] == 1 and not report["accounting_complete"]
    assert "offline-test-secret" not in (tmp_path / "calls/one/h0/result.json").read_text()


@pytest.mark.parametrize("http_status", [401, 402, 403])
def test_fatal_auth_or_payment_stops_queued_calls(tmp_path, http_status):
    client = calls(tmp_path, lambda *a, **k: response(status=http_status, cost=None))
    assert client.call("one", "h0", base()) is None
    assert client.call("two", "h0", base()) is None
    assert client.used == 1 and client.stopped == f"FATAL_HTTP_{http_status}"


def test_shared_wire_schema_rejection_stops_but_individual_bad_response_does_not(tmp_path):
    client = calls(tmp_path, lambda *a, **k: response(status=400, cost=None,
                                                    error={"message": "unsupported schema uniqueItems"}))
    assert client.call("one", "h0", base()) is None
    assert client.call("two", "h0", base()) is None
    assert client.stopped == "FATAL_SHARED_REQUEST_FORMAT"
    other = calls(tmp_path / "other", lambda *a, **k: response(payload={"wrong": True}))
    assert other.call("one", "h0", base()) is None
    assert other.call("two", "h0", base()) is None
    assert other.used == 2 and other.stopped is None


@pytest.mark.parametrize("kwargs", [{"model": "another-model"}, {"provider": "other-provider"}])
def test_model_provider_identity_mismatch_is_not_accepted(tmp_path, kwargs):
    client = calls(tmp_path, lambda *a, **k: response(**kwargs))
    assert client.call("one", "h0", base()) is None
    assert client.records[0]["error"] == "model_or_provider_identity_mismatch"


def test_wire_schema_removes_only_unique_items_without_mutating_full_schema():
    schema = {"uniqueItems": True, "items": {"type": "integer", "minimum": 0},
              "allOf": [{"uniqueItems": True, "maxItems": 3}]}
    saved = deepcopy(schema)
    assert runner.strip_unique_items(schema) == {"items": {"type": "integer", "minimum": 0},
                                               "allOf": [{"maxItems": 3}]}
    assert schema == saved


def test_nonobject_native_response_is_a_persisted_failure_not_a_batch_abort(tmp_path):
    client = calls(tmp_path, lambda *a, **k: SimpleNamespace(status_code=200, text="[]", json=list))
    assert client.call("one", "h0", base()) is None
    assert client.call("two", "h0", base()) is None
    assert client.used == 2 and client.stopped is None
    assert all(r["error"] == "TypeError" for r in client.records)


def test_preflight_mask_read_never_accesses_gt_label_values():
    class MaskOnly:
        mask = SimpleNamespace(instrument=True, verb=False, target=True, ivt=False, phase=True)

        def __getattr__(self, name):
            raise AssertionError("GT label accessed before inference: " + name)

    result = runner._mask_only(SimpleNamespace(frame_supervision=None, evaluation=SimpleNamespace(
        instance_supervision_available=True, instances=[MaskOnly(), MaskOnly()])))
    assert result == {"instrument": True, "verb": False, "target": True, "ivt": False, "phase": True}


def test_paired_scoring_retains_failed_targets_and_excludes_masked_heads():
    target = SimpleNamespace(mask=SimpleNamespace(instrument=True, verb=False, target=False,
                                                  ivt=False, phase=True),
                             instrument_ids=(2,), phase_id=3)

    class Adapter:
        def iter_video(self, video, frame_ids):
            for frame in frame_ids:
                yield SimpleNamespace(inference=SimpleNamespace(target_frame_id=frame),
                                      frame_supervision=target, evaluation=None)

    row = {"video_id": "VID103", "frame_id": 101, "h0": None, "h1": None,
           "old_final": None, "final_a": None, "final_b": None}
    comparisons, scored = runner.score_all(Adapter(), [row])
    for arm in runner.ARMS:
        metrics = comparisons[arm]["arms"]["final"]["tasks"]
        assert metrics["phase"]["valid_targets"] == 1
        assert metrics["phase"]["exact_set_accuracy"] == 0
        assert metrics["verb"]["valid_targets"] == 0 and metrics["verb"]["micro_f1"] is None
    assert scored[0]["gt"]["verb"] is None


def test_unexpected_implementation_error_preserves_known_h0_and_halts(tmp_path, monkeypatch):
    client = calls(tmp_path, lambda *a, **k: response())
    selected = {"key": "VID103_101", "video_id": "VID103", "frame_id": 101, "proposal_slot": "FIRST"}

    def broken(base, callback, **kwargs):
        callback("h0", base)
        raise RuntimeError("deliberate offline implementation failure")

    monkeypatch.setattr(runner, "run_target", broken)
    index, row = runner.run_work(3, selected, base(), client)
    assert index == 3 and row["status"] == "IMPLEMENTATION_ERROR"
    assert row["h0"]["ivt"] == [60]
    assert row["final_a"] == row["final_b"] == row["old_final"] == row["h0"]
    assert client.stopped == "UNEXPECTED_IMPLEMENTATION_ERROR"
    assert client.call("second", "h0", base()) is None
    assert client.used == 1


def test_review_limit_versions_match_wire_local_and_admission_without_changing_inputs():
    frozen_requests = {}
    for limit in (300, 1000):
        sent = {}
        responses = _responses()

        def call(stage, request, *, sent=sent, limit=limit, responses=responses):
            sent[stage] = request
            wire = runner.wire_body(request)
            schema = wire["response_format"]["json_schema"]["schema"]
            if stage == "presence_review":
                assert schema["properties"]["assessments"]["items"]["properties"]["observation"]["maxLength"] == limit
                payload = neutral_response(request)
                for item in payload["assessments"]:
                    item["observation"] = "e" * limit
            else:
                payload = stage_response(responses, stage, request)
                if stage == "review":
                    assert schema["properties"]["instances"]["items"]["properties"]["distinguishing_observation"]["maxLength"] == limit
                    for item in payload["instances"]:
                        item["distinguishing_observation"] = "e" * limit
                elif stage == "proposal":
                    assert schema["properties"]["instances"]["items"]["properties"]["contact_observation"]["maxLength"] == 300
            runner.validator_for(request.response_schema_version)(payload)
            return payload

        row = runner.run_target(base(), call, proposal_slot="FIRST", review_observation_max_chars=limit)
        assert row["presence_status"] == "OK" and row["old_final"] == row["h1"]
        assert row["final_a"] == row["final_b"] == row["h0"]
        frozen_requests[limit] = sent
    for stage in ("h0", "locator", "proposal"):
        assert canonical_request_metadata(frozen_requests[300][stage]).request_hash == canonical_request_metadata(frozen_requests[1000][stage]).request_hash
    for stage in ("review", "presence_review"):
        old, new = frozen_requests[300][stage], frozen_requests[1000][stage]
        assert old.payload["input_text"] == new.payload["input_text"]
        assert old.images == new.images and old.generation_parameters == new.generation_parameters
        old_instruction = old.payload["system_text"].split("Required output schema:")[0]
        new_instruction = new.payload["system_text"].split("Required output schema:")[0]
        assert new_instruction == old_instruction.replace(old.response_schema_version, new.response_schema_version)
