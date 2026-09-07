import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from surgical_agent.api.contracts import canonical_json_bytes
from surgical_agent.research.verification.final_only_grounded import (
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_repair import REVIEW_VERSION
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_VERSION,
    build_presence_review_input,
)

SCRIPT = Path(__file__).resolve().parents[2] / "tools/audit/replay_review_length_only.py"
SPEC = importlib.util.spec_from_file_location("review_length_only_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _fixture(tmp_path, stage, *, invalid=False, approve=False):
    h0 = {"instrument": [2], "verb": [2], "target": [0], "ivt": [60], "phase": [3]}
    locator = {"schema_version": "contact_locator_v1", "all_visible_tools_covered": True,
               "instances": [{"instance_id": 1, "tip_box": [.2, .2, .4, .4]}]}
    proposal = {"schema_version": "instance_interaction_proposal_v1", "instances": [{
        "instance_id": 1, "instrument_id": 2, "ivt_ids": [59], "support": "SUPPORTED",
        "contact_observation": "Visible contact."}]}
    prepared = prepare_grounded_review(h0, locator, proposal, proposal_slot="FIRST")
    row = {"video_id": "VID02", "frame_id": 101, "h0": h0, "h1": prepared["h1"],
           "locator": locator, "proposal": proposal, "proposal_slot": "FIRST", "crop_manifest": []}
    body = {"video_id": "VID02", "target_frame_id": 101, "crop_manifest": []}
    if stage == "review":
        body.update(hypotheses=prepared["hypotheses"], predicted_instance_regions=locator)
        version = REVIEW_VERSION
        candidate = {"schema_version": version, "preferred": "FIRST", "all_visible_tools_covered": True,
                     "instances": [{"instance_id": 1, "crop_relevant": True,
                                    "instrument_identity_supported": True, "target_identity_or_oov_supported": True,
                                    "action_or_oov_supported": True, "distinguishing_observation": "x" * 301}]}
        if invalid:
            candidate["instances"][0]["instance_id"] = 2
    else:
        version = PRESENCE_REVIEW_VERSION
        evidence = [{"ref": "frame:101"}]
        neutral = build_presence_review_input(h0, row["h1"],
                                             allowed_evidence_refs=["frame:101"], full_frame_ref="frame:101")
        body.update(**neutral, evidence_manifest=evidence)
        candidate = {"schema_version": version, "assessments": [{
            "proposition_id": p["proposition_id"],
            "presence": (("PRESENT" if p["label_id"] in row["h1"][p["task"]] else "ABSENT")
                         if approve else "UNCLEAR"), "observation": "x" * 301,
            "evidence_refs": ["other"] if invalid else ["frame:101"], "scope": "FRAME",
            "full_frame_reviewed": True} for p in neutral["propositions"]]}
    payload = {"input_text": json.dumps(body)}
    metadata = {"provider": "openrouter", "endpoint_identifier": "endpoint",
                "requested_model_identifier": "qwen/test", "prompt_version": "test",
                "response_schema_version": version, "generation_parameters": {}, "images": []}
    metadata["request_hash"] = hashlib.sha256(canonical_json_bytes({**metadata, "payload": payload})).hexdigest()
    metadata["payload_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if stage == "review":
        row["review_binding"] = {"request_hash": metadata["request_hash"],
                                 "hypotheses": prepared["hypotheses"], "proposal_slot": "FIRST"}
    else:
        row["presence_binding"] = {"request_metadata": metadata, "evidence_manifest": evidence,
                                   "full_frame_ref": "frame:101"}
    source = tmp_path / "run/calls/VID02_101" / stage
    source.mkdir(parents=True)
    native = {"model": "qwen/test", "provider": "Alibaba", "id": "one",
              "choices": [{"message": {"content": json.dumps(candidate)}}]}
    replay.write_new(source / "request.json", {"metadata": metadata, "payload": payload})
    replay.write_new(source / "result.json", {"provider_calls": 1, "provider_request_id": "one",
                                              "request_hash": metadata["request_hash"]})
    replay.write_new(source / "http_response.json", {"status_code": 200, "body": json.dumps(native)})
    (source / "dispatch.lock").write_text("one", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    return row, candidate, tmp_path / "run", output


@pytest.mark.parametrize("stage", ["review", "presence_review"])
def test_only_observation_is_trimmed_and_full_contract_revalidated(tmp_path, stage):
    row, candidate, source, output = _fixture(tmp_path, stage)
    original = deepcopy(candidate)
    updates, audit = replay.replay_stage(row, stage, source, output)
    assert audit["status"] == "LENGTH_ONLY_REPLAY_VALID"
    assert candidate == original
    assert all(item["original_length"] == 301 and item["retained_length"] == 300
               for item in audit["changes"])
    if stage == "review":
        assert updates["old_final"] == row["h1"]
    else:
        assert updates["final_a"] == updates["final_b"] == row["h0"]
        assert len(updates["claims"]) == len(updates["claim_decisions"]) > 0


@pytest.mark.parametrize("stage", ["review", "presence_review"])
def test_other_invalid_binding_is_not_repaired(tmp_path, stage):
    row, _, source, output = _fixture(tmp_path, stage, invalid=True)
    updates, audit = replay.replay_stage(row, stage, source, output)
    assert updates is None
    assert audit["reason"] == "OTHER_INVALID_OR_BINDING_FAILURE"
    assert not list(output.iterdir())


def test_no_length_change_keeps_original_and_phase_ids_are_never_normalized():
    candidate = {"assessments": [{"observation": "short", "presence": "INVALID",
                                  "proposition_id": "invalid", "evidence_refs": ["unknown"]}]}
    normalized, changes = replay.trim_only(candidate, "presence_review")
    assert normalized == candidate and changes == []


def test_incomplete_run_refused_before_predictions_or_gt_read(tmp_path):
    with pytest.raises(ValueError, match="main run must be complete"):
        replay.run_replay(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_derived_decisions_and_claims_are_saved_before_gt_and_sources_unchanged(tmp_path, monkeypatch):
    row, _, source, _ = _fixture(tmp_path, "presence_review", approve=True)
    row.update(old_final=deepcopy(row["h0"]), final_a=deepcopy(row["h0"]),
               final_b=deepcopy(row["h0"]), presence_status="REVIEW_FAILURE_KEEP",
               decision_a="KEEP", decision_b="KEEP", claims=[], claim_decisions=[],
               stage_errors={"presence_review": "NO_VALID_RESPONSE"})
    replay.write_new(source / "predictions.json", [row])
    truth = {**row, "gt": deepcopy(row["h0"]), "mask": {task: True for task in row["h0"]}}
    replay.write_new(source / "scored_predictions.json", [truth])
    plan = {"source_sha256": {str(SCRIPT.relative_to(replay.ROOT)): replay.sha(SCRIPT)}}
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    replay.write_new(source / "plan.json", plan)
    replay.write_new(source / "summary.json", {
        "plan_sha256": plan["plan_sha256"], "predictions_sha256": replay.sha(source / "predictions.json"),
        "cost_usd": .01, "known_cost_usd": .01,
    })
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen fixture protocol: length only.", encoding="utf-8")
    before = {str(path): replay.sha(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "derived"
    original_read = replay.read
    observed = {"gt_reads": 0}

    def guarded_read(path):
        if Path(path) == source / "scored_predictions.json":
            observed["gt_reads"] += 1
            assert (output / "predictions.json").is_file()
            assert (output / "replay_changes.json").is_file()
            saved = original_read(output / "predictions.json")[0]
            assert "gt" not in saved and "mask" not in saved
            assert saved["final_a"] == row["h1"]
            assert saved["decision_a"] == "ACCEPT"
            assert len(saved["claims"]) == len(saved["claim_decisions"]) > 0
            assert any(item["applied_b"] for item in saved["claim_decisions"])
            assert saved["strict_original_review_metadata"]["claim_decisions"] == []
            assert saved["strict_original_review_metadata"]["final_a"] == row["h0"]
            assert saved["length_replay_metadata"]["stages"]["presence_review"]["status"] == "LENGTH_ONLY_REPLAY_VALID"
            assert before == {str(p): replay.sha(p) for p in source.rglob("*") if p.is_file()}
        return original_read(path)

    monkeypatch.setattr(replay, "read", guarded_read)
    report = replay.run_replay(source, output, protocol)
    assert observed["gt_reads"] == 1
    assert report["predictions_saved_before_gt_read"] is True
    assert before == {str(path): replay.sha(path) for path in source.rglob("*") if path.is_file()}
    assert original_read(output / "scored_predictions.json")[0]["gt"] == truth["gt"]
