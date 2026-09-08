"""Actual archived replay plus corruption checks; no GT, images or network."""
import json
from copy import deepcopy

import pytest

from scripts.replay_verified_repair_candidate import DEFAULT, ROOT, digest, replay


@pytest.fixture(scope="module")
def bundle():
    return json.loads(DEFAULT.read_text(encoding="utf-8"))


def reseal(bundle):
    bundle["payload_sha256"] = digest(bundle["payload"])
    return bundle


def test_actual_four_target_single_arm_replay(bundle, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("offline replay attempted network")

    monkeypatch.setattr(socket, "socket", forbidden)
    result = replay(bundle)
    assert result["verified"] and result["api_calls"] == 0
    assert result["targets"] == 4 and result["replayed_rounds"] == 6
    assert result["gt_independently_rescored"] is False
    assert result["metrics_from_archived_counts"]["evidence_feedback"]["ivt"]["micro_f1"] == 8 / 17


def test_payload_checksum_detects_edits(bundle):
    changed = deepcopy(bundle)
    changed["payload"]["targets"][0]["h0"]["ivt"] = []
    with pytest.raises(ValueError, match="checksum"):
        replay(changed)


def test_core_review_change_cannot_keep_archived_means(bundle):
    changed = deepcopy(bundle)
    record = changed["payload"]["targets"][0]["rounds"][0]
    raw = record["raw_reviews"]["grok"]
    item = next(iter(raw["judgments"].values()))
    item.update(rating=3, finding="UNCLEAR", scope="UNCERTAIN")
    record["reviewer_model_text"]["grok"] = json.dumps(raw)
    with pytest.raises(ValueError, match="normalized reviews"):
        replay(reseal(changed))


def test_truncated_unresolved_history_rejected(bundle):
    changed = deepcopy(bundle)
    target = next(t for t in changed["payload"]["targets"] if len(t["rounds"]) == 2)
    target["rounds"] = target["rounds"][:1]
    with pytest.raises(ValueError, match="truncated"):
        replay(reseal(changed))


def test_image_binding_rejects_future_frame(bundle):
    changed = deepcopy(bundle)
    changed["payload"]["targets"][0]["causal_frame_ids"][0] += 100
    with pytest.raises(ValueError, match="causal window"):
        replay(reseal(changed))


def test_feedback_text_must_match_actual_previous_judgment(bundle):
    changed = deepcopy(bundle)
    target = next(t for t in changed["payload"]["targets"] if len(t["rounds"]) == 2)
    target["rounds"][1]["review_evidence_feedback"]["instruction"] = "Assume reviewers are correct."
    with pytest.raises(ValueError, match="visual feedback"):
        replay(reseal(changed))


def test_identical_duplicate_core_json_is_compatible_without_new_vote(bundle):
    changed = deepcopy(bundle)
    record = changed["payload"]["targets"][0]["rounds"][0]
    raw = record["raw_reviews"]["grok"]
    value = next(iter(raw["judgments"].values()))["scope"]
    text = json.dumps(raw)
    needle = f'"scope": "{value}"'
    record["reviewer_model_text"]["grok"] = text.replace(needle, needle + ", " + needle, 1)
    assert replay(reseal(changed))["verified"]


def test_wrong_metric_arithmetic_is_rejected(bundle):
    changed = deepcopy(bundle)
    changed["payload"]["scores"]["evidence_feedback"]["ivt"]["metrics"]["micro_f1"] = 1.0
    with pytest.raises(ValueError, match="metric arithmetic"):
        replay(reseal(changed))


def test_release_source_guard_covers_runtime_and_resource(bundle):
    source = bundle["payload"]["replay_source_sha256_lf"]
    assert "src/surgical_agent/research/verification/recent_mean_panel.py" in source
    assert "src/surgical_agent/research/signals/resources/ivt_components_v1.csv" in source
    assert "src/surgical_agent/perception/prompts/perception_schema_gate_owned_compact.json" in source
    assert not any("credentials" in name or "run_evidence_feedback_trial" in name for name in source)
    changed = deepcopy(bundle)
    changed["payload"]["replay_source_sha256_lf"]["scripts/replay_verified_repair_candidate.py"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        replay(reseal(changed), root=ROOT)
