"""Exercise paid-stage bounds and exact full-panel sharing without a network."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_action_cue_candidate_trial as trial

H0 = {"schema_version": "joint_perception_final_only_v1", "instrument": {"selected_ids": [2]},
      "verb": {"selected_ids": [2]}, "target": {"selected_ids": [0]},
      "ivt": {"selected_ids": [60]}, "phase": {"selected_id": 3}}
EMPTY = {t: [] for t in trial.TASKS[:-1]}


class FakeCalls:
    def __init__(self, *, invalid_h0=False, different=False, invalid_seat=None):
        self.rows = []
        self.invalid_h0 = invalid_h0
        self.different = different
        self.invalid_seat = invalid_seat

    def call(self, target, stage, seat, body):
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        if stage == "h0":
            return None if self.invalid_h0 else deepcopy(H0)
        if stage == "shared_phase":
            return {"phase_id": 1, "image_indices": [2], "observation": "Visible current phase evidence."}
        if stage.endswith("_proposal"):
            return {**EMPTY, "verb": [1]} if self.different and stage.startswith("action_cue") else deepcopy(EMPTY)
        if seat == self.invalid_seat:
            return None
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [2],
                "observation": "Insufficient evidence."}
        if seat == "gemini":
            return {"rows": [{"candidate_id": p["id"], **item} for p in body["pool"]["propositions"]]}
        return {"judgments": {p["id"]: deepcopy(item) for p in body["pool"]["propositions"]}}


@pytest.fixture
def setup_wires(monkeypatch):
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda base: {"model": trial.PROPOSER})
    monkeypatch.setattr(trial, "retrieve_candidate_hints", lambda *a, **k: {"packet": {"frozen": True}})
    monkeypatch.setattr(trial, "proposal_pair", lambda *a: (
        {"control": {"arm": "control", "messages": []},
         "action_cue": {"arm": "action_cue", "messages": []}}, {"proxy": True}))
    monkeypatch.setattr(trial, "review_wire", lambda seat, base, selected, pool: {
        "seat": seat, "pool": pool, "messages": []})
    monkeypatch.setattr(trial, "phase_wire", lambda seat, selected: {"seat": seat, "messages": []})
    selected = {"key": "VID31_1001", "video_id": "VID31", "frame_id": 1001,
                "images": [{}, {}, {}], "arm_order": list(trial.ARMS)}
    return SimpleNamespace(images=[1, 2, 3]), selected


def test_same_panel_is_shared_including_failure(setup_wires):
    base, selected = setup_wires
    calls = FakeCalls(invalid_seat="gpt")
    row, records, shared = trial.run_target(calls, base, selected, {})
    assert len(calls.rows) == 13  # H0 + Phase5 + proposals2 + one review5.
    assert records["action_cue"]["shared_from"] == "control"
    assert records["control"]["raw"]["gpt"] is None
    assert records["action_cue"]["raw"]["gpt"] is None
    assert row["control"] == row["action_cue"]
    assert row["control"]["phase"] == [1]
    assert records["control"]["prediction"]["phase"] == [3]
    assert shared["phase"]["status"] == "REVIEWED"


def test_changed_pool_requires_all_five_fresh_reviews(setup_wires):
    base, selected = setup_wires
    calls = FakeCalls(different=True)
    row, records, _ = trial.run_target(calls, base, selected, {})
    assert len(calls.rows) == 18
    assert "shared_from" not in records["action_cue"]
    for arm in trial.ARMS:
        assert sum(c["stage"] == arm + "_review" for c in calls.rows) == 5
    assert row["control"]["phase"] == row["action_cue"]["phase"]


def test_failed_h0_retains_target_without_repair_calls(setup_wires):
    base, selected = setup_wires
    calls = FakeCalls(invalid_h0=True)
    row, records, shared = trial.run_target(calls, base, selected, {})
    assert len(calls.rows) == 1
    assert row["h0"] is None
    assert all(row[a] == trial.empty_prediction() for a in trial.ARMS)
    assert all(row["statuses"][a] == "H0_FAILED" for a in trial.ARMS)
    assert records == {}
    assert shared["status"] == "H0_FAILED"


def test_action_cue_first_shares_back_to_control(setup_wires):
    base, selected = setup_wires
    selected["arm_order"] = list(reversed(trial.ARMS))
    calls = FakeCalls()
    _, records, _ = trial.run_target(calls, base, selected, {})
    assert len(calls.rows) == 13
    assert records["control"]["shared_from"] == "action_cue"


def test_runtime_identity_comparison_accepts_json_roundtrip():
    trial.same(trial.ARMS, list(trial.ARMS), "JSON arrays")
    with pytest.raises(ValueError):
        trial.same({"model": "one"}, {"model": "two"}, "changed model")
