from copy import deepcopy

import pytest

from scripts import run_joint_phase_feedback_trial as trial
from surgical_agent.research.verification.candidate_coordinator import make_pool


def setup():
    state = {"instrument": [0], "verb": [2], "target": [1], "ivt": [0], "phase": [2]}
    pool = trial.joint_pool(make_pool(state))
    means = {p["id"]: 4 if p["task"] != "phase" else 1 for p in pool["propositions"]}
    means["phase_3"] = 4.4
    return state, pool, means


def test_phase_requires_complete_panel_not_just_old_and_winner():
    state, pool, means = setup()
    means["phase_6"] = None
    out, decision = trial.select_joint(state, pool, means)
    assert out == state
    assert not decision["valid_panel"]


@pytest.mark.parametrize("changes,expected", [({}, 3), ({"phase_2": 4.4}, 2),
    ({"phase_4": 4.4}, 2), ({"phase_3": 3.9}, 2)])
def test_phase_unique_supported_winner(changes, expected):
    state, pool, means = setup()
    original = deepcopy(state)
    means.update(changes)
    out, _ = trial.select_joint(state, pool, means)
    assert out["phase"] == [expected]
    assert trial.four_pool(pool) == make_pool(state)
    assert state == original


def test_uncertainty_keeps_interactions_and_never_imputes_votes():
    state, pool, means = setup()
    means["verb_2"] = None
    means["ivt_0"] = None
    out, _ = trial.select_joint(state, pool, means)
    assert out["verb"] == [2] and out["ivt"] == [0]


def test_old_ivt_component_protection_is_retained():
    state, pool, means = setup()
    means["instrument_0"] = 1
    out, _ = trial.select_joint(state, pool, means)
    assert out["instrument"] == [0]


def test_new_pool_contains_all_seven_phases_without_duplicates():
    _, pool, _ = setup()
    assert trial.joint_pool(pool) == pool
    assert len([p for p in pool["propositions"] if p["task"] == "phase"]) == 7


def test_targeted_blocks_unproposed_additions_and_phase_drift():
    state, _, _ = setup()
    reviewed = deepcopy(state)
    reviewed["instrument"].append(4)
    reviewed["phase"] = [3]
    out, rejected = trial.apply_targeted_round(state, state, reviewed)
    assert out == state and len(rejected) == 2


def test_targeted_requires_both_repair_and_review():
    state, _, _ = setup()
    state["target"].append(14)
    proposed = deepcopy(state)
    proposed["target"].remove(14)
    proposed["verb"].append(1)
    reviewed = deepcopy(state)
    reviewed["target"].remove(14)
    out, _ = trial.apply_targeted_round(state, proposed, reviewed)
    assert out["target"] == [1] and out["verb"] == [2]


def test_targeted_retained_ivt_protects_components():
    state, _, _ = setup()
    proposed = deepcopy(state)
    proposed["instrument"] = []
    out, rejected = trial.apply_targeted_round(state, proposed, proposed)
    assert out == state and rejected[0]["reason"] == "RETAINED_IVT_COMPONENT"


def test_targeted_new_ivt_requires_accepted_components():
    empty = {k: [] for k in trial.TASKS}
    empty["phase"] = [2]
    proposed, _, _ = setup()
    reviewed = deepcopy(proposed)
    reviewed["target"] = []
    out, rejected = trial.apply_targeted_round(empty, proposed, reviewed)
    assert out["ivt"] == [] and rejected[0]["reason"] == "COMPONENT_NOT_ACCEPTED"


@pytest.mark.parametrize("variant", ["v2", "v3", "v4"])
def test_independent_repair_hides_previous_answers_preserves_schema(monkeypatch, variant):
    import json
    packet = {"current_prediction": "old answer", "candidate_pool": "old labels",
        "full_ontology": "ontology", "issues": [], "label_boundaries": "boundaries"}
    monkeypatch.setattr(trial, "gemini_proposal", lambda *a: {"messages": [{"content": [{"text": json.dumps(packet)}]}]})
    args = (None, None, None, {"propositions": []}, [{"candidate": {"task": "verb"}}], {"packet": "old hints"})
    old = trial.packet_of(trial.repair_wire(*args, "v1"))
    new = trial.packet_of(trial.repair_wire(*args, variant))
    assert new["response_schema"] == old["response_schema"]
    assert new["full_ontology"] == old["full_ontology"]
    assert new["tasks_to_reinspect"] == ["verb"]
    assert not set(new) & {"current_prediction", "candidate_pool", "review_feedback", "candidate_relation_hints", "issues"}
    assert (trial.PROMPTS / "joint_phase_feedback_v1.txt").read_bytes() == (trial.PROMPTS / f"joint_phase_feedback_{variant}.txt").read_bytes()


@pytest.mark.parametrize("stage,blind", [("joint_r1", False), ("joint_r2", True)])
def test_v3_blinds_second_review_only(monkeypatch, stage, blind):
    import json
    packet = {"current_prediction_hypothesis": {"phase": [1]}, "phase_recommendation_hypothesis": 2,
        "propositions": ["same pool"], "response_schema": "same schema", "instructions": "same instructions"}
    monkeypatch.setattr(trial, "joint_wire", lambda *a: {"messages": [{"content": [{"text": json.dumps(packet)}]}]})
    def inspect(calls, key, stage, bodies):
        for body in bodies.values():
            actual = trial.packet_of(body)
            assert ("current_prediction_hypothesis" not in actual) == blind
            assert ("phase_recommendation_hypothesis" not in actual) == blind
            for field in ("propositions", "response_schema", "instructions"):
                assert actual[field] == packet[field]
        raise RuntimeError("wire validated; no API")
    monkeypatch.setattr(trial, "panel_call", inspect)
    with pytest.raises(RuntimeError, match="wire validated"):
        trial.review_joint(None, None, {"key": "sample"}, None, None, 2, stage, "v3")
