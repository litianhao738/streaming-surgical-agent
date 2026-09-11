"""Unified admission: prior-gated four heads plus a jointly verified Phase; leakage guards."""
from copy import deepcopy

import pytest

from scripts import run_prior_gated_joint_confirmation as confirmation
from scripts import run_split_review_trial as common
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_gated_joint import (
    ARMS,
    PRIMARY,
    assert_ungated_request,
    decide,
    decide_phase,
    four_pool,
    joint_pool,
)
from surgical_agent.research.verification.prior_panel import BOUNDS


def prior(phase_rates):
    def rows(rates, eligible):
        return [{"id": c, "rate": rates.get(c, 0.0), "eligible": eligible and c in rates} for c in range(BOUNDS["ivt"])]
    table = {"excluded_video": "VIDX", "fit_videos": ["VIDA", "VIDB"], "tasks": {}}
    for task in ("instrument", "verb", "target"):
        table["tasks"][task] = {"global": [{"id": c, "rate": 0.5, "eligible": True} for c in range(BOUNDS[task])],
                                "phase": {}}
    table["tasks"]["ivt"] = {"global": rows({}, True), "phase": {"3": rows(phase_rates, True)}}
    return table


H0 = {"instrument": [2], "verb": [2], "target": [1], "ivt": [59], "phase": [3]}
GATE = {"veto_rate": 0.01, "add_rate": 0.7, "prune": []}


def review(pool, rating=3, phase_scores=None):
    """Five identical seats; compact reviews ignore phase rows, joint reviews rate all."""
    item = {"rating": rating, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [],
            "observation": "fixture"}
    judgments = {}
    for p in pool["propositions"]:
        if p["task"] == "phase":
            score = (phase_scores or {}).get(p["label_id"], 1)
            if score == 3:
                judgments[p["id"]] = dict(item, rating=3)
            else:
                judgments[p["id"]] = {"rating": score, "finding": "MATCH" if score >= 4 else "REFUTED",
                                      "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "fixture"}
        else:
            judgments[p["id"]] = dict(item)
    return {s: {"judgments": deepcopy(judgments)} for s in SEATS}


def votes(phase):
    return {s: {"phase_id": phase, "image_indices": [2], "observation": "fixture"} for s in SEATS}


def run(phase_scores, compact_rating=3, phase_raw=None):
    pool = make_pool(H0, {"instrument": [], "verb": [], "target": [], "ivt": [60]})
    return decide(H0, pool, prior({59: 0.001, 60: 0.8}), compact_raw=review(pool, compact_rating),
                  joint_raw=review(joint_pool(pool), phase_scores=phase_scores),
                  phase_raw=phase_raw if phase_raw is not None else votes(3), gate=GATE,
                  apply_phase_choices=common.apply_phase_choices, normalize_compact=common.normalize_five)


def test_joint_pool_adds_exactly_seven_phases_and_round_trips():
    pool = make_pool(H0)
    jpool = joint_pool(pool)
    assert [p["id"] for p in jpool["propositions"] if p["task"] == "phase"] == [f"phase_{i}" for i in range(7)]
    assert four_pool(jpool) == pool
    with pytest.raises(ValueError):
        joint_pool(jpool)


def test_primary_arm_keeps_gated_four_heads_and_takes_verified_phase():
    predictions, detail = run({1: 5})
    assert set(predictions) == set(ARMS)
    gated, primary = predictions["gated_control"], predictions[PRIMARY]
    assert gated["ivt"] == [60] and gated["phase"] == [3]  # 59 vetoed, 60 admitted, Phase frozen to H0
    assert {t: primary[t] for t in ("instrument", "verb", "target", "ivt")} == {t: gated[t] for t in ("instrument", "verb", "target", "ivt")}
    assert primary["phase"] == [1]
    assert detail["phase_decision"]["reason"] == "UNIQUE_BETTER_SUPPORTED_PHASE"
    assert detail["phase_decision"]["valid_panel"]


@pytest.mark.parametrize("scores,expected,reason", [
    ({3: 5}, 3, "CURRENT_PHASE_BEST"),
    ({1: 4, 5: 4}, 3, "TIED_PHASE_SUPPORT"),
    ({1: 3}, 3, "NO_SUPPORTED_ALTERNATIVE"),
    ({1: 5, 3: 5}, 3, "TIED_PHASE_SUPPORT"),
    ({1: 4, 3: 5}, 3, "CURRENT_PHASE_BEST"),
])
def test_phase_admission_keeps_current_without_unique_better_support(scores, expected, reason):
    predictions, detail = run(scores)
    assert predictions[PRIMARY]["phase"] == [expected]
    assert detail["phase_decision"]["reason"] == reason


def test_incomplete_phase_panel_never_switches():
    means = {f"phase_{i}": 1.0 for i in range(7)}
    means["phase_1"] = 5.0
    means["phase_6"] = None
    phase, decision = decide_phase(H0, means)
    assert phase == [3] and not decision["valid_panel"]
    assert decision["reason"] == "INVALID_CURRENT_PHASE_EVIDENCE"


def test_gate_bucket_is_h0_phase_even_when_verified_phase_differs():
    # Phase 1 bucket is absent from the fixture prior: a Phase-1 bucket would fall back to the global
    # table where 60 is ineligible, so admission of 60 proves the gate used the H0 Phase 3 bucket.
    predictions, detail = run({1: 5})
    assert predictions[PRIMARY]["ivt"] == [60] and detail["gate_log_control"]["phase_used"] == 3


def test_control_arm_still_uses_blind_majority_vote():
    predictions, _ = run({1: 5}, phase_raw=votes(5))
    assert predictions["control"]["phase"] == [5]
    assert predictions["gated_control"]["phase"] == [3] and predictions[PRIMARY]["phase"] == [1]


def test_request_guard_rejects_gated_labels_hints_and_ground_truth():
    assert_ungated_request({"current_prediction_hypothesis": deepcopy(H0), "propositions": []}, H0)
    gated = {**H0, "ivt": [60]}
    with pytest.raises(ValueError, match="other than H0"):
        assert_ungated_request({"current_prediction_hypothesis": gated}, H0)
    with pytest.raises(ValueError, match="prior hints"):
        assert_ungated_request({"candidate_relation_hints": []}, H0)
    with pytest.raises(ValueError, match="prior hints"):
        assert_ungated_request({"gt": {}}, H0)
    with pytest.raises(TypeError):
        assert_ungated_request(["not", "an", "object"], H0)


def test_recommendation_wire_strips_prior_hints(monkeypatch):
    captured = {}

    def fake_phase_proposal(base, selected, initial):
        packet = {"task": "x", "candidate_relation_hints": ["leak"], "current_prediction": initial["h0"]}
        return {"messages": [{"content": [{"text": confirmation.json.dumps(packet)}]}]}

    monkeypatch.setattr(confirmation.joint, "phase_proposal", fake_phase_proposal)
    body = confirmation.phase_recommendation_wire(None, None, H0, make_pool(H0), {"packet": ["leak"]})
    packet = confirmation.joint.packet_of(body)
    assert "candidate_relation_hints" not in packet
    assert_ungated_request(packet, H0)
    captured["ok"] = True
    assert captured["ok"]


@pytest.mark.parametrize("fatal,calls,keys", [(None, 17, ["a"]), ("ValueError", 18, ["a"]), (None, 18, [])])
def test_incomplete_confirmation_cannot_be_scored(tmp_path, fatal, calls, keys):
    confirmation.save(tmp_path / "plan.json", {"max_calls": 18, "selection": [{"key": "a"}], "video": "VIDX"})
    confirmation.save(tmp_path / "completion.json", {"fatal_error": fatal, "calls": calls})
    confirmation.save(tmp_path / "predictions.json", {"targets": [{"key": k} for k in keys]})
    with pytest.raises(ValueError, match="complete declared cohort"):
        confirmation.score(tmp_path, None)  # No truth adapter may be reached.


def test_plan_freezes_thresholds_and_stage_counts():
    assert confirmation.CALLS_PER_TARGET == 18
    assert confirmation.STAGES["joint_r1"] == 5 and confirmation.STAGES["phase_recommendation"] == 1
    assert confirmation.GATE == {"veto_rate": 0.01, "add_rate": 0.7, "prune": []}
    with pytest.raises(ValueError, match="frozen thresholds"):
        confirmation.verify_plan({"profile": confirmation.PROFILE, "models": confirmation.joint.roster.MODELS,
                                  "version": confirmation.VERSION, "gate": {**confirmation.GATE, "add_rate": 0.5,
                                                                            "version": confirmation.gated.GATE_VERSION},
                                  "phase_threshold": 4.0}, None)


@pytest.mark.parametrize("ledger_class", ["LedgerCalls", "ResumeCalls"])
def test_budget_write_retries_transient_file_lock(monkeypatch, ledger_class):
    from scripts import resume_prior_gated_joint_confirmation as recovery

    attempts = []

    def persist(_):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("transient file lock")
        return "saved"

    monkeypatch.setattr(confirmation.joint.roster.GLMCalls, "persist", persist)
    monkeypatch.setattr(confirmation.time, "sleep", lambda _: None)
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    cls = confirmation.LedgerCalls if ledger_class == "LedgerCalls" else recovery.ResumeCalls
    ledger = object.__new__(cls)
    assert ledger.persist() == "saved"
    assert len(attempts) == 2


def test_resume_replays_only_recorded_attempts_and_rejects_duplicates(tmp_path):
    from scripts import resume_prior_gated_joint_confirmation as recovery

    source = tmp_path / "source"
    limits = {"openrouter_usd": "3", "aliyun_cny": "2", "xai_usd": "0"}
    row = {"index": 0, "target": "a", "stage": "joint_r1", "seat": "gpt", "status": "API_FAILED"}
    confirmation.save(source / "budget.json", {"stopped": True, "limits": limits,
                      "occupied": {k: "0" for k in limits}, "calls": [row]})
    body = {"model": "example", "messages": []}
    confirmation.save(source / "calls/000_a_joint_r1_gpt/request.json", body)
    ledger = recovery.ResumeCalls(tmp_path / "out", source, {"limits": limits, "max_calls": 18}, offline=True)
    assert ledger.call("a", "joint_r1", "gpt", body) is None  # recorded failure is kept, never retried
    with pytest.raises(ValueError, match="duplicate"):
        ledger.call("a", "joint_r1", "gpt", body)
    with pytest.raises(ValueError, match="cannot dispatch"):
        ledger.call("b", "joint_r1", "gpt", body)
    changed = recovery.ResumeCalls(tmp_path / "out2", source, {"limits": limits, "max_calls": 18}, offline=True)
    with pytest.raises(ValueError, match="saved request differs"):
        changed.call("a", "joint_r1", "gpt", {**body, "model": "different"})
