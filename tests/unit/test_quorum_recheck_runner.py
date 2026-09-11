"""Bounded shared rechecks use each policy's own state, with no API or GT."""
import json
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import pytest

from scripts import run_quorum_recheck_trial as trial
from surgical_agent.research.verification import recent_mean_panel as original
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def judgment(rating):
    return {"rating": rating,
            "finding": "MATCH" if rating >= 4 else "REFUTED" if rating <= 2 else "UNCLEAR",
            "scope": "WHOLE_FRAME", "image_indices": [2],
            "observation": "Synthetic visual judgment for deterministic coordinator tests."}


def raw_panel(pool, rating):
    return {seat: {"judgments": {p["id"]: judgment(rating) for p in pool["propositions"]}}
            for seat in SEATS}


def source_case(*, missing_targets=(), weak_existing=False):
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    proposal = {"instrument": [], "verb": [], "target": [8], "ivt": [19]}
    pool = make_pool(h0, proposal, make_pool(h0))
    raw = raw_panel(pool, 4)
    for seat in SEATS:
        for p in pool["propositions"]:
            if p["label_id"] not in h0[p["task"]]:
                raw[seat]["judgments"][p["id"]] = judgment(3)
        if weak_existing:
            raw[seat]["judgments"]["target_0"] = judgment(3)
        for pid in missing_targets:
            raw[seat]["judgments"][pid] = judgment(4)
    for pid in missing_targets:
        raw["deepseek"]["judgments"].pop(pid)
    reviews, formats = trial.normalize_five(raw, pool, 3)
    means, diagnostics = original.aggregate(reviews, pool, image_count=3)
    prediction = original.select(h0, pool, means, threshold=4)
    initial = {"key": "VID_SYNTHETIC_101", "h0": h0, "graph_r1": {
        "proposal": proposal, "pool": pool, "raw": raw, "reviews": reviews,
        "format_diagnostics": formats, "means": means, "diagnostics": diagnostics,
        "prediction": prediction}}
    return initial, trial.initial_policies(initial)


class FakeCalls:
    def __init__(self, responses, *, undispatched=(), stopped=False):
        self.responses = deepcopy(responses)
        self.undispatched = set(undispatched)
        self.stopped = stopped
        self.rows, self.invocations = [], []
        self.lock = Lock()

    def call(self, target, stage, seat, wire):
        with self.lock:
            self.invocations.append((target, stage, seat, deepcopy(wire)))
            if seat in self.undispatched:
                return None
            self.rows.append({"target": target, "stage": stage, "seat": seat})
        return deepcopy(self.responses[seat])


@pytest.fixture
def wire_spy(monkeypatch):
    seen = []

    def wire(seat, base, selected, pool):
        body = {"model": seat, "messages": [{"role": "user", "content": [
            {"type": "text", "text": json.dumps({"candidate_pool": pool})}]}]}
        seen.append({"seat": seat, "pool": deepcopy(pool), "body": deepcopy(body),
                     "selected": deepcopy(selected), "images": deepcopy(base.images)})
        return body

    monkeypatch.setattr(trial, "review_wire", wire)
    return seen


def run(initial, state, calls):
    return trial.run_target(calls, SimpleNamespace(images=["past2", "past1", "target"]),
                            {"key": initial["key"]}, initial, state)


def test_no_queue_does_not_build_requests_or_dispatch_api(wire_spy):
    initial, state = source_case()
    calls = FakeCalls(raw_panel(initial["graph_r1"]["pool"], 1))
    result = run(initial, state, calls)
    assert not calls.invocations and not calls.rows and not wire_spy
    assert result["review_calls_observed"] == 0
    assert result["reviewed"] is False and result["attempted"] is False
    for policy in trial.POLICIES:
        assert result["policies"][policy]["status"] == "NO_RECHECK_NEEDED"
        assert result["policies"][policy]["prediction"] == state["policies"][policy]["prediction"]


def test_only_eligible_policy_changes_other_keeps_its_own_different_r1(wire_spy):
    initial, state = source_case(missing_targets=("target_0", "target_8"))
    assert state["policies"]["q5"]["queue"] and not state["policies"]["q4"]["queue"]
    assert state["policies"]["q5"]["prediction"]["target"] == [0]
    assert state["policies"]["q4"]["prediction"]["target"] == [0, 8]
    frozen = deepcopy((initial, state))
    calls = FakeCalls(raw_panel(initial["graph_r1"]["pool"], 1))
    result = run(initial, state, calls)
    assert len(calls.rows) == 5
    assert result["policies"]["q5"]["prediction"]["target"] == []
    assert result["policies"]["q4"]["prediction"] == state["policies"]["q4"]["prediction"]
    assert result["policies"]["q4"]["status"] == "NO_RECHECK_NEEDED"
    assert (initial, state) == frozen


def test_both_eligible_share_exactly_one_original_full_pool_five_seat_panel(wire_spy):
    initial, state = source_case(weak_existing=True)
    assert all(data["queue"] for data in state["policies"].values())
    pool = deepcopy(initial["graph_r1"]["pool"])
    calls = FakeCalls(raw_panel(pool, 4))
    result = run(initial, state, calls)
    assert len(calls.rows) == len(calls.invocations) == len(wire_spy) == 5
    assert {r["seat"] for r in calls.rows} == set(SEATS)
    assert all(r["target"] == initial["key"] and r["stage"] == trial.STAGE for r in calls.rows)
    assert all(row["pool"] == pool for row in wire_spy)
    assert all(row["images"] == ["past2", "past1", "target"] for row in wire_spy)
    assert result["pool"] == pool and result["reviewed"] is True
    bodies = {row["seat"]: row["body"] for row in wire_spy}
    assert all(body == bodies[seat] for _, _, seat, body in calls.invocations)
    assert result["request_fingerprints"] == {s: trial.fingerprint(bodies[s]) for s in SEATS}
    for policy in trial.POLICIES:
        record = result["policies"][policy]
        assert record["before"] == state["policies"][policy]["prediction"]
        assert record["prediction"]["ivt"] == [7, 19]
        assert record["prediction"]["phase"] == initial["h0"]["phase"]
        assert record["status"] == "REVIEWED_NO_PENDING"


def test_incomplete_dispatch_falls_back_even_with_four_usable_responses(wire_spy):
    initial, state = source_case(weak_existing=True)
    calls = FakeCalls(raw_panel(initial["graph_r1"]["pool"], 4), undispatched=("deepseek",))
    result = run(initial, state, calls)
    assert len(calls.invocations) == 5 and result["review_calls_observed"] == 4
    assert result["attempted"] is True and result["reviewed"] is False
    assert result["raw"]["deepseek"] is None
    for policy in trial.POLICIES:
        record = result["policies"][policy]
        assert record["status"] == "INCOMPLETE_PANEL"
        assert record["prediction"] == state["policies"][policy]["prediction"]
        assert record["queue_after"] == state["policies"][policy]["queue"]


@pytest.mark.parametrize("missing", ["whole_response", "candidate_judgment"])
def test_five_dispatched_with_one_missing_semantic_vote_allows_only_quorum4(wire_spy, missing):
    initial, state = source_case(weak_existing=True)
    raw = raw_panel(initial["graph_r1"]["pool"], 4)
    if missing == "whole_response":
        raw["deepseek"] = None
    else:
        raw["deepseek"]["judgments"].pop("ivt_19")
    result = run(initial, state, FakeCalls(raw))
    assert result["review_calls_observed"] == 5 and result["reviewed"] is True
    assert result["policies"]["q4"]["means"]["ivt_19"] == 4
    assert result["policies"]["q5"]["means"]["ivt_19"] is None
    assert result["policies"]["q4"]["diagnostics"]["ivt_19"]["valid_count"] == 4
    assert result["policies"]["q4"]["prediction"]["ivt"] == [7, 19]
    assert result["policies"]["q5"]["prediction"]["ivt"] == [7]


def test_budget_stop_preserves_each_policy_without_dispatch(wire_spy):
    initial, state = source_case(weak_existing=True)
    calls = FakeCalls({}, stopped=True)
    result = run(initial, state, calls)
    assert not calls.invocations and not wire_spy
    for policy in trial.POLICIES:
        assert result["policies"][policy]["status"] == "BUDGET_STOPPED"
        assert result["policies"][policy]["prediction"] == state["policies"][policy]["prediction"]


def test_initial_policy_replay_binds_original_quorum5_and_derives_quorum4_from_h0():
    initial, state = source_case(missing_targets=("target_8",))
    assert state["policies"]["q5"]["prediction"] == initial["graph_r1"]["prediction"]
    assert state["policies"]["q4"]["prediction"]["target"] == [0, 8]
    altered = deepcopy(initial)
    altered["graph_r1"]["prediction"]["target"] = [0, 8]
    with pytest.raises(ValueError, match="quorum5 original prediction"):
        trial.initial_policies(altered)


def test_archived_feedback_is_one_replay_from_original_graph_r1_not_quorum4_r1():
    initial, state = source_case(missing_targets=("target_8",))
    before = deepcopy(initial["graph_r1"]["prediction"])
    assert before != state["policies"]["q4"]["prediction"]
    pool = initial["graph_r1"]["pool"]
    record = {"reviewed": True, "review_calls_observed": 5, "status": "UNRESOLVED",
              "raw": raw_panel(pool, 3), "pool": pool,
              "prediction": state["policies"]["q4"]["prediction"]}
    frozen = deepcopy((initial, record))
    result = trial.archived_feedback_replay(initial, record)
    assert result == before and result != state["policies"]["q4"]["prediction"]
    assert (initial, record) == frozen


@pytest.mark.parametrize("reviewed,count,status", [
    (False, 0, "NOT_ATTEMPTED"), (True, 4, "UNRESOLVED"),
    (True, 5, "SELECTION_FAILED"), (True, 5, "INCOMPLETE_PANEL"),
])
def test_ineligible_archived_replay_keeps_original_r1_without_parsing(reviewed, count, status):
    initial, _ = source_case(missing_targets=("target_8",))
    record = {"reviewed": reviewed, "review_calls_observed": count, "status": status}
    result = trial.archived_feedback_replay(initial, record)
    assert result == initial["graph_r1"]["prediction"]
    assert result is not initial["graph_r1"]["prediction"]
