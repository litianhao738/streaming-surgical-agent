"""Continuation safety checks using archived inference only and zero transport."""

import json
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest

from scripts import run_prior_feedback_continuation as trial
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.candidate_coordinator import SEATS


@pytest.fixture(scope="module")
def archived():
    source = trial.DEFAULT_SOURCE
    if not (source / "predictions.json").is_file():
        pytest.skip("local archived eight-target inference is required")
    rows = []
    selected = {r["key"]: r for r in trial.read(source / "plan.json")["selection"]}
    for old in trial.read(source / "predictions.json")["targets"]:
        row = {k: deepcopy(old[k]) for k in ("key", "video_id", "frame_id", "h0")}
        row.update(control_r1=trial.read(source / "targets" / old["key"] / "control.json"),
                   graph_r1=trial.read(source / "targets" / old["key"] / "prior_graph.json"),
                   hints=trial.read(source / "targets" / old["key"] / "hints.json"))
        rows.append((row, selected[row["key"]]))
    return rows


@pytest.fixture
def unresolved(archived):
    return deepcopy(next(pair for pair in archived if pair[0]["graph_r1"]["status"] == "UNRESOLVED"))


@pytest.fixture
def base():
    return SimpleNamespace(
        images=[SimpleNamespace(mime_type="image/jpeg", content=b"offline-image") for _ in range(3)],
        payload={"image_details": ["low", "low", "high"]},
    )


def packet_wire(packet, *, model="fake-model"):
    return {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": json.dumps(packet, sort_keys=True)},
    ]}]}


@pytest.fixture
def tiny_wires(monkeypatch):
    def proposer(base, selected, initial, arm):
        return packet_wire({"target": selected["key"], "arm": arm})

    def reviewer(seat, base, selected, pool):
        return packet_wire({"target": selected["key"], "pool": pool}, model=seat)

    monkeypatch.setattr(trial, "proposal_wire", proposer)
    monkeypatch.setattr(trial, "review_wire", reviewer)
    return reviewer


def proposal(initial, offset=0):
    existing = {p["label_id"] for p in initial["graph_r1"]["pool"]["propositions"] if p["task"] == "target"}
    novel = [label for label in range(15) if label not in existing][offset]
    return {"instrument": [], "verb": [], "target": [novel], "ivt": []}


class FakeCalls:
    def __init__(self, proposed, *, failures=(), max_calls=999, stopped=False, rating=3):
        self.proposed, self.failures, self.max_calls = proposed, set(failures), max_calls
        self.stopped, self.rating = stopped, rating
        self.rows, self.requests, self.lock = [], [], RLock()

    def call(self, target, stage, seat, body):
        with self.lock:
            if self.stopped or len(self.rows) >= self.max_calls:
                self.stopped = True
                return None
            self.rows.append({"target": target, "stage": stage, "seat": seat})
            self.requests.append(deepcopy(body))
        if seat == "base":
            return deepcopy(self.proposed)
        if seat in self.failures:
            return None
        pool = json.loads(body["messages"][0]["content"][0]["text"])["pool"]
        return {"judgments": {p["id"]: {
            "rating": self.rating,
            "finding": "UNCLEAR" if self.rating == 3 else "MATCH",
            "scope": "UNCERTAIN" if self.rating == 3 else "LOCAL_REGION",
            "image_indices": [2], "observation": "Synthetic response for transport-free safety testing.",
        } for p in pool["propositions"]}}


def test_all_eight_archived_first_rounds_replay_without_gt(archived):
    assert len(archived) == 8
    before = deepcopy(archived)
    for row, _ in archived:
        trial.initial_row_replay(row)
    assert archived == before
    assert sum(row["graph_r1"]["status"] == "MODEL_PASS" for row, _ in archived) == 1


def test_inherited_model_pass_never_dispatches(archived, base, tiny_wires):
    initial, selected = next(pair for pair in archived if pair[0]["graph_r1"]["status"] == "MODEL_PASS")
    calls = FakeCalls(None)
    result, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "INHERITED_MODEL_PASS"
    assert result["prediction"] == initial["graph_r1"]["prediction"]
    assert calls.rows == [] and cache is None and result["round2_attempted"] is False


def test_unchanged_pool_stops_after_one_proposal(unresolved, base, tiny_wires):
    initial, selected = unresolved
    original = deepcopy(initial)
    calls = FakeCalls({task: [] for task in ("instrument", "verb", "target", "ivt")})
    result, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "NO_NEW_CANDIDATES"
    assert len(calls.rows) == 1 and result["round2_attempted"] is True
    assert result["reviewed"] is False and cache is None
    assert result["prediction"] == initial["graph_r1"]["prediction"] and initial == original


def test_novel_pool_dispatches_five_complete_reviews(unresolved, base, tiny_wires):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial))
    result, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert len(calls.rows) == 6 and {row["seat"] for row in calls.rows[1:]} == set(SEATS)
    assert result["pool"] != initial["graph_r1"]["pool"]
    assert result["reviewed"] is True and result["review_calls_observed"] == result["new_review_calls"] == 5
    assert result["prediction"] == initial["graph_r1"]["prediction"]
    assert result["prediction"]["phase"] == initial["h0"]["phase"]
    assert cache is not None


def test_identical_complete_requests_share_original_failures(unresolved, base, tiny_wires):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial), failures=("gpt",))
    first, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    second, _ = trial.run_target(calls, base, selected, initial, "issues_only", cache)
    assert len(calls.rows) == 7 and second["shared_from"] == "evidence_feedback"
    assert second["raw"]["gpt"] is None and second["raw"] == first["raw"]
    assert second["new_review_calls"] == 0 and second["review_calls_observed"] == 5
    assert second["prediction"] == first["prediction"] == initial["graph_r1"]["prediction"]
    assert second["panel_seconds"] is None


def test_changed_pool_cannot_reuse_any_seat(unresolved, base, tiny_wires):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial))
    first, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    calls.proposed = proposal(initial, 1)
    second, _ = trial.run_target(calls, base, selected, initial, "issues_only", cache)
    assert first["pool"] != second["pool"] and len(calls.rows) == 12
    assert second["shared_from"] is None and second["new_review_calls"] == 5


def test_one_changed_model_request_forces_entire_new_panel(unresolved, base, tiny_wires, monkeypatch):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial))
    _, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")

    def changed(seat, *args):
        body = tiny_wires(seat, *args)
        if seat == "gpt":
            body["model"] = "different-model"
        return body

    monkeypatch.setattr(trial, "review_wire", changed)
    second, _ = trial.run_target(calls, base, selected, initial, "issues_only", cache)
    assert len(calls.rows) == 12 and second["shared_from"] is None


def test_cross_target_cache_is_not_reused_even_for_identical_wires(unresolved, base, tiny_wires):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial))
    _, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    cache["target"] = "different_target"
    second, _ = trial.run_target(calls, base, selected, initial, "issues_only", cache)
    assert len(calls.rows) == 12 and second["shared_from"] is None


def test_observations_and_prior_hints_only_enter_proposer(unresolved, base):
    initial, selected = unresolved
    original = deepcopy(initial)
    feedback = trial.proposal_wire(base, selected, initial, "evidence_feedback")
    issues = trial.proposal_wire(base, selected, initial, "issues_only")
    packet = json.loads(feedback["messages"][0]["content"][0]["text"])
    assert packet.pop("review_evidence_feedback")["candidates"]
    feedback["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    assert feedback == issues
    assert packet["candidate_relation_hints"] == initial["hints"]["packet"]
    assert packet["current_prediction"] == initial["graph_r1"]["prediction"]
    for seat in SEATS:
        body = trial.review_wire(seat, base, selected, initial["graph_r1"]["pool"])
        reviewer_packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert "review_evidence_feedback" not in reviewer_packet
        assert "candidate_relation_hints" not in reviewer_packet
        assert "current_prediction" not in reviewer_packet
        assert "academic_context" in reviewer_packet
    assert initial == original


def test_incomplete_dispatch_preserves_first_round(unresolved, base, tiny_wires):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial), max_calls=4, rating=4)
    result, _ = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "INCOMPLETE_PANEL" and len(calls.rows) == 4
    assert result["reviewed"] is False and result["new_review_calls"] == 3
    assert result["prediction"] == initial["graph_r1"]["prediction"]


@pytest.mark.parametrize("stopped,max_calls", [(True, 999), (False, 0)])
def test_budget_stop_never_invents_proposal_or_change(unresolved, base, tiny_wires, stopped, max_calls):
    initial, selected = unresolved
    calls = FakeCalls(proposal(initial), stopped=stopped, max_calls=max_calls)
    result, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "BUDGET_STOPPED" and calls.rows == [] and cache is None
    assert result["prediction"] == initial["graph_r1"]["prediction"]
    assert result["round2_attempted"] is False


@pytest.mark.parametrize("bad", [None, {}, {"instrument": [], "verb": [], "target": [999], "ivt": []}])
def test_invalid_proposal_preserves_first_round(unresolved, base, tiny_wires, bad):
    initial, selected = unresolved
    calls = FakeCalls(bad)
    result, cache = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "PROPOSAL_FAILED" and len(calls.rows) == 1 and cache is None
    assert result["prediction"] == initial["graph_r1"]["prediction"]


def test_failed_selection_is_recorded_without_losing_prior_prediction(unresolved, base, tiny_wires, monkeypatch):
    initial, selected = unresolved

    def invalid_selection(*args, **kwargs):
        raise ApiSchemaError("selected labels exceed the final-only contract")

    monkeypatch.setattr(trial.panel, "select", invalid_selection)
    calls = FakeCalls(proposal(initial))
    result, _ = trial.run_target(calls, base, selected, initial, "evidence_feedback")
    assert result["status"] == "SELECTION_FAILED" and len(calls.rows) == 6
    assert result["prediction"] == initial["graph_r1"]["prediction"]
