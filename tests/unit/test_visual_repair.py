import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.run_repair_revision_trial import normalize_five
from scripts.run_visual_repair_trial import (
    REPAIR_STAGE,
    REVIEW_STAGE,
    repair_wire,
    run_target,
)
from scripts.score_visual_repair_trial import summarize
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.repair_feedback_v2 import (
    build_repair_feedback,
)
from surgical_agent.research.verification.visual_repair import (
    apply_reviewed_repair,
    compile_repair,
)


def current():
    return {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}


def edit(pid, operation="ADD", indices=None):
    return {"candidate_id": pid, "operation": operation, "image_indices": [2] if indices is None else indices,
            "observation": "A plausible tool contact relation needs independent checking."}


def raw_reviews(pool, score=3):
    return {seat: {"judgments": {p["id"]: {"rating": score,
        "finding": "MATCH" if score >= 4 else "REFUTED" if score <= 2 else "UNCLEAR",
        "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "Whole frame was inspected."}
        for p in pool["propositions"]}} for seat in SEATS}


def initial_state():
    prediction = current()
    pool = make_pool(prediction)
    raw = raw_reviews(pool)
    reviews, formatting = normalize_five(raw, pool, 3)
    means, diagnostics = panel.aggregate(reviews, pool)
    return {"key": "VID00_51", "hints": {"packet": None}, "graph_r1": {
        "prediction": prediction, "pool": pool, "raw": raw, "reviews": reviews,
        "format_diagnostics": formatting, "issues": panel.unresolved(prediction, pool, means, diagnostics)}}


@pytest.mark.parametrize("kind,expected", [("local", "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"),
                                         ("conflict", "RATING_FINDING_CONFLICT"), ("missing", "MISSING_CANDIDATE_ID")])
def test_feedback_preserves_actual_invalid_reason_without_restoring_a_vote(kind, expected):
    old = initial_state()["graph_r1"]
    raw = old["raw"]
    if kind == "missing":
        raw["gpt"]["judgments"].pop("instrument_0")
    elif kind == "local":
        raw["gpt"]["judgments"]["instrument_0"].update(rating=1, finding="REFUTED", scope="LOCAL_REGION")
    else:
        raw["gpt"]["judgments"]["instrument_0"].update(rating=1, finding="MATCH")
    reviews, _ = normalize_five(raw, old["pool"], 3)
    means, diagnostics = panel.aggregate(reviews, old["pool"])
    issues = panel.unresolved(old["prediction"], old["pool"], means, diagnostics)
    before = deepcopy((raw, reviews, issues))
    result = build_repair_feedback(old["pool"], raw, reviews, issues)
    invalid = result["candidates"][0]["invalid_reviewers"][0]
    assert invalid["reason_codes"] == [expected]
    assert invalid["normalization_reason"] == "SCHEMA_INVALID"
    assert "observation" not in invalid
    assert len(result["candidates"][0]["valid_observations"]) == 4
    assert means["instrument_0"] is None
    assert before == (raw, reviews, issues)


def test_feedback_rejects_mismatched_raw_binding():
    old = initial_state()["graph_r1"]
    old["raw"]["gpt"]["judgments"]["instrument_0"]["observation"] = "Different response"
    with pytest.raises(ValueError, match="bindings disagree"):
        build_repair_feedback(old["pool"], old["raw"], old["reviews"], old["issues"])


def test_pool_external_relation_needs_no_self_rating_and_components_are_reviewed():
    state = current()
    snapshot = deepcopy(state)
    compiled = compile_repair(state, make_pool(state), {"changes": [edit("ivt_60")], "recheck": []}, [])
    assert 60 in compiled["temporary"]["ivt"]
    assert state == snapshot
    assert "rating" not in compiled["requested_changes"]["ivt_60"]
    means = {p["id"]: 4 for p in compiled["review_pool"]["propositions"]}
    assert apply_reviewed_repair(compiled, means) == compiled["temporary"]
    means["target_0"] = 3
    assert 60 not in apply_reviewed_repair(compiled, means)["ivt"]


def test_no_new_candidates_still_has_pending_review():
    old = initial_state()["graph_r1"]
    compiled = compile_repair(old["prediction"], old["pool"], {"changes": [], "recheck": []}, old["issues"])
    assert compiled["pool"] == old["pool"]
    assert compiled["review_pool"] == old["pool"]
    assert apply_reviewed_repair(compiled, {"instrument_0": 1})["instrument"] == []


def test_explicit_recheck_with_no_pending_issue_and_no_new_pool():
    state = current()
    compiled = compile_repair(state, make_pool(state), {"changes": [], "recheck": ["instrument_0"]}, [])
    assert compiled["review_pool"]["propositions"]


def test_replacement_removes_only_explicit_relation_and_keeps_independent_heads():
    state = {"instrument": [3], "verb": [3], "target": [0, 2], "ivt": [60], "phase": [2]}
    raw = {"changes": [edit("ivt_60", "REMOVE"), edit("ivt_59")], "recheck": []}
    compiled = compile_repair(state, make_pool(state), raw, [])
    assert compiled["temporary"]["ivt"] == [59]
    assert 0 in compiled["temporary"]["target"]
    means = {p["id"]: 5 for p in compiled["review_pool"]["propositions"]}
    means["ivt_60"] = 1
    final = apply_reviewed_repair(compiled, means)
    assert final["ivt"] == [59] and final["phase"] == [2]
    assert set(state["target"]) <= set(final["target"])


def test_component_removal_cannot_break_retained_relation():
    comp = COMPONENTS[60]
    state = {t: [v] for t, v in comp.items()} | {"ivt": [60], "phase": [2]}
    compiled = compile_repair(state, make_pool(state), {"changes": [edit("target_0", "REMOVE")], "recheck": []}, [])
    assert compiled["blocked_temporary_changes"]
    assert apply_reviewed_repair(compiled, {"target_0": 1}) == state


def test_dependency_only_negative_cannot_delete_unrequested_component():
    comp = COMPONENTS[60]
    state = {t: [v] for t, v in comp.items()} | {"ivt": [60], "phase": [2]}
    compiled = compile_repair(state, make_pool(state), {"changes": [edit("ivt_60", "REMOVE")], "recheck": []}, [])
    result = apply_reviewed_repair(compiled, {p["id"]: 1 for p in compiled["review_pool"]["propositions"]})
    assert result["ivt"] == []
    assert all(result[t] == state[t] for t in comp)


@pytest.mark.parametrize("raw", [None, {}, {"changes": [edit("phase_0")], "recheck": []},
    {"changes": [edit("ivt_100")], "recheck": []}, {"changes": [edit("ivt_60", indices=[0])], "recheck": []},
    {"changes": [edit("ivt_60"), edit("ivt_60")], "recheck": []},
    {"changes": [edit("instrument_0")], "recheck": []},
    {"changes": [edit("ivt_60", "REMOVE")], "recheck": []},
    {"changes": [edit("ivt_60") | {"rating": 5}], "recheck": []}])
def test_invalid_or_ambiguous_changes_fail_closed(raw):
    with pytest.raises(ValueError):
        compile_repair(current(), make_pool(current()), raw, [])


@pytest.mark.parametrize("score", [None, True, float("nan"), 6, 0])
def test_invalid_or_missing_review_scores_never_invent_support(score):
    compiled = compile_repair(current(), make_pool(current()), {"changes": [edit("verb_0")], "recheck": []}, [])
    if score is None:
        assert apply_reviewed_repair(compiled, {"verb_0": None}) == current()
    else:
        with pytest.raises(ValueError):
            apply_reviewed_repair(compiled, {"verb_0": score})


class MockCalls:
    def __init__(self, repair, dispatch_limit=5):
        self.repair, self.dispatch_limit = repair, dispatch_limit
        self.rows, self.stopped = [], False

    def call(self, target, stage, seat, body):
        if stage == REPAIR_STAGE:
            self.rows.append({"target": target, "stage": stage, "seat": seat})
            return self.repair
        if SEATS.index(seat) >= self.dispatch_limit:
            return None
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        assert stage == REVIEW_STAGE
        return raw_reviews(body["pool"], 1)[seat]


@pytest.mark.parametrize("dispatch,expected", [(5, "REVIEWED"), (4, "INCOMPLETE_PANEL")])
def test_runner_does_not_stop_on_empty_patch_or_apply_incomplete_panel(monkeypatch, dispatch, expected):
    initial = initial_state()
    monkeypatch.setattr("scripts.run_visual_repair_trial.repair_wire", lambda *a: {"model": "mock", "messages": []})
    monkeypatch.setattr("scripts.run_visual_repair_trial.review_wire", lambda seat, base, selected, pool: {"pool": pool, "messages": []})
    calls = MockCalls({"changes": [], "recheck": []}, dispatch)
    record = run_target(calls, SimpleNamespace(images=[1, 2, 3]), {"key": initial["key"]}, initial)
    assert record["status"] == expected
    assert len(calls.rows) == 1 + dispatch
    assert record["prediction"]["instrument"] == ([] if dispatch == 5 else [0])


def test_runner_schema_failure_keeps_original_and_makes_no_review(monkeypatch):
    initial = initial_state()
    monkeypatch.setattr("scripts.run_visual_repair_trial.repair_wire", lambda *a: {"messages": []})
    calls = MockCalls({"changes": [edit("phase_0")], "recheck": []})
    record = run_target(calls, SimpleNamespace(images=[1, 2, 3]), {"key": initial["key"]}, initial)
    assert record["status"] == "REPAIR_FAILED"
    assert record["prediction"] == initial["graph_r1"]["prediction"]
    assert len(calls.rows) == 1


def test_visual_wire_keeps_causal_images_ontology_and_academic_prefix_without_self_ratings():
    initial = initial_state()
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test")] * 3,
                           payload={"image_details": ["low", "low", "high"]})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    body = repair_wire(base, selected, initial)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context" and "academic" in packet["academic_context"]
    assert packet["full_ontology"]
    assert not {"gt", "ground_truth", "task_masks", "issues"} & packet.keys()
    assert set(packet["response_schema"]["properties"]) == {"changes", "recheck"}
    assert "rating" not in packet["response_schema"]["properties"]["changes"]["items"]["properties"]
    assert all("rating" not in o for c in packet["review_feedback"]["candidates"] for o in c["valid_observations"])
    assert [b["image_url"]["detail"] for b in body["messages"][0]["content"][1:]] == ["low", "low", "high"]
    assert body["provider"]["only"] == ["google-ai-studio"]
    assert body["model"] == "google/gemini-3.8-flash"


def test_scoring_masks_missing_target_and_retains_failed_repair_in_denominator():
    state = current()
    row = {"key": "VID00_51", "video_id": "VID00", "frame_id": 51,
           **{version: deepcopy(state) for version in ("h0", "graph_r1", "temporary", "final")}}
    row["temporary"]["target"] = [0]
    truth = [{"video_id": "VID00", "frame_id": 51, "gt": deepcopy(state),
              "mask": {t: t != "target" for t in state}}]
    result, details = summarize([row], truth, [initial_state()], {row["key"]: {"compiled": None, "status": "REPAIR_FAILED"}})
    assert result["metrics"]["final"]["tasks"]["instrument"]["valid_targets"] == 1
    assert result["metrics"]["temporary"]["tasks"]["target"]["valid_targets"] == 0
    assert result["comparisons"]["graph_r1_to_temporary"]["introduced_label_errors"] == 0
    assert details[0]["status"] == "REPAIR_FAILED"
