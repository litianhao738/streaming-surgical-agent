"""Protocol boundaries for one-shot, evidence-bound group recovery."""
from copy import deepcopy

import pytest

from scripts import run_disputed_relation_trial as trial
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.disputed_relation import (
    apply_groups,
    eligible_groups,
)
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.verb_guard import select_with_verb_guard


def case(ivts=(7,)):
    h0 = {"instrument": [0], "verb": [2], "target": [0], "ivt": [1], "phase": [1]}
    pool = make_pool(h0, {"instrument": [], "verb": [0], "target": [], "ivt": list(ivts)})
    reviews = {seat: {"judgments": {p["id"]: {"rating": 5, "finding": "MATCH", "scope": "LOCAL_REGION",
               "image_indices": [2], "observation": "Tool visibly acts on the directly contacted tissue."}
               for p in pool["propositions"]}} for seat in SEATS}
    reviews["gpt"]["judgments"]["verb_0"].update(rating=3, finding="UNCLEAR", scope="UNCERTAIN")
    return h0, pool, reviews


def select(h0, pool, reviews):
    means, diagnostics = panel.aggregate(reviews, pool)
    graph = {"prediction": panel.select(h0, pool, means), "pool": pool, "reviews": reviews,
             "means": means, "diagnostics": diagnostics}
    guard = select_with_verb_guard(h0, pool, means, diagnostics)
    eligible = eligible_groups(h0, graph, guard)
    return graph, guard, eligible


def answer(groups, verdict="SUPPORT"):
    return {"judgments": {g["group_id"]: {"verdict": verdict, "image_indices": [2],
            "observation": "The visible tool tip closes on the named tissue in the current image."} for g in groups}}


def test_support_restores_relation_and_verb_together_without_rewriting_scores():
    h0, pool, reviews = case()
    graph, guard, eligible = select(h0, pool, reviews)
    snapshot = deepcopy((h0, graph, guard))
    assert [g["group_id"] for g in eligible["groups"]] == ["ivt_7"]
    result = apply_groups(guard["prediction"], graph["prediction"], eligible["groups"], answer(eligible["groups"]))
    assert result["prediction"] == graph["prediction"]
    assert 0 in result["prediction"]["verb"] and 7 in result["prediction"]["ivt"]
    assert (h0, graph, guard) == snapshot
    assert guard["guarded_means"]["verb_0"] is None


@pytest.mark.parametrize("pid,finding", [("verb_0", "REFUTED"), ("ivt_7", "REFUTED"),
    ("target_0", "VISIBLE_ONLY"), ("target_0", "INDIRECT_EFFECT"), ("instrument_0", "REFUTED")])
def test_any_related_valid_negative_blocks_recheck_even_when_mean_passes(pid, finding):
    h0, pool, reviews = case()
    reviews["grok"]["judgments"][pid].update(rating=2, finding=finding, scope="WHOLE_FRAME")
    graph, guard, eligible = select(h0, pool, reviews)
    if 7 in graph["prediction"]["ivt"] and 7 not in guard["prediction"]["ivt"]:
        assert eligible["excluded"] == [{"group_id": "ivt_7", "reason": "EXPLICIT_RELATED_REFUTATION"}]
    assert eligible["groups"] == []


def test_invalid_related_vote_is_not_treated_as_unclear():
    h0, pool, reviews = case()
    reviews["grok"]["judgments"]["target_0"]["image_indices"] = []
    graph, _guard, eligible = select(h0, pool, reviews)
    assert graph["means"]["target_0"] is None
    assert eligible["groups"] == []


@pytest.mark.parametrize("raw_kind", ["none", "wrong_id", "extra_key", "history_only", "boolean_index", "too_long"])
def test_bad_recheck_preserves_strict_output(raw_kind):
    graph, guard, eligible = select(*case())
    raw = answer(eligible["groups"])
    if raw_kind == "none":
        raw = None
    elif raw_kind == "wrong_id":
        raw["judgments"]["ivt_999"] = raw["judgments"].pop("ivt_7")
    elif raw_kind == "extra_key":
        raw["phase"] = 3
    elif raw_kind == "history_only":
        raw["judgments"]["ivt_7"]["image_indices"] = [0, 1]
    elif raw_kind == "boolean_index":
        raw["judgments"]["ivt_7"]["image_indices"] = [True, 2]
    else:
        raw["judgments"]["ivt_7"]["observation"] = "x" * 1001
    result = apply_groups(guard["prediction"], graph["prediction"], eligible["groups"], raw)
    assert result["prediction"] == guard["prediction"]
    assert result["status"] == "INVALID_RESPONSE"


@pytest.mark.parametrize("verdict", ["UNCLEAR", "REFUTE"])
def test_no_support_is_not_a_frame_wide_deletion(verdict):
    graph, guard, eligible = select(*case())
    result = apply_groups(guard["prediction"], graph["prediction"], eligible["groups"], answer(eligible["groups"], verdict))
    assert result["prediction"] == guard["prediction"]
    assert result["accepted_groups"] == []


def test_refuting_one_relation_does_not_remove_verb_supported_by_another():
    other = next(c for c, comp in COMPONENTS.items() if c != 7 and comp["verb"] == 0 and comp["instrument"] == 0)
    graph, guard, eligible = select(*case((7, other)))
    assert len(eligible["groups"]) == 2
    raw = answer(eligible["groups"])
    raw["judgments"][f"ivt_{other}"]["verdict"] = "REFUTE"
    result = apply_groups(guard["prediction"], graph["prediction"], eligible["groups"], raw)
    assert 7 in result["prediction"]["ivt"] and other not in result["prediction"]["ivt"]
    assert 0 in result["prediction"]["verb"]


def test_rechecker_wire_contains_images_and_relations_but_no_votes_h0_or_priors(monkeypatch):
    _graph, _guard, eligible = select(*case())
    template = {"model": trial.PROPOSER, "messages": [
        {"role": "system", "content": "HIDDEN_H0_PROMPT"},
        {"role": "user", "content": [{"type": "text", "text": "HIDDEN_VOTES_PRIORS"},
         *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}",
                   "detail": "high" if i == 2 else "low"}} for i in range(3)]]}],
        "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}}
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda _: deepcopy(template))
    body = trial.recheck_wire(None, eligible["groups"])
    import json
    text = body["messages"][0]["content"][0]["text"]
    packet = json.loads(text)
    assert next(iter(packet)) == "academic_context"
    assert "HIDDEN" not in text and "scores" not in packet and "h0" not in packet
    assert body["messages"][0]["content"][1:] == template["messages"][1]["content"][1:]
    assert body["model"] == trial.PROPOSER and body["max_tokens"] == 4096


def test_closed_audit_failure_prevents_gt_scoring(monkeypatch, tmp_path):
    def bad(*args):
        raise ValueError("frozen source changed")
    def forbidden(*args):
        pytest.fail("GT loaded despite failed inference audit")
    monkeypatch.setattr(trial, "audit", bad)
    monkeypatch.setattr(trial, "score_saved", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        trial.score(tmp_path, None)


def test_paid_rerun_never_reaches_transport(monkeypatch, tmp_path):
    (tmp_path / "execution.lock").write_text("already executed")
    monkeypatch.setattr(trial, "verify_plan", lambda _: {})
    monkeypatch.setattr(trial, "BoundCalls", lambda *args: pytest.fail("second paid dispatch"))
    with pytest.raises(ValueError, match="single-use"):
        trial.execute(tmp_path, None)


def test_one_failed_extra_call_keeps_strict_answer_without_retry(monkeypatch):
    h0, pool, reviews = case()
    graph, guard, _eligible = select(h0, pool, reviews)
    baseline = {"h0": h0, "original": graph["prediction"], "verb_guard": guard["prediction"],
                "graph": graph, "guard": guard, "guard_error": None, "status": "UNRESOLVED"}
    monkeypatch.setattr(trial.base_trial, "run_target", lambda *args: deepcopy(baseline))
    monkeypatch.setattr(trial, "recheck_wire", lambda *args: {"model": trial.PROPOSER, "messages": []})

    class FailedCall:
        def __init__(self):
            self.calls = []

        def call(self, *args):
            self.calls.append(args)

    calls = FailedCall()
    result = trial.run_target(calls, None, {"key": "synthetic"}, {})
    assert len(calls.calls) == 1
    assert calls.calls[0][:3] == ("synthetic", trial.STAGE, "base")
    assert result["joint_recheck"] == guard["prediction"]
    assert result["recheck_status"] == "INVALID_RESPONSE"
    assert result["recheck_error"] is None
