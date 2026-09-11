import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_prior_candidate_trial as trial
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.prior_panel import (
    COMPONENTS,
    fit_prior,
    video_counts,
)


def counts(ivts=(17, 19)):
    rows = []
    for f in range(120):
        ivt = ivts[f % len(ivts)]
        rows.append({"frame_id": f, "mask": {t: True for t in trial.TASKS},
                     "gt": {**{t: [v] for t, v in COMPONENTS[ivt].items()}, "ivt": [ivt], "phase": [1]}})
    return video_counts(rows)


def prior():
    return fit_prior({"query": counts((7,)), "a": counts(), "b": counts(), "c": counts()}, "query")


def h0():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def test_query_video_statistics_cannot_affect_its_own_prior():
    left = prior()
    right = fit_prior({"query": counts((94,)), "a": counts(), "b": counts(), "c": counts()}, "query")
    assert left == right
    out = retrieve_candidate_hints(h0(), left, video_id="query")
    assert "query" not in out["audit"]["fit_videos"]
    assert 1 <= len(out["packet"]["relations"]) <= 2
    assert out["audit"]["selected"][0]["ivt"] == 17
    assert len(json.dumps(out["packet"], ensure_ascii=False)) <= 1800


def test_wrong_query_and_tampered_prior_rejected():
    with pytest.raises(ValueError, match="excluded"):
        retrieve_candidate_hints(h0(), prior(), video_id="a")
    damaged = prior()
    damaged["tasks"]["ivt"]["global"][17]["rate"] = 1.0
    with pytest.raises(ValueError, match="checksum"):
        retrieve_candidate_hints(h0(), damaged, video_id="query")


def test_null_h0_can_receive_non_null_hypotheses_without_mutation():
    original = {"instrument": [0], "verb": [9], "target": [14], "ivt": [94], "phase": [1]}
    before = deepcopy(original)
    result = retrieve_candidate_hints(original, prior(), video_id="query")
    assert original == before
    assert all(r["ivt_id"] < 94 for r in result["packet"]["relations"])
    assert result["audit"]["prior_is_not_visual_evidence"] is True


def test_missing_gt_mask_is_not_a_negative_training_observation():
    mask = {t: False for t in trial.TASKS}
    stats = video_counts([{"frame_id": 1, "mask": mask, "gt": {t: None for t in trial.TASKS}}])
    assert stats["ivt"]["n"] == 0 and stats["ivt"]["x"] == [0] * 100
    model = fit_prior({"query": counts(), "a": stats, "b": stats, "c": stats}, "query")
    assert retrieve_candidate_hints(h0(), model, video_id="query")["packet"] is None


def test_hints_only_change_proposal_text_not_images_or_output(monkeypatch):
    original = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": json.dumps({"academic_context": "Academic surgical video analysis", "current_prediction": h0()})},
        {"type": "image_url", "image_url": {"url": "test", "detail": "high"}}]}],
        "max_tokens": 4096, "response_format": {"type": "json_object"}}
    monkeypatch.setattr(trial, "gemini_proposal", lambda *args: deepcopy(original))
    assert trial.proposal_wire(None, {}, h0(), make_pool(h0())) == original
    result = trial.proposal_wire(None, {}, h0(), make_pool(h0()), {"relations": [{"ivt_id": 17}]})
    text = json.loads(result["messages"][0]["content"][0]["text"])
    text.pop("candidate_relation_hints")
    result["messages"][0]["content"][0]["text"] = json.dumps(text)
    assert result == original


def test_same_pool_panels_reuse_original_failed_replies(monkeypatch):
    monkeypatch.setattr(trial, "review_wire", lambda *args: {"messages": []})

    class Failed:
        def __init__(self):
            self.rows = []

        def call(self, target, stage, seat, body):
            self.rows.append({"target": target, "stage": stage, "seat": seat})

    calls, reuse = Failed(), []
    a = trial.review(calls, SimpleNamespace(images=[0, 1, 2]), {"key": "test"}, h0(), make_pool(h0()), "control", reuse)
    b = trial.review(calls, SimpleNamespace(images=[0, 1, 2]), {"key": "test"}, h0(), make_pool(h0()), "prior_graph", reuse)
    assert len(calls.rows) == 5 and a["prediction"] == b["prediction"] == h0()
    assert b["shared_from"] == "control" and b["panel_seconds"] is None


def test_distinct_candidate_pools_are_not_shared(monkeypatch):
    monkeypatch.setattr(trial, "review_wire", lambda seat, base, selected, pool: {"messages": [], "pool": pool})

    class Failed:
        def __init__(self):
            self.rows = []

        def call(self, target, stage, seat, body):
            self.rows.append({"target": target, "stage": stage, "seat": seat})

    calls, reuse = Failed(), []
    args = (calls, SimpleNamespace(images=[0, 1, 2]), {"key": "test"}, h0())
    trial.review(*args, make_pool(h0()), "control", reuse)
    pool = make_pool(h0(), {"instrument": [], "verb": [], "target": [], "ivt": [17]})
    trial.review(*args, pool, "prior_graph", reuse)
    assert len(calls.rows) == 10
