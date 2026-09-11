import pytest

from surgical_agent.research.verification import weighted_review as weighted


def test_fitting_excludes_query_video_and_weights_are_monotone():
    examples = [{"video_id": "A", "task": "verb", "scores": [1] * 5, "present": 0}] * 12
    examples += [{"video_id": "B", "task": "verb", "scores": [5] * 5, "present": 1}] * 12
    examples += [{"video_id": "QUERY", "task": "verb", "scores": [5] * 5, "present": 0}] * 99
    model = weighted.fit(examples, "QUERY")
    assert model["fit_videos"] == ["A", "B"]
    assert model["models"]["verb"]["count"] == 24
    assert all(w >= 0 for w in model["models"]["verb"]["weights"])
    assert weighted.fit(examples[:-99], "QUERY") == model


def test_wrong_video_model_is_rejected_before_review():
    model = weighted.fit([], "OTHER")
    with pytest.raises(ValueError, match="exclude query"):
        weighted.select({}, {}, {}, model, "QUERY")


@pytest.mark.parametrize("intercept,scores,mean,expected", [
    (20, [3] * 5, 3, None),
    (-20, [3] * 5, 3, None),
    (20, [5, 5, 5, 5, None], None, None),
    (20, [4, 3, 3, 3, 3], 3.2, 4.0),
    (-20, [2, 3, 3, 3, 3], 2.8, 2.0),
])
def test_learned_weights_cannot_replace_required_visual_votes(monkeypatch, intercept, scores, mean, expected):
    model = weighted.fit([], "QUERY")
    model["models"]["verb"] = {"fallback": None, "intercept": intercept, "weights": [0] * 5}
    monkeypatch.setattr(weighted.panel, "aggregate", lambda *a, **k: (
        {"v": mean}, {"v": {"scores": scores}}))
    monkeypatch.setattr(weighted.panel, "select", lambda h0, pool, adjusted, **k: adjusted)
    result = weighted.select({}, {"propositions": [{"id": "v", "task": "verb"}]}, {}, model, "QUERY")
    assert result["prediction"]["v"] == expected
