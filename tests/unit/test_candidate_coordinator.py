from copy import deepcopy

import pytest

from surgical_agent.research.verification import candidate_coordinator as c


def baseline():
    return {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]}


def proposal(**values):
    return {q: values.get(q, []) for q in c.TASKS}


def reviews(pool, value=3):
    return {s: {"scores": {p["id"]: value for p in pool["propositions"]}} for s in c.SEATS}


def test_keyed_scores_and_long_auxiliary_text():
    pool = c.make_pool(baseline(), proposal(ivt=[0, 1]))
    raw = reviews(pool)
    raw["grok"]["explanation"] = "x" * 10001
    raw["grok"]["scores"] = dict(reversed(list(raw["grok"]["scores"].items())))
    means, normalized = c.aggregate(raw, pool)
    assert set(means.values()) == {3}
    assert normalized["grok"]["ignored_auxiliary_fields"] == ["explanation"]


@pytest.mark.parametrize("bad", [True, 4.5, 0, 6, "4"])
def test_invalid_numeric_fields_rejected(bad):
    pool = c.make_pool(baseline())
    raw = reviews(pool)
    raw["qwen"]["scores"]["instrument_0"] = bad
    with pytest.raises(ValueError):
        c.aggregate(raw, pool)


def test_missing_id_unknown_id_and_missing_seat_rejected():
    pool = c.make_pool(baseline())
    for change in (lambda r: r.pop("deepseek"),
                   lambda r: r["gpt"]["scores"].pop("instrument_0"),
                   lambda r: r["gemini"]["scores"].update(ivt_99=4)):
        raw = reviews(pool)
        change(raw)
        with pytest.raises(ValueError):
            c.aggregate(raw, pool)


def test_multiple_ivts_and_independent_component_protection():
    h0 = baseline()
    pool = c.make_pool(h0, proposal(ivt=[0, 1]))
    means, _ = c.aggregate(reviews(pool, 4), pool)
    selected = c.select(h0, pool, means)
    assert selected["ivt"] == [0, 1]
    assert h0 == baseline()
    means["instrument_0"] = 1
    assert c.select(selected, pool, means)["instrument"] == [0]
    means["ivt_0"] = 1
    result = c.select(selected, pool, means)
    assert result["ivt"] == [1] and result["instrument"] == [0]


def test_proposal_cannot_publish_and_no_novel_candidates_stops():
    original = baseline()
    def propose(round_no, state, pool, issues):
        state["instrument"] = [6]
        return proposal()
    out = c.run(original, propose, lambda n, s, p: reviews(p), lambda *args: None)
    assert out["final"] == original == baseline()
    assert out["stop_reason"] == "NO_NEW_CANDIDATES"
    assert len(out["history"]) == 1


def test_failed_later_panel_keeps_last_valid_state():
    def review(n, state, pool):
        if n == 2:
            return None
        raw = reviews(pool)
        for r in raw.values():
            r["scores"]["target_1"] = 4
        return raw
    out = c.run(baseline(), lambda n, *args: proposal(target=[n + 1]), review, lambda *args: None)
    assert out["stop_reason"] == "REVIEW_FAILED"
    assert out["final"]["target"] == [1]
    assert all(s == out["final"] for s in out["snapshots"])


def test_three_round_cap_and_uncertainty_keeps_h0():
    out = c.run(baseline(), lambda n, *args: proposal(target=[n + 1]),
                lambda n, s, p: reviews(p), lambda *args: None)
    assert out["stop_reason"] == "MAX_ROUNDS"
    assert len(out["history"]) == 3
    assert out["final"] == baseline()


def test_invalid_proposal_and_selected_cap_fail_closed():
    bad = proposal(ivt=[[0]])
    with pytest.raises(ValueError):
        c.make_pool(baseline(), bad)
    out = c.run(baseline(), lambda *args: bad, lambda *args: None, lambda *args: None)
    assert out["stop_reason"] == "PROPOSAL_FAILED"
    assert out["final"] == baseline()
    h0 = deepcopy(baseline())
    h0["ivt"] = list(range(6))
    # Existing wire cap is independently read, so this fixture stays meaningful.
    from surgical_agent.perception.final_only import final_only_schema
    cap = final_only_schema()["properties"]["ivt"]["properties"]["selected_ids"]["maxItems"]
    h0["ivt"] = list(range(cap))
    out = c.run(h0, lambda *args: proposal(ivt=[cap]),
                lambda n, s, p: reviews(p, 5), lambda *args: None)
    assert out["stop_reason"] == "REVIEW_FAILED"
    assert out["final"] == h0
