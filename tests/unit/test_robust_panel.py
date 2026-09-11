"""Valid-subset aggregation stays a drop-in for the published panel."""
from copy import deepcopy

import pytest

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification import robust_panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool

H0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def judgment(rating, finding=None, scope="LOCAL_REGION", indices=(1, 2)):
    if finding is None:
        finding = "MATCH" if rating >= 4 else "UNCLEAR" if rating == 3 else "REFUTED"
    if rating <= 2:
        scope = "WHOLE_FRAME"
    return {"rating": rating, "finding": finding, "scope": scope,
            "image_indices": list(indices), "observation": "Deterministic reviewer text."}


def pool_of(**proposals):
    return make_pool(H0, {q: proposals.get(q, []) for q in
                          ("instrument", "verb", "target", "ivt")}, make_pool(H0))


def reviews_for(pool, per_seat):
    """per_seat maps a seat to {pid: rating} or None for a missing response."""
    out = {}
    for seat, ratings in per_seat.items():
        if ratings is None:
            out[seat] = None
            continue
        out[seat] = {"judgments": {p["id"]: judgment(ratings.get(p["id"], 3))
                                   for p in pool["propositions"]}}
    return out


def uniform(pool, rating, seats=SEATS):
    return reviews_for(pool, {s: {p["id"]: rating for p in pool["propositions"]} for s in seats})


@pytest.mark.parametrize("bad", [("grok", "grok", "gpt"), ("grok", "qwen"), ("grok", "qwen", "nobody")])
def test_a_roster_must_be_three_or_more_distinct_published_seats(bad):
    with pytest.raises(ValueError):
        robust_panel.resolve_roster(bad)


def test_five_valid_mean_reproduces_the_published_aggregator_exactly():
    pool = pool_of(instrument=[2], ivt=[60])
    raw = uniform(pool, 5)
    published_means, published_diag = panel.aggregate(raw, pool, image_count=3)
    means, diagnostics = robust_panel.aggregate(raw, pool, image_count=3,
                                                seats=SEATS, min_valid=5, statistic="mean")
    assert means == published_means
    for pid in published_means:
        assert diagnostics[pid]["scores"] == published_diag[pid]["scores"]
        assert diagnostics[pid]["invalid"] == published_diag[pid]["invalid"]
    assert panel.select(H0, pool, means) == panel.select(H0, pool, published_means)


def test_one_broken_seat_blocks_the_published_rule_but_not_the_relaxed_one():
    pool = pool_of(instrument=[2])
    per_seat = {s: {p["id"]: 5 for p in pool["propositions"]} for s in SEATS}
    per_seat[SEATS[0]] = None
    raw = reviews_for(pool, per_seat)

    blocked, _ = panel.aggregate(raw, pool, image_count=3)
    assert blocked["instrument_2"] is None

    means, diagnostics = robust_panel.aggregate(raw, pool, image_count=3, min_valid=3)
    assert means["instrument_2"] == 5
    assert diagnostics["instrument_2"]["valid_seats"] == 4
    assert set(diagnostics["instrument_2"]["invalid"]) == {SEATS[0]}
    assert 2 in panel.select(H0, pool, means)["instrument"]


def test_min_valid_is_a_floor_not_a_suggestion():
    pool = pool_of(instrument=[2])
    per_seat = {s: {p["id"]: 5 for p in pool["propositions"]} for s in SEATS}
    for seat in SEATS[:3]:
        per_seat[seat] = None
    means, diagnostics = robust_panel.aggregate(reviews_for(pool, per_seat), pool, min_valid=3)
    assert diagnostics["instrument_2"]["valid_seats"] == 2
    assert means["instrument_2"] is None
    assert panel.select(H0, pool, means)["instrument"] == [0]


def test_median_ignores_a_lone_outlier_that_drags_the_mean_below_threshold():
    pool = pool_of(instrument=[2])
    ratings = dict(zip(SEATS, (5, 5, 5, 5, 1), strict=True))
    raw = reviews_for(pool, {s: {p["id"]: r for p in pool["propositions"]}
                             for s, r in ratings.items()})
    by_mean, _ = robust_panel.aggregate(raw, pool, min_valid=5, statistic="mean")
    by_median, _ = robust_panel.aggregate(raw, pool, min_valid=5, statistic="median")
    assert by_mean["instrument_2"] == 4.2
    assert by_median["instrument_2"] == 5
    assert 2 in panel.select(H0, pool, by_mean)["instrument"]
    assert 2 in panel.select(H0, pool, by_median)["instrument"]


def test_a_three_seat_roster_only_reads_its_own_seats():
    pool = pool_of(instrument=[2])
    roster = SEATS[:3]
    raw = uniform(pool, 5, seats=roster)
    means, diagnostics = robust_panel.aggregate(raw, pool, seats=roster, min_valid=3)
    assert means["instrument_2"] == 5
    assert diagnostics["instrument_2"]["roster"] == list(roster)
    assert len(diagnostics["instrument_2"]["scores"]) == 3


def test_a_roster_mismatch_is_rejected_rather_than_silently_scored():
    pool = pool_of(instrument=[2])
    raw = uniform(pool, 5)
    with pytest.raises(ValueError, match="exactly the configured roster"):
        robust_panel.aggregate(raw, pool, seats=SEATS[:3], min_valid=3)


@pytest.mark.parametrize("statistic,expected", [("mean", None), ("median", None)])
def test_invalid_items_never_become_a_score(statistic, expected):
    pool = pool_of(instrument=[2])
    # A local-scope refutation is invalid; it must not be counted as a low vote.
    per_seat = {s: {p["id"]: 5 for p in pool["propositions"]} for s in SEATS}
    raw = reviews_for(pool, per_seat)
    for seat in SEATS[:4]:
        raw[seat]["judgments"]["instrument_2"] = {
            "rating": 1, "finding": "REFUTED", "scope": "LOCAL_REGION",
            "image_indices": [1, 2], "observation": "Only checked one region."}
    means, diagnostics = robust_panel.aggregate(raw, pool, min_valid=3, statistic=statistic)
    assert diagnostics["instrument_2"]["valid_seats"] == 1
    assert means["instrument_2"] is expected
    assert panel.select(H0, pool, means)["instrument"] == [0]


def test_aggregation_does_not_mutate_the_pool_or_the_reviews():
    pool = pool_of(instrument=[2], ivt=[60])
    raw = uniform(pool, 4)
    snapshot = deepcopy((pool, raw))
    robust_panel.aggregate(raw, pool, min_valid=3, statistic="median")
    assert (pool, raw) == snapshot


@pytest.mark.parametrize("bad", [("min_valid", 2), ("min_valid", 6), ("statistic", "mode")])
def test_out_of_contract_settings_are_rejected(bad):
    pool = pool_of(instrument=[2])
    raw = uniform(pool, 5)
    kwargs = {bad[0]: bad[1]}
    with pytest.raises(ValueError):
        robust_panel.aggregate(raw, pool, **kwargs)


def test_unresolved_reporting_matches_the_published_contract():
    pool = pool_of(instrument=[2])
    raw = uniform(pool, 5)
    means, diagnostics = robust_panel.aggregate(raw, pool, min_valid=3)
    state = panel.select(H0, pool, means)
    assert robust_panel.unresolved(state, pool, means, diagnostics) == panel.unresolved(
        state, pool, means, diagnostics)
