"""Prior-gated IVT admission: pure Python over a frozen selector and a LOVO prior."""
from copy import deepcopy

import pytest

from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.prior_gated_repair import (
    phase_rate,
    select_prior_gated,
)
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS


def prior(phase_rates, global_rates=None, eligible_phase=True):
    """Minimal table: ivt phase bucket 3 carries `phase_rates`; others default to 0."""
    def rows(rates, eligible):
        return [{"id": c, "rate": rates.get(c, 0.0), "eligible": eligible and c in rates}
                for c in range(BOUNDS["ivt"])]
    table = {"excluded_video": "VIDX", "fit_videos": ["VIDA", "VIDB"], "tasks": {}}
    for task in ("instrument", "verb", "target"):
        table["tasks"][task] = {"global": [{"id": c, "rate": 0.5, "eligible": True} for c in range(BOUNDS[task])],
                                "phase": {}}
    table["tasks"]["ivt"] = {"global": rows(global_rates or {}, True),
                             "phase": {"3": rows(phase_rates, eligible_phase)}}
    return table


def state():
    # hook/dissect/cystic_plate (59) is the H0 relation; hook/dissect/gallbladder (60) is in the pool.
    return {"instrument": [2], "verb": [2], "target": [1], "ivt": [59], "phase": [3]}


def pool_with(*ivts):
    return make_pool(state(), {"instrument": [], "verb": [], "target": [], "ivt": list(ivts)})


def neutral_means(pool):
    return {p["id"]: 3.0 for p in pool["propositions"]}


def test_phase_rate_falls_back_to_global_when_bucket_has_no_eligible_class():
    table = prior({59: 0.001}, global_rates={59: 0.2}, eligible_phase=False)
    assert phase_rate(table, "ivt", 59, 3) == 0.2
    assert phase_rate(table, "ivt", 59, 5) == 0.2  # missing bucket


def test_veto_removes_rare_relation_but_keeps_components_without_prune():
    pool = pool_with(60)
    out, log = select_prior_gated(state(), pool, neutral_means(pool), prior({59: 0.001, 60: 0.3}), phase=3, add_rate=None)
    assert out["ivt"] == [] and log["vetoed"] == [59]
    assert out["verb"] == [2] and out["target"] == [1] and out["instrument"] == [2]
    assert out["phase"] == [3]


def test_prune_removes_components_left_unanchored_by_veto():
    pool = pool_with(60)
    out, log = select_prior_gated(state(), pool, neutral_means(pool), prior({59: 0.001, 60: 0.3}), phase=3,
                                  add_rate=None, prune=("verb", "target"))
    assert out["verb"] == [] and out["target"] == [] and out["instrument"] == [2]
    assert log["pruned"] == {"verb": [2], "target": [1]}


def test_add_admits_frequent_pool_relation_with_its_components():
    pool = pool_with(60)
    out, log = select_prior_gated(state(), pool, neutral_means(pool), prior({59: 0.001, 60: 0.76}), phase=3)
    assert out["ivt"] == [60] and log["prior_added"] == [60]
    assert out["target"] == [0, 1]  # gallbladder added; cystic_plate kept (no prune)
    assert out["verb"] == [2] and out["instrument"] == [2]


def test_add_only_from_pool_unless_universe_requested():
    pool = pool_with()  # 60 not proposed
    table = prior({59: 0.5, 60: 0.9})
    out, _ = select_prior_gated(state(), pool, neutral_means(pool), table, phase=3)
    assert out["ivt"] == [59]
    out, log = select_prior_gated(state(), pool, neutral_means(pool), table, phase=3, universe_add=True)
    assert 60 in out["ivt"] and log["prior_added"] == [60]


def test_null_relations_are_never_vetoed_or_added_by_prior():
    current = {"instrument": [2], "verb": [9], "target": [14], "ivt": [96], "phase": [3]}
    pool = make_pool(current, {"instrument": [], "verb": [], "target": [], "ivt": [95]})
    table = prior({95: 0.99, 96: 0.0})
    out, log = select_prior_gated(current, pool, neutral_means(pool), table, phase=3)
    assert out["ivt"] == [96] and log == {**log, "vetoed": [], "prior_added": []}


def test_panel_selector_runs_first_and_gate_applies_to_its_output():
    pool = pool_with(60)
    means = neutral_means(pool)
    means.update({"ivt_60": 4.4, "instrument_2": 4.4, "verb_2": 4.4, "target_0": 4.4})
    out, log = select_prior_gated(state(), pool, means, prior({59: 0.001, 60: 0.3}), phase=3, add_rate=None)
    assert out["ivt"] == [60] and log["vetoed"] == [59]  # panel added 60, prior vetoed 59


def test_means_none_gates_frozen_h0_without_any_panel():
    pool = pool_with(60)
    out, _ = select_prior_gated(state(), pool, None, prior({59: 0.001, 60: 0.76}), phase=3)
    assert out["ivt"] == [60]


def test_inputs_are_not_mutated_and_shape_is_five_sorted_heads():
    pool = pool_with(60)
    current, before = state(), deepcopy(state())
    means = neutral_means(pool)
    out, _ = select_prior_gated(current, pool, means, prior({59: 0.001, 60: 0.76}), phase=3)
    assert current == before
    assert set(out) == {"instrument", "verb", "target", "ivt", "phase"}
    assert all(out[k] == sorted(out[k]) for k in out)
    for c in out["ivt"]:
        assert all(COMPONENTS[c][t] in out[t] for t in ("instrument", "verb", "target"))


@pytest.mark.parametrize("kwargs", [{"phase": None}, {"phase": 7}, {"phase": 3, "veto_rate": 1.5},
                                    {"phase": 3, "add_rate": 0}, {"phase": 3, "prune": ("ivt",)}])
def test_invalid_configuration_is_rejected(kwargs):
    pool = pool_with(60)
    with pytest.raises(ValueError):
        select_prior_gated(state(), pool, neutral_means(pool), prior({59: 0.5}), **kwargs)


def test_missing_prior_is_rejected():
    pool = pool_with(60)
    with pytest.raises(ValueError):
        select_prior_gated(state(), pool, neutral_means(pool), None, phase=3)
