"""Synthetic branch execution/cost tests; no Training/Testing data is loaded."""
from copy import deepcopy

import numpy as np
import pytest

from tools.audit import pgp_gate_replay_20260913 as replay


def labels(phase=0):
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [phase]}


def test_four_actions_keep_interaction_heads_grouped():
    cheap = labels()
    full = {t: [1] for t in replay.TASKS}
    a = replay.branch_outputs(cheap, full)
    assert a[0] == cheap and a[3] == full
    assert all(a[1][t] == full[t] and a[2][t] == cheap[t] for t in replay.TASKS[:4])
    assert a[1]["phase"] == cheap["phase"] and a[2]["phase"] == full["phase"]
    a[1]["verb"].append(9)
    assert cheap == labels() and full["verb"] == [1]


def test_cost_prefix_reuses_qwen_and_only_queries_active_branch():
    keys = list(replay.PREFIX_KEYS) + [stage + "|" + seat
                                    for stage in ("control_graph", "joint_r1")
                                    for seat in replay.base.QORDER[1:]]
    ledger = {k: {"usd_per_call": .1} for k in keys}
    c = replay.branch_costs((5, 3), ledger)
    assert c[:, 0].tolist() == [5, 9, 7, 11]
    assert c[:, 4].tolist() == [2, 6, 4, 8]
    assert np.allclose(c[:, 1], [.5, .9, .7, 1.1])
    assert replay.branch_costs((0, 0), ledger)[:, 0].tolist() == [5] * 4
    with pytest.raises(ValueError, match="charged twice"):
        replay.cost_vector([keys[0], keys[0]], ledger)


def test_partial_bounds_include_equality_boundaries_and_invalid_answer():
    assert not replay.partial_settled([2] * 4, True)  # min total 9: removal possible
    assert replay.partial_settled([3] * 4, True)
    assert not replay.partial_settled([5] * 3, False)  # max total 25: addition possible
    assert not replay.partial_settled([5, 5, 5, 1], False)
    assert replay.partial_settled([1] * 4, False)
    assert replay.partial_settled([None], False)
    with pytest.raises(ValueError):
        replay.partial_settled([0], False)


class PrefixScores:
    def __init__(self, allowed):
        self.allowed = allowed

    def __getitem__(self, index):
        assert index in self.allowed, "unqueried future rating was inspected"
        return self.allowed[index]


def test_streaming_stop_never_reads_future_seat_scores():
    # An observed invalid Qwen response settles both no-change branches.
    invalid = {"scores": PrefixScores({}), "invalid": {"qwen": "bad"}}
    result = {"pool": {"propositions": [{"id": "verb_0", "task": "verb", "label_id": 0}]},
              "diagnostics": {"verb_0": deepcopy(invalid)},
              "joint_diagnostics": {f"phase_{p}": deepcopy(invalid) for p in range(7)}}
    assert replay.streaming_stop_plan(result, labels()) == (1, 1)


def test_phase_bound_and_all_future_completion_extremes():
    observed = [[5] * 3] + [[1] * 3 for _ in range(6)]
    assert replay.partial_phase_settled(observed, 0)
    observed[1] = [5] * 3
    assert not replay.partial_phase_settled(observed, 0)


def test_feature_schema_has_no_tracker_and_no_duplicate_columns():
    names = replay.feature_schema()
    assert not any("tracker" in name for name in names)
    assert len(names) == len(set(names))
    assert names[-1].startswith("qwen_joint_")
