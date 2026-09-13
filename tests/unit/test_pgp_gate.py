"""Synthetic unit checks for PGP; no dataset split or external call is loaded."""

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from surgical_agent.research.gate import official_net_training as old
from surgical_agent.research.gate import pgp_gate as pgp


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_four_actions_preserve_unselected_branches_exactly(action):
    cheap = {task: [i, i + 10] for i, task in enumerate(pgp.TASKS)}
    full = {task: [i + 20] for i, task in enumerate(pgp.TASKS)}
    before = deepcopy((cheap, full))
    out = pgp.compose_output(cheap, full, action)
    for i, task in enumerate(pgp.TASKS):
        source = full if action & (1 if i < 4 else 2) else cheap
        assert out[task] == source[task]
        assert out[task] is not source[task]
    out["verb"].append(999)
    assert (cheap, full) == before


def test_high_help_cannot_cancel_high_harm_and_thresholds_are_inclusive():
    scores = np.array([
        [[1., .9], [.5, .2]],
        [[.5, .2], [1., .9]],
        [[.5, .2], [.5, .2]],
        [[.49, 0.], [.9, .21]],
    ])
    assert pgp.select_actions(scores, [[.5, .2], [.5, .2]]).tolist() == [2, 1, 3, 0]


def test_semantic_benefit_can_coexist_with_original_harm():
    cheap = np.zeros((1, 5, 3), dtype=int)
    full = cheap.copy()
    cheap[:, 1] = [1, 0, 1]
    full[:, 1] = [1, 2, 1]
    merged_cheap = cheap.copy()
    merged_full = merged_cheap.copy()
    merged_full[:, 1] = [2, 0, 0]
    main = pgp.branch_targets(cheap, full, merged_cheap, merged_full)
    original = pgp.branch_targets(cheap, full, merged_cheap, merged_full, "original")
    merged = pgp.branch_targets(cheap, full, merged_cheap, merged_full, "merged")
    assert main[0, 0].tolist() == [True, True]
    assert original[0, 0].tolist() == [False, True]
    assert merged[0, 0].tolist() == [True, False]
    assert not main[0, 1].any()


def test_original_harm_also_detects_more_errors_when_branch_mean_f1_improves():
    cheap = np.zeros((1, 5, 3), dtype=int)
    full = cheap.copy()
    cheap[:, 0] = [0, 0, 1]
    full[:, 0] = [1, 0, 0]
    cheap[:, 1] = [100, 0, 0]
    full[:, 1] = [100, 3, 0]
    assert old.frame_utility(cheap, full)[0] > 0
    labels = pgp.branch_targets(cheap, full, cheap, full, "original")
    assert labels[0, 0].tolist() == [True, True]


def test_empty_head_f1_matches_old_utility_and_branches_reconstruct_it():
    cheap = np.zeros((3, 5, 3), dtype=int)
    full = cheap.copy()
    full[0, 4] = [0, 1, 0]
    cheap[1, 2] = [0, 0, 1]
    full[1, 2] = [1, 0, 0]
    branch_gain = pgp._branch_utilities(cheap, full)
    assert branch_gain[0].tolist() == [0., -1.]
    assert branch_gain[1].tolist() == [.25, 0.]
    np.testing.assert_allclose(
        (4 * branch_gain[:, 0] + branch_gain[:, 1]) / 5,
        old.frame_utility(cheap, full),
    )
    assert not pgp.branch_targets(cheap[2:], full[2:], cheap[2:], full[2:]).any()


def fitting_example():
    X = np.array([[0., 1., 2.], [1., 0., 2.], [2., 1., 2.], [4., 0., 2.], [6., 1., 2.], [8., 0., 2.]])
    y = np.array([
        [[0, 1], [1, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]],
        [[1, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 1], [1, 0]],
    ], dtype=bool)
    return X, y, np.array(["a", "a", "a", "a", "b", "b"]), ["h0_count", "qwen_valid", "proposal_rule"]


def test_json_export_matches_sklearn_with_video_and_unweighted_class_balance():
    X, y, videos, names = fitting_example()
    model = pgp.fit_model(X, y, videos, names)
    restored = json.loads(json.dumps(model, allow_nan=False))
    probe = np.vstack((X, [19., 4., 2.]))
    actual = pgp.predict_scores(restored, probe, names)
    weights = np.array([.75, .75, .75, .75, 1.5, 1.5])
    scaler = StandardScaler().fit(X, sample_weight=weights)
    np.testing.assert_allclose(restored["scaler"]["mean"], scaler.mean_)
    for branch in range(2):
        for target in range(2):
            target_y = y[:, branch, target]
            if len(np.unique(target_y)) == 1:
                expected = np.full(len(probe), float(target_y[0]))
            else:
                reference = LogisticRegression(
                    C=1., class_weight="balanced", solver="liblinear",
                    max_iter=2000, random_state=3407,
                ).fit(scaler.transform(X), target_y, sample_weight=weights)
                expected = reference.predict_proba(scaler.transform(probe))[:, 1]
            np.testing.assert_allclose(actual[:, branch, target], expected, rtol=1e-12, atol=1e-12)
    assert restored["estimators"][1][1] == {"kind": "constant", "constant": 0.}
    assert restored["tracker_enabled"] is False
    # Inference sees an out-of-fit distribution but cannot refit the scaler.
    assert restored == model


@pytest.mark.parametrize("name", ["tracker_count", "Qwen_Tracker_hint", "TRACKER_AVAILABLE"])
def test_tracker_feature_names_are_rejected_even_if_values_zero(name):
    with pytest.raises(ValueError, match="Tracker"):
        pgp.fit_model(np.zeros((2, 1)), np.zeros((2, 2, 2)), ["a", "b"], [name])


def test_inference_rejects_reordered_schema_and_bad_weights():
    X, y, videos, names = fitting_example()
    model = pgp.fit_model(X, y, videos, names)
    with pytest.raises(ValueError, match="schema/order"):
        pgp.predict_scores(model, X[:, ::-1], names[::-1])
    bad = deepcopy(model)
    bad["estimators"][0][0]["weights"][0] = float("nan")
    with pytest.raises(ValueError, match="weights"):
        pgp.predict_scores(bad, X, names)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_features_scores_and_thresholds_rejected(invalid):
    with pytest.raises(ValueError, match="finite"):
        pgp.fit_model([[invalid]], np.zeros((1, 2, 2)), ["a"], ["h0_count"])
    scores = np.zeros((1, 2, 2))
    scores[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        pgp.select_actions(scores, [[.5, .5], [.5, .5]])
    with pytest.raises(ValueError, match="finite"):
        pgp.select_actions(np.zeros((1, 2, 2)), [[invalid, .5], [.5, .5]])


@pytest.mark.parametrize("action", [-1, 4, True, 1.5])
def test_invalid_actions_rejected(action):
    labels = {task: [] for task in pgp.TASKS}
    with pytest.raises(ValueError, match="bitmask"):
        pgp.compose_output(labels, labels, action)


def test_count_and_target_shape_validation():
    counts = np.zeros((2, 5, 3))
    with pytest.raises(ValueError, match="identical shape"):
        pgp.branch_targets(counts, counts[:1], counts, counts)
    with pytest.raises(ValueError, match="unknown supervision"):
        pgp.branch_targets(counts, counts, counts, counts, "oracle")
    bad = counts.copy()
    bad[0, 0, 0] = -.1
    with pytest.raises(ValueError, match="nonnegative"):
        pgp.branch_targets(bad, counts, counts, counts)
