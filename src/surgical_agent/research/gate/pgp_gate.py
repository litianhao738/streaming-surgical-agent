"""Offline, Tracker-free branch decisions for Prior-Guided Progressive Gate.

This module consumes features already available after the frozen Qwen probes.
It neither queries reviewers nor loads checkpoints or training data. A caller
must supply fitting rows only and choose thresholds on separate calibration
videos. The four logistic outputs are uncalibrated decision scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TASKS = ("instrument", "verb", "target", "ivt", "phase")
BRANCHES = ("interaction", "phase")
TARGETS = ("help", "harm")
SUPERVISIONS = ("semantic_benefit_original_harm", "original", "merged")
MODEL_SCHEMA = "surgical_agent.pgp_gate.no_tracker"
MODEL_VERSION = 1
EPS = 1e-12


def compose_output(cheap: Mapping, full: Mapping, action: int) -> dict:
    """Compose one frame; bit 0 selects interaction and bit 1 selects phase.

    Values are copied without sorting, filtering, or normalizing: unselected
    branches preserve the cheap answer exactly, including representation.
    """
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer bitmask in 0..3")
    if not 0 <= action <= 3:
        raise ValueError("action must be an integer bitmask in 0..3")
    if not isinstance(cheap, Mapping) or not isinstance(full, Mapping):
        raise ValueError("cheap and full must be five-task mappings")
    if any(task not in cheap or task not in full for task in TASKS):
        raise ValueError("cheap and full must contain all five tasks")
    return {
        task: deepcopy((full if action & (1 if i < 4 else 2) else cheap)[task])
        for i, task in enumerate(TASKS)
    }


def _counts(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (5, 3):
        raise ValueError(f"{name} must have shape (N,5,3), ordered TP,FP,FN")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative counts")
    if np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain integer-valued counts")
    return array


def _head_f1(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    # Match official_net_training.frame_utility: an empty head scores one.
    return np.divide(2 * tp, denominator, out=np.ones_like(tp, dtype=float), where=denominator != 0)


def _branch_utilities(cheap: np.ndarray, full: np.ndarray) -> np.ndarray:
    delta = _head_f1(full) - _head_f1(cheap)
    return np.column_stack((delta[:, :4].mean(axis=1), delta[:, 4]))


def _branch_error_changes(cheap: np.ndarray, full: np.ndarray) -> np.ndarray:
    delta = (full - cheap)[:, :, 1:].sum(axis=2)
    return np.column_stack((delta[:, :4].sum(axis=1), delta[:, 4]))


def branch_targets(
    cheap_counts: Any,
    full_counts: Any,
    cheap_merged: Any,
    full_merged: Any,
    supervision: str = "semantic_benefit_original_harm",
) -> np.ndarray:
    """Return bool (N,2,2), ordered interaction/phase then help/harm.

    The main target permits simultaneous help and harm: semantic improvement
    does not erase original-label damage. For each harm target, either lower
    branch frame F1 or more branch FP+FN is sufficient. Branch frame F1 is the
    mean of four interaction head F1s or the single phase head F1; it is not
    pooled dataset F1. The two ablations use one label view for both targets.
    """
    if supervision not in SUPERVISIONS:
        raise ValueError(f"unknown supervision: {supervision}")
    cheap = _counts(cheap_counts, "cheap_counts")
    full = _counts(full_counts, "full_counts")
    merged_cheap = _counts(cheap_merged, "cheap_merged")
    merged_full = _counts(full_merged, "full_merged")
    if any(x.shape != cheap.shape for x in (full, merged_cheap, merged_full)):
        raise ValueError("all count arrays must have identical shape")
    original_gain = _branch_utilities(cheap, full)
    merged_gain = _branch_utilities(merged_cheap, merged_full)
    benefit = original_gain if supervision == "original" else merged_gain
    harm_gain = merged_gain if supervision == "merged" else original_gain
    harm_errors = _branch_error_changes(
        merged_cheap if supervision == "merged" else cheap,
        merged_full if supervision == "merged" else full,
    )
    return np.stack((benefit > EPS, (harm_gain < -EPS) | (harm_errors > 0)), axis=2)


def _feature_names(feature_names: Sequence[str], width: int) -> list[str]:
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must be an ordered sequence of names")
    names = list(feature_names)
    if len(names) != width or not width:
        raise ValueError("feature_names length must match nonempty feature width")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")
    if any("tracker" in name.casefold() for name in names):
        raise ValueError("Tracker features are forbidden in this model schema")
    return names


def _features(X: Any, feature_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("X must be a finite two-dimensional feature matrix")
    return array, _feature_names(feature_names, array.shape[1])


def fit_model(
    X: Any, targets: Any, videos: Any, feature_names: Sequence[str]
) -> dict:
    """Fit four models from scratch and export JSON-compatible raw weights.

    Each fitting video receives equal total sample weight. The scaler uses
    those video weights; each classifier additionally uses sklearn's balanced
    class weights based on unweighted class frequency, matching the archived
    whole-frame baseline. No prior model or Tracker state is read or updated.
    """
    features, names = _features(X, feature_names)
    if not len(features):
        raise ValueError("fitting requires at least one row")
    labels = np.asarray(targets)
    if labels.shape != (len(features), 2, 2):
        raise ValueError("targets must have shape (N,2,2)")
    try:
        labels = labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets must be binary") from exc
    if not np.all(np.isfinite(labels)) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("targets must be finite binary values")
    labels = labels.astype(np.int8)
    groups = np.asarray(videos)
    if groups.shape != (len(features),) or any(
        not isinstance(v, (str, np.str_)) or not v.strip() for v in groups
    ):
        raise ValueError("videos must contain one nonempty video name per fitting row")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    video_weights = len(features) / (len(unique) * counts[inverse])
    scaler = StandardScaler().fit(features, sample_weight=video_weights)
    transformed = scaler.transform(features)
    estimators = []
    for branch in range(2):
        pair = []
        for target in range(2):
            y = labels[:, branch, target]
            if len(np.unique(y)) == 1:
                pair.append({"kind": "constant", "constant": float(y[0])})
                continue
            estimator = LogisticRegression(
                C=1., class_weight="balanced", solver="liblinear",
                max_iter=2000, random_state=3407,
            ).fit(transformed, y, sample_weight=video_weights)
            weights = estimator.coef_[0] / scaler.scale_
            bias = float(estimator.intercept_[0] - weights @ scaler.mean_)
            if not np.allclose(
                features @ weights + bias, estimator.decision_function(transformed),
                rtol=1e-10, atol=1e-9,
            ):
                raise ValueError("raw-feature weight export changed model scores")
            pair.append({
                "kind": "logistic", "weights": weights.tolist(), "bias": bias,
                "iterations": int(estimator.n_iter_[0]),
            })
        estimators.append(pair)
    return {
        "schema": MODEL_SCHEMA, "version": MODEL_VERSION,
        "tracker_enabled": False, "feature_names": names,
        "branches": list(BRANCHES), "targets": list(TARGETS),
        "score_semantics": "uncalibrated_logistic_decision_scores",
        "training": {
            "rows": len(features), "videos": unique.tolist(),
            "video_counts": counts.tolist(), "seed": 3407,
            "C": 1., "solver": "liblinear", "max_iter": 2000,
            "sample_weight": "N / (number_of_videos * rows_in_video)",
            "class_weight": "balanced_unweighted_class_frequency",
            "scaler_weight": "video_sample_weight_fit_rows_only",
        },
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "estimators": estimators,
    }


def predict_scores(model: Mapping, X: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Apply an exported model with exact feature-order and version checks."""
    features, names = _features(X, feature_names)
    if not isinstance(model, Mapping) or model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported PGP model schema")
    if model.get("version") != MODEL_VERSION or model.get("tracker_enabled") is not False:
        raise ValueError("unsupported PGP version or Tracker-enabled model")
    if model.get("feature_names") != names:
        raise ValueError("model feature schema/order differs from inference input")
    if model.get("branches") != list(BRANCHES) or model.get("targets") != list(TARGETS):
        raise ValueError("model branch/target order mismatch")
    estimators = model.get("estimators")
    if not isinstance(estimators, list) or len(estimators) != 2:
        raise ValueError("model must contain two branch estimator pairs")
    scores = np.empty((len(features), 2, 2), dtype=np.float64)
    for branch, pair in enumerate(estimators):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("model must contain help and harm estimators")
        for target, estimator in enumerate(pair):
            if not isinstance(estimator, Mapping):
                raise ValueError("invalid estimator")
            if estimator.get("kind") == "constant":
                value = float(estimator["constant"])
                if not np.isfinite(value) or value not in (0., 1.):
                    raise ValueError("invalid constant estimator")
                scores[:, branch, target] = value
            elif estimator.get("kind") == "logistic":
                weights = np.asarray(estimator["weights"], dtype=np.float64)
                bias = float(estimator["bias"])
                if weights.shape != (features.shape[1],) or not np.all(np.isfinite(weights)) or not np.isfinite(bias):
                    raise ValueError("invalid logistic estimator weights")
                raw = features @ weights + bias
                if not np.all(np.isfinite(raw)):
                    raise ValueError("nonfinite logits from feature/weight overflow")
                scores[:, branch, target] = 1 / (1 + np.exp(-np.clip(raw, -700, 700)))
            else:
                raise ValueError("unsupported estimator kind")
    return scores


def select_actions(scores: Any, thresholds: Any) -> np.ndarray:
    """Require both help >= minimum and harm <= maximum for each branch.

    Thresholds are finite (2,2), ordered interaction/phase then min_help/max_harm.
    Values outside [0,1] permit fixed always/never actions during calibration.
    """
    values = np.asarray(scores, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("scores must have shape (N,2,2)")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("scores must be finite and within [0,1]")
    if limits.shape != (2, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("thresholds must be finite with shape (2,2)")
    selected = (values[:, :, 0] >= limits[:, 0]) & (values[:, :, 1] <= limits[:, 1])
    return selected[:, 0].astype(np.uint8) + 2 * selected[:, 1].astype(np.uint8)
