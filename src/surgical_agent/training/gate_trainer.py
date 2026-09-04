"""Train one shared three-scope Benefit Gate from video-OOF examples."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.research.gate.contracts import (
    FORMAL_GATE_FEATURE_ORDER,
    REPAIR_SCOPE_ORDER,
)
from surgical_agent.research.gate.oof_dataset import GateOOFExample


def _fit_scope(
    examples: tuple[GateOOFExample, ...],
    *,
    scope_index: int,
    seed: int,
) -> tuple[tuple[float, ...], float, dict[str, object]]:
    observed = tuple(
        item for item in examples if item.benefit_labels[scope_index] is not None
    )
    if not observed:
        raise ValueError("every formal Gate scope needs observed labels")
    labels = np.asarray(
        [int(item.benefit_labels[scope_index]) for item in observed], dtype=np.int64
    )
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("every formal Gate scope needs both benefit classes")
    matrix = np.asarray([item.feature_values for item in observed], dtype=np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(scaled, labels)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    standardized = np.asarray(model.coef_[0], dtype=np.float64)
    weights = standardized / scale
    bias = float(model.intercept_[0] - np.sum(standardized * mean / scale))
    raw_scores = bias + matrix @ weights
    if not np.allclose(raw_scores, model.decision_function(scaled), atol=1e-9):
        raise RuntimeError("raw Gate weights differ from the fitted model")
    predictions = (raw_scores >= 0.0).astype(np.int64)
    return (
        tuple(float(value) for value in weights),
        bias,
        {
            "observed_examples": len(observed),
            "positive_examples": int(labels.sum()),
            "negative_examples": int(len(labels) - labels.sum()),
            "training_accuracy_at_0_5": float(np.mean(predictions == labels)),
        },
    )


def fit_formal_gate(
    examples: tuple[GateOOFExample, ...],
    *,
    output_dir: str | Path,
    seed: int = 3407,
) -> MappingProxyType:
    """Fit a Training-only bootstrap G0; it is never directly deployable."""

    if not examples:
        raise ValueError("formal Gate training requires examples")
    if any(item.feature_order != FORMAL_GATE_FEATURE_ORDER for item in examples):
        raise ValueError("formal Gate examples use an incompatible feature schema")
    identities = tuple(item.sample_id for item in examples)
    if len(set(identities)) != len(identities):
        raise ValueError("formal Gate example IDs must be unique")
    views = Counter(item.tracker_view for item in examples)
    if set(views) != {"T0", "T1"} or views["T0"] != views["T1"]:
        raise ValueError("formal Gate training requires balanced paired T0/T1 views")
    by_base: dict[tuple[str, int], set[str]] = {}
    for item in examples:
        by_base.setdefault((item.video_id, item.frame_id), set()).add(item.tracker_view)
    if any(value != {"T0", "T1"} for value in by_base.values()):
        raise ValueError("every Gate observation must contain both tracker views")

    weights: dict[str, tuple[float, ...]] = {}
    biases: dict[str, float] = {}
    metrics: dict[str, object] = {}
    values = tuple(examples)
    for scope_index, scope in enumerate(REPAIR_SCOPE_ORDER):
        scope_weights, scope_bias, scope_metrics = _fit_scope(
            values,
            scope_index=scope_index,
            seed=seed + scope_index,
        )
        weights[scope] = scope_weights
        biases[scope] = scope_bias
        metrics[scope] = scope_metrics

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "bootstrap_gate.json"
    artifact = {
        "schema_version": "benefit_gate_linear_v1",
        "gate_stage": "bootstrap_g0",
        "source_split": "training",
        "feature_order": list(FORMAL_GATE_FEATURE_ORDER),
        "weights_by_scope": {scope: list(weights[scope]) for scope in REPAIR_SCOPE_ORDER},
        "bias_by_scope": biases,
        "scope_order": list(REPAIR_SCOPE_ORDER),
        "threshold": 0.0,
        "training_dataset_ids": sorted({item.video_id for item in values}),
        "training_recipe_id": "d0_video_grouped_paired_t0_t1_logistic_v1",
        "normalization_version": "formal_gate_features_v1_raw",
        "rollout_policy_id": "d0_same_snapshot_counterfactual_v1",
    }
    atomic_write_json(artifact_path, artifact)
    summary = {
        "schema_version": "formal_gate_bootstrap_training_summary_v2",
        "gate_stage": "bootstrap_g0",
        "paper_final": False,
        "deployable": False,
        "requires_cross_fitted_d1_rollout": True,
        "seed": seed,
        "example_count": len(values),
        "observation_count": len(by_base),
        "video_ids": sorted({item.video_id for item in values}),
        "folds": sorted({item.fold for item in values}),
        "tracker_views": dict(views),
        "scope_metrics": metrics,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
    }
    atomic_write_json(destination / "training_summary.json", summary)
    return MappingProxyType(summary)


def fit_cross_fitted_bootstrap_gates(
    examples: tuple[GateOOFExample, ...],
    *,
    output_dir: str | Path,
    seed: int = 3407,
) -> MappingProxyType:
    """Fit one bootstrap G0 per held-out video fold for the future D1 rollout."""

    folds = tuple(sorted({item.fold for item in examples}))
    if len(folds) < 2:
        raise ValueError("cross-fitted bootstrap training requires at least two folds")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    for fold in folds:
        fitting = tuple(item for item in examples if item.fold != fold)
        held_out = tuple(item for item in examples if item.fold == fold)
        if not fitting or not held_out:
            raise ValueError("every bootstrap fold needs fit and held-out examples")
        fit_videos = {item.video_id for item in fitting}
        held_out_videos = {item.video_id for item in held_out}
        if fit_videos & held_out_videos:
            raise ValueError("bootstrap folds must be grouped by video")
        fold_dir = destination / f"fold_{fold:02d}"
        summary = fit_formal_gate(
            fitting,
            output_dir=fold_dir,
            seed=seed + fold * len(REPAIR_SCOPE_ORDER),
        )
        artifact_path = fold_dir / "bootstrap_gate.json"
        fold_rows.append(
            {
                "fold": fold,
                "fit_video_ids": sorted(fit_videos),
                "held_out_video_ids": sorted(held_out_videos),
                "artifact": artifact_path.relative_to(destination).as_posix(),
                "artifact_sha256": sha256_file(artifact_path),
                "fit_example_count": len(fitting),
                "held_out_example_count": len(held_out),
                "training_summary_sha256": sha256_file(
                    Path(summary["artifact"]).with_name("training_summary.json")
                ),
            }
        )
    manifest = {
        "schema_version": "cross_fitted_bootstrap_gate_manifest_v1",
        "source_split": "Training",
        "gate_stage": "bootstrap_g0",
        "paper_final": False,
        "deployable": False,
        "held_out_policy_use": "D1_CAUSAL_ROLLOUT_ONLY",
        "folds": fold_rows,
    }
    manifest_path = destination / "cross_fitted_bootstrap_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return MappingProxyType(manifest)


__all__ = ["fit_cross_fitted_bootstrap_gates", "fit_formal_gate"]
