"""Training-held-out threshold diagnostics for a bootstrap Benefit Gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.oof_dataset import GateOOFExample


def _score(
    artifact: FrozenLinearBenefitArtifact,
    example: GateOOFExample,
) -> tuple[str, float, int | None]:
    values = example.feature_values
    scores = {
        scope: artifact.bias_by_scope[scope]
        + sum(
            weight * value
            for weight, value in zip(
                artifact.weights_by_scope[scope], values, strict=True
            )
        )
        for scope in artifact.scope_order
    }
    scope = max(
        artifact.scope_order,
        key=lambda value: (scores[value], -artifact.scope_order.index(value)),
    )
    label = example.benefit_labels[artifact.scope_order.index(scope)]
    return scope, scores[scope], label


def calibrate_formal_gate(
    examples: tuple[GateOOFExample, ...],
    *,
    artifact_path: str | Path,
    output_dir: str | Path,
    verification_cost: float = 0.05,
    false_positive_harm: float = 1.0,
) -> MappingProxyType:
    """Choose a diagnostic Training threshold; never claim final calibration."""

    if not examples:
        raise ValueError("Gate calibration requires Training-held-out examples")
    if not 0.0 <= verification_cost <= 1.0 or false_positive_harm < 0.0:
        raise ValueError("calibration utility parameters are invalid")
    artifact_source = Path(artifact_path).expanduser().resolve()
    artifact = FrozenLinearBenefitArtifact.from_json(
        artifact_source, allow_bootstrap=True
    )
    if artifact.gate_stage != "bootstrap_g0":
        raise ValueError("Training diagnostics require a bootstrap_g0 artifact")
    calibration_video_ids = {item.video_id for item in examples}
    if calibration_video_ids & set(artifact.training_dataset_ids):
        raise ValueError(
            "Gate fit videos and Training-held-out calibration videos must be disjoint"
        )
    scored = tuple(_score(artifact, item) for item in examples)
    observed = tuple(item for item in scored if item[2] is not None)
    if not observed:
        raise ValueError("calibration has no observed chosen-scope labels")
    thresholds = sorted({score for _, score, _ in observed}, reverse=True)
    thresholds = [float("inf"), *thresholds, float("-inf")]
    best: tuple[float, float, int, int, int] | None = None
    for threshold in thresholds:
        utility = 0.0
        selected = true_positive = false_positive = 0
        for _, score, raw_label in observed:
            label = int(raw_label)
            if score < threshold:
                continue
            selected += 1
            utility -= verification_cost
            if label == 1:
                utility += 1.0
                true_positive += 1
            else:
                utility -= false_positive_harm
                false_positive += 1
        mean_utility = utility / len(observed)
        candidate = (mean_utility, threshold, selected, true_positive, false_positive)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    _, threshold, selected, true_positive, false_positive = best
    if threshold == float("inf"):
        threshold = max(score for _, score, _ in observed) + 1.0
    elif threshold == float("-inf"):
        threshold = min(score for _, score, _ in observed) - 1.0

    raw = json.loads(artifact_source.read_text(encoding="utf-8"))
    raw["threshold"] = threshold
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    calibrated_path = destination / "bootstrap_gate_diagnostic.json"
    atomic_write_json(calibrated_path, raw)
    summary = {
        "schema_version": "bootstrap_gate_threshold_diagnostic_v2",
        "source_split": "Training",
        "partition": "training_internal_diagnostic",
        "test_data_seen": False,
        "paper_final": False,
        "deployable": False,
        "requires_cross_fitted_d1_rollout": True,
        "requires_validation_operating_point": True,
        "verification_cost": verification_cost,
        "false_positive_harm": false_positive_harm,
        "threshold": threshold,
        "observed_examples": len(observed),
        "selected_examples": selected,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "mean_utility": best[0],
        "calibration_video_ids": sorted(calibration_video_ids),
        "training_artifact_sha256": sha256_file(artifact_source),
        "calibrated_artifact": str(calibrated_path),
        "calibrated_artifact_sha256": sha256_file(calibrated_path),
    }
    atomic_write_json(destination / "calibration_summary.json", summary)
    return MappingProxyType(summary)


__all__ = ["calibrate_formal_gate"]
