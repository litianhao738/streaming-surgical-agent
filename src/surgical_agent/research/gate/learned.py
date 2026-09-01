"""Training-only utilities for the demo Full Learned Benefit Gate.

The online runtime loads only the exported JSON linear artifact.  Pickled
scikit-learn objects are retained strictly for reproducibility and analysis.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from surgical_agent.data.schemas import FrameSupervisionTarget
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import TASK_NAMES
from surgical_agent.research.gate.features import (
    NO_TRACKER_FEATURE_ORDER,
    WITH_TRACKER_FEATURE_ORDER,
)

GATE_EXAMPLE_SCHEMA_VERSION = "learned_gate_example_v1"
GATE_TRAINING_SCHEMA_VERSION = "learned_gate_training_v1"


def _finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _freeze_features(
    value: Mapping[str, float],
    *,
    expected: tuple[str, ...],
    name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly the frozen feature schema")
    frozen = {
        feature: _finite(value[feature], name=f"{name}.{feature}")
        for feature in expected
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class LearnedGateExample:
    """One GT-isolated counterfactual record used only by the trainer."""

    sample_id: str
    video_id: str
    frame_id: int
    partition: str
    no_tracker_features: Mapping[str, float]
    with_tracker_features: Mapping[str, float]
    initial_utility: float
    verified_utility: float
    benefit_delta: float
    benefit_label: int
    valid_tasks: tuple[str, ...]
    schema_version: str = GATE_EXAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id or not self.video_id:
            raise ValueError("sample_id and video_id must not be empty")
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool):
            raise TypeError("frame_id must be an integer")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.partition not in {"train", "dev"}:
            raise ValueError("partition must be train or dev")
        if self.schema_version != GATE_EXAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported learned Gate example schema")
        no_tracker = _freeze_features(
            self.no_tracker_features,
            expected=NO_TRACKER_FEATURE_ORDER,
            name="no_tracker_features",
        )
        with_tracker = _freeze_features(
            self.with_tracker_features,
            expected=WITH_TRACKER_FEATURE_ORDER,
            name="with_tracker_features",
        )
        initial = _finite(self.initial_utility, name="initial_utility")
        verified = _finite(self.verified_utility, name="verified_utility")
        delta = _finite(self.benefit_delta, name="benefit_delta")
        if not all(0.0 <= value <= 1.0 for value in (initial, verified)):
            raise ValueError("utilities must lie in [0, 1]")
        if not math.isclose(delta, verified - initial, abs_tol=1e-9):
            raise ValueError("benefit_delta must equal verified_utility-initial_utility")
        if self.benefit_label not in {0, 1}:
            raise ValueError("benefit_label must be binary")
        if self.benefit_label != int(delta > 0.0):
            raise ValueError("benefit_label must reflect a strictly positive utility delta")
        valid_tasks = tuple(self.valid_tasks)
        if not valid_tasks or any(task not in TASK_NAMES for task in valid_tasks):
            raise ValueError("valid_tasks must contain known task names")
        if len(set(valid_tasks)) != len(valid_tasks):
            raise ValueError("valid_tasks must be unique")
        object.__setattr__(self, "no_tracker_features", no_tracker)
        object.__setattr__(self, "with_tracker_features", with_tracker)
        object.__setattr__(self, "initial_utility", initial)
        object.__setattr__(self, "verified_utility", verified)
        object.__setattr__(self, "benefit_delta", delta)
        object.__setattr__(self, "valid_tasks", valid_tasks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "partition": self.partition,
            "no_tracker_features": dict(self.no_tracker_features),
            "with_tracker_features": dict(self.with_tracker_features),
            "initial_utility": self.initial_utility,
            "verified_utility": self.verified_utility,
            "benefit_delta": self.benefit_delta,
            "benefit_label": self.benefit_label,
            "valid_tasks": list(self.valid_tasks),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LearnedGateExample:
        expected = {
            "schema_version",
            "sample_id",
            "video_id",
            "frame_id",
            "partition",
            "no_tracker_features",
            "with_tracker_features",
            "initial_utility",
            "verified_utility",
            "benefit_delta",
            "benefit_label",
            "valid_tasks",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("learned Gate example fields do not match the schema")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            sample_id=value["sample_id"],  # type: ignore[arg-type]
            video_id=value["video_id"],  # type: ignore[arg-type]
            frame_id=value["frame_id"],  # type: ignore[arg-type]
            partition=value["partition"],  # type: ignore[arg-type]
            no_tracker_features=value["no_tracker_features"],  # type: ignore[arg-type]
            with_tracker_features=value["with_tracker_features"],  # type: ignore[arg-type]
            initial_utility=value["initial_utility"],  # type: ignore[arg-type]
            verified_utility=value["verified_utility"],  # type: ignore[arg-type]
            benefit_delta=value["benefit_delta"],  # type: ignore[arg-type]
            benefit_label=value["benefit_label"],  # type: ignore[arg-type]
            valid_tasks=tuple(value["valid_tasks"]),  # type: ignore[arg-type]
        )


def _set_f1(predicted: Sequence[int], expected: Sequence[int]) -> float:
    predicted_set = set(predicted)
    expected_set = set(expected)
    if not predicted_set and not expected_set:
        return 1.0
    denominator = len(predicted_set) + len(expected_set)
    return 2.0 * len(predicted_set & expected_set) / denominator if denominator else 0.0


def prediction_utility(
    prediction: InitialPrediction,
    target: FrameSupervisionTarget,
) -> tuple[float, Mapping[str, float]]:
    """Macro utility over only label heads explicitly marked valid by GT masks."""

    if not isinstance(prediction, InitialPrediction):
        raise TypeError("prediction must be an InitialPrediction")
    if not isinstance(target, FrameSupervisionTarget):
        raise TypeError("target must be a FrameSupervisionTarget")
    values: dict[str, float] = {}
    pairs = {
        "instrument": (prediction.instrument_ids, target.instrument_ids),
        "verb": (prediction.verb_ids, target.verb_ids),
        "target": (prediction.target_ids, target.target_ids),
        "ivt": (prediction.triplet_ids, target.triplet_ids),
    }
    for task, (predicted, expected) in pairs.items():
        if getattr(target.mask, task):
            values[task] = _set_f1(predicted, expected)
    if target.mask.phase:
        assert target.phase_id is not None
        values["phase"] = float(prediction.phase_id == target.phase_id)
    if not values:
        raise ValueError("target has no valid task supervision")
    return sum(values.values()) / len(values), MappingProxyType(values)


def build_example(
    *,
    sample_id: str,
    video_id: str,
    frame_id: int,
    partition: str,
    no_tracker_features: Mapping[str, float],
    with_tracker_features: Mapping[str, float],
    initial_prediction: InitialPrediction,
    verified_prediction: InitialPrediction,
    target: FrameSupervisionTarget,
) -> LearnedGateExample:
    initial, task_values = prediction_utility(initial_prediction, target)
    verified, _ = prediction_utility(verified_prediction, target)
    delta = verified - initial
    return LearnedGateExample(
        sample_id=sample_id,
        video_id=video_id,
        frame_id=frame_id,
        partition=partition,
        no_tracker_features=no_tracker_features,
        with_tracker_features=with_tracker_features,
        initial_utility=initial,
        verified_utility=verified,
        benefit_delta=delta,
        benefit_label=int(delta > 0.0),
        valid_tasks=tuple(task_values),
    )


def write_examples(path: str | Path, examples: Iterable[LearnedGateExample]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(examples)
    if not values:
        raise ValueError("at least one learned Gate example is required")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in values:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
    return destination


def load_examples(path: str | Path) -> tuple[LearnedGateExample, ...]:
    source = Path(path)
    records: list[LearnedGateExample] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            records.append(LearnedGateExample.from_dict(value))
    if not records:
        raise ValueError("learned Gate example file is empty")
    identities = tuple(item.sample_id for item in records)
    if len(set(identities)) != len(identities):
        raise ValueError("learned Gate sample IDs must be unique")
    return tuple(records)


def _binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in probabilities)})
    best: tuple[tuple[float, float, float, float], float, dict[str, float | int]] | None = None
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(np.int64)
        metrics = _binary_metrics(labels, predicted)
        key = (
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fit_one(
    examples: tuple[LearnedGateExample, ...],
    *,
    feature_order: tuple[str, ...],
    feature_field: str,
    variant: str,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    train = tuple(item for item in examples if item.partition == "train")
    dev = tuple(item for item in examples if item.partition == "dev")
    if not train or not dev:
        raise ValueError("learned Gate requires non-empty train and dev partitions")
    train_labels = np.asarray([item.benefit_label for item in train], dtype=np.int64)
    dev_labels = np.asarray([item.benefit_label for item in dev], dtype=np.int64)
    if len(set(train_labels.tolist())) != 2:
        raise ValueError("training partition must contain both benefit classes")
    if len(set(dev_labels.tolist())) != 2:
        raise ValueError("dev partition must contain both benefit classes")

    def matrix(items: tuple[LearnedGateExample, ...]) -> np.ndarray:
        return np.asarray(
            [
                [float(getattr(item, feature_field)[name]) for name in feature_order]
                for item in items
            ],
            dtype=np.float64,
        )

    train_x = matrix(train)
    dev_x = matrix(dev)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(scaled_train, train_labels)
    dev_probabilities = model.predict_proba(scaler.transform(dev_x))[:, 1]
    probability_threshold, dev_metrics = _select_threshold(
        dev_labels, dev_probabilities
    )

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    standardized_weights = np.asarray(model.coef_[0], dtype=np.float64)
    raw_weights = standardized_weights / scale
    raw_bias = float(model.intercept_[0] - np.sum(standardized_weights * mean / scale))
    raw_threshold = _logit(probability_threshold)

    suffix = variant.removeprefix("gate_")
    model_path = output_dir / f"{variant}.pkl"
    scaler_path = output_dir / f"scaler_{suffix}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    deploy_path = output_dir / f"{variant}.json"
    deploy_payload = {
        "schema_version": "benefit_gate_linear_v1",
        "gate_stage": "final_g1",
        "source_split": "training",
        "feature_order": list(feature_order),
        "weights_by_scope": {"joint": raw_weights.tolist()},
        "bias_by_scope": {"joint": raw_bias},
        "scope_order": ["joint"],
        "threshold": raw_threshold,
        "training_dataset_ids": sorted({item.video_id for item in train}),
        "training_recipe_id": "standard_scaler_logistic_balanced_seeded_v1",
        "normalization_version": "demo_gate_features_v1_raw",
        "rollout_policy_id": "forced_joint_targeted_verification_v1",
    }
    _write_json(deploy_path, deploy_payload)
    feature_schema_path = output_dir / f"feature_schema_{suffix}.json"
    _write_json(
        feature_schema_path,
        {
            "schema_version": "demo_gate_feature_schema_v1",
            "variant": variant,
            "feature_order": list(feature_order),
            "feature_count": len(feature_order),
            "uses_tracker": feature_order == WITH_TRACKER_FEATURE_ORDER,
            "gold_free_at_inference": True,
            "demo_only": True,
            "paper_final": False,
        },
    )
    raw_scores = raw_bias + dev_x @ raw_weights
    if not np.allclose(raw_scores, model.decision_function(scaler.transform(dev_x)), atol=1e-9):
        raise RuntimeError("raw deploy artifact is not equivalent to sklearn model")
    return {
        "variant": variant,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "train_positive": int(np.sum(train_labels)),
        "dev_positive": int(np.sum(dev_labels)),
        "probability_threshold": probability_threshold,
        "raw_score_threshold": raw_threshold,
        "dev_metrics": dev_metrics,
        "deploy_artifact": str(deploy_path.resolve()),
        "model_pickle": str(model_path.resolve()),
        "scaler_pickle": str(scaler_path.resolve()),
        "feature_schema": str(feature_schema_path.resolve()),
    }


def fit_learned_gate_pair(
    examples: Sequence[LearnedGateExample],
    *,
    output_dir: str | Path,
    seed: int = 20260831,
) -> Mapping[str, object]:
    """Fit no-tracker and with-tracker gates and export runtime artifacts."""

    values = tuple(examples)
    if not values:
        raise ValueError("at least one learned Gate example is required")
    identities = tuple(item.sample_id for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("learned Gate sample IDs must be unique")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    no_tracker = _fit_one(
        values,
        feature_order=NO_TRACKER_FEATURE_ORDER,
        feature_field="no_tracker_features",
        variant="gate_no_tracker",
        output_dir=destination,
        seed=seed,
    )
    with_tracker = _fit_one(
        values,
        feature_order=WITH_TRACKER_FEATURE_ORDER,
        feature_field="with_tracker_features",
        variant="gate_with_tracker",
        output_dir=destination,
        seed=seed,
    )
    summary: dict[str, object] = {
        "schema_version": GATE_TRAINING_SCHEMA_VERSION,
        "seed": seed,
        "source_split": "Training",
        "demo_only": True,
        "paper_final": False,
        "label_definition": "verified_macro_valid_task_utility > initial_macro_valid_task_utility",
        "example_count": len(values),
        "source_video_ids": sorted({item.video_id for item in values}),
        "no_tracker": no_tracker,
        "with_tracker": with_tracker,
    }
    _write_json(destination / "training_summary.json", summary)
    return MappingProxyType(summary)


__all__ = [
    "GATE_EXAMPLE_SCHEMA_VERSION",
    "LearnedGateExample",
    "build_example",
    "fit_learned_gate_pair",
    "load_examples",
    "prediction_utility",
    "write_examples",
]
