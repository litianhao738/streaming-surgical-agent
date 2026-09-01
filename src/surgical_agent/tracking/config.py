"""Frozen configuration for the predicted instrument tracker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.config.loader import load_yaml

TRACKER_CONFIG_SCHEMA_VERSION = "predicted_tracker_training_v1"
TRACKER_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"

_FIELDS = {
    "schema_version",
    "architecture",
    "initial_weights",
    "num_classes",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_workers",
    "min_size",
    "max_size",
    "score_threshold",
    "nms_threshold",
    "association_iou_threshold",
    "max_age",
    "oof_folds",
}


def _integer(raw: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = raw[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"tracker.{name} must be an integer >= {minimum}")
    return value


def _float(
    raw: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = raw[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"tracker.{name} must be numeric")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"tracker.{name} is outside the supported range")
    return result


@dataclass(frozen=True)
class TrackerTrainingConfig:
    architecture: str
    initial_weights: str
    num_classes: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    min_size: int
    max_size: int
    score_threshold: float
    nms_threshold: float
    association_iou_threshold: float
    max_age: int
    oof_folds: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> TrackerTrainingConfig:
        if set(raw) != _FIELDS:
            raise ValueError("tracker config fields do not match the frozen schema")
        if raw["schema_version"] != TRACKER_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported tracker configuration schema")
        if raw["architecture"] != TRACKER_ARCHITECTURE:
            raise ValueError("unsupported tracker detector architecture")
        if raw["initial_weights"] not in {"COCO_V1", "NONE"}:
            raise ValueError("initial_weights must be COCO_V1 or NONE")
        if raw["num_classes"] != 8:
            raise ValueError("num_classes must be 8 (background plus seven instruments)")
        min_size = _integer(raw, "min_size", minimum=32)
        max_size = _integer(raw, "max_size", minimum=min_size)
        return cls(
            architecture=str(raw["architecture"]),
            initial_weights=str(raw["initial_weights"]),
            num_classes=_integer(raw, "num_classes", minimum=8),
            seed=_integer(raw, "seed", minimum=0),
            epochs=_integer(raw, "epochs", minimum=1),
            batch_size=_integer(raw, "batch_size", minimum=1),
            learning_rate=_float(raw, "learning_rate", minimum=1e-12),
            weight_decay=_float(raw, "weight_decay", minimum=0.0),
            num_workers=_integer(raw, "num_workers", minimum=0),
            min_size=min_size,
            max_size=max_size,
            score_threshold=_float(
                raw, "score_threshold", minimum=0.0, maximum=1.0
            ),
            nms_threshold=_float(raw, "nms_threshold", minimum=0.0, maximum=1.0),
            association_iou_threshold=_float(
                raw, "association_iou_threshold", minimum=0.0, maximum=1.0
            ),
            max_age=_integer(raw, "max_age", minimum=0),
            oof_folds=_integer(raw, "oof_folds", minimum=2),
        )


def load_tracker_training_config(path: str | Path) -> TrackerTrainingConfig:
    raw = load_yaml(Path(path).expanduser().resolve())
    if not isinstance(raw, Mapping):
        raise TypeError("tracker config must be a mapping")
    tracker = raw.get("tracker")
    if not isinstance(tracker, Mapping):
        raise TypeError("tracker config requires a tracker mapping")
    return TrackerTrainingConfig.from_mapping(tracker)


__all__ = [
    "TRACKER_ARCHITECTURE",
    "TRACKER_CONFIG_SCHEMA_VERSION",
    "TrackerTrainingConfig",
    "load_tracker_training_config",
]
