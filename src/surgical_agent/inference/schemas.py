"""Gold-free prediction contracts shared by all pipeline backends."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from surgical_agent.data.constants import TASK_CLASS_COUNTS, TASK_ID_BOUNDS
from surgical_agent.data.schemas import DatasetSplit

PREDICTION_SCHEMA_VERSION = "prediction_record_v1"


def _validate_ids(name: str, values: tuple[int, ...], task: str) -> None:
    lower, upper = TASK_ID_BOUNDS[task]
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be sorted and unique")
    if any(not isinstance(value, int) or not lower <= value <= upper for value in values):
        raise ValueError(f"{name} contains an ID outside {lower}..{upper}")


def _validate_probabilities(probabilities: Mapping[str, tuple[float, ...]]) -> None:
    if set(probabilities) != set(TASK_CLASS_COUNTS):
        raise ValueError("probabilities must contain exactly the five task heads")
    for task, class_count in TASK_CLASS_COUNTS.items():
        values = probabilities[task]
        if len(values) != class_count:
            raise ValueError(f"{task} probabilities must contain {class_count} values")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{task} probabilities must be finite values in [0, 1]")


@dataclass(frozen=True)
class InitialPrediction:
    """Validated frame-level hypothesis emitted by a perception backend."""

    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    triplet_ids: tuple[int, ...]
    phase_id: int
    probabilities: Mapping[str, tuple[float, ...]]
    granularity: str = "frame_multilabel"
    backend: str = "local_smoke"

    def __post_init__(self) -> None:
        for name, values, task in (
            ("instrument_ids", self.instrument_ids, "instrument"),
            ("verb_ids", self.verb_ids, "verb"),
            ("target_ids", self.target_ids, "target"),
            ("triplet_ids", self.triplet_ids, "ivt"),
        ):
            _validate_ids(name, values, task)
        phase_lower, phase_upper = TASK_ID_BOUNDS["phase"]
        if not phase_lower <= self.phase_id <= phase_upper:
            raise ValueError(f"phase_id must lie in {phase_lower}..{phase_upper}")
        if self.granularity != "frame_multilabel":
            raise ValueError("P2 predictions must declare frame_multilabel granularity")
        _validate_probabilities(self.probabilities)


@dataclass(frozen=True)
class PredictionRecord:
    """Serializable runtime result that cannot carry labels or target masks."""

    run_id: str
    video_id: str
    frame_id: int
    source_split: DatasetSplit
    causal_frame_ids: tuple[int, ...]
    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    triplet_ids: tuple[int, ...]
    phase_id: int
    granularity: str
    backend: str
    gate_action: str
    verification_status: str
    alignment_version: str
    probabilities: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    trace: tuple[str, ...] = ()
    failure_reason: str | None = None
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id or not self.video_id:
            raise ValueError("run_id and video_id must not be empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.causal_frame_ids or self.causal_frame_ids[-1] != self.frame_id:
            raise ValueError("causal_frame_ids must end at frame_id")
        for name, values, task in (
            ("instrument_ids", self.instrument_ids, "instrument"),
            ("verb_ids", self.verb_ids, "verb"),
            ("target_ids", self.target_ids, "target"),
            ("triplet_ids", self.triplet_ids, "ivt"),
        ):
            _validate_ids(name, values, task)
        phase_lower, phase_upper = TASK_ID_BOUNDS["phase"]
        if not phase_lower <= self.phase_id <= phase_upper:
            raise ValueError(f"phase_id must lie in {phase_lower}..{phase_upper}")
        forbidden = {"ground_truth", "label_mask", "frame_task_mask"}
        if forbidden & set(self.probabilities):
            raise ValueError("Prediction probabilities contain a GT-bearing key")
        _validate_probabilities(self.probabilities)
