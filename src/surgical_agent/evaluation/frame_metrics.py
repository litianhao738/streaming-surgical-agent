"""Formal video-wise recognition metrics for durable frame predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from surgical_agent.data.constants import TASK_CLASS_COUNTS, TASK_ID_BOUNDS
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from surgical_agent.inference.schemas import PredictionRecord

_MULTILABEL_TASKS = ("instrument", "verb", "target", "ivt")
_IVT_NULL_CLASSES = (94, 95, 96, 97, 98, 99)


@dataclass(frozen=True)
class TaskMapReport:
    class_ap: Mapping[int, float | None]
    class_video_support: Mapping[int, int]
    video_wise_map: float | None
    excluded_classes: tuple[int, ...] = ()


@dataclass(frozen=True)
class PhaseVideoReport:
    accuracy: float
    macro_f1: float
    class_f1: Mapping[int, float | None]
    class_support: Mapping[int, int]
    excluded_classes: tuple[int, ...]


@dataclass(frozen=True)
class PhaseMetricReport:
    video_wise_accuracy: float | None
    video_wise_macro_f1: float | None
    per_video: Mapping[str, PhaseVideoReport]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resampling_unit: str = "video"


@dataclass(frozen=True)
class FrameMetricReport:
    tasks: Mapping[str, TaskMapReport]
    phase: PhaseMetricReport
    score_semantics: tuple[str, ...]
    schema_version: str = "frame_recognition_metrics_v1"


class FrameMetricAccumulator:
    def __init__(self) -> None:
        self._records: list[tuple[PredictionRecord, FrameSupervisionTarget]] = []
        self._identities: set[tuple[str, int]] = set()

    def update(
        self,
        prediction: PredictionRecord,
        target: FrameSupervisionTarget,
    ) -> None:
        snapshot = _snapshot_pair(prediction, target)
        identity = (snapshot[0].video_id, snapshot[0].frame_id)
        if identity in self._identities:
            raise ValueError("duplicate prediction and target identity")
        self._records.append(snapshot)
        self._identities.add(identity)

    def compute(self) -> FrameMetricReport:
        return compute_frame_metric_report(tuple(self._records))


def _phase_id(value: object, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{owner} phase_id must be a non-boolean integer")
    lower, upper = TASK_ID_BOUNDS["phase"]
    if not lower <= value <= upper:
        raise ValueError(f"{owner} phase_id must lie in {lower}..{upper}")
    return int(value)


def _snapshot_pair(
    prediction: PredictionRecord,
    target: FrameSupervisionTarget,
) -> tuple[PredictionRecord, FrameSupervisionTarget]:
    if (prediction.video_id, prediction.frame_id) != (
        target.video_id,
        target.frame_id,
    ):
        raise ValueError("prediction and target identity mismatch")

    mask_values = tuple(
        getattr(target.mask, task) for task in (*_MULTILABEL_TASKS, "phase")
    )
    if any(type(value) is not bool for value in mask_values):
        raise TypeError("task masks must contain exact bool values")
    mask = FrameTaskMask(*mask_values)
    prediction_phase_id = _phase_id(prediction.phase_id, "prediction")
    if mask.phase:
        target_phase_id: int | None = _phase_id(target.phase_id, "target")
    else:
        if target.phase_id is not None:
            raise ValueError("target phase_id must be absent when its mask is false")
        target_phase_id = None

    probabilities = MappingProxyType(
        {task: tuple(prediction.probabilities[task]) for task in TASK_CLASS_COUNTS}
    )
    prediction_snapshot = replace(
        prediction,
        probabilities=probabilities,
        phase_id=prediction_phase_id,
    )
    target_snapshot = replace(
        target,
        instrument_ids=tuple(target.instrument_ids),
        verb_ids=tuple(target.verb_ids),
        target_ids=tuple(target.target_ids),
        triplet_ids=tuple(target.triplet_ids),
        phase_id=target_phase_id,
        mask=mask,
    )
    return prediction_snapshot, target_snapshot


def _snapshot_records(
    records: Iterable[tuple[PredictionRecord, FrameSupervisionTarget]],
) -> tuple[tuple[PredictionRecord, FrameSupervisionTarget], ...]:
    snapshots: list[tuple[PredictionRecord, FrameSupervisionTarget]] = []
    identities: set[tuple[str, int]] = set()
    for prediction, target in records:
        snapshot = _snapshot_pair(prediction, target)
        identity = (snapshot[0].video_id, snapshot[0].frame_id)
        if identity in identities:
            raise ValueError("duplicate prediction and target identity")
        snapshots.append(snapshot)
        identities.add(identity)
    return tuple(snapshots)


def _task_target_ids(target: FrameSupervisionTarget, task: str) -> tuple[int, ...]:
    if task == "instrument":
        return target.instrument_ids
    if task == "verb":
        return target.verb_ids
    if task == "target":
        return target.target_ids
    return target.triplet_ids


def _compute_task_map(
    task: str,
    records: Iterable[tuple[PredictionRecord, FrameSupervisionTarget]],
) -> TaskMapReport:
    by_video: dict[str, list[tuple[PredictionRecord, FrameSupervisionTarget]]] = (
        defaultdict(list)
    )
    for prediction, target in records:
        if getattr(target.mask, task):
            by_video[prediction.video_id].append((prediction, target))

    class_count = TASK_CLASS_COUNTS[task]
    excluded_classes = _IVT_NULL_CLASSES if task == "ivt" else ()
    excluded_set = set(excluded_classes)
    class_ap: dict[int, float | None] = {}
    class_video_support: dict[int, int] = {}
    for class_id in range(class_count):
        video_ap: list[float] = []
        for video_id in sorted(by_video):
            video_records = by_video[video_id]
            targets = np.asarray(
                [
                    class_id in _task_target_ids(target, task)
                    for _, target in video_records
                ],
                dtype=np.int8,
            )
            if not np.any(targets):
                continue
            scores = np.asarray(
                [
                    prediction.probabilities[task][class_id]
                    for prediction, _ in video_records
                ],
                dtype=np.float64,
            )
            video_ap.append(float(average_precision_score(targets, scores)))
        class_video_support[class_id] = len(video_ap)
        class_ap[class_id] = (
            None
            if class_id in excluded_set or not video_ap
            else float(np.mean(video_ap))
        )

    defined_ap = [value for value in class_ap.values() if value is not None]
    return TaskMapReport(
        class_ap=class_ap,
        class_video_support=class_video_support,
        video_wise_map=float(np.mean(defined_ap)) if defined_ap else None,
        excluded_classes=excluded_classes,
    )


def _compute_phase(
    records: Iterable[tuple[PredictionRecord, FrameSupervisionTarget]],
) -> PhaseMetricReport:
    by_video: dict[str, list[tuple[PredictionRecord, FrameSupervisionTarget]]] = (
        defaultdict(list)
    )
    for prediction, target in records:
        if target.mask.phase:
            by_video[prediction.video_id].append((prediction, target))

    per_video: dict[str, PhaseVideoReport] = {}
    for video_id in sorted(by_video):
        video_records = by_video[video_id]
        targets = np.asarray(
            [target.phase_id for _, target in video_records],
            dtype=np.int64,
        )
        predictions = np.asarray(
            [prediction.phase_id for prediction, _ in video_records],
            dtype=np.int64,
        )
        class_f1: dict[int, float | None] = {}
        class_support: dict[int, int] = {}
        for class_id in range(TASK_CLASS_COUNTS["phase"]):
            target_binary = targets == class_id
            support = int(np.sum(target_binary))
            class_support[class_id] = support
            class_f1[class_id] = (
                float(
                    f1_score(
                        target_binary,
                        predictions == class_id,
                        zero_division=0,
                    )
                )
                if support
                else None
            )
        excluded_classes = tuple(
            class_id for class_id, support in class_support.items() if support == 0
        )
        defined_f1 = [value for value in class_f1.values() if value is not None]
        per_video[video_id] = PhaseVideoReport(
            accuracy=float(np.mean(predictions == targets)),
            macro_f1=float(np.mean(defined_f1)),
            class_f1=class_f1,
            class_support=class_support,
            excluded_classes=excluded_classes,
        )

    videos = tuple(per_video.values())
    return PhaseMetricReport(
        video_wise_accuracy=(
            float(np.mean([video.accuracy for video in videos])) if videos else None
        ),
        video_wise_macro_f1=(
            float(np.mean([video.macro_f1 for video in videos])) if videos else None
        ),
        per_video=per_video,
    )


def compute_frame_metric_report(
    records: Iterable[tuple[PredictionRecord, FrameSupervisionTarget]],
) -> FrameMetricReport:
    record_tuple = _snapshot_records(records)
    return FrameMetricReport(
        tasks={
            task: _compute_task_map(task, record_tuple) for task in _MULTILABEL_TASKS
        },
        phase=_compute_phase(record_tuple),
        score_semantics=tuple(
            sorted({prediction.score_semantics for prediction, _ in record_tuple})
        ),
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def framework_gain(full: float, baseline: float) -> float:
    """Return the formal full-framework metric minus its baseline."""

    return _finite_number(full, "full") - _finite_number(baseline, "baseline")


def paired_video_bootstrap(
    full_by_video: Mapping[str, float],
    baseline_by_video: Mapping[str, float],
    *,
    seed: int,
    resamples: int,
) -> BootstrapInterval:
    """Bootstrap paired metric gains, resampling complete named videos only."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral):
        raise TypeError("resamples must be an integer")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not full_by_video or not baseline_by_video:
        raise ValueError("paired bootstrap requires at least one video")
    if set(full_by_video) != set(baseline_by_video):
        raise ValueError("full and baseline video IDs must match exactly")
    video_ids = tuple(sorted(full_by_video))
    if any(not isinstance(video_id, str) for video_id in video_ids):
        raise TypeError("video IDs must be strings")

    gains = np.asarray(
        [
            framework_gain(full_by_video[video_id], baseline_by_video[video_id])
            for video_id in video_ids
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(int(seed))
    sampled_indices = generator.integers(
        0,
        len(video_ids),
        size=(int(resamples), len(video_ids)),
    )
    sampled_means = np.mean(gains[sampled_indices], axis=1)
    lower, upper = np.percentile(sampled_means, (2.5, 97.5), method="linear")
    return BootstrapInterval(
        estimate=float(np.mean(gains)),
        lower=float(lower),
        upper=float(upper),
        confidence=0.95,
    )
