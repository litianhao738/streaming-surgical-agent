"""Training-only causal windows and frame-level Joint Perception targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter, ResolvedSample
from surgical_agent.data.schemas import (
    DatasetSplit,
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
    InferenceSample,
)

_MULTILABEL_SPECS = {
    "instrument": "instrument_ids",
    "verb": "verb_ids",
    "target": "target_ids",
    "ivt": "triplet_ids",
}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PerceptionTrainingRecord:
    inference: InferenceSample
    target: FrameSupervisionTarget
    source: str

    @property
    def identity(self) -> tuple[str, int]:
        return self.inference.video_id, self.inference.target_frame_id


@dataclass(frozen=True)
class PerceptionTrainingItem:
    frames: Tensor
    record: PerceptionTrainingRecord


@dataclass(frozen=True)
class PerceptionTrainingBatch:
    frames: Tensor
    targets: tuple[FrameSupervisionTarget, ...]
    identities: tuple[tuple[str, int], ...]


def aggregate_evaluation_target(
    evaluation: EvaluationTarget,
    *,
    source: str,
) -> FrameSupervisionTarget:
    """Convert fully available instance labels into one multi-label frame target."""

    if not isinstance(evaluation, EvaluationTarget):
        raise TypeError("evaluation must be an EvaluationTarget")
    instances = tuple(evaluation.instances)
    availability = {
        task: bool(instances)
        and all(getattr(instance.mask, task) for instance in instances)
        for task in TASK_CLASS_COUNTS
    }

    def ids(task: str, attribute: str) -> tuple[int, ...]:
        if not availability[task]:
            return ()
        return tuple(sorted({getattr(instance, attribute) for instance in instances}))

    phase_id: int | None = None
    if availability["phase"]:
        phases = {instance.phase_id for instance in instances}
        if len(phases) != 1:
            raise ValueError("evaluation instances contain conflicting phases")
        phase_id = next(iter(phases))
    return FrameSupervisionTarget(
        video_id=evaluation.video_id,
        frame_id=evaluation.frame_id,
        instrument_ids=ids("instrument", "instrument_id"),
        verb_ids=ids("verb", "verb_id"),
        target_ids=ids("target", "target_id"),
        triplet_ids=ids("ivt", "triplet_id"),
        phase_id=phase_id,
        mask=FrameTaskMask(
            instrument=availability["instrument"],
            verb=availability["verb"],
            target=availability["target"],
            ivt=availability["ivt"],
            phase=availability["phase"],
        ),
        source_granularity="frame_multilabel",
        source=source,
    )


def _has_all_tasks(target: FrameSupervisionTarget | None) -> bool:
    return target is not None and all(
        getattr(target.mask, task) for task in TASK_CLASS_COUNTS
    )


def target_for_training(sample: ResolvedSample) -> FrameSupervisionTarget | None:
    """Resolve frame labels without allowing Testing or inventing masked tasks."""

    if sample.inference.source_split is DatasetSplit.TESTING:
        raise ValueError("Testing data is forbidden in perception training")
    if _has_all_tasks(sample.frame_supervision):
        return sample.frame_supervision
    if sample.evaluation is not None and sample.evaluation.instances:
        source = sample.provenance.annotation_source or "instance_annotation"
        aggregated = aggregate_evaluation_target(sample.evaluation, source=source)
        if any(getattr(aggregated.mask, task) for task in TASK_CLASS_COUNTS):
            return aggregated
    return sample.frame_supervision


def build_perception_records(
    adapter: CholecTrack20DatasetAdapter,
    video_ids: Sequence[str],
    *,
    allowed_split: DatasetSplit,
    max_samples_per_video: int | None = None,
) -> tuple[PerceptionTrainingRecord, ...]:
    """Build a deterministic video-major train or validation record set."""

    if allowed_split is DatasetSplit.TESTING:
        raise ValueError("Testing cannot be a supervised perception split")
    records: list[PerceptionTrainingRecord] = []
    for video_id in video_ids:
        for sample in adapter.iter_video(video_id, max_samples=max_samples_per_video):
            if sample.inference.source_split is not allowed_split:
                raise ValueError("perception records crossed the requested split")
            target = target_for_training(sample)
            if target is None or not any(
                getattr(target.mask, task) for task in TASK_CLASS_COUNTS
            ):
                continue
            records.append(
                PerceptionTrainingRecord(
                    inference=sample.inference,
                    target=target,
                    source=target.source,
                )
            )
    if not records:
        raise ValueError("no supervised perception records were resolved")
    identities = tuple(item.identity for item in records)
    if len(set(identities)) != len(identities):
        raise ValueError("perception record identities must be unique")
    return tuple(records)


def preprocess_causal_frames(
    frames: Tensor,
    *,
    image_size: int,
    window_size: int,
) -> Tensor:
    """Resize, left-pad and normalize a ``[T,3,H,W]`` RGB window."""

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [T,3,H,W]")
    if not frames.is_floating_point():
        raise TypeError("frames must be floating point")
    if image_size <= 0 or window_size <= 0:
        raise ValueError("image_size and window_size must be positive")
    if not 1 <= frames.shape[0] <= window_size:
        raise ValueError("frame count must lie in 1..window_size")
    if not torch.isfinite(frames).all() or torch.any(frames < 0) or torch.any(frames > 1):
        raise ValueError("frames must contain finite RGB values in [0,1]")
    resized = functional.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    if resized.shape[0] < window_size:
        padding = resized[0:1].expand(window_size - resized.shape[0], -1, -1, -1)
        resized = torch.cat((padding, resized), dim=0)
    mean = resized.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


class CausalFrameTensorDataset:
    """Load only past-to-current PNG windows for train/validation."""

    def __init__(
        self,
        records: Sequence[PerceptionTrainingRecord],
        *,
        image_size: int,
        window_size: int,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        self.records = tuple(records)
        self.image_size = image_size
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PerceptionTrainingItem:
        record = self.records[index]
        arrays: list[Tensor] = []
        for reference in record.inference.media_refs:
            path = Path(reference)
            if path.suffix.lower() != ".png":
                raise ValueError("supervised perception training requires PNG frames")
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            arrays.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        frames = preprocess_causal_frames(
            torch.stack(arrays),
            image_size=self.image_size,
            window_size=self.window_size,
        )
        return PerceptionTrainingItem(frames=frames, record=record)


def collate_perception_batch(
    items: Sequence[PerceptionTrainingItem],
) -> PerceptionTrainingBatch:
    if not items:
        raise ValueError("cannot collate an empty perception batch")
    return PerceptionTrainingBatch(
        frames=torch.stack([item.frames for item in items]),
        targets=tuple(item.record.target for item in items),
        identities=tuple(item.record.identity for item in items),
    )


def compute_class_weights(
    records: Sequence[PerceptionTrainingRecord],
    *,
    max_positive_weight: float = 20.0,
) -> dict[str, tuple[float, ...]]:
    """Compute Training-only positive/phase weights for class imbalance."""

    if not records or max_positive_weight < 1.0:
        raise ValueError("records and max_positive_weight are invalid")
    result: dict[str, tuple[float, ...]] = {}
    for task, attribute in _MULTILABEL_SPECS.items():
        valid = [item.target for item in records if getattr(item.target.mask, task)]
        counts = [0] * TASK_CLASS_COUNTS[task]
        for target in valid:
            for class_id in getattr(target, attribute):
                counts[class_id] += 1
        weights = []
        for count in counts:
            value = 1.0 if count == 0 else (len(valid) - count) / count
            weights.append(min(max(float(value), 1.0), max_positive_weight))
        result[task] = tuple(weights)
    valid_phase = [
        item.target.phase_id
        for item in records
        if item.target.mask.phase and item.target.phase_id is not None
    ]
    phase_counts = [valid_phase.count(class_id) for class_id in range(TASK_CLASS_COUNTS["phase"])]
    phase_weights = [
        1.0
        if count == 0
        else min(max(len(valid_phase) / (len(phase_counts) * count), 0.25), 10.0)
        for count in phase_counts
    ]
    result["phase"] = tuple(float(value) for value in phase_weights)
    return result


__all__ = [
    "CausalFrameTensorDataset",
    "PerceptionTrainingBatch",
    "PerceptionTrainingRecord",
    "aggregate_evaluation_target",
    "build_perception_records",
    "collate_perception_batch",
    "compute_class_weights",
    "preprocess_causal_frames",
    "target_for_training",
]
