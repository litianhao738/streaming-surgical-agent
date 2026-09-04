"""Training-only target construction and deterministic Gate collection sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from surgical_agent.data.dataset import ResolvedSample
from surgical_agent.data.schemas import DatasetSplit, FrameSupervisionTarget
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target

T = TypeVar("T")


def gate_training_target(sample: ResolvedSample) -> FrameSupervisionTarget:
    """Return the richest conservative Training target without inventing negatives."""

    if sample.inference.source_split is not DatasetSplit.TRAINING:
        raise ValueError("formal Gate targets are Training-only")
    evaluation = sample.evaluation
    if evaluation is not None and evaluation.instance_supervision_available:
        source = sample.provenance.annotation_source
        if source is None:
            raise ValueError("instance supervision has no annotation provenance")
        return aggregate_evaluation_target(evaluation, source=source)
    if sample.frame_supervision is None:
        raise ValueError("Training observation has no usable supervision")
    return sample.frame_supervision


def deterministic_timeline_sample(
    values: Sequence[T], max_samples: int | None
) -> tuple[T, ...]:
    """Select an ordered, endpoint-covering sample instead of a prefix."""

    items = tuple(values)
    if max_samples is None or max_samples >= len(items):
        return items
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer or None")
    if max_samples == 1:
        return (items[(len(items) - 1) // 2],)
    last = len(items) - 1
    indices = tuple(index * last // (max_samples - 1) for index in range(max_samples))
    return tuple(items[index] for index in indices)


__all__ = ["deterministic_timeline_sample", "gate_training_target"]
