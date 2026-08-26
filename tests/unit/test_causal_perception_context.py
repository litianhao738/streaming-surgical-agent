"""Tests for the only gold-free context admitted to joint perception."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import (
    DatasetSplit,
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
    InferenceSample,
    LabelMask,
)
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.context_builder import (
    CausalPerceptionContextBuilder,
    PerceptionContextError,
)


def sample(
    *,
    causal_frame_ids: tuple[int, ...] = (10, 11, 12),
) -> InferenceSample:
    return InferenceSample(
        video_id="VID01",
        target_frame_id=causal_frame_ids[-1],
        causal_frame_ids=causal_frame_ids,
        media_refs=tuple(f"synthetic:{frame_id}" for frame_id in causal_frame_ids),
        source_split=DatasetSplit.TESTING,
        alignment_version="unit-test",
    )


def prediction_record(
    *,
    frame_id: int,
    video_id: str = "VID01",
) -> PredictionRecord:
    return PredictionRecord(
        run_id="unit-run",
        video_id=video_id,
        frame_id=frame_id,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(frame_id,),
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        granularity="frame_multilabel",
        backend="unit-test",
        gate_action="ACCEPT",
        verification_status="SKIPPED",
        alignment_version="unit-test",
        probabilities={
            task: (0.0,) * count for task, count in TASK_CLASS_COUNTS.items()
        },
    )


def red_tensor() -> torch.Tensor:
    return torch.tensor([[[1.0]], [[0.0]], [[0.0]]])


def green_tensor() -> torch.Tensor:
    return torch.tensor([[[0.0]], [[1.0]], [[0.0]]])


def blue_tensor() -> torch.Tensor:
    return torch.tensor([[[0.0]], [[0.0]], [[1.0]]])


def three_frames() -> torch.Tensor:
    return torch.stack((red_tensor(), green_tensor(), blue_tensor()))


def builder() -> CausalPerceptionContextBuilder:
    return CausalPerceptionContextBuilder(max_frames=3)


def evaluation_target() -> EvaluationTarget:
    return EvaluationTarget(video_id="VID01", frame_id=12, instances=())


def frame_supervision_target() -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id="VID01",
        frame_id=12,
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit-test",
    )


@dataclass(frozen=True)
class SnapshotEnvelope:
    payload: object


def test_builder_binds_three_ordered_frames_to_three_image_hashes() -> None:
    context = builder().build(
        sample=sample(causal_frame_ids=(10, 11, 12)),
        frames=three_frames(),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=prediction_record(frame_id=11),
    )

    assert [image.identifier for image in context.images] == [
        "synthetic:10",
        "synthetic:11",
        "synthetic:12",
    ]
    assert len({image.sha256 for image in context.images}) == 3
    assert context.frames.shape == (3, 3, 1, 1)


def test_builder_normalizes_a_singleton_batch_dimension() -> None:
    context = builder().build(
        sample=sample(),
        frames=three_frames().unsqueeze(0),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )

    assert context.frames.shape == (3, 3, 1, 1)


def test_builder_rejects_noncausal_prior_before_encoding() -> None:
    with pytest.raises(PerceptionContextError, match="strictly earlier"):
        builder().build(
            sample=sample(causal_frame_ids=(10, 11, 12)),
            frames=three_frames(),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=prediction_record(frame_id=12),
        )


def test_builder_rejects_prior_from_another_video() -> None:
    with pytest.raises(PerceptionContextError, match="same video"):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=prediction_record(frame_id=11, video_id="VID02"),
        )


def test_builder_rejects_frame_count_mismatch() -> None:
    with pytest.raises(PerceptionContextError, match="frame count"):
        builder().build(
            sample=sample(),
            frames=torch.stack((red_tensor(), green_tensor())),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (torch.full((3, 3, 1, 1), float("nan")), "finite"),
        (torch.full((3, 3, 1, 1), 1.1), "[0, 1]"),
        (torch.zeros((3, 1, 1, 1)), "RGB"),
    ],
)
def test_builder_rejects_invalid_image_tensors(
    frames: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(PerceptionContextError, match=message):
        builder().build(
            sample=sample(),
            frames=frames,
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )


def test_builder_rejects_windows_longer_than_three_frames() -> None:
    with pytest.raises(PerceptionContextError, match="at most 3"):
        CausalPerceptionContextBuilder(max_frames=3).build(
            sample=sample(causal_frame_ids=(9, 10, 11, 12)),
            frames=torch.zeros((4, 3, 1, 1)),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )


def test_builder_rejects_invalid_maximum_frame_count() -> None:
    with pytest.raises(ValueError, match="1..3"):
        CausalPerceptionContextBuilder(max_frames=4)


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "ground_truth",
        "evaluation_target",
        "frame_supervision",
        "label_mask",
        "frame_task_mask",
        "labels",
        "future_state",
    ],
)
def test_builder_rejects_gold_bearing_snapshot_mapping_keys(
    forbidden_key: str,
) -> None:
    with pytest.raises(PerceptionContextError, match=forbidden_key):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot={"safe": {forbidden_key: "not admitted"}},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )


@pytest.mark.parametrize(
    "gold_bearing_value",
    [
        evaluation_target(),
        frame_supervision_target(),
        LabelMask(True, True, True, True, True),
    ],
)
def test_builder_rejects_gold_bearing_dataclasses_nested_in_snapshots(
    gold_bearing_value: object,
) -> None:
    with pytest.raises(PerceptionContextError, match="GT-bearing"):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot={"safe": SnapshotEnvelope(gold_bearing_value)},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )


def test_builder_allows_predicted_target_fields_in_snapshot() -> None:
    context = builder().build(
        sample=sample(),
        frames=three_frames(),
        workflow_snapshot={"target_ids": (1, 2), "target_frame_id": 12},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )

    assert context.workflow_snapshot == {
        "target_ids": (1, 2),
        "target_frame_id": 12,
    }
