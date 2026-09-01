"""Tests for the only gold-free context admitted to joint perception."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO

import pytest
import torch
from PIL import Image

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
    causal_frame_ids: tuple[int, ...] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="unit-run",
        video_id=video_id,
        frame_id=frame_id,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(frame_id,) if causal_frame_ids is None else causal_frame_ids,
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


class ChangingSnapshot(Mapping[str, object]):
    """A mapping that exposes a forbidden key only on a later traversal."""

    def __init__(self) -> None:
        self.traversal_count = 0

    def __iter__(self) -> Iterator[str]:
        self.traversal_count += 1
        if self.traversal_count == 1:
            return iter(("safe",))
        return iter(("ground_truth",))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        if key == "safe":
            return "safe value"
        if key == "ground_truth":
            return "injected value"
        raise KeyError(key)


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


def test_builder_rejects_prior_with_future_causal_history() -> None:
    with pytest.raises(PerceptionContextError, match="strictly earlier"):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=prediction_record(
                frame_id=11,
                causal_frame_ids=(10, 12, 11),
            ),
        )


def test_builder_rejects_prior_with_unordered_causal_history() -> None:
    with pytest.raises(PerceptionContextError, match="unique and increasing"):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=prediction_record(
                frame_id=11,
                causal_frame_ids=(10, 9, 11),
            ),
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
    with pytest.raises(ValueError, match="1..6"):
        CausalPerceptionContextBuilder(max_frames=7)


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


@pytest.mark.parametrize("snapshot_name", ["workflow_snapshot", "memory_snapshot"])
def test_builder_rejects_non_mapping_snapshot_before_pair_conversion(
    snapshot_name: str,
) -> None:
    snapshots: dict[str, object] = {
        "workflow_snapshot": {},
        "memory_snapshot": {},
    }
    snapshots[snapshot_name] = [("ground_truth", "must not enter context")]

    with pytest.raises(PerceptionContextError, match="must be a mapping"):
        builder().build(
            sample=sample(),
            frames=three_frames(),
            workflow_snapshot=snapshots["workflow_snapshot"],  # type: ignore[arg-type]
            memory_snapshot=snapshots["memory_snapshot"],  # type: ignore[arg-type]
            prior_finalized_prediction=None,
        )


def test_builder_validates_the_exact_frozen_snapshot_representation() -> None:
    changing_snapshot = ChangingSnapshot()

    context = builder().build(
        sample=sample(),
        frames=three_frames(),
        workflow_snapshot=changing_snapshot,
        memory_snapshot={},
        prior_finalized_prediction=None,
    )

    assert context.workflow_snapshot == {"safe": "safe value"}
    assert changing_snapshot.traversal_count == 1


def test_builder_freezes_snapshot_values_against_source_and_consumer_mutation() -> None:
    workflow_snapshot = {"nested": {"phase": 1}}
    memory_snapshot = {"events": ["event-1"]}
    context = builder().build(
        sample=sample(),
        frames=three_frames(),
        workflow_snapshot=workflow_snapshot,
        memory_snapshot=memory_snapshot,
        prior_finalized_prediction=None,
    )

    workflow_snapshot["ground_truth"] = "late mutation"
    workflow_snapshot["nested"]["phase"] = 2
    memory_snapshot["events"].append("event-2")

    assert context.workflow_snapshot == {"nested": {"phase": 1}}
    assert context.memory_snapshot == {"events": ("event-1",)}
    with pytest.raises(TypeError):
        context.workflow_snapshot["ground_truth"] = "consumer mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        context.workflow_snapshot["nested"]["phase"] = 2  # type: ignore[index]


def test_builder_snapshots_frames_before_encoding_images() -> None:
    frames = three_frames()
    context = builder().build(
        sample=sample(),
        frames=frames,
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )

    frames.zero_()

    assert context.frames.tolist() == three_frames().tolist()
    with Image.open(BytesIO(context.images[0].content)) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
