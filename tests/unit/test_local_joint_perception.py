"""Focused contracts for the trainable five-task local perception path."""

from __future__ import annotations

import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from surgical_agent.models.perception import (
    CausalJointPerceptionModel,
    JointPerceptionModelConfig,
)
from surgical_agent.training.perception_data import preprocess_causal_frames
from surgical_agent.training.perception_trainer import weighted_joint_loss


def _target() -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id="VID00",
        frame_id=5,
        instrument_ids=(0, 2),
        verb_ids=(1,),
        target_ids=(3,),
        triplet_ids=(4,),
        phase_id=2,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit_test",
    )


def test_causal_joint_model_outputs_all_five_task_shapes_and_loss() -> None:
    model = CausalJointPerceptionModel(
        JointPerceptionModelConfig(
            architecture="mobilenet_v3_small",
            visual_feature_dim=32,
            temporal_hidden_dim=24,
            dropout=0.0,
            pretrained=False,
        )
    )
    logits = model(torch.rand(2, 3, 3, 64, 64))

    for task, class_count in TASK_CLASS_COUNTS.items():
        assert getattr(logits, task).shape == (2, class_count)
    result = weighted_joint_loss(
        logits,
        (_target(), _target()),
        class_weights={
            task: torch.ones(class_count)
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
    )
    assert torch.isfinite(result.total)
    result.total.backward()


def test_preprocess_left_pads_short_causal_windows() -> None:
    frames = torch.rand(2, 3, 40, 50)

    processed = preprocess_causal_frames(
        frames,
        image_size=32,
        window_size=6,
    )

    assert processed.shape == (6, 3, 32, 32)
    assert processed.dtype == torch.float32
    assert torch.equal(processed[0], processed[4])
    assert not torch.equal(processed[4], processed[5])
