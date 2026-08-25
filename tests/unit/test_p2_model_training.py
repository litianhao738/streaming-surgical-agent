"""P2 model, mask normalization, trainer, and checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import torch

from surgical_agent.data.schemas import (
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.models.baseline import (
    TASK_CLASS_COUNTS,
    LocalSmokeModel,
    decode_frame_logits,
)
from surgical_agent.training.checkpoint import load_checkpoint, save_checkpoint
from surgical_agent.training.losses import masked_multitask_loss


def _target(
    video_id: str,
    frame_id: int,
    *,
    complete: bool,
) -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id=video_id,
        frame_id=frame_id,
        instrument_ids=(0,) if complete else (),
        verb_ids=(1,) if complete else (),
        target_ids=(2,) if complete else (),
        triplet_ids=(3,) if complete else (),
        phase_id=4,
        mask=FrameTaskMask(complete, complete, complete, complete, True),
        source_granularity="frame_multilabel" if complete else "phase_only",
        source="synthetic-test",
    )


def test_local_smoke_model_has_fixed_shapes_and_supports_causal_tensor() -> None:
    model = LocalSmokeModel(pretrained=False)
    current = torch.rand(2, 3, 64, 64)
    causal = torch.rand(2, 3, 3, 64, 64)

    current_logits = model(current)
    causal_logits = model(causal)

    for task, count in TASK_CLASS_COUNTS.items():
        assert getattr(current_logits, task).shape == (2, count)
        assert getattr(causal_logits, task).shape == (2, count)
    decoded = decode_frame_logits(current_logits)
    assert len(decoded) == 2
    assert all(item.granularity == "frame_multilabel" for item in decoded)


def test_pretrained_weights_are_forbidden() -> None:
    try:
        LocalSmokeModel(pretrained=True)
    except ValueError as error:
        assert "forbids pretrained" in str(error)
    else:
        raise AssertionError("P2 accepted pretrained weights")


def test_masked_loss_normalizes_each_task_over_valid_samples() -> None:
    model = LocalSmokeModel()
    logits = model(torch.rand(2, 3, 48, 48))
    result = masked_multitask_loss(
        logits,
        (
            _target("VID02", 1, complete=False),
            _target("VID31", 2, complete=True),
        ),
    )

    assert result.valid_counts == {
        "instrument": 1,
        "verb": 1,
        "target": 1,
        "ivt": 1,
        "phase": 2,
    }
    assert torch.isfinite(result.total)
    result.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_checkpoint_round_trip_restores_exact_parameters(tmp_path: Path) -> None:
    model = LocalSmokeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = model(torch.rand(1, 3, 32, 32)).phase.sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=1,
        metadata={"purpose": "unit-test"},
    )

    restored = LocalSmokeModel()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    state = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
    )

    assert state.step == 1
    assert state.metadata == {"purpose": "unit-test"}
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
