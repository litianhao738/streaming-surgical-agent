from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from surgical_agent.tracking.training_checkpoint import (
    load_tracker_training_state,
    save_tracker_training_state,
)


def _objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    generator = torch.Generator().manual_seed(17)
    return model, optimizer, generator


def test_tracker_training_state_restores_model_optimizer_and_epoch(tmp_path) -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    model, optimizer, generator = _objects()
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    identity = {"fold": 1, "videos": ["VID01", "VID02"]}
    path = save_tracker_training_state(
        tmp_path / "training_state.pt",
        model=model,
        optimizer=optimizer,
        loader_generator=generator,
        epoch_completed=2,
        losses=[1.2, 0.8],
        identity=identity,
    )
    restored, restored_optimizer, restored_generator = _objects()

    epoch, losses = load_tracker_training_state(
        path,
        model=restored,
        optimizer=restored_optimizer,
        loader_generator=restored_generator,
        expected_identity=identity,
        map_location="cpu",
    )

    assert epoch == 2
    assert losses == [1.2, 0.8]
    assert all(
        torch.equal(value, expected[name])
        for name, value in restored.state_dict().items()
    )
    assert restored_optimizer.state_dict()["state"]


def test_tracker_resume_rejects_a_different_fold_identity(tmp_path) -> None:
    model, optimizer, generator = _objects()
    path = save_tracker_training_state(
        tmp_path / "training_state.pt",
        model=model,
        optimizer=optimizer,
        loader_generator=generator,
        epoch_completed=1,
        losses=[1.0],
        identity={"fold": 1},
    )
    restored, restored_optimizer, restored_generator = _objects()

    with pytest.raises(ValueError, match="identity"):
        load_tracker_training_state(
            path,
            model=restored,
            optimizer=restored_optimizer,
            loader_generator=restored_generator,
            expected_identity={"fold": 2},
            map_location="cpu",
        )


def test_resume_rejects_changed_supervision_fingerprint(tmp_path) -> None:
    model, optimizer, generator = _objects()
    identity = {"bbox_policy": "clip_to_frame_v2", "data_sha256": "original"}
    path = save_tracker_training_state(
        tmp_path / "training_state.pt", model=model, optimizer=optimizer,
        loader_generator=generator, epoch_completed=1, losses=[1.0], identity=identity,
    )
    with pytest.raises(ValueError, match="identity"):
        load_tracker_training_state(
            path, model=model, optimizer=optimizer, loader_generator=generator,
            expected_identity={**identity, "data_sha256": "changed"}, map_location="cpu",
        )


def test_legacy_state_cannot_be_mixed_into_clipped_training(tmp_path) -> None:
    model, optimizer, generator = _objects()
    path = save_tracker_training_state(
        tmp_path / "training_state.pt", model=model, optimizer=optimizer,
        loader_generator=generator, epoch_completed=1, losses=[1.0], identity={},
    )
    payload = torch.load(path, weights_only=False)
    payload["schema_version"] = "tracker_training_state_v1"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="legacy_strict_v1"):
        load_tracker_training_state(
            path, model=model, optimizer=optimizer, loader_generator=generator,
            expected_identity={"bbox_policy": "clip_to_frame_v2"}, map_location="cpu",
            allow_legacy_identity=True,
        )


def test_resume_rejects_changed_trainable_parameter_identity(tmp_path) -> None:
    model, optimizer, generator = _objects()
    path = save_tracker_training_state(
        tmp_path / "training_state.pt", model=model, optimizer=optimizer,
        loader_generator=generator, epoch_completed=1, losses=[1.0], identity={},
    )
    model.bias.requires_grad_(False)
    with pytest.raises(ValueError, match="model contract"):
        load_tracker_training_state(
            path, model=model, optimizer=optimizer, loader_generator=generator,
            expected_identity={}, map_location="cpu",
        )


def test_coco_detector_next_step_matches_after_state_restore(tmp_path) -> None:
    """A real detector update after resume must match uninterrupted optimization."""
    from dataclasses import replace

    from surgical_agent.tracking.config import load_tracker_training_config
    from surgical_agent.tracking.detector import build_instrument_detector

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(13)
        config = replace(
            load_tracker_training_config("configs/tracker/fasterrcnn_mobilenet_v3.yaml"),
            min_size=64, max_size=64, num_workers=0,
        )
        model = build_instrument_detector(config, use_pretrained=False)
        optimizer = torch.optim.SGD(
            [value for value in model.parameters() if value.requires_grad],
            lr=0.001, momentum=0.9,
        )
        generator = torch.Generator().manual_seed(15)
        image = torch.rand(3, 64, 64)
        target = {"boxes": torch.tensor([[8., 8., 48., 48.]]), "labels": torch.tensor([1])}

        def step(selected_model, selected_optimizer):
            selected_model.train()
            selected_optimizer.zero_grad(set_to_none=True)
            loss = sum(selected_model([image], [target]).values())
            assert torch.isfinite(loss)
            loss.backward()
            selected_optimizer.step()
            return loss.detach().clone()

        first_loss = step(model, optimizer)
        identity = {"bbox_policy": "clip_to_frame_v2", "training_data": "synthetic_fixed_v1"}
        path = save_tracker_training_state(
            tmp_path / "training_state.pt", model=model, optimizer=optimizer,
            loader_generator=generator, epoch_completed=1,
            losses=[float(first_loss)], identity=identity,
        )
        second_loss = step(model, optimizer)
        restored = build_instrument_detector(config, use_pretrained=False)
        restored_optimizer = torch.optim.SGD(
            [value for value in restored.parameters() if value.requires_grad],
            lr=0.001, momentum=0.9,
        )
        load_tracker_training_state(
            path, model=restored, optimizer=restored_optimizer,
            loader_generator=torch.Generator(), expected_identity=identity,
            map_location="cpu", restore_cuda_rng=False,
        )
        assert torch.equal(step(restored, restored_optimizer), second_loss)
        assert all(
            torch.equal(value, restored.state_dict()[name])
            for name, value in model.state_dict().items()
        )
    finally:
        torch.set_num_threads(previous_threads)
