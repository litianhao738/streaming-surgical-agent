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
