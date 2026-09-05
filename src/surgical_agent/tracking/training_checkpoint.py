"""Atomic end-of-epoch Tracker training state for AutoDL resume."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from surgical_agent.tracking.detector import tracker_model_contract

TRACKER_TRAINING_STATE_VERSION = "tracker_training_state_v2"
LEGACY_TRACKER_TRAINING_STATE_VERSION = "tracker_training_state_v1"
_LEGACY_IDENTITY_FIELDS = (
    "tracker_config_sha256", "training_video_ids", "excluded_video_ids",
    "mode", "initial_weights", "max_train_batches",
)


def save_tracker_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    epoch_completed: int,
    losses: list[float],
    identity: Mapping[str, object],
) -> Path:
    if epoch_completed <= 0:
        raise ValueError("epoch_completed must be positive")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRACKER_TRAINING_STATE_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "epoch_completed": epoch_completed,
        "losses": list(losses),
        "identity": dict(identity),
        "model_contract": tracker_model_contract(model),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_tracker_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    expected_identity: Mapping[str, object],
    map_location: str | torch.device,
    allow_legacy_identity: bool = False,
    restore_cuda_rng: bool = True,
) -> tuple[int, list[float]]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Tracker training state must be a mapping")
    version = payload.get("schema_version")
    if version not in {TRACKER_TRAINING_STATE_VERSION, LEGACY_TRACKER_TRAINING_STATE_VERSION}:
        raise ValueError("unsupported Tracker training-state schema")
    expected = dict(expected_identity)
    if version == LEGACY_TRACKER_TRAINING_STATE_VERSION:
        if not allow_legacy_identity or expected.get("bbox_policy") != "legacy_strict_v1":
            raise ValueError("legacy resume requires the explicit legacy_strict_v1 option")
        expected = {name: expected[name] for name in _LEGACY_IDENTITY_FIELDS}
    elif payload.get("model_contract") != tracker_model_contract(model):
        raise ValueError("Tracker resume model contract does not match this training job")
    if payload.get("identity") != expected:
        raise ValueError("Tracker resume identity does not match this training job")
    epoch_completed = payload.get("epoch_completed")
    losses = payload.get("losses")
    rng = payload.get("rng")
    if (
        not isinstance(epoch_completed, int)
        or isinstance(epoch_completed, bool)
        or epoch_completed <= 0
        or not isinstance(losses, list)
        or not isinstance(rng, Mapping)
    ):
        raise ValueError("Tracker training state is incomplete")
    model.load_state_dict(payload["model_state"], strict=True)  # type: ignore[arg-type]
    optimizer.load_state_dict(payload["optimizer_state"])  # type: ignore[arg-type]
    loader_generator.set_state(payload["loader_generator_state"].cpu())  # type: ignore[union-attr]
    random.setstate(rng["python"])  # type: ignore[arg-type]
    np.random.set_state(rng["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(rng["torch_cpu"].cpu())  # type: ignore[union-attr]
    cuda_state = rng["torch_cuda"]
    if cuda_state is not None and restore_cuda_rng:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA resume state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    normalized_losses = [float(value) for value in losses]
    if not normalized_losses:
        raise ValueError("Tracker resume state contains no optimizer losses")
    return epoch_completed, normalized_losses


__all__ = [
    "TRACKER_TRAINING_STATE_VERSION",
    "load_tracker_training_state",
    "save_tracker_training_state",
]
