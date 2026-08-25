"""Explicit CPU/CUDA device selection with fail-closed overrides."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device without silently downgrading an explicit CUDA request."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested!r}")
    return torch.device(normalized)
