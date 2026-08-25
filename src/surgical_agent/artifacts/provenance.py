"""Sanitized runtime provenance for reproducible local smoke runs."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeProvenance:
    python_version: str
    platform: str
    torch_version: str
    numpy_version: str
    device: str
    dtype: str
    seed: int
    deterministic: bool


def capture_runtime_provenance(
    *,
    device: torch.device,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    """Return runtime versions without environment variables or credentials."""

    return asdict(
        RuntimeProvenance(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            device=str(device),
            dtype=str(torch.get_default_dtype()),
            seed=seed,
            deterministic=deterministic,
        )
    )
