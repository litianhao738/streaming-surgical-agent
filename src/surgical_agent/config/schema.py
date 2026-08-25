"""Small typed configuration contracts used during repository bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location and read-only policy.

    A missing root is intentional until a CLI option or resolved config supplies
    it. Production source code must not embed a workstation-specific path.
    """

    root: Path | None = None
    read_only: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Bootstrap-level experiment identity and safety switches."""

    name: str
    seed: int = 42
    research_enabled: bool = False


@dataclass(frozen=True)
class ApiConfig:
    """P0 API identity; credentials are never represented in config."""

    enabled: bool = False
    provider: str | None = None
    model_identifier: str | None = None
    cache_required: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    """Independent compact workflow-state switch."""

    enabled: bool = True
    finalized_only: bool = True


@dataclass(frozen=True)
class EventMemoryConfig:
    """Independent episodic-memory switch and per-video bound."""

    enabled: bool = True
    finalized_only: bool = True
    max_events_per_video: int | None = None
