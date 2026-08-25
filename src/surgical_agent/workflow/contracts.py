"""Workflow state boundary kept independent from episodic EventMemory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class WorkflowState:
    """Compact finalized history available before the current commit."""

    video_id: str
    recent_finalized_phases: tuple[str, ...] = ()
    phase_stability: float | None = None
    observed_transitions: tuple[tuple[str, str], ...] = ()
    transition_tendency: Mapping[str, float] = field(default_factory=dict)
    temporal_state: Mapping[str, float | int | str] = field(default_factory=dict)
    source_max_frame_id: int | None = None


class FinalizedEventView(Protocol):
    """Minimum finalized-event fields consumed by workflow state."""

    video_id: str
    frame_id: int


class WorkflowStateStore(Protocol):
    """Stateful per-video service; similarity retrieval is intentionally absent."""

    def reset(self, video_id: str) -> None:
        """Clear all state at a video boundary."""

    def snapshot(self) -> WorkflowState:
        """Return the compact state built only from prior finalized events."""

    def update(self, event: FinalizedEventView) -> None:
        """Consume one finalized event after current-state finalization."""
