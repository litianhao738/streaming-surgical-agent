"""Compact finalized-only workflow history for causal context."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

from surgical_agent.perception.context_builder import freeze_snapshot
from surgical_agent.systems.pipeline import FinalizedEvent, PipelineContractError


class FinalizedWorkflowStateStore:
    """Track recent committed phases without similarity retrieval or GT."""

    def __init__(self, *, max_recent_phases: int = 16) -> None:
        if (
            not isinstance(max_recent_phases, int)
            or isinstance(max_recent_phases, bool)
            or max_recent_phases <= 0
        ):
            raise ValueError("max_recent_phases must be a positive integer")
        self.max_recent_phases = max_recent_phases
        self.video_id: str | None = None
        self._frame_ids: list[int] = []
        self._phases: list[int] = []
        self._last_seen_frame_id: int | None = None

    def reset(self, video_id: str) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must not be empty")
        self.video_id = video_id
        self._frame_ids = []
        self._phases = []
        self._last_seen_frame_id = None

    def snapshot(self) -> Mapping[str, object]:
        phases = tuple(str(phase_id) for phase_id in self._phases)
        transitions = tuple(pairwise(phases))
        stability = None
        if phases:
            stability = phases.count(phases[-1]) / len(phases)
        return freeze_snapshot(
            {
                "component": "workflow_state",
                "status": "FINALIZED_ONLY",
                "video_id": self.video_id,
                "recent_finalized_phases": phases,
                "phase_stability": stability,
                "observed_transitions": transitions,
                "source_max_frame_id": self._last_seen_frame_id,
            },
            name="workflow",
        )

    def update(self, event: FinalizedEvent) -> None:
        if not isinstance(event, FinalizedEvent):
            raise TypeError("workflow state accepts only FinalizedEvent values")
        if self.video_id != event.video_id:
            raise PipelineContractError("workflow commit crossed a video boundary")
        if (
            self._last_seen_frame_id is not None
            and event.frame_id <= self._last_seen_frame_id
        ):
            raise PipelineContractError(
                "workflow state requires strictly increasing frame IDs"
            )
        self._last_seen_frame_id = event.frame_id
        if event.final_status not in {"Accepted", "Verified"}:
            return
        self._frame_ids.append(event.frame_id)
        self._phases.append(event.phase_id)
        if len(self._phases) > self.max_recent_phases:
            excess = len(self._phases) - self.max_recent_phases
            del self._phases[:excess]
            del self._frame_ids[:excess]
