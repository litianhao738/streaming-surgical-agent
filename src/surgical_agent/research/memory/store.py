"""Bounded finalized-only per-video event memory."""

from __future__ import annotations

from collections.abc import Mapping

from surgical_agent.perception.context_builder import freeze_snapshot
from surgical_agent.research.reliability.state import validate_status_memory_action
from surgical_agent.systems.pipeline import FinalizedEvent, PipelineContractError


class BoundedEventMemory:
    """Store only committed events and expose immutable prior snapshots."""

    def __init__(self, *, max_events_per_video: int = 64) -> None:
        if (
            not isinstance(max_events_per_video, int)
            or isinstance(max_events_per_video, bool)
            or max_events_per_video <= 0
        ):
            raise ValueError("max_events_per_video must be a positive integer")
        self.max_events_per_video = max_events_per_video
        self.video_id: str | None = None
        self._reliable_events: list[FinalizedEvent] = []
        self._short_term_events: list[FinalizedEvent] = []
        self._pending_events: list[FinalizedEvent] = []
        self._last_seen_frame_id: int | None = None

    def reset(self, video_id: str) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must not be empty")
        self.video_id = video_id
        self._reliable_events = []
        self._short_term_events = []
        self._pending_events = []
        self._last_seen_frame_id = None

    def snapshot(self) -> Mapping[str, object]:
        reliable_events = tuple(
            self._event_payload(event) for event in self._reliable_events
        )
        short_term_events = tuple(
            self._event_payload(event) for event in self._short_term_events
        )
        pending_events = tuple(
            self._event_payload(event) for event in self._pending_events
        )
        events = tuple(
            self._event_payload(event)
            for event in sorted(
                (*self._reliable_events, *self._short_term_events),
                key=lambda item: item.frame_id,
            )
        )
        return freeze_snapshot(
            {
                "component": "event_memory",
                "status": "FINALIZED_ONLY",
                "video_id": self.video_id,
                "max_events_per_video": self.max_events_per_video,
                "source_max_frame_id": self._last_seen_frame_id,
                "events": events,
                "reliable_events": reliable_events,
                "short_term_events": short_term_events,
                "pending_events": pending_events,
            },
            name="event_memory",
        )

    def update(self, event: FinalizedEvent) -> None:
        if not isinstance(event, FinalizedEvent):
            raise TypeError("event memory accepts only FinalizedEvent values")
        if self.video_id != event.video_id:
            raise PipelineContractError("event memory commit crossed a video boundary")
        try:
            validate_status_memory_action(event.final_status, event.memory_action)
        except ValueError:
            raise PipelineContractError(
                "event memory rejected an inconsistent reliability state"
            ) from None
        if (
            self._last_seen_frame_id is not None
            and event.frame_id <= self._last_seen_frame_id
        ):
            raise PipelineContractError(
                "event memory requires strictly increasing frame IDs"
            )
        self._last_seen_frame_id = event.frame_id
        if event.final_status in {"Candidate", "Rejected"}:
            return
        collection: list[FinalizedEvent] | None = None
        if event.memory_action == "WRITE_RELIABLE":
            collection = self._reliable_events
        elif event.memory_action == "WRITE_SHORT_TERM":
            collection = self._short_term_events
        elif event.memory_action == "BUFFER_PENDING":
            collection = self._pending_events
        if collection is not None:
            collection.append(event)
            if len(collection) > self.max_events_per_video:
                del collection[: len(collection) - self.max_events_per_video]

    @staticmethod
    def _event_payload(event: FinalizedEvent) -> dict[str, object]:
        return {
            "video_id": event.video_id,
            "frame_id": event.frame_id,
            "instrument_ids": event.instrument_ids,
            "verb_ids": event.verb_ids,
            "target_ids": event.target_ids,
            "ivt_ids": event.triplet_ids,
            "phase_id": event.phase_id,
            "backend": event.backend,
            "gate_action": event.gate_action,
            "verification_status": event.verification_status,
            "score_semantics": event.score_semantics,
            "initial_state": event.initial_state,
            "gate_reasons": event.gate_reasons,
            "flagged_fields": event.flagged_fields,
            "repaired_fields": event.repaired_fields,
            "final_status": event.final_status,
            "memory_action": event.memory_action,
        }
