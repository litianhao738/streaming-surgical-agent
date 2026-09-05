"""Deterministic phase/major-IVT event aggregation and reliability-aware reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from surgical_agent.research.outcome import ExecutionOutcome, FinalOutcome
from surgical_agent.runtime.finalization import FinalizationRecord


@dataclass(frozen=True)
class TemporalEvent:
    event_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int
    phase_id: int | None
    major_ivt_ids: tuple[int, ...]
    frame_count: int
    state_counts: dict[str, int]
    task_state_counts: dict[str, dict[str, int]]

    @property
    def reliability(self) -> str:
        semantic_counts = tuple(
            self.task_state_counts.get(task, {}) for task in ("phase", "ivt")
        )
        if self.state_counts.get("Pending", 0) or any(
            counts.get("Pending", 0) for counts in semantic_counts
        ):
            return "UNCERTAIN"
        if semantic_counts and all(
            counts.get("Verified", 0) == self.frame_count
            for counts in semantic_counts
        ):
            return "DEFINITE"
        return "OBSERVED"


class TemporalEventAggregator:
    """Group adjacent semantic outputs by exact phase and major-IVT signature."""

    def aggregate(
        self,
        records: tuple[FinalizationRecord, ...],
    ) -> tuple[TemporalEvent, ...]:
        usable = tuple(
            record
            for record in sorted(records, key=lambda item: item.observation)
            if isinstance(record.outcome, FinalOutcome)
            and record.outcome.state in {"Accepted", "Verified", "Pending"}
        )
        events: list[TemporalEvent] = []
        group: list[FinalizationRecord] = []
        key: tuple[str, int | None, tuple[int, ...]] | None = None
        for record in usable:
            outcome = record.outcome
            assert not isinstance(outcome, ExecutionOutcome)
            hypothesis = outcome.hypothesis
            current_key = (
                record.observation.video_id,
                None if hypothesis is None else hypothesis.phase_id,
                () if hypothesis is None else hypothesis.triplet_ids,
            )
            if group and current_key != key:
                events.append(self._event(group, len(events)))
                group = []
            key = current_key
            group.append(record)
        if group:
            events.append(self._event(group, len(events)))
        return tuple(events)

    @staticmethod
    def _event(records: list[FinalizationRecord], index: int) -> TemporalEvent:
        first = records[0]
        last = records[-1]
        outcome = first.outcome
        assert isinstance(outcome, FinalOutcome)
        hypothesis = outcome.hypothesis
        return TemporalEvent(
            event_id=f"{first.observation.video_id}:event-{index:05d}",
            video_id=first.observation.video_id,
            start_frame_id=first.observation.frame_id,
            end_frame_id=last.observation.frame_id,
            phase_id=None if hypothesis is None else hypothesis.phase_id,
            major_ivt_ids=() if hypothesis is None else hypothesis.triplet_ids,
            frame_count=len(records),
            state_counts=dict(
                Counter(
                    item.outcome.state
                    for item in records
                    if isinstance(item.outcome, FinalOutcome)
                )
            ),
            task_state_counts={
                task: dict(
                    Counter(
                        item.outcome.task_states[task]
                        for item in records
                        if isinstance(item.outcome, FinalOutcome)
                        and item.outcome.task_states is not None
                    )
                )
                for task in ("phase", "ivt")
            },
        )


class ReliabilityAwareTemplateReporter:
    """Render fixed wording; it never calls a model or introduces new semantics."""

    def render(self, events: tuple[TemporalEvent, ...]) -> dict[str, object]:
        rows = []
        for event in events:
            if event.reliability == "DEFINITE":
                wording = "Verified surgical event"
            elif event.reliability == "OBSERVED":
                wording = "Observed surgical event"
            else:
                wording = "Uncertain surgical event requiring review"
            rows.append(
                {
                    "event_id": event.event_id,
                    "video_id": event.video_id,
                    "start_frame_id": event.start_frame_id,
                    "end_frame_id": event.end_frame_id,
                    "phase_id": event.phase_id,
                    "major_ivt_ids": list(event.major_ivt_ids),
                    "frame_count": event.frame_count,
                    "state_counts": event.state_counts,
                    "task_state_counts": event.task_state_counts,
                    "reliability": event.reliability,
                    "text": wording,
                }
            )
        return {
            "schema_version": "reliability_aware_event_report_v2",
            "generator": "deterministic_template",
            "events": rows,
        }


__all__ = [
    "ReliabilityAwareTemplateReporter",
    "TemporalEvent",
    "TemporalEventAggregator",
]
