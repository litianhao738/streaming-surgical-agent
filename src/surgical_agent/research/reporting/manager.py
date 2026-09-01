"""Boundary-aware event report manager."""

from __future__ import annotations

from surgical_agent.research.reporting.contracts import (
    REPORT_MODES,
    EventReportGenerator,
    ReportRecord,
)
from surgical_agent.research.reporting.template import TemplateReportGenerator
from surgical_agent.systems.pipeline import FinalizedEvent

_ELIGIBLE_STATUSES = frozenset({"Verified", "Accepted", "Pending"})
_STATUS_ORDER = ("Verified", "Accepted", "Pending")


class EventReportManager:
    """Buffer eligible finalized events and flush only at allowed boundaries."""

    def __init__(
        self,
        *,
        window_size: int = 30,
        mode: str = "template_report",
        generator: EventReportGenerator | None = None,
    ) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if mode not in REPORT_MODES:
            raise ValueError("unsupported event report mode")
        if mode == "llm_report" and generator is None:
            raise ValueError("llm_report requires an injected generator")
        if mode == "template_report" and generator is not None:
            raise ValueError("template_report uses the deterministic local generator")
        self.window_size = window_size
        self.mode = mode
        self.generator = generator or TemplateReportGenerator()
        self._buffer: list[FinalizedEvent] = []
        self._records: list[ReportRecord] = []
        self._last_frame_by_video: dict[str, int] = {}
        self._current_video_id: str | None = None
        self._current_phase_id: int | None = None
        self._window_observations = 0
        self._finalized = False

    @property
    def records(self) -> tuple[ReportRecord, ...]:
        return tuple(self._records)

    def observe(self, event: FinalizedEvent) -> tuple[ReportRecord, ...]:
        if self._finalized:
            raise RuntimeError("event report manager was already finalized")
        if not isinstance(event, FinalizedEvent):
            raise TypeError("event report manager accepts only FinalizedEvent values")
        prior_frame = self._last_frame_by_video.get(event.video_id)
        if prior_frame is not None and event.frame_id <= prior_frame:
            raise ValueError("frame IDs must be strictly increasing within each video")
        emitted: list[ReportRecord] = []
        if self._current_video_id is not None and event.video_id != self._current_video_id:
            emitted.extend(self._flush(("video_end",)))
            self._window_observations = 0
            self._current_phase_id = None
        elif (
            self._current_phase_id is not None
            and event.phase_id != self._current_phase_id
        ):
            emitted.extend(self._flush(("phase_change", "event_segment_end")))
            self._window_observations = 0

        self._current_video_id = event.video_id
        self._current_phase_id = event.phase_id
        self._last_frame_by_video[event.video_id] = event.frame_id
        self._window_observations += 1
        if event.final_status in _ELIGIBLE_STATUSES:
            self._buffer.append(event)
        if self._window_observations == self.window_size:
            emitted.extend(self._flush(("fixed_window",)))
            self._window_observations = 0
        return tuple(emitted)

    def finalize(self) -> tuple[ReportRecord, ...]:
        if self._finalized:
            raise RuntimeError("event report manager was already finalized")
        emitted = self._flush(("video_end",))
        self._finalized = True
        return emitted

    def _flush(self, trigger_reasons: tuple[str, ...]) -> tuple[ReportRecord, ...]:
        if not self._buffer:
            return ()
        events = tuple(self._buffer)
        self._buffer.clear()
        text = self.generator.generate(events, trigger_reasons=trigger_reasons)
        statuses = tuple(
            status
            for status in _STATUS_ORDER
            if any(event.final_status == status for event in events)
        )
        report = ReportRecord.create(
            video_id=events[0].video_id,
            start_frame_id=events[0].frame_id,
            end_frame_id=events[-1].frame_id,
            phase_ids=tuple(dict.fromkeys(event.phase_id for event in events)),
            trigger_reasons=trigger_reasons,
            included_final_statuses=statuses,
            mode=self.mode,
            text=text,
        )
        self._records.append(report)
        return (report,)


__all__ = ["EventReportManager"]
