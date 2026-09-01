"""Contracts for deterministic event-level report artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from surgical_agent.systems.pipeline import FinalizedEvent

REPORT_SCHEMA_VERSION = "event_report_v1"
REPORT_MODES = frozenset({"template_report", "llm_report"})
REPORTABLE_STATUSES = frozenset({"Verified", "Accepted", "Pending"})
_CLAUSE_PREFIXES = {
    "Verified": re.compile(r"^confirmed\b", flags=re.IGNORECASE),
    "Accepted": re.compile(
        r"^high-confidence observation\b",
        flags=re.IGNORECASE,
    ),
    "Pending": re.compile(
        r"^(?:may|possible|unresolved)\b",
        flags=re.IGNORECASE,
    ),
}
_CERTAINTY_LANGUAGE = re.compile(
    r"\b(?:confirmed|high-confidence observation)\b",
    flags=re.IGNORECASE,
)


def _validate_status_aware_clauses(text: str, statuses: tuple[str, ...]) -> None:
    clauses = tuple(
        clause.strip()
        for clause in re.split(r"[.;]|\r?\n", text)
        if clause.strip()
    )
    classified: list[str] = []
    for clause in clauses:
        normalized = re.sub(
            r"^(?:[-*]+|\d+[.)])\s*",
            "",
            clause,
        )
        matches = tuple(
            status
            for status, prefix in _CLAUSE_PREFIXES.items()
            if prefix.search(normalized)
        )
        if len(matches) != 1:
            raise ValueError(
                "report text requires independent status-aware clauses with "
                "explicit uncertainty language"
            )
        status = matches[0]
        if status == "Pending" and _CERTAINTY_LANGUAGE.search(normalized):
            raise ValueError(
                "Pending uncertainty language cannot contain certainty claims"
            )
        classified.append(status)
    if set(classified) != set(statuses):
        raise ValueError(
            "report text requires independent status-aware clauses with "
            "explicit uncertainty language"
        )


class EventReportGenerator(Protocol):
    """Generate prose once for a completed event segment."""

    def generate(
        self,
        events: tuple[FinalizedEvent, ...],
        *,
        trigger_reasons: tuple[str, ...],
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ReportRecord:
    """Self-identifying, status-aware event report."""

    report_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int
    phase_ids: tuple[int, ...]
    trigger_reasons: tuple[str, ...]
    included_final_statuses: tuple[str, ...]
    mode: str
    text: str
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in REPORT_MODES:
            raise ValueError("unsupported event report mode")
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("event report frame range is invalid")
        if not self.video_id or not self.text:
            raise ValueError("event report video_id and text must not be empty")
        statuses = tuple(self.included_final_statuses)
        if (
            not statuses
            or len(set(statuses)) != len(statuses)
            or any(status not in REPORTABLE_STATUSES for status in statuses)
        ):
            raise ValueError("event reports require unique eligible final statuses")
        _validate_status_aware_clauses(self.text, statuses)
        expected = self.content_sha256(
            video_id=self.video_id,
            start_frame_id=self.start_frame_id,
            end_frame_id=self.end_frame_id,
            phase_ids=self.phase_ids,
            trigger_reasons=self.trigger_reasons,
            included_final_statuses=self.included_final_statuses,
            mode=self.mode,
            text=self.text,
            schema_version=self.schema_version,
        )
        if self.report_id != expected:
            raise ValueError("report_id does not match canonical report content")

    @classmethod
    def create(
        cls,
        *,
        video_id: str,
        start_frame_id: int,
        end_frame_id: int,
        phase_ids: tuple[int, ...],
        trigger_reasons: tuple[str, ...],
        included_final_statuses: tuple[str, ...],
        mode: str,
        text: str,
        schema_version: str = REPORT_SCHEMA_VERSION,
    ) -> ReportRecord:
        report_id = cls.content_sha256(
            video_id=video_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            phase_ids=phase_ids,
            trigger_reasons=trigger_reasons,
            included_final_statuses=included_final_statuses,
            mode=mode,
            text=text,
            schema_version=schema_version,
        )
        return cls(
            report_id=report_id,
            video_id=video_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            phase_ids=phase_ids,
            trigger_reasons=trigger_reasons,
            included_final_statuses=included_final_statuses,
            mode=mode,
            text=text,
            schema_version=schema_version,
        )

    @staticmethod
    def content_sha256(**content: object) -> str:
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "video_id": self.video_id,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "phase_ids": list(self.phase_ids),
            "trigger_reasons": list(self.trigger_reasons),
            "included_final_statuses": list(self.included_final_statuses),
            "mode": self.mode,
            "text": self.text,
            "schema_version": self.schema_version,
        }


__all__ = [
    "REPORTABLE_STATUSES",
    "REPORT_MODES",
    "REPORT_SCHEMA_VERSION",
    "EventReportGenerator",
    "ReportRecord",
]
