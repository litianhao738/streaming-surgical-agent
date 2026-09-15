"""Contracts for deterministic event-level report artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from surgical_agent.systems.pipeline import FinalizedEvent
from surgical_agent.perception.ontology_prompt import _TASK_NAMES, _ivt_rows

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


# Frame-level GSR contracts extend the existing event-report API.

NAMES = {k: tuple(v) for k, v in _TASK_NAMES.items()}
TRIPLETS = {r[0]: tuple(NAMES[t][v] for t, v in zip(("instrument", "verb", "target"), r[1:])) for r in _ivt_rows()}
ONTOLOGY_HASH = hashlib.sha256(json.dumps([NAMES, TRIPLETS], sort_keys=True).encode()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON field")
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    endpoint: str = ""
    temperature: float = 0.0
    max_tokens: int = 400

    def __post_init__(self):
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider/model must be fixed explicitly")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("invalid token limit")
        if isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature != 0:
            raise ValueError("v1 freezes temperature at zero")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ReportInput:
    video_id: str
    frame_id: int
    source_prediction: str
    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    ivt_ids: tuple[int, ...]
    phase_id: int

    def __post_init__(self):
        if not self.video_id or type(self.frame_id) is not int or self.frame_id < 0 or self.source_prediction not in ("H0", "FULL"):
            raise ValueError("invalid report identity")
        for task, upper in (("instrument", 6), ("verb", 9), ("target", 14), ("ivt", 99)):
            values = getattr(self, task + "_ids")
            if not isinstance(values, tuple) or len(set(values)) != len(values) or any(type(v) is not int or not 0 <= v <= upper for v in values):
                raise ValueError("invalid canonical " + task)
        if type(self.phase_id) is not int or not 0 <= self.phase_id < 7:
            raise ValueError("invalid phase")

    @classmethod
    def from_prediction(cls, video_id, frame_id, source_prediction, prediction):
        # Whitelist only five heads: unrelated GT/image/routing metadata cannot enter.
        heads = {k: tuple(prediction[k]) for k in ("instrument", "verb", "target", "ivt", "phase")}
        if len(heads["phase"]) != 1:
            raise ValueError("phase must be a singleton")
        return cls(video_id, frame_id, source_prediction, *(heads[k] for k in ("instrument", "verb", "target", "ivt")), heads["phase"][0])

    def appendix(self):
        return {**{t + "_ids": list(getattr(self, t + "_ids")) for t in ("instrument", "verb", "target", "ivt")}, "phase_id": self.phase_id}

    def state(self):
        return {**{t: [NAMES[t][i] for i in getattr(self, t + "_ids")] for t in ("instrument", "verb", "target")},
                "ivt": [",".join(TRIPLETS[i]) for i in self.ivt_ids], "phase": NAMES["phase"][self.phase_id]}


@dataclass(frozen=True)
class GroundTruth:
    ivt_ids: tuple[int, ...] | None
    phase_id: int | None
    ivt_valid: bool
    phase_valid: bool

    def __post_init__(self):
        if type(self.ivt_valid) is not bool or type(self.phase_valid) is not bool:
            raise ValueError("mask must be boolean")
        if self.ivt_valid and (self.ivt_ids is None or any(type(i) is not int or i not in TRIPLETS for i in self.ivt_ids)):
            raise ValueError("invalid supervised IVT")
        if self.phase_valid and (type(self.phase_id) is not int or not 0 <= self.phase_id < 7):
            raise ValueError("invalid supervised phase")

    @property
    def report_valid(self):
        return self.ivt_valid and self.phase_valid

    def state(self):
        if not self.report_valid:
            raise ValueError("incomplete report supervision")
        return {"ivt": [",".join(TRIPLETS[i]) for i in self.ivt_ids], "phase": NAMES["phase"][self.phase_id]}


__all__ += ["ReportInput", "GroundTruth", "ModelConfig"]
