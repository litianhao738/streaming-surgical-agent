"""Immutable contracts for deterministic frame evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.contracts import TASK_NAMES

EVIDENCE_SCHEMA_VERSION = "evidence_frame_v1"
EVIDENCE_RECORD_SCHEMA_VERSION = "evidence_record_v1"
_TASK_SIGNAL_NAMES = {
    "instrument": frozenset(
        {
            "candidate_ambiguity",
            "ivt_internal_conflict",
            "self_reported_uncertainty",
            "temporal_set_change",
        }
    ),
    "verb": frozenset(
        {
            "candidate_ambiguity",
            "ivt_internal_conflict",
            "self_reported_uncertainty",
            "temporal_set_change",
        }
    ),
    "target": frozenset(
        {
            "candidate_ambiguity",
            "ivt_internal_conflict",
            "self_reported_uncertainty",
            "temporal_set_change",
        }
    ),
    "ivt": frozenset(
        {
            "candidate_ambiguity",
            "ivt_internal_conflict",
            "self_reported_uncertainty",
            "temporal_set_change",
        }
    ),
    "phase": frozenset(
        {
            "candidate_ambiguity",
            "phase_change_anomaly",
            "self_reported_uncertainty",
        }
    ),
}
_GLOBAL_SIGNAL_NAMES = frozenset({"ivt_internal_conflict"})
GATE_FEATURE_NAMES = frozenset(
    {
        f"{task}.{signal}.{attribute}"
        for task, signals in _TASK_SIGNAL_NAMES.items()
        for signal in signals
        for attribute in ("value", "available")
    }
    | {
        f"global.{signal}.{attribute}"
        for signal in _GLOBAL_SIGNAL_NAMES
        for attribute in ("value", "available")
    }
)
_SIGNAL_SOURCES = {
    "candidate_ambiguity": "joint_rank_margin",
    "ivt_internal_conflict": "ivt_component_map_v1",
    "phase_change_anomaly": "frozen_phase_transition_graph",
    "self_reported_uncertainty": "joint_self_reported_confidence",
    "temporal_set_change": "finalized_prior_jaccard",
}


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_nonempty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class EvidenceValue:
    """One normalized evidence signal with explicit causal provenance."""

    value: float | None
    available: bool
    source: str
    source_max_frame_id: int

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("available must be a boolean")
        _require_nonempty_string(self.source, name="evidence source")
        _require_nonnegative_int(
            self.source_max_frame_id, name="source_max_frame_id"
        )
        if self.available:
            if (
                not isinstance(self.value, (int, float))
                or isinstance(self.value, bool)
                or not math.isfinite(float(self.value))
                or not 0.0 <= float(self.value) <= 1.0
            ):
                raise ValueError("available evidence values must be finite in [0, 1]")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError("unavailable evidence must use value=None")


def unavailable(source: str, source_max_frame_id: int) -> EvidenceValue:
    """Represent absent evidence without fabricating a numeric value."""

    return EvidenceValue(None, False, source, source_max_frame_id)


@dataclass(frozen=True)
class EvidenceProfile:
    """The closed-vocabulary evidence emitted for one causal frame."""

    video_id: str
    frame_id: int
    task_values: Mapping[str, Mapping[str, EvidenceValue]]
    global_values: Mapping[str, EvidenceValue]
    evidence_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.video_id, name="video_id")
        _require_nonnegative_int(self.frame_id, name="frame_id")
        if self.evidence_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence_version: {self.evidence_version!r}")
        if not isinstance(self.task_values, Mapping):
            raise TypeError("task_values must be a mapping")
        if set(self.task_values) != set(TASK_NAMES):
            raise ValueError("task_values must contain exactly the five task heads")
        if not isinstance(self.global_values, Mapping):
            raise TypeError("global_values must be a mapping")

        frozen_task_values: dict[str, Mapping[str, EvidenceValue]] = {}
        for task in TASK_NAMES:
            values = self.task_values[task]
            if not isinstance(values, Mapping):
                raise TypeError("each task evidence value collection must be a mapping")
            frozen_values = dict(values)
            _validate_values(
                frozen_values,
                frame_id=self.frame_id,
                allowed_names=_TASK_SIGNAL_NAMES[task],
            )
            frozen_task_values[task] = MappingProxyType(frozen_values)
        frozen_global_values = dict(self.global_values)
        _validate_values(
            frozen_global_values,
            frame_id=self.frame_id,
            allowed_names=_GLOBAL_SIGNAL_NAMES,
        )
        object.__setattr__(self, "task_values", MappingProxyType(frozen_task_values))
        object.__setattr__(self, "global_values", MappingProxyType(frozen_global_values))


def _validate_values(
    values: Mapping[str, EvidenceValue],
    *,
    frame_id: int,
    allowed_names: frozenset[str],
) -> None:
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("evidence signal names must be non-empty strings")
    if any(not isinstance(value, EvidenceValue) for value in values.values()):
        raise TypeError("evidence values must be EvidenceValue instances")
    if set(values) != allowed_names:
        raise ValueError("evidence signal names must match the allowlisted schema")
    if any(value.source != _SIGNAL_SOURCES[name] for name, value in values.items()):
        raise ValueError("evidence sources must match the allowlisted signal sources")
    if any(value.source_max_frame_id > frame_id for value in values.values()):
        raise ValueError("evidence provenance must not reference a future frame")


@dataclass(frozen=True)
class EvidenceRecord:
    """Persisted evidence paired to one canonical prediction hash."""

    run_id: str
    video_id: str
    frame_id: int
    prediction_sha256: str
    task_values: Mapping[str, Mapping[str, EvidenceValue]]
    global_values: Mapping[str, EvidenceValue]
    evidence_version: str
    schema_version: str = EVIDENCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.run_id, name="run_id")
        if self.schema_version != EVIDENCE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if (
            len(self.prediction_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.prediction_sha256)
        ):
            raise ValueError("prediction_sha256 must be a lowercase SHA-256 hex digest")
        profile = EvidenceProfile(
            video_id=self.video_id,
            frame_id=self.frame_id,
            task_values=self.task_values,
            global_values=self.global_values,
            evidence_version=self.evidence_version,
        )
        object.__setattr__(self, "task_values", profile.task_values)
        object.__setattr__(self, "global_values", profile.global_values)


@dataclass(frozen=True)
class PhaseTransitionGraph:
    """A frozen train-only graph of allowed directed phase transitions."""

    transitions: tuple[tuple[int, int], ...]
    source_video_ids: tuple[str, ...]
    version: str
    sha256: str

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if any(
            not isinstance(transition, tuple) or len(transition) != 2
            for transition in transitions
        ):
            raise ValueError("transitions must contain phase ID pairs")
        lower, upper = TASK_ID_BOUNDS["phase"]
        if any(
            not all(
                isinstance(phase_id, int)
                and not isinstance(phase_id, bool)
                and lower <= phase_id <= upper
                for phase_id in transition
            )
            for transition in transitions
        ):
            raise ValueError("transition phase IDs are outside the phase range")
        if tuple(sorted(set(transitions))) != transitions:
            raise ValueError("transitions must be sorted and unique")
        source_video_ids = tuple(self.source_video_ids)
        if not source_video_ids:
            raise ValueError("source_video_ids must contain at least one video ID")
        if any(
            not isinstance(video_id, str)
            or not video_id
            or video_id != video_id.strip()
            for video_id in source_video_ids
        ):
            raise ValueError(
                "source_video_ids must contain stripped non-empty strings"
            )
        if tuple(sorted(set(source_video_ids))) != source_video_ids:
            raise ValueError("source_video_ids must be sorted and unique")
        _require_nonempty_string(self.version, name="version")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "source_video_ids", source_video_ids)

    def allows(self, previous: int, current: int) -> bool:
        return (previous, current) in self.transitions
