"""Gold-free contracts shared by verification and deterministic coordination."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import (
    FIELD_UNCERTAINTY_PATHS,
    FIELD_UNCERTAINTY_REASONS,
    TASK_NAMES,
    ApiCallProvenance,
    RankedCandidate,
)
from surgical_agent.research.reliability.state import canonical_tasks

VERIFICATION_SCHEMA_VERSION = "verification_runtime_v1"


def _selected_ids(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    if task == "instrument":
        return prediction.instrument_ids
    if task == "verb":
        return prediction.verb_ids
    if task == "target":
        return prediction.target_ids
    if task == "ivt":
        return prediction.triplet_ids
    if task == "phase":
        return (prediction.phase_id,)
    raise KeyError(task)


def changed_tasks(
    initial: InitialPrediction,
    proposed: InitialPrediction,
) -> tuple[str, ...]:
    """Return the frozen task order whose selected labels changed."""

    return tuple(
        task
        for task in TASK_NAMES
        if _selected_ids(initial, task) != _selected_ids(proposed, task)
    )


def candidate_id_for(prediction: InitialPrediction) -> str:
    """Create an opaque stable identifier without serializing probabilities."""

    payload = {task: list(_selected_ids(prediction, task)) for task in TASK_NAMES}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"H_{digest[:16]}"


@dataclass(frozen=True)
class CandidateSet:
    """Factorized bounded label pool derived only from current perception ranks."""

    initial_prediction: InitialPrediction
    allowed_ids: Mapping[str, tuple[int, ...]]
    source_frame_id: int
    candidate_records: Mapping[str, tuple[RankedCandidate, ...]] | None = None
    candidate_version: str = "factorized_topk_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.initial_prediction, InitialPrediction):
            raise TypeError("initial_prediction must be an InitialPrediction")
        if (
            not isinstance(self.source_frame_id, int)
            or isinstance(self.source_frame_id, bool)
            or self.source_frame_id < 0
        ):
            raise ValueError("source_frame_id must be a non-negative integer")
        if not isinstance(self.allowed_ids, Mapping):
            raise TypeError("allowed_ids must be a mapping")
        if set(self.allowed_ids) != set(TASK_NAMES):
            raise ValueError("allowed_ids must contain exactly the five task heads")
        frozen: dict[str, tuple[int, ...]] = {}
        for task in TASK_NAMES:
            values = tuple(self.allowed_ids[task])
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{task} candidate IDs must be sorted and unique")
            lower, upper = TASK_ID_BOUNDS[task]
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not lower <= value <= upper
                for value in values
            ):
                raise ValueError(f"{task} candidate ID is outside the ontology")
            if not set(_selected_ids(self.initial_prediction, task)).issubset(values):
                raise ValueError("candidate pools must retain the initial hypothesis")
            frozen[task] = values
        object.__setattr__(self, "allowed_ids", MappingProxyType(frozen))
        records = self.candidate_records
        if records is None:
            records = {task: () for task in TASK_NAMES}
        if not isinstance(records, Mapping) or set(records) != set(TASK_NAMES):
            raise ValueError("candidate_records must contain exactly the five task heads")
        frozen_records: dict[str, tuple[RankedCandidate, ...]] = {}
        for task in TASK_NAMES:
            values = tuple(records[task])
            if any(not isinstance(value, RankedCandidate) for value in values):
                raise TypeError("candidate_records must contain RankedCandidate values")
            ids = tuple(value.class_id for value in values)
            if values and set(ids) != set(frozen[task]):
                raise ValueError("candidate_records must describe the allowed IDs")
            if len(set(ids)) != len(ids) or any(
                values[index].confidence < values[index + 1].confidence
                for index in range(len(values) - 1)
            ):
                raise ValueError("candidate_records must be unique and descending")
            frozen_records[task] = values
        object.__setattr__(
            self, "candidate_records", MappingProxyType(frozen_records)
        )

    def admits(self, prediction: InitialPrediction) -> bool:
        """Return whether every selected label stays inside the frozen pool."""

        if not isinstance(prediction, InitialPrediction):
            return False
        return all(
            set(_selected_ids(prediction, task)).issubset(self.allowed_ids[task])
            for task in TASK_NAMES
        )


@dataclass(frozen=True)
class VerificationUncertainty:
    """One exact bounded uncertainty returned for a targeted field."""

    reason: str
    alternative_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.reason not in FIELD_UNCERTAINTY_REASONS:
            raise ValueError("verification uncertainty reason is unsupported")
        alternatives = tuple(self.alternative_ids)
        if len(alternatives) > 8 or len(set(alternatives)) != len(alternatives):
            raise ValueError("verification uncertainty alternatives must be bounded and unique")
        object.__setattr__(self, "alternative_ids", alternatives)


@dataclass(frozen=True)
class FieldVerificationOutcome:
    """One ordered, strictly parsed response for a requested field."""

    path: str
    selected_ids: tuple[int, ...]
    candidate_records: tuple[RankedCandidate, ...]
    status: str
    uncertainty: VerificationUncertainty | None

    def __post_init__(self) -> None:
        if self.path not in FIELD_UNCERTAINTY_PATHS:
            raise ValueError("verification outcome path is unsupported")
        if self.status not in {"Verified", "Pending", "Rejected"}:
            raise ValueError("verification outcome status is unsupported")
        selected = tuple(self.selected_ids)
        records = tuple(self.candidate_records)
        task = self.task
        lower, upper = TASK_ID_BOUNDS[task]
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not lower <= value <= upper
            for value in selected
        ):
            raise ValueError("verification selected ID is outside the ontology")
        if task == "phase" and len(selected) != 1:
            raise ValueError("phase verification must select exactly one ID")
        if len(set(selected)) != len(selected):
            raise ValueError("verification selected IDs must be unique")
        if any(not isinstance(value, RankedCandidate) for value in records):
            raise TypeError("candidate_records must contain RankedCandidate values")
        if not set(selected).issubset(value.class_id for value in records):
            raise ValueError("verification selection must be represented in top-k")
        if self.uncertainty is not None and not isinstance(
            self.uncertainty, VerificationUncertainty
        ):
            raise TypeError("uncertainty must be VerificationUncertainty or None")
        object.__setattr__(self, "selected_ids", selected)
        object.__setattr__(self, "candidate_records", records)

    @property
    def task(self) -> str:
        return FIELD_UNCERTAINTY_PATHS[self.path]


@dataclass(frozen=True)
class VerificationResult:
    """One same-backbone verification proposal before coordinator validation."""

    scope: str
    prediction: InitialPrediction
    provenance: ApiCallProvenance
    requested_fields: tuple[str, ...] = ()
    field_outcomes: tuple[FieldVerificationOutcome, ...] = ()
    repaired_fields: tuple[str, ...] = ()
    schema_version: str = VERIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("verification scope must not be empty")
        if not isinstance(self.prediction, InitialPrediction):
            raise TypeError("verification prediction must be an InitialPrediction")
        if not isinstance(self.provenance, ApiCallProvenance):
            raise TypeError("verification provenance must be ApiCallProvenance")
        if self.schema_version != VERIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported verification schema version")
        requested = canonical_tasks(tuple(self.requested_fields))
        outcomes = tuple(self.field_outcomes)
        if any(not isinstance(item, FieldVerificationOutcome) for item in outcomes):
            raise TypeError("field_outcomes must contain FieldVerificationOutcome values")
        if outcomes and tuple(item.task for item in outcomes) != requested:
            raise ValueError("field outcomes must match requested fields in order")
        repaired = canonical_tasks(tuple(self.repaired_fields))
        if not set(repaired).issubset(requested):
            raise ValueError("repaired_fields must be a subset of requested_fields")
        object.__setattr__(self, "requested_fields", requested)
        object.__setattr__(self, "field_outcomes", outcomes)
        object.__setattr__(self, "repaired_fields", repaired)


@dataclass(frozen=True)
class CoordinationResult:
    """Validated final semantic state and an auditable fixed-vocabulary outcome."""

    prediction: InitialPrediction
    verification_status: str
    reason: str
    selected_candidate_id: str | None = None
    touched_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, InitialPrediction):
            raise TypeError("coordinated prediction must be an InitialPrediction")
        if self.verification_status not in {
            "NOT_REQUESTED",
            "VERIFIED_KEEP",
            "VERIFIED_REPAIR",
            "VERIFIED_PENDING",
            "VERIFIED_REJECT",
            "FALLBACK_KEEP",
        }:
            raise ValueError("unsupported verification status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("coordination reason must not be empty")
        tasks = tuple(self.touched_tasks)
        if any(task not in TASK_NAMES for task in tasks) or len(set(tasks)) != len(tasks):
            raise ValueError("touched_tasks must be unique known task names")
        if self.verification_status == "VERIFIED_REPAIR":
            if self.selected_candidate_id is None or not tasks:
                raise ValueError("verified repairs require a candidate and touched tasks")
        elif self.selected_candidate_id is not None or tasks:
            raise ValueError("non-repair outcomes cannot carry a candidate or touched tasks")
        object.__setattr__(self, "touched_tasks", tasks)
