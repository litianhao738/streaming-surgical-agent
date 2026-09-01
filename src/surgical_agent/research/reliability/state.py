"""Closed-vocabulary reliability state shared by routing and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.contracts import (
    FIELD_UNCERTAINTY_PATHS,
    FIELD_UNCERTAINTY_REASONS,
    TASK_NAMES,
)

CANONICAL_TASK_ORDER = tuple(TASK_NAMES)
TASK_PATHS = {
    task: path for path, task in FIELD_UNCERTAINTY_PATHS.items()
}
ALL_TASK_FIELDS = CANONICAL_TASK_ORDER

INITIAL_STATE = "Candidate"
FINAL_STATUSES = frozenset(
    {"Candidate", "Accepted", "Verified", "Pending", "Rejected"}
)
MEMORY_ACTIONS = frozenset(
    {"WRITE_RELIABLE", "WRITE_SHORT_TERM", "BUFFER_PENDING", "SKIP"}
)
ALLOWED_MEMORY_ACTIONS_BY_FINAL_STATUS = MappingProxyType(
    {
        "Candidate": frozenset({"SKIP"}),
        "Rejected": frozenset({"SKIP"}),
        "Pending": frozenset({"BUFFER_PENDING"}),
        "Verified": frozenset({"WRITE_RELIABLE"}),
        "Accepted": frozenset({"WRITE_RELIABLE", "WRITE_SHORT_TERM"}),
    }
)
GATE_REASONS = frozenset(
    {
        "ONTOLOGY_VIOLATION",
        "IVT_CLOSURE",
        "TRIPLET_COMPATIBILITY",
        "PHASE_TRIPLET_CONFLICT",
        "TEMPORAL_JUMP",
        "LOW_CONFIDENCE",
        "TRACKER_CONFLICT",
    }
)
FINDING_REASONS = GATE_REASONS | FIELD_UNCERTAINTY_REASONS


@dataclass(frozen=True)
class GateFinding:
    """One deterministic reason for verifying one canonical task path."""

    path: str
    reason: str
    alternative_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.path not in FIELD_UNCERTAINTY_PATHS:
            raise ValueError("Gate finding path is unsupported")
        if self.reason not in FINDING_REASONS:
            raise ValueError("Gate finding reason is outside the closed vocabulary")
        task = FIELD_UNCERTAINTY_PATHS[self.path]
        lower, upper = TASK_ID_BOUNDS[task]
        alternatives = tuple(self.alternative_ids)
        if len(set(alternatives)) != len(alternatives):
            raise ValueError("Gate finding alternative IDs must be unique")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not lower <= value <= upper
            for value in alternatives
        ):
            raise ValueError("Gate finding alternative ID is outside its ontology")
        object.__setattr__(self, "alternative_ids", alternatives)


def canonical_tasks(tasks: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate task names in the frozen five-head order."""

    if any(task not in CANONICAL_TASK_ORDER for task in tasks):
        raise ValueError("flagged fields must use canonical task names")
    selected = set(tasks)
    return tuple(task for task in CANONICAL_TASK_ORDER if task in selected)


def validate_status_memory_action(final_status: str, memory_action: str) -> None:
    """Enforce the closed final-status to memory-action state invariant."""

    if final_status not in FINAL_STATUSES:
        raise ValueError("final_status is unsupported")
    if memory_action not in MEMORY_ACTIONS:
        raise ValueError("memory_action is unsupported")
    if memory_action not in ALLOWED_MEMORY_ACTIONS_BY_FINAL_STATUS[final_status]:
        raise ValueError("memory_action is inconsistent with final_status")


def final_status_for(verification_status: str) -> str:
    """Map a coordinator outcome to the reliability state machine."""

    if verification_status == "NOT_REQUESTED":
        return "Accepted"
    if verification_status in {"VERIFIED_KEEP", "VERIFIED_REPAIR"}:
        return "Verified"
    if verification_status == "VERIFIED_REJECT":
        return "Rejected"
    return "Pending"


def memory_action_for(
    final_status: str,
    *,
    selected_confidence_floor: float | None,
    has_gate_findings: bool,
    reliable_confidence_threshold: float = 0.85,
) -> str:
    """Route one final state into the only eligible memory collection."""

    if final_status == "Verified":
        return "WRITE_RELIABLE"
    if final_status == "Accepted":
        if (
            selected_confidence_floor is not None
            and selected_confidence_floor >= reliable_confidence_threshold
            and not has_gate_findings
        ):
            return "WRITE_RELIABLE"
        return "WRITE_SHORT_TERM"
    if final_status == "Pending":
        return "BUFFER_PENDING"
    return "SKIP"
