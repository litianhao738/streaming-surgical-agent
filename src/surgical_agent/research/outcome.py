"""The single authority that assigns final semantic states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import TASK_NAMES
from surgical_agent.research.gate.contracts import SCOPE_TASKS
from surgical_agent.research.reliability.taskwise import (
    normalize_task_states,
    tasks_in_state,
)
from surgical_agent.research.verification.contracts import _selected_ids, changed_tasks
from surgical_agent.research.verification.repair import RepairProposal

SemanticState = Literal["Accepted", "Verified", "Pending", "Rejected"]
ExecutionStatus = Literal[
    "EXECUTION_ERROR",
    "INVALID_RESPONSE",
    "PENDING_NO_H0",
    "POLICY_FAILURE",
]


@dataclass(frozen=True)
class FinalOutcome:
    state: SemanticState
    hypothesis: InitialPrediction | None
    provenance: str
    lower_reliability: bool = False
    task_states: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.state not in {"Accepted", "Verified", "Pending", "Rejected"}:
            raise ValueError("unsupported final semantic state")
        if self.state in {"Accepted", "Verified"} and not isinstance(
            self.hypothesis, InitialPrediction
        ):
            raise ValueError("Accepted and Verified outcomes require a hypothesis")
        if self.state == "Rejected" and self.hypothesis is not None:
            raise ValueError("Rejected outcomes cannot carry a hypothesis")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("final outcomes require provenance")
        if self.lower_reliability and self.state != "Accepted":
            raise ValueError("only Accepted outcomes may be marked lower reliability")
        default_task_state = {
            "Accepted": "Accepted",
            "Verified": "Verified",
            "Pending": "Pending",
            "Rejected": "Rejected",
        }[self.state]
        task_states = normalize_task_states(
            self.task_states,
            default=default_task_state,  # type: ignore[arg-type]
        )
        if self.state == "Verified" and set(task_states.values()) != {"Verified"}:
            raise ValueError("frame-level Verified requires all five tasks Verified")
        if self.state == "Accepted" and any(
            value in {"Pending", "Rejected"} for value in task_states.values()
        ):
            raise ValueError(
                "Accepted outcomes cannot contain Pending or Rejected tasks"
            )
        if self.state == "Pending" and "Pending" not in task_states.values():
            raise ValueError("Pending outcomes require at least one Pending task")
        if self.state == "Rejected" and set(task_states.values()) != {"Rejected"}:
            raise ValueError("Rejected outcomes require all five tasks Rejected")
        object.__setattr__(self, "task_states", task_states)

    @property
    def verified_tasks(self) -> tuple[str, ...]:
        assert self.task_states is not None
        return tasks_in_state(self.task_states, "Verified")

    @property
    def checked_tasks(self) -> tuple[str, ...]:
        assert self.task_states is not None
        return tasks_in_state(self.task_states, "Checked")

    @property
    def derived_tasks(self) -> tuple[str, ...]:
        assert self.task_states is not None
        return tasks_in_state(self.task_states, "Derived")


@dataclass(frozen=True)
class ExecutionOutcome:
    status: ExecutionStatus
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {
            "EXECUTION_ERROR",
            "INVALID_RESPONSE",
            "PENDING_NO_H0",
            "POLICY_FAILURE",
        }:
            raise ValueError("unsupported execution status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("execution outcomes require a reason")


class OutcomeFinalizer:
    """Map a validated proposal to exactly one authoritative semantic state."""

    @staticmethod
    def _scoped_task_states(
        *,
        h0: InitialPrediction,
        proposal: RepairProposal,
    ) -> Mapping[str, str]:
        states = {task: "Accepted" for task in TASK_NAMES}
        checked: set[str] = set()
        verified: set[str] = set()
        for attempt in proposal.attempts:
            if attempt.specialist_status not in {"KEEP", "REPAIR"}:
                continue
            scope_tasks = set(SCOPE_TASKS[attempt.scope])
            checked.update(scope_tasks)
            if (
                attempt.accepted
                and attempt.postcheck_hard_valid
                and proposal.hypothesis is not None
            ):
                verified.update(
                    task
                    for task, values in attempt.certified_task_values
                    if task in scope_tasks
                    and _selected_ids(proposal.hypothesis, task) == values
                )
        for task in checked:
            states[task] = "Checked"
        for task in verified:
            states[task] = "Verified"
        if proposal.hypothesis is not None:
            for task in changed_tasks(h0, proposal.hypothesis):
                if task not in checked:
                    states[task] = "Derived"
        return states

    def finalize(
        self,
        *,
        h0: InitialPrediction,
        proposal: RepairProposal,
    ) -> FinalOutcome:
        if not isinstance(h0, InitialPrediction):
            raise TypeError("OutcomeFinalizer requires H0")
        if not isinstance(proposal, RepairProposal):
            raise TypeError("OutcomeFinalizer requires a RepairProposal")
        if proposal.status == "GATE_ACCEPTED":
            if proposal.hypothesis is not h0:
                raise ValueError("GATE_ACCEPTED must preserve H0 object identity")
            return FinalOutcome("Accepted", h0, "GATE_ACCEPTED")
        if proposal.status == "FALLBACK_KEEP":
            if proposal.hypothesis is not h0:
                raise ValueError("FALLBACK_KEEP must preserve H0 object identity")
            return FinalOutcome(
                "Accepted",
                h0,
                "FALLBACK_KEEP",
                lower_reliability=True,
            )
        if proposal.status == "VERIFIED_KEEP":
            states = self._scoped_task_states(h0=h0, proposal=proposal)
            fully_verified = set(states.values()) == {"Verified"}
            return FinalOutcome(
                "Verified" if fully_verified else "Accepted",
                proposal.hypothesis,
                "VERIFIED_KEEP",
                lower_reliability=not fully_verified
                and any(not attempt.accepted for attempt in proposal.attempts),
                task_states=states,
            )
        if proposal.status == "VERIFIED_REPAIR":
            task_states = self._scoped_task_states(h0=h0, proposal=proposal)
            frame_state: SemanticState = (
                "Verified" if set(task_states.values()) == {"Verified"} else "Accepted"
            )
            return FinalOutcome(
                frame_state,
                proposal.hypothesis,
                "VERIFIED_REPAIR",
                lower_reliability=frame_state == "Accepted"
                and any(not attempt.accepted for attempt in proposal.attempts),
                task_states=task_states,
            )
        if proposal.status == "PENDING_UNRESOLVED":
            task_states = {task: "Accepted" for task in TASK_NAMES}
            pending_tasks = set(TASK_NAMES)
            if proposal.attempts:
                pending_tasks = set(SCOPE_TASKS[proposal.attempts[-1].scope])
            for task in pending_tasks:
                task_states[task] = "Pending"
            return FinalOutcome(
                "Pending",
                proposal.hypothesis,
                "PENDING_UNRESOLVED",
                task_states=task_states,
            )
        if proposal.status == "REJECTED_BY_VERIFY":
            return FinalOutcome("Rejected", None, "REJECTED_BY_VERIFY")
        raise AssertionError(f"unhandled proposal status: {proposal.status}")


__all__ = [
    "ExecutionOutcome",
    "ExecutionStatus",
    "FinalOutcome",
    "OutcomeFinalizer",
    "SemanticState",
]
