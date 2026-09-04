"""The single authority that assigns final semantic states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from surgical_agent.inference.schemas import InitialPrediction
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
            return FinalOutcome("Verified", proposal.hypothesis, "VERIFIED_KEEP")
        if proposal.status == "VERIFIED_REPAIR":
            return FinalOutcome("Verified", proposal.hypothesis, "VERIFIED_REPAIR")
        if proposal.status == "PENDING_UNRESOLVED":
            return FinalOutcome("Pending", proposal.hypothesis, "PENDING_UNRESOLVED")
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
