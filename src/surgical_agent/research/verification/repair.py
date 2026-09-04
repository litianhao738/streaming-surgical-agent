"""Bounded targeted verification and deterministic repair admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.gate.budget import BudgetGrant, VerificationBudgetManager
from surgical_agent.research.gate.contracts import (
    SCOPE_TASKS,
    RepairScope,
    VerificationPriority,
)
from surgical_agent.research.safety import (
    SafetyValidator,
    choose_covering_scope,
    legal_repair_scopes,
)
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    candidate_id_for,
    changed_tasks,
)

SpecialistStatus = Literal[
    "NO_RESPONSE",
    "INVALID_RESPONSE",
    "UNRESOLVED",
    "REJECT",
    "KEEP",
    "REPAIR",
]
ProposalStatus = Literal[
    "GATE_ACCEPTED",
    "FALLBACK_KEEP",
    "VERIFIED_KEEP",
    "VERIFIED_REPAIR",
    "PENDING_UNRESOLVED",
    "REJECTED_BY_VERIFY",
]


@dataclass(frozen=True)
class SpecialistResult:
    """Strictly parsed semantic result from one Specialist call."""

    status: SpecialistStatus
    hypothesis: InitialPrediction | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {
            "NO_RESPONSE",
            "INVALID_RESPONSE",
            "UNRESOLVED",
            "REJECT",
            "KEEP",
            "REPAIR",
        }:
            raise ValueError("unsupported Specialist status")
        if self.status == "REPAIR":
            if not isinstance(self.hypothesis, InitialPrediction):
                raise ValueError("REPAIR requires one proposed hypothesis")
        elif self.hypothesis is not None:
            raise ValueError("only REPAIR may carry a replacement hypothesis")
        if not isinstance(self.reason, str):
            raise TypeError("Specialist reason must be text")


class TargetedSpecialist(Protocol):
    def verify(
        self,
        *,
        hypothesis: InitialPrediction,
        scope: RepairScope,
        attempt: int,
    ) -> SpecialistResult: ...


@dataclass(frozen=True)
class VerificationAttempt:
    attempt: int
    scope: RepairScope
    budget: BudgetGrant
    specialist_status: str
    candidate_id: str | None = None
    postcheck_hard_valid: bool | None = None
    violation_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if self.scope not in SCOPE_TASKS:
            raise ValueError("attempt scope is unsupported")
        if not isinstance(self.budget, BudgetGrant):
            raise TypeError("attempt budget must be BudgetGrant")


@dataclass(frozen=True)
class RepairProposal:
    """Pre-finalization result; it does not assign a semantic state."""

    status: ProposalStatus
    hypothesis: InitialPrediction | None
    reason: str
    attempts: tuple[VerificationAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            "GATE_ACCEPTED",
            "FALLBACK_KEEP",
            "VERIFIED_KEEP",
            "VERIFIED_REPAIR",
            "PENDING_UNRESOLVED",
            "REJECTED_BY_VERIFY",
        }:
            raise ValueError("unsupported Repair proposal status")
        if self.status in {
            "GATE_ACCEPTED",
            "FALLBACK_KEEP",
            "VERIFIED_KEEP",
            "VERIFIED_REPAIR",
        } and not isinstance(self.hypothesis, InitialPrediction):
            raise ValueError("this proposal status requires a hypothesis")
        if self.status == "REJECTED_BY_VERIFY" and self.hypothesis is not None:
            raise ValueError("Rejected proposals cannot carry a hypothesis")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Repair proposals require a reason")
        attempts = tuple(self.attempts)
        if any(not isinstance(item, VerificationAttempt) for item in attempts):
            raise TypeError("attempts must contain VerificationAttempt values")
        object.__setattr__(self, "attempts", attempts)


def gate_accepted(h0: InitialPrediction) -> RepairProposal:
    return RepairProposal("GATE_ACCEPTED", h0, "GATE_SELECTED_USE_H0")


class BoundedVerifyRepairLoop:
    """Run at most two charged Specialist attempts and revalidate every result."""

    def __init__(
        self,
        *,
        specialist: TargetedSpecialist,
        validator: SafetyValidator,
        budget: VerificationBudgetManager,
        max_attempts: int = 1,
    ) -> None:
        if not hasattr(specialist, "verify") or not callable(specialist.verify):
            raise TypeError("specialist must implement verify")
        if not isinstance(validator, SafetyValidator):
            raise TypeError("validator must be SafetyValidator")
        if not isinstance(budget, VerificationBudgetManager):
            raise TypeError("budget must be VerificationBudgetManager")
        if max_attempts not in {1, 2}:
            raise ValueError("max_attempts must be 1 or 2")
        self.specialist = specialist
        self.validator = validator
        self.budget = budget
        self.max_attempts = max_attempts

    def run(
        self,
        *,
        h0: InitialPrediction,
        candidates: CandidateSet,
        scope: RepairScope,
        priority: VerificationPriority,
        fallback_h0_allowed: bool,
    ) -> RepairProposal:
        if priority == "MANDATORY" and fallback_h0_allowed:
            raise ValueError("mandatory verification cannot fall back to H0")
        if candidates.initial_prediction is not h0:
            raise ValueError("candidate pool must be bound to H0")
        current = h0
        current_scope = scope
        initial_violations = self.validator.validate(h0)
        attempts: list[VerificationAttempt] = []
        seen_states: set[tuple[str, RepairScope, tuple[str, ...]]] = set()

        for attempt_number in range(1, self.max_attempts + 1):
            grant = self.budget.request(priority)
            if not grant.granted:
                attempts.append(
                    VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status="BUDGET_DENIED",
                    )
                )
                return self._failure(
                    h0,
                    attempts,
                    fallback_h0_allowed,
                    grant.reason,
                )

            try:
                result = self.specialist.verify(
                    hypothesis=current,
                    scope=current_scope,
                    attempt=attempt_number,
                )
            except Exception as exc:  # noqa: BLE001 - provider failure is an execution status
                result = SpecialistResult("NO_RESPONSE", reason=type(exc).__name__)
            if not isinstance(result, SpecialistResult):
                result = SpecialistResult("INVALID_RESPONSE", reason="WRONG_RESULT_TYPE")

            if result.status in {"NO_RESPONSE", "INVALID_RESPONSE", "UNRESOLVED"}:
                attempts.append(
                    VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status=result.status,
                    )
                )
                continue
            if result.status == "REJECT":
                attempts.append(
                    VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status="REJECT",
                    )
                )
                return RepairProposal(
                    "REJECTED_BY_VERIFY",
                    None,
                    result.reason or "SPECIALIST_REJECTED_HYPOTHESIS",
                    tuple(attempts),
                )

            candidate = current if result.status == "KEEP" else result.hypothesis
            assert candidate is not None
            touched = changed_tasks(current, candidate)
            admitted = candidates.admits(candidate) and set(touched).issubset(
                SCOPE_TASKS[current_scope]
            )
            if not admitted:
                attempts.append(
                    VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status="INVALID_RESPONSE",
                    )
                )
                continue

            violations = self.validator.validate(candidate)
            candidate_id = candidate_id_for(candidate)
            attempts.append(
                VerificationAttempt(
                    attempt=attempt_number,
                    scope=current_scope,
                    budget=grant,
                    specialist_status=result.status,
                    candidate_id=candidate_id,
                    postcheck_hard_valid=not violations,
                    violation_codes=tuple(item.code for item in violations),
                )
            )
            if not violations:
                if result.status == "REPAIR" and not initial_violations:
                    attempts[-1] = VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status="UNPROVEN_REPAIR",
                        candidate_id=candidate_id,
                        postcheck_hard_valid=True,
                    )
                    return self._failure(
                        h0,
                        attempts,
                        fallback_h0_allowed,
                        "HARD_VALID_H0_PROTECTED",
                    )
                final_status: ProposalStatus = (
                    "VERIFIED_KEEP" if candidate_id == candidate_id_for(h0) else "VERIFIED_REPAIR"
                )
                return RepairProposal(
                    final_status,
                    candidate,
                    final_status,
                    tuple(attempts),
                )

            state_key = (
                candidate_id,
                current_scope,
                tuple(item.code for item in violations),
            )
            if state_key in seen_states:
                break
            seen_states.add(state_key)
            next_scopes = legal_repair_scopes(violations, candidates)
            next_scope = choose_covering_scope(violations, next_scopes)
            if next_scope is None:
                break
            current = candidate
            current_scope = next_scope

        return self._failure(
            h0,
            attempts,
            fallback_h0_allowed,
            "VERIFY_ATTEMPTS_EXHAUSTED",
        )

    @staticmethod
    def _failure(
        h0: InitialPrediction,
        attempts: list[VerificationAttempt],
        fallback_h0_allowed: bool,
        reason: str,
    ) -> RepairProposal:
        if fallback_h0_allowed:
            return RepairProposal("FALLBACK_KEEP", h0, reason, tuple(attempts))
        return RepairProposal("PENDING_UNRESOLVED", h0, reason, tuple(attempts))


__all__ = [
    "BoundedVerifyRepairLoop",
    "ProposalStatus",
    "RepairProposal",
    "SpecialistResult",
    "SpecialistStatus",
    "TargetedSpecialist",
    "VerificationAttempt",
    "gate_accepted",
]
