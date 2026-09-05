"""Bounded targeted verification and deterministic repair admission."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    _selected_ids,
    candidate_id_for,
    changed_tasks,
)


def _scope_admits_candidate(
    *,
    scope: RepairScope,
    current: InitialPrediction,
    candidate: InitialPrediction,
    touched: tuple[str, ...],
) -> bool:
    """Admit semantic scope changes plus one deterministic closure cascade."""

    allowed_tasks = set(SCOPE_TASKS[scope])
    if scope != "instrument_presence":
        return set(touched).issubset(allowed_tasks)
    if not set(touched).issubset(allowed_tasks | {"ivt"}):
        return False
    components = load_ivt_components()
    expected_ivts = tuple(
        ivt_id
        for ivt_id in current.triplet_ids
        if components[ivt_id][0] in candidate.instrument_ids
    )
    return candidate.triplet_ids == expected_ivts


SpecialistStatus = Literal[
    "NO_RESPONSE",
    "INVALID_RESPONSE",
    "UNRESOLVED",
    "REJECT",
    "KEEP",
    "REPAIR",
]
RepairEvidenceKind = Literal[
    "TRACKER_TEMPORAL_CONSENSUS",
    "COMPONENT_EXPERT_CONSENSUS",
    "WORKFLOW_TRANSITION_DOMINANCE",
    "VISUAL_INSTRUMENT_AGREEMENT",
    "VISUAL_INTERACTION_AGREEMENT",
    "VISUAL_PHASE_AGREEMENT",
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
class RepairEvidence:
    """Local evidence certificate that cannot be supplied by the wire schema."""

    kind: RepairEvidenceKind
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in {
            "TRACKER_TEMPORAL_CONSENSUS",
            "COMPONENT_EXPERT_CONSENSUS",
            "WORKFLOW_TRANSITION_DOMINANCE",
            "VISUAL_INSTRUMENT_AGREEMENT",
            "VISUAL_INTERACTION_AGREEMENT",
            "VISUAL_PHASE_AGREEMENT",
        }:
            raise ValueError("unsupported Repair evidence kind")
        sources = tuple(self.sources)
        if len(sources) < 2 or len(set(sources)) != len(sources):
            raise ValueError("Repair evidence requires distinct corroborating sources")
        if any(not isinstance(source, str) or not source for source in sources):
            raise ValueError("Repair evidence sources must be non-empty text")
        if self.kind.startswith("VISUAL_") and (
            len(sources) != 2
            or any(not source.startswith("api:review:") for source in sources)
        ):
            raise ValueError("visual agreement needs two distinct API review hashes")
        if self.kind == "TRACKER_TEMPORAL_CONSENSUS" and (
            len(sources) != 2
            or any(not source.startswith("tracker:frame:") for source in sources)
        ):
            raise ValueError("Tracker evidence requires exactly two frame sources")
        if self.kind == "COMPONENT_EXPERT_CONSENSUS":
            required = {
                "api:verb_component",
                "api:target_component",
                "ontology:exact_ivt",
            }
            instrument_sources = {
                "tracker:target_frame",
                "api:instrument_component",
            }
            if (
                not required.issubset(sources)
                or len(instrument_sources & set(sources)) != 1
            ):
                raise ValueError(
                    "Component evidence requires I/V/T sources and exact IVT closure"
                )
        if self.kind == "WORKFLOW_TRANSITION_DOMINANCE" and (
            len(sources) != 2
            or not any(source.startswith("workflow:frame:") for source in sources)
            or not any(source.startswith("phase_graph:") for source in sources)
        ):
            raise ValueError(
                "Workflow evidence requires history and phase graph sources"
            )
        object.__setattr__(self, "sources", sources)


_EVIDENCE_KIND_BY_SCOPE: dict[RepairScope, RepairEvidenceKind] = {
    "instrument_presence": "TRACKER_TEMPORAL_CONSENSUS",
    "interaction": "COMPONENT_EXPERT_CONSENSUS",
    "workflow": "WORKFLOW_TRANSITION_DOMINANCE",
}

_VISUAL_EVIDENCE_BY_SCOPE: dict[RepairScope, RepairEvidenceKind] = {
    "instrument_presence": "VISUAL_INSTRUMENT_AGREEMENT",
    "interaction": "VISUAL_INTERACTION_AGREEMENT",
    "workflow": "VISUAL_PHASE_AGREEMENT",
}


def evidence_matches_scope(evidence: RepairEvidence | None, scope: RepairScope) -> bool:
    return evidence is not None and evidence.kind in {
        _EVIDENCE_KIND_BY_SCOPE[scope],
        _VISUAL_EVIDENCE_BY_SCOPE[scope],
    }


@dataclass(frozen=True)
class SpecialistResult:
    """Strictly parsed semantic result from one Specialist call."""

    status: SpecialistStatus
    hypothesis: InitialPrediction | None = None
    reason: str = ""
    repair_evidence: RepairEvidence | None = None

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
        if self.repair_evidence is not None:
            if self.status not in {"KEEP", "REPAIR"}:
                raise ValueError(
                    "only KEEP or REPAIR may carry an evidence certificate"
                )
            if not isinstance(self.repair_evidence, RepairEvidence):
                raise TypeError("repair_evidence must be RepairEvidence or None")
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
    repair_evidence_kind: RepairEvidenceKind | None = None
    repair_evidence_sources: tuple[str, ...] = ()
    accepted: bool = False
    certified_task_values: tuple[tuple[str, tuple[int, ...]], ...] = ()
    specialist_reason: str = ""
    candidate_labels: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt <= 0
        ):
            raise ValueError("attempt must be a positive integer")
        if self.scope not in SCOPE_TASKS:
            raise ValueError("attempt scope is unsupported")
        if not isinstance(self.budget, BudgetGrant):
            raise TypeError("attempt budget must be BudgetGrant")
        sources = tuple(self.repair_evidence_sources)
        if (self.repair_evidence_kind is None) != (not sources):
            raise ValueError("attempt evidence kind and sources must occur together")
        if self.repair_evidence_kind is not None:
            RepairEvidence(self.repair_evidence_kind, sources)
        object.__setattr__(self, "repair_evidence_sources", sources)


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
        require_repair_evidence: bool = False,
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
        self.require_repair_evidence = require_repair_evidence

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
                result = SpecialistResult(
                    "INVALID_RESPONSE", reason="WRONG_RESULT_TYPE"
                )

            if result.status in {"NO_RESPONSE", "INVALID_RESPONSE", "UNRESOLVED"}:
                attempts.append(
                    VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status=result.status,
                        specialist_reason=result.reason,
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
            repair_evidence = result.repair_evidence
            touched = changed_tasks(current, candidate)
            admitted = candidates.admits(candidate) and _scope_admits_candidate(
                scope=current_scope,
                current=current,
                candidate=candidate,
                touched=touched,
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
                    specialist_reason=result.reason,
                    candidate_labels=tuple(
                        (task, _selected_ids(candidate, task))
                        for task in ("instrument", "verb", "target", "ivt", "phase")
                    ),
                    candidate_id=candidate_id,
                    postcheck_hard_valid=not violations,
                    violation_codes=tuple(item.code for item in violations),
                    repair_evidence_kind=(
                        None if repair_evidence is None else repair_evidence.kind
                    ),
                    repair_evidence_sources=(
                        () if repair_evidence is None else repair_evidence.sources
                    ),
                )
            )
            if not violations:
                evidence_supports_scope = evidence_matches_scope(
                    repair_evidence, current_scope
                )
                if (
                    result.status == "REPAIR"
                    and (not initial_violations or self.require_repair_evidence)
                    and not evidence_supports_scope
                ):
                    attempts[-1] = VerificationAttempt(
                        attempt=attempt_number,
                        scope=current_scope,
                        budget=grant,
                        specialist_status="UNPROVEN_REPAIR",
                        specialist_reason=result.reason,
                        candidate_labels=tuple(
                            (task, _selected_ids(candidate, task))
                            for task in ("instrument", "verb", "target", "ivt", "phase")
                        ),
                        candidate_id=candidate_id,
                        postcheck_hard_valid=True,
                        repair_evidence_kind=(
                            None if repair_evidence is None else repair_evidence.kind
                        ),
                        repair_evidence_sources=(
                            () if repair_evidence is None else repair_evidence.sources
                        ),
                    )
                    return self._failure(
                        h0,
                        attempts,
                        fallback_h0_allowed,
                        "HARD_VALID_H0_PROTECTED",
                    )
                final_status: ProposalStatus = (
                    "VERIFIED_KEEP"
                    if candidate_id == candidate_id_for(h0)
                    else "VERIFIED_REPAIR"
                )
                attempts[-1] = replace(
                    attempts[-1],
                    accepted=True,
                    certified_task_values=tuple(
                        (task, _selected_ids(candidate, task))
                        for task in sorted(SCOPE_TASKS[current_scope])
                    )
                    if evidence_supports_scope
                    else (),
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
    "RepairEvidence",
    "RepairEvidenceKind",
    "RepairProposal",
    "SpecialistResult",
    "SpecialistStatus",
    "TargetedSpecialist",
    "VerificationAttempt",
    "gate_accepted",
]
