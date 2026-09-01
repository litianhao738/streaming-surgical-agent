"""Deterministic KEEP/REPAIR validation with no model or GT access."""

from __future__ import annotations

from dataclasses import replace

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    CoordinationResult,
    VerificationResult,
    candidate_id_for,
    changed_tasks,
)
from surgical_agent.systems.pipeline import GateDecision

_SCOPE_TASKS = {
    "joint": frozenset({"instrument", "verb", "target", "ivt", "phase"}),
    "targeted": frozenset({"instrument", "verb", "target", "ivt", "phase"}),
    "spatial_track": frozenset({"instrument"}),
    "interaction": frozenset({"verb", "target", "ivt"}),
    "workflow": frozenset({"phase"}),
}


def _fallback(initial: InitialPrediction, reason: str) -> CoordinationResult:
    return CoordinationResult(
        prediction=initial,
        verification_status="FALLBACK_KEEP",
        reason=reason,
    )


class DeterministicCoordinator:
    """Admit only in-pool, scope-authorized semantic changes."""

    def coordinate(
        self,
        prediction: InitialPrediction,
        decision: GateDecision,
        candidates: object,
        verification: VerificationResult | None,
    ) -> CoordinationResult:
        if decision.action == "ACCEPT":
            return CoordinationResult(
                prediction=prediction,
                verification_status="NOT_REQUESTED",
                reason="GATE_ACCEPTED",
            )
        if not isinstance(candidates, CandidateSet):
            return _fallback(prediction, "CANDIDATE_SET_UNAVAILABLE")
        if verification is None:
            return _fallback(prediction, "VERIFICATION_RESULT_MISSING")
        if decision.scope != verification.scope:
            return _fallback(prediction, "SCOPE_MISMATCH")
        allowed_tasks = _SCOPE_TASKS.get(verification.scope)
        if allowed_tasks is None:
            return _fallback(prediction, "UNKNOWN_SCOPE")
        if verification.requested_fields:
            if verification.requested_fields != decision.flagged_fields:
                return _fallback(prediction, "REQUESTED_FIELDS_MISMATCH")
            outcomes = verification.field_outcomes
            if tuple(outcome.task for outcome in outcomes) != verification.requested_fields:
                return _fallback(prediction, "REQUESTED_FIELDS_MISMATCH")
            if any(outcome.status == "Rejected" for outcome in outcomes):
                return CoordinationResult(
                    prediction=prediction,
                    verification_status="VERIFIED_REJECT",
                    reason="VERIFIER_REJECTED_REQUESTED_FIELD",
                )
            if any(outcome.status == "Pending" for outcome in outcomes):
                return CoordinationResult(
                    prediction=prediction,
                    verification_status="VERIFIED_PENDING",
                    reason="VERIFIER_LEFT_REQUESTED_FIELD_PENDING",
                )
            if not outcomes or any(outcome.status != "Verified" for outcome in outcomes):
                return _fallback(prediction, "FIELD_OUTCOMES_INVALID")
        proposed = verification.prediction
        if proposed.score_semantics != prediction.score_semantics:
            return _fallback(prediction, "SCORE_SEMANTICS_MISMATCH")
        touched = changed_tasks(prediction, proposed)
        if verification.requested_fields and not set(touched).issubset(
            verification.requested_fields
        ):
            return _fallback(prediction, "REQUESTED_TASK_VIOLATION")
        if verification.requested_fields and touched != verification.repaired_fields:
            return _fallback(prediction, "REPAIRED_FIELDS_MISMATCH")
        if not touched:
            return CoordinationResult(
                prediction=prediction,
                verification_status="VERIFIED_KEEP",
                reason="VERIFIER_CONFIRMED_CURRENT",
            )
        if not set(touched).issubset(allowed_tasks):
            return _fallback(prediction, "SCOPE_TASK_VIOLATION")
        if not candidates.admits(proposed):
            return _fallback(prediction, "CANDIDATE_OUT_OF_POOL")

        probabilities = dict(prediction.probabilities)
        for task in touched:
            probabilities[task] = tuple(proposed.probabilities[task])
        updates: dict[str, object] = {"probabilities": probabilities}
        if "instrument" in touched:
            updates["instrument_ids"] = proposed.instrument_ids
        if "verb" in touched:
            updates["verb_ids"] = proposed.verb_ids
        if "target" in touched:
            updates["target_ids"] = proposed.target_ids
        if "ivt" in touched:
            updates["triplet_ids"] = proposed.triplet_ids
        if "phase" in touched:
            updates["phase_id"] = proposed.phase_id
        coordinated = replace(prediction, **updates)
        return CoordinationResult(
            prediction=coordinated,
            verification_status="VERIFIED_REPAIR",
            reason="REPAIR_WITHIN_CANDIDATE_POOL",
            selected_candidate_id=candidate_id_for(coordinated),
            touched_tasks=touched,
        )
