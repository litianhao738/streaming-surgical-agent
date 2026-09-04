"""Adapters that expose a strict KEEP/REPAIR Specialist contract."""

from __future__ import annotations

from dataclasses import dataclass, replace

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.research.gate.contracts import SCOPE_TASKS, RepairScope
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    VerificationResult,
)
from surgical_agent.research.verification.repair import SpecialistResult


@dataclass
class BoundTargetedSpecialist:
    """Bind observation-local inputs around the existing targeted API verifier."""

    verifier: object
    context: PerceptionContext
    candidates: CandidateSet
    evidence: EvidenceProfile

    def __post_init__(self) -> None:
        if not hasattr(self.verifier, "verify") or not callable(self.verifier.verify):
            raise TypeError("verifier must implement verify")
        if not isinstance(self.context, PerceptionContext):
            raise TypeError("context must be PerceptionContext")
        if not isinstance(self.candidates, CandidateSet):
            raise TypeError("candidates must be CandidateSet")
        if not isinstance(self.evidence, EvidenceProfile):
            raise TypeError("evidence must be EvidenceProfile")

    def verify(
        self,
        *,
        hypothesis: InitialPrediction,
        scope: RepairScope,
        attempt: int,
    ) -> SpecialistResult:
        del attempt  # attempt identity is logged by the bounded loop, not sent as semantics
        if not self.candidates.admits(hypothesis):
            return SpecialistResult("INVALID_RESPONSE", reason="INPUT_OUTSIDE_POOL")
        # Interaction is selected IVT-first.  Instrument, verb and target are
        # then derived from the selected IVTs, so a factorized model response
        # cannot construct a non-closed joint hypothesis.
        requested = (
            ("ivt",)
            if scope == "interaction"
            else tuple(task for task in SCOPE_TASKS[scope])
        )
        # Preserve the canonical five-head order expected by the parser.
        requested = tuple(
            task for task in ("instrument", "verb", "target", "ivt", "phase")
            if task in requested
        )
        try:
            result = self.verifier.verify(
                scope,
                self.context,
                hypothesis,
                self.candidates,
                self.evidence,
                requested_fields=requested,
            )
        except Exception as exc:  # noqa: BLE001 - provider failure is not semantic rejection
            return SpecialistResult("NO_RESPONSE", reason=type(exc).__name__)
        if not isinstance(result, VerificationResult):
            return SpecialistResult("INVALID_RESPONSE", reason="WRONG_RESULT_TYPE")
        statuses = {item.status for item in result.field_outcomes}
        if any(item.uncertainty is not None for item in result.field_outcomes):
            return SpecialistResult("UNRESOLVED", reason="SPECIALIST_UNCERTAIN")
        if "Rejected" in statuses:
            return SpecialistResult("REJECT", reason="SPECIALIST_REJECTED")
        if "Pending" in statuses:
            return SpecialistResult("UNRESOLVED", reason="SPECIALIST_PENDING")
        if statuses and statuses != {"Verified"}:
            return SpecialistResult("INVALID_RESPONSE", reason="INVALID_FIELD_STATUS")
        proposed = result.prediction
        if scope == "interaction":
            represented = {"instrument": set(), "verb": set(), "target": set()}
            components = load_ivt_components()
            for ivt_id in proposed.triplet_ids:
                instrument_id, verb_id, target_id = components[ivt_id]
                represented["instrument"].add(instrument_id)
                represented["verb"].add(verb_id)
                represented["target"].add(target_id)
            proposed = replace(
                proposed,
                instrument_ids=tuple(sorted(represented["instrument"])),
                verb_ids=tuple(sorted(represented["verb"])),
                target_ids=tuple(sorted(represented["target"])),
            )
            if not self.candidates.admits(proposed):
                return SpecialistResult(
                    "INVALID_RESPONSE",
                    reason="DERIVED_INTERACTION_OUTSIDE_POOL",
                )
        if proposed == hypothesis:
            return SpecialistResult("KEEP", reason="SPECIALIST_KEEP")
        return SpecialistResult(
            "REPAIR",
            hypothesis=proposed,
            reason="SPECIALIST_REPAIR",
        )


__all__ = ["BoundTargetedSpecialist"]
