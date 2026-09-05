"""Executable Pipeline matching the uploaded minimal research architecture."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

from torch import Tensor

from surgical_agent.data.schemas import InferenceSample
from surgical_agent.perception.context_builder import (
    CausalPerceptionContextBuilder,
    PerceptionContext,
)
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.research.gate.budget import BudgetBucket, VerificationBudgetManager
from surgical_agent.research.gate.contracts import (
    GateAction,
    RouteDecision,
    SafetySupport,
)
from surgical_agent.research.gate.formal_policy import BenefitGate
from surgical_agent.research.memory.pending import BoundedPendingResolver
from surgical_agent.research.outcome import (
    ExecutionOutcome,
    FinalOutcome,
    OutcomeFinalizer,
)
from surgical_agent.research.safety import (
    DecisionSupportBuilder,
    MandatorySafetyGuard,
    SafetyValidator,
)
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.verification.contracts import CandidateSet
from surgical_agent.research.verification.coverage import run_scope_coverage
from surgical_agent.research.verification.repair import (
    BoundedVerifyRepairLoop,
    RepairProposal,
    TargetedSpecialist,
    gate_accepted,
)
from surgical_agent.runtime.finalization import (
    AtomicFinalizationStore,
    FinalizationRecord,
)
from surgical_agent.runtime.state import ObservationClock, ObservationIdentity


class _Perception(Protocol):
    def predict(self, context: PerceptionContext) -> JointPerceptionResult: ...


class _CandidateGenerator(Protocol):
    def build(self, result: JointPerceptionResult) -> CandidateSet: ...


class _SignalExtractor(Protocol):
    def extract(
        self, context: PerceptionContext, result: JointPerceptionResult
    ) -> EvidenceProfile: ...


class _TrackProvider(Protocol):
    def reset(self, video_id: str) -> None: ...

    def snapshot(self, sample: InferenceSample) -> Mapping[str, object]: ...


SpecialistFactory = Callable[
    [PerceptionContext, CandidateSet, EvidenceProfile], TargetedSpecialist
]


@dataclass(frozen=True)
class FinalPipelineComponents:
    context_builder: CausalPerceptionContextBuilder
    perception: _Perception
    candidate_generator: _CandidateGenerator
    signal_extractor: _SignalExtractor
    safety_validator: SafetyValidator
    support_builder: DecisionSupportBuilder
    mandatory_guard: MandatorySafetyGuard
    benefit_gate: BenefitGate
    budget: VerificationBudgetManager
    specialist_factory: SpecialistFactory
    outcome_finalizer: OutcomeFinalizer
    finalization_store: AtomicFinalizationStore
    initial_model_requested: str | None = None
    verification_model_requested: str | None = None
    track_provider: _TrackProvider | None = None
    pending_resolver: BoundedPendingResolver | None = None
    max_verify_attempts: int = 1
    max_coverage_scopes: int = 1

    def __post_init__(self) -> None:
        builder = self.context_builder
        if (
            builder.max_frames != 3
            or builder.max_images != 3
            or builder.selection_strategy != "fixed_all"
        ):
            raise ValueError(
                "final Pipeline requires fixed_all causal input with max_frames=max_images=3"
            )
        if self.max_verify_attempts not in {1, 2}:
            raise ValueError("max_verify_attempts must be 1 or 2")
        if self.max_coverage_scopes not in {1, 2, 3}:
            raise ValueError("max_coverage_scopes must be one to three")
        if (self.initial_model_requested is None) != (
            self.verification_model_requested is None
        ):
            raise ValueError("Pipeline model identities must be recorded as a pair")


@dataclass(frozen=True)
class FinalPipelineRunResult:
    observation: ObservationIdentity
    outcome: FinalOutcome | ExecutionOutcome
    finalization: FinalizationRecord
    tracker_snapshot: Mapping[str, object]
    context: PerceptionContext | None = None
    evidence: EvidenceProfile | None = None
    support: SafetySupport | None = None
    route: RouteDecision | None = None
    proposal: RepairProposal | None = None
    resolved_pending: tuple[FinalizationRecord, ...] = ()


class FinalStreamingPipeline:
    """Run the uploaded Pipeline without allowing GT-bearing inputs."""

    def __init__(self, components: FinalPipelineComponents) -> None:
        if not isinstance(components, FinalPipelineComponents):
            raise TypeError("components must be FinalPipelineComponents")
        self.components = components
        self.clock = ObservationClock()
        self._video_id: str | None = None

    def reset(self, video_id: str, *, recover: bool = True) -> None:
        self.clock.reset(video_id)
        self.components.finalization_store.reset(video_id, recover=recover)
        if recover:
            self.components.budget.restore_charged_buckets(
                video_id,
                self._committed_budget_buckets(
                    self.components.finalization_store.records
                ),
            )
        else:
            self.components.budget.reset(video_id)
        if self.components.track_provider is not None:
            self.components.track_provider.reset(video_id)
        self._video_id = video_id

    @staticmethod
    def _committed_budget_buckets(
        records: tuple[FinalizationRecord, ...],
    ) -> tuple[BudgetBucket, ...]:
        buckets: list[BudgetBucket] = []
        for record in records:
            proposal = record.audit.get("proposal")
            if not isinstance(proposal, Mapping):
                continue
            attempts = proposal.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt in attempts:
                if (
                    not isinstance(attempt, Mapping)
                    or attempt.get("budget_granted") is not True
                ):
                    continue
                bucket = attempt.get("budget_bucket")
                if bucket in {"SAFETY_RESERVE", "OPTIONAL", "SHARED"}:
                    buckets.append(bucket)  # type: ignore[arg-type]
        return tuple(buckets)

    def run(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        segment_id: str = "segment-0",
        observation_time: float | None = None,
    ) -> FinalPipelineRunResult:
        if self._video_id != sample.video_id:
            self.reset(sample.video_id)
        observation = ObservationIdentity.from_sample(
            sample,
            segment_id=segment_id,
            observation_time=observation_time,
        )
        self.clock.accept(observation)
        tracker_snapshot = self.tracker_snapshot(sample)
        memory = self.components.finalization_store.snapshot()
        previous_entry = memory.trusted[-1] if memory.trusted else None
        previous = (
            previous_entry.hypothesis
            if previous_entry is not None
            and previous_entry.observation.frame_id in sample.causal_frame_ids[:-1]
            else None
        )
        previous_verified_tasks = (
            ()
            if previous is None or previous_entry is None
            else previous_entry.verified_tasks
        )
        workflow_snapshot = self._workflow_snapshot(memory.trusted)
        try:
            context = self.components.context_builder.build(
                sample,
                frames,
                workflow_snapshot=workflow_snapshot,
                memory_snapshot=memory.as_mapping(),
                prior_finalized_prediction=None,
                track_snapshot=tracker_snapshot,
            )
            perception = self.components.perception.predict(context)
        except Exception as exc:  # noqa: BLE001 - runtime errors become auditable statuses
            return self._execution_failure(
                observation,
                tracker_snapshot,
                "EXECUTION_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(perception, JointPerceptionResult):
            return self._execution_failure(
                observation,
                tracker_snapshot,
                "INVALID_RESPONSE",
                "PERCEPTION_RESULT_TYPE",
                context=context,
            )

        try:
            candidates = self.components.candidate_generator.build(perception)
            evidence = self.components.signal_extractor.extract(context, perception)
            if not isinstance(candidates, CandidateSet):
                raise TypeError("candidate generator returned the wrong type")
            if not isinstance(evidence, EvidenceProfile):
                raise TypeError("signal extractor returned the wrong type")
            violations = self.components.safety_validator.validate(
                perception.prediction
            )
            support = self.components.support_builder.build(
                perception=perception,
                candidates=candidates,
                violations=violations,
                tracker_snapshot=tracker_snapshot,
                previous=previous,
                previous_verified_tasks=previous_verified_tasks,
            )
        except Exception as exc:  # noqa: BLE001 - malformed support is an execution status
            return self._execution_failure(
                observation,
                tracker_snapshot,
                "INVALID_RESPONSE",
                f"{type(exc).__name__}: {exc}",
                context=context,
            )

        route = self.components.mandatory_guard.route(support)
        if route is None:
            try:
                action = self.components.benefit_gate.decide(support)
                route = self._gate_route(action, support)
            except Exception as exc:  # noqa: BLE001 - policy failure is explicitly modeled
                return self._execution_failure(
                    observation,
                    tracker_snapshot,
                    "POLICY_FAILURE",
                    f"{type(exc).__name__}: {exc}",
                    context=context,
                    evidence=evidence,
                    support=support,
                )

        h0 = perception.prediction
        if route.kind == "PENDING":
            proposal = RepairProposal(
                "PENDING_UNRESOLVED",
                h0,
                route.reason,
            )
        elif route.kind == "USE_H0":
            proposal = gate_accepted(h0)
        else:
            assert route.scope is not None and route.priority is not None
            specialist = self.components.specialist_factory(
                context, candidates, evidence
            )
            loop = BoundedVerifyRepairLoop(
                specialist=specialist,
                validator=self.components.safety_validator,
                budget=self.components.budget,
                max_attempts=self.components.max_verify_attempts,
            )
            proposal = (
                loop.run(
                    h0=h0,
                    candidates=candidates,
                    scope=route.scope,
                    priority=route.priority,
                    fallback_h0_allowed=route.fallback_h0_allowed,
                )
                if self.components.max_coverage_scopes == 1
                else run_scope_coverage(
                    h0=h0,
                    candidates=candidates,
                    initial_scope=route.scope,
                    specialist=specialist,
                    validator=self.components.safety_validator,
                    budget=self.components.budget,
                    max_scopes=self.components.max_coverage_scopes,
                )
            )

        outcome = self.components.outcome_finalizer.finalize(h0=h0, proposal=proposal)
        audit = self._audit_payload(
            tracker_snapshot=tracker_snapshot,
            context=context,
            candidates=candidates,
            support=support,
            route=route,
            proposal=proposal,
        )
        phase_support = getattr(
            self.components.support_builder, "phase_ivt_support", None
        )
        final_hypothesis = outcome.hypothesis
        audit["phase_ivt_check"] = {
            "soft_training_support_enabled": phase_support is not None,
            "hard_constraints_enabled": self.components.safety_validator.strict_phase_allowed_ivt
            is not None,
            "final_unobserved_ivt_ids": sorted(
                set(final_hypothesis.triplet_ids)
                - set(phase_support.get(final_hypothesis.phase_id, ()))
            )
            if phase_support is not None and final_hypothesis is not None
            else [],
        }
        finalization = self.components.finalization_store.commit(
            observation=observation,
            outcome=outcome,
            pending_scope=route.scope,
            audit=audit,
        )
        resolved_pending = (
            ()
            if self.components.pending_resolver is None
            else self.components.pending_resolver.resolve_eligible(
                now=observation,
                store=self.components.finalization_store,
            )
        )
        return FinalPipelineRunResult(
            observation=observation,
            outcome=outcome,
            finalization=finalization,
            tracker_snapshot=tracker_snapshot,
            context=context,
            evidence=evidence,
            support=support,
            route=route,
            proposal=proposal,
            resolved_pending=resolved_pending,
        )

    def tracker_snapshot(self, sample: InferenceSample) -> Mapping[str, object]:
        """Return current evidence or an explicit non-stale unavailability status."""
        provider = self.components.track_provider
        if provider is None:
            return {
                "status": "UNAVAILABLE",
                "runtime_status": "DISABLED_BY_ABLATION",
                "source_max_frame_id": None,
                "frames": (),
            }
        try:
            raw = dict(provider.snapshot(sample))
        except Exception as exc:  # noqa: BLE001 - Tracker failure must not reuse stale state
            return {
                "status": "UNAVAILABLE",
                "runtime_status": "ERROR",
                "source_max_frame_id": None,
                "frames": (),
                "error_type": type(exc).__name__,
            }
        frames = raw.get("frames", ())
        last_tracks: Any = None
        if (
            isinstance(frames, (tuple, list))
            and frames
            and isinstance(frames[-1], Mapping)
        ):
            last_tracks = frames[-1].get("tracks", ())
        empty_current = isinstance(last_tracks, (tuple, list)) and not last_tracks
        raw["runtime_status"] = "AVAILABLE_EMPTY" if empty_current else "OK"
        return raw

    @staticmethod
    def _gate_route(action: GateAction, support: SafetySupport) -> RouteDecision:
        if not isinstance(action, GateAction):
            raise TypeError("Benefit Gate returned the wrong type")
        if action.kind == "USE_H0":
            return RouteDecision(
                "USE_H0",
                "BENEFIT_GATE",
                action.reason,
                fallback_h0_allowed=True,
            )
        if action.scope not in support.legal_scopes:
            raise ValueError("Benefit Gate requested an unavailable scope")
        return RouteDecision(
            "VERIFY",
            "BENEFIT_GATE",
            action.reason,
            scope=action.scope,
            priority="OPTIONAL",
            fallback_h0_allowed=True,
        )

    @staticmethod
    def _workflow_snapshot(trusted: tuple[Any, ...]) -> Mapping[str, object]:
        phase_entries = trusted[-16:]
        phases = tuple(str(item.hypothesis.phase_id) for item in phase_entries)
        phase_states = tuple(item.task_states["phase"] for item in phase_entries)
        transitions = tuple(pairwise(phases))
        stability = None if not phases else phases.count(phases[-1]) / len(phases)
        return {
            "status": "FINALIZED_ONLY",
            "source_max_frame_id": (
                phase_entries[-1].observation.frame_id if phase_entries else None
            ),
            "recent_finalized_phases": phases,
            "recent_phase_states": phase_states,
            "phase_stability": stability,
            "observed_transitions": transitions,
        }

    @staticmethod
    def _audit_payload(
        *,
        tracker_snapshot: Mapping[str, object],
        context: PerceptionContext,
        candidates: CandidateSet,
        support: SafetySupport,
        route: RouteDecision,
        proposal: RepairProposal,
    ) -> Mapping[str, object]:
        return {
            "schema_version": "final_pipeline_frame_audit_v1",
            "causal_frame_ids": list(context.sample.causal_frame_ids),
            "selected_image_frame_ids": list(context.selected_image_frame_ids),
            "candidate_set": {
                "version": candidates.candidate_version,
                "allowed_ids": {
                    task: list(candidates.allowed_ids[task])
                    for task in candidates.allowed_ids
                },
            },
            "tracker_runtime_status": tracker_snapshot.get("runtime_status"),
            "safety_class": support.safety_class,
            "hard_violations": [
                {"code": item.code, "tasks": list(item.tasks)}
                for item in support.hard_violations
            ],
            "soft_risks": [
                {"code": item.code, "tasks": list(item.tasks), "value": item.value}
                for item in support.soft_risks
            ],
            "gate_features": dict(support.gate_features),
            "route": {
                "kind": route.kind,
                "source": route.source,
                "scope": route.scope,
                "priority": route.priority,
                "fallback_h0_allowed": route.fallback_h0_allowed,
                "reason": route.reason,
            },
            "proposal": {
                "status": proposal.status,
                "reason": proposal.reason,
                "attempts": [
                    {
                        "attempt": item.attempt,
                        "scope": item.scope,
                        "budget_granted": item.budget.granted,
                        "budget_bucket": item.budget.bucket,
                        "budget_reason": item.budget.reason,
                        "specialist_status": item.specialist_status,
                        "specialist_reason": item.specialist_reason,
                        "accepted": item.accepted,
                        "candidate_labels": {
                            task: list(values) for task, values in item.candidate_labels
                        },
                        "certified_task_values": {
                            task: list(values)
                            for task, values in item.certified_task_values
                        },
                        "candidate_id": item.candidate_id,
                        "postcheck_hard_valid": item.postcheck_hard_valid,
                        "violation_codes": list(item.violation_codes),
                        "repair_evidence_kind": item.repair_evidence_kind,
                        "repair_evidence_sources": list(item.repair_evidence_sources),
                    }
                    for item in proposal.attempts
                ],
            },
        }

    def _execution_failure(
        self,
        observation: ObservationIdentity,
        tracker_snapshot: Mapping[str, object],
        status: str,
        reason: str,
        *,
        context: PerceptionContext | None = None,
        evidence: EvidenceProfile | None = None,
        support: SafetySupport | None = None,
    ) -> FinalPipelineRunResult:
        outcome = ExecutionOutcome(status, reason)  # type: ignore[arg-type]
        finalization = self.components.finalization_store.commit(
            observation=observation,
            outcome=outcome,
            audit={
                "schema_version": "final_pipeline_frame_audit_v1",
                "tracker_runtime_status": tracker_snapshot.get("runtime_status"),
                "execution_failure": {"status": status, "reason": reason},
            },
        )
        return FinalPipelineRunResult(
            observation=observation,
            outcome=outcome,
            finalization=finalization,
            tracker_snapshot=tracker_snapshot,
            context=context,
            evidence=evidence,
            support=support,
        )


__all__ = [
    "FinalPipelineComponents",
    "FinalPipelineRunResult",
    "FinalStreamingPipeline",
]
