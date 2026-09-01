"""The single canonical streaming pipeline used by every experiment system."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol

import torch
from torch import Tensor

from surgical_agent.api.errors import ApiProviderCallBudgetError
from surgical_agent.data.schemas import InferenceSample
from surgical_agent.inference.engine import require_inference_sample
from surgical_agent.inference.frame_result_writer import FrameResultSink
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.models.baseline import LocalSmokeModel, decode_frame_logits
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionBackend,
    PerceptionEvidence,
)
from surgical_agent.research.reliability.state import (
    ALL_TASK_FIELDS,
    FINDING_REASONS,
    INITIAL_STATE,
    GateFinding,
    canonical_tasks,
    final_status_for,
    memory_action_for,
    validate_status_memory_action,
)
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.verification.contracts import (
    CoordinationResult,
    VerificationResult,
)


class PipelineContractError(RuntimeError):
    """Raised when a component violates the frozen runner contract."""


class GateObserver(Protocol):
    """Optional gold-free research hook invoked before persistence."""

    def observe(
        self,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
        evidence: EvidenceProfile,
        decision: GateDecision,
        coordinated: CoordinationResult,
    ) -> None: ...


@dataclass(frozen=True)
class ContextBundle:
    """Causal visual context and immutable snapshots; no GT is representable."""

    sample: InferenceSample
    frames: Tensor
    track_snapshot: Mapping[str, Any]
    workflow_snapshot: Mapping[str, Any]
    memory_snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateTrace:
    status: str
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    action: str
    scope: str | None
    reason: str
    findings: tuple[GateFinding, ...] = ()
    flagged_fields: tuple[str, ...] = ()
    selected_confidence_floor: float | None = None
    benefit_probability: float | None = None

    def __post_init__(self) -> None:
        if self.action not in {"ACCEPT", "VERIFY"}:
            raise ValueError(f"Unsupported Gate action: {self.action}")
        if (self.action == "VERIFY") != (self.scope is not None):
            raise ValueError("VERIFY requires one scope and ACCEPT forbids scope")
        findings = tuple(self.findings)
        if any(not isinstance(finding, GateFinding) for finding in findings):
            raise TypeError("findings must contain GateFinding values")
        flagged_fields = canonical_tasks(tuple(self.flagged_fields))
        if self.action == "VERIFY" and not flagged_fields:
            flagged_fields = ALL_TASK_FIELDS
        if self.action == "ACCEPT" and (findings or flagged_fields):
            raise ValueError("ACCEPT forbids findings and flagged fields")
        confidence = self.selected_confidence_floor
        if confidence is not None and (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("selected confidence floor must lie in [0, 1]")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "flagged_fields", flagged_fields)
        if confidence is not None:
            object.__setattr__(self, "selected_confidence_floor", float(confidence))
        probability = self.benefit_probability
        if probability is not None and (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("benefit_probability must lie in [0, 1]")
        if probability is not None:
            object.__setattr__(self, "benefit_probability", float(probability))


@dataclass(frozen=True)
class FinalizedEvent:
    video_id: str
    frame_id: int
    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    triplet_ids: tuple[int, ...]
    phase_id: int
    backend: str
    gate_action: str
    verification_status: str
    score_semantics: str
    initial_state: str = INITIAL_STATE
    gate_reasons: tuple[str, ...] = ()
    flagged_fields: tuple[str, ...] = ()
    repaired_fields: tuple[str, ...] = ()
    final_status: str = INITIAL_STATE
    memory_action: str = "SKIP"

    def __post_init__(self) -> None:
        if self.initial_state != INITIAL_STATE:
            raise ValueError("initial_state must be Candidate")
        gate_reasons = tuple(self.gate_reasons)
        if any(reason not in FINDING_REASONS for reason in gate_reasons):
            raise ValueError("gate_reasons contains an unsupported reason")
        flagged_fields = canonical_tasks(tuple(self.flagged_fields))
        repaired_fields = canonical_tasks(tuple(self.repaired_fields))
        if not set(repaired_fields).issubset(flagged_fields):
            raise ValueError("repaired_fields must be a subset of flagged_fields")
        validate_status_memory_action(self.final_status, self.memory_action)
        object.__setattr__(self, "gate_reasons", gate_reasons)
        object.__setattr__(self, "flagged_fields", flagged_fields)
        object.__setattr__(self, "repaired_fields", repaired_fields)


@dataclass(frozen=True)
class PipelineRunResult:
    prediction: PredictionRecord
    evidence: EvidenceProfile
    event: FinalizedEvent
    runtime_trace: tuple[str, ...]
    selected_image_frame_ids: tuple[int, ...] = ()
    temporal_evidence: Mapping[str, Any] = field(default_factory=dict)


class ContextBuilder(Protocol):
    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
        track_snapshot: Mapping[str, Any],
    ) -> PerceptionContext:
        raise NotImplementedError


class CandidateGenerator(Protocol):
    def build(self, result: JointPerceptionResult) -> object:
        raise NotImplementedError


class EvidenceSignalExtractor(Protocol):
    def extract(
        self,
        context: PerceptionContext,
        result: JointPerceptionResult,
    ) -> EvidenceProfile:
        raise NotImplementedError


class GatePolicy(Protocol):
    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
    ) -> GateDecision:
        raise NotImplementedError


class SpecialistRegistry(Protocol):
    enabled_scopes: tuple[str, ...]

    def verify(
        self,
        scope: str,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: object,
        evidence: EvidenceProfile,
    ) -> VerificationResult:
        raise NotImplementedError


class Coordinator(Protocol):
    def coordinate(
        self,
        prediction: InitialPrediction,
        decision: GateDecision,
        candidates: object,
        verification: VerificationResult | None,
    ) -> CoordinationResult:
        raise NotImplementedError


class Finalizer(Protocol):
    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        coordinated: CoordinationResult,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
        raise NotImplementedError


class CausalStore(Protocol):
    def reset(self, video_id: str) -> None:
        raise NotImplementedError

    def snapshot(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def update(self, event: FinalizedEvent) -> None:
        raise NotImplementedError


class TrackContextProvider(Protocol):
    def reset(self, video_id: str) -> None:
        raise NotImplementedError

    def snapshot(self, sample: InferenceSample) -> Mapping[str, Any]:
        raise NotImplementedError


class FramesOnlyContextBuilder:
    """P2 context implementation: causal frames and prior snapshots only."""

    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
        track_snapshot: Mapping[str, Any],
    ) -> PerceptionContext:
        from surgical_agent.perception.context_builder import (
            CausalPerceptionContextBuilder,
        )

        return CausalPerceptionContextBuilder(max_frames=3).build(
            sample,
            frames,
            workflow_snapshot=workflow_snapshot,
            memory_snapshot=memory_snapshot,
            prior_finalized_prediction=prior_finalized_prediction,
            track_snapshot=track_snapshot,
        )


class LocalSmokePerception:
    """Adapter exposing the local P2 model through the perception slot."""

    def __init__(
        self,
        model: LocalSmokeModel,
        *,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold

    def predict(self, context: PerceptionContext) -> JointPerceptionResult:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(context.frames.unsqueeze(0).to(self.device))
        decoded = decode_frame_logits(logits, threshold=self.threshold)
        if len(decoded) != 1:
            raise PipelineContractError("Canonical per-sample runner expects batch size 1")
        return JointPerceptionResult(
            prediction=decoded[0],
            raw_evidence=PerceptionEvidence.local_unavailable(
                context.sample.target_frame_id
            ),
            api_provenance=ApiCallProvenance.local(),
        )


class DisabledCandidateGenerator:
    def build(self, result: JointPerceptionResult) -> CandidateTrace:
        del result
        return CandidateTrace(status="DISABLED_TRACEABLE")


class NeverVerify:
    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext | None = None,
        perception_result: JointPerceptionResult | None = None,
    ) -> GateDecision:
        del signals, context, perception_result
        return GateDecision(
            action="ACCEPT",
            scope=None,
            reason="P2_NEVER_VERIFY_POLICY",
        )


class DisabledSpecialistRegistry:
    enabled_scopes: tuple[str, ...] = ()

    def verify(
        self,
        scope: str,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: object,
        evidence: EvidenceProfile,
    ) -> VerificationResult:
        del scope, context, prediction, candidates, evidence
        raise PipelineContractError("Specialists are disabled in P2")


class NoOpCoordinator:
    """KEEP-only coordinator; it cannot generate or alter semantic content."""

    def coordinate(
        self,
        prediction: InitialPrediction,
        decision: GateDecision,
        candidates: object,
        verification: VerificationResult | None,
    ) -> CoordinationResult:
        del candidates, verification
        if decision.action == "ACCEPT":
            return CoordinationResult(
                prediction=prediction,
                verification_status="NOT_REQUESTED",
                reason="GATE_ACCEPTED",
            )
        return CoordinationResult(
            prediction=prediction,
            verification_status="FALLBACK_KEEP",
            reason="VERIFICATION_NOT_CONFIGURED",
        )


class NoOpCausalStore:
    """Traceable per-video state slot with no semantic payload."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.video_id: str | None = None
        self.reset_history: list[str] = []
        self.committed_frames: list[int] = []

    def reset(self, video_id: str) -> None:
        self.video_id = video_id
        self.committed_frames = []
        self.reset_history.append(video_id)

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "component": self.name,
            "video_id": self.video_id,
            "committed_frame_count": len(self.committed_frames),
            "status": "NOOP_TRACEABLE",
        }

    def update(self, event: FinalizedEvent) -> None:
        if self.video_id != event.video_id:
            raise PipelineContractError(f"{self.name} commit crossed a video boundary")
        if self.committed_frames and event.frame_id <= self.committed_frames[-1]:
            raise PipelineContractError(f"{self.name} received non-increasing frame IDs")
        self.committed_frames.append(event.frame_id)


class PredictionFinalizer:
    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        coordinated: CoordinationResult | None = None,
        prediction: InitialPrediction | None = None,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
        if coordinated is None:
            if prediction is None:
                raise TypeError("finalize requires coordinated or prediction")
            if decision.action != "ACCEPT":
                raise ValueError(
                    "legacy prediction finalization is valid only for ACCEPT"
                )
            coordinated = CoordinationResult(
                prediction=prediction,
                verification_status="NOT_REQUESTED",
                reason="GATE_ACCEPTED",
            )
        elif prediction is not None:
            raise ValueError("finalize accepts only one prediction input")
        prediction = coordinated.prediction
        final_status = final_status_for(coordinated.verification_status)
        gate_reasons = tuple(finding.reason for finding in decision.findings)
        flagged_fields = decision.flagged_fields
        repaired_fields = canonical_tasks(tuple(coordinated.touched_tasks))
        memory_action = memory_action_for(
            final_status,
            selected_confidence_floor=decision.selected_confidence_floor,
            has_gate_findings=bool(decision.findings),
        )
        record = PredictionRecord(
            run_id=run_id,
            video_id=sample.video_id,
            frame_id=sample.target_frame_id,
            source_split=sample.source_split,
            causal_frame_ids=sample.causal_frame_ids,
            instrument_ids=prediction.instrument_ids,
            verb_ids=prediction.verb_ids,
            target_ids=prediction.target_ids,
            triplet_ids=prediction.triplet_ids,
            phase_id=prediction.phase_id,
            granularity=prediction.granularity,
            backend=prediction.backend,
            gate_action=decision.action,
            verification_status=coordinated.verification_status,
            alignment_version=sample.alignment_version,
            probabilities=prediction.probabilities,
            trace=trace,
            score_semantics=prediction.score_semantics,
            initial_state=INITIAL_STATE,
            gate_reasons=gate_reasons,
            flagged_fields=flagged_fields,
            repaired_fields=repaired_fields,
            final_status=final_status,
            memory_action=memory_action,
        )
        event = FinalizedEvent(
            video_id=sample.video_id,
            frame_id=sample.target_frame_id,
            instrument_ids=prediction.instrument_ids,
            verb_ids=prediction.verb_ids,
            target_ids=prediction.target_ids,
            triplet_ids=prediction.triplet_ids,
            phase_id=prediction.phase_id,
            backend=prediction.backend,
            gate_action=decision.action,
            verification_status=coordinated.verification_status,
            score_semantics=prediction.score_semantics,
            initial_state=INITIAL_STATE,
            gate_reasons=gate_reasons,
            flagged_fields=flagged_fields,
            repaired_fields=repaired_fields,
            final_status=final_status,
            memory_action=memory_action,
        )
        return record, event


def _freeze_prediction_record(record: PredictionRecord) -> PredictionRecord:
    """Detach every container before the record crosses the durable boundary."""

    return replace(
        record,
        causal_frame_ids=tuple(record.causal_frame_ids),
        instrument_ids=tuple(record.instrument_ids),
        verb_ids=tuple(record.verb_ids),
        target_ids=tuple(record.target_ids),
        triplet_ids=tuple(record.triplet_ids),
        probabilities=MappingProxyType(
            {
                task: tuple(values)
                for task, values in record.probabilities.items()
            }
        ),
        trace=tuple(record.trace),
        gate_reasons=tuple(record.gate_reasons),
        flagged_fields=tuple(record.flagged_fields),
        repaired_fields=tuple(record.repaired_fields),
    )


def _decide_gate(
    policy: GatePolicy,
    signals: EvidenceProfile,
    *,
    context: PerceptionContext,
    perception_result: JointPerceptionResult,
) -> GateDecision:
    """Pass the richer contract when supported without masking policy errors."""

    method = policy.decide
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(signals)  # type: ignore[call-arg]
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or "context" in parameters:
        kwargs["context"] = context
    if accepts_kwargs or "perception_result" in parameters:
        kwargs["perception_result"] = perception_result
    return method(signals, **kwargs)  # type: ignore[arg-type]


def _validate_finalizer_output(
    record: object,
    event: object,
    *,
    run_id: str,
    sample: InferenceSample,
) -> tuple[PredictionRecord, FinalizedEvent]:
    if not isinstance(record, PredictionRecord) or not isinstance(
        event, FinalizedEvent
    ):
        raise PipelineContractError(
            "finalizer output must contain a PredictionRecord and FinalizedEvent"
        )
    sample_identity = (sample.video_id, sample.target_frame_id)
    if (
        record.run_id != run_id
        or (record.video_id, record.frame_id) != sample_identity
        or (event.video_id, event.frame_id) != sample_identity
        or (record.video_id, record.frame_id)
        != (event.video_id, event.frame_id)
    ):
        raise PipelineContractError(
            "finalizer output identity does not match the current run and sample"
        )
    record_state = (
        record.initial_state,
        record.gate_reasons,
        record.flagged_fields,
        record.repaired_fields,
        record.final_status,
        record.memory_action,
    )
    event_state = (
        event.initial_state,
        event.gate_reasons,
        event.flagged_fields,
        event.repaired_fields,
        event.final_status,
        event.memory_action,
    )
    if record_state != event_state:
        raise PipelineContractError(
            "finalizer prediction/event reliability state does not match"
        )
    try:
        validate_status_memory_action(record.final_status, record.memory_action)
    except ValueError:
        raise PipelineContractError(
            "finalizer output reliability state is invalid"
        ) from None
    try:
        frozen_record = _freeze_prediction_record(record)
    except (TypeError, ValueError):
        raise PipelineContractError(
            "finalizer output could not be normalized safely"
        ) from None
    return frozen_record, event


@dataclass(frozen=True)
class PipelineComponents:
    context_builder: ContextBuilder
    perception: PerceptionBackend
    candidate_generator: CandidateGenerator
    signal_extractor: EvidenceSignalExtractor
    gate_policy: GatePolicy
    specialist_registry: SpecialistRegistry
    coordinator: Coordinator
    finalizer: Finalizer
    workflow_store: CausalStore
    event_memory: CausalStore
    result_sink: FrameResultSink
    track_provider: TrackContextProvider | None = None
    backbone_policy: str = "shared"
    gate_observer: GateObserver | None = None

    def __post_init__(self) -> None:
        if self.backbone_policy not in {"shared", "cascade_efficiency"}:
            raise ValueError("unsupported backbone policy")


class CanonicalStreamingPipeline:
    """Frozen per-sample runner; component assembly is the only variation point."""

    def __init__(self, components: PipelineComponents) -> None:
        self.components = components
        self._active_video_id: str | None = None
        self._prior_finalized_prediction: PredictionRecord | None = None
        self._faulted = False

    @property
    def prior_finalized_prediction(self) -> PredictionRecord | None:
        return self._prior_finalized_prediction

    def _reset_for_video(self, video_id: str) -> None:
        try:
            self.components.workflow_store.reset(video_id)
            self.components.event_memory.reset(video_id)
            if self.components.track_provider is not None:
                self.components.track_provider.reset(video_id)
        except Exception:  # noqa: BLE001 - every store failure is fail-stop
            self._faulted = True
            raise PipelineContractError(
                "Video boundary reset failed; pipeline is permanently faulted"
            ) from None
        self._prior_finalized_prediction = None
        self._active_video_id = video_id

    def run(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        run_id: str,
    ) -> PipelineRunResult:
        """Execute the canonical order without accepting any GT-bearing object."""

        if self._faulted:
            raise PipelineContractError("Canonical pipeline is permanently faulted")
        sample = require_inference_sample(sample)
        trace: list[str] = []
        if sample.video_id != self._active_video_id:
            self._reset_for_video(sample.video_id)
            trace.append("01_video_boundary_reset")
        else:
            trace.append("01_video_boundary_continue")
        trace.append("02_gold_free_sample_resolved")

        workflow_snapshot = self.components.workflow_store.snapshot()
        memory_snapshot = self.components.event_memory.snapshot()
        track_snapshot: Mapping[str, Any] = {}
        if self.components.track_provider is not None:
            try:
                track_snapshot = self.components.track_provider.snapshot(sample)
            except Exception:  # noqa: BLE001 - invalid track state is fail-stop
                self._faulted = True
                raise PipelineContractError(
                    "Predicted track context failed; pipeline is permanently faulted"
                ) from None
        trace.append("03_prior_state_snapshotted")
        context = self.components.context_builder.build(
            sample,
            frames,
            workflow_snapshot=workflow_snapshot,
            memory_snapshot=memory_snapshot,
            prior_finalized_prediction=self._prior_finalized_prediction,
            track_snapshot=track_snapshot,
        )
        trace.append("04_causal_context_built")
        perception_result = self.components.perception.predict(context)
        if not isinstance(perception_result, JointPerceptionResult):
            raise PipelineContractError(
                "Perception backend must return a JointPerceptionResult"
            )
        trace.append("05_perception_validated")
        evidence = self.components.signal_extractor.extract(
            context,
            perception_result,
        )
        if not isinstance(evidence, EvidenceProfile):
            raise PipelineContractError(
                "Evidence signal extractor must return an EvidenceProfile"
            )
        trace.append("06_evidence_profile_built")
        initial = perception_result.prediction
        candidates = self.components.candidate_generator.build(perception_result)
        trace.append("07_candidates_built")
        decision = _decide_gate(
            self.components.gate_policy,
            evidence,
            context=context,
            perception_result=perception_result,
        )
        if not isinstance(decision, GateDecision):
            raise PipelineContractError("Gate policy must return a GateDecision")
        trace.append(f"08_gate_{decision.action.lower()}")
        verification: VerificationResult | None = None
        if decision.action == "ACCEPT":
            trace.extend(("09_specialist_skipped", "10_keep_initial_prediction"))
        elif decision.scope not in self.components.specialist_registry.enabled_scopes:
            trace.extend(("09_specialist_unavailable", "10_keep_initial_prediction"))
        else:
            assert decision.scope is not None
            try:
                verify = self.components.specialist_registry.verify
                if "requested_fields" in inspect.signature(verify).parameters:
                    verification = verify(
                        decision.scope,
                        context,
                        initial,
                        candidates,
                        evidence,
                        requested_fields=decision.flagged_fields,
                    )
                else:
                    verification = verify(
                        decision.scope,
                        context,
                        initial,
                        candidates,
                        evidence,
                    )
            except ApiProviderCallBudgetError:
                raise
            except Exception:  # noqa: BLE001 - fixed-vocabulary fallback hides provider detail
                trace.extend(
                    ("09_specialist_call_failed", "10_keep_initial_prediction")
                )
            else:
                if not isinstance(verification, VerificationResult):
                    verification = None
                    trace.extend(
                        ("09_specialist_result_invalid", "10_keep_initial_prediction")
                    )
                elif not _same_backend_identity(
                    perception_result.api_provenance,
                    verification.provenance,
                    backbone_policy=self.components.backbone_policy,
                ):
                    verification = None
                    trace.extend(
                        (
                            "09_specialist_backend_mismatch",
                            "10_keep_initial_prediction",
                        )
                    )
                else:
                    trace.extend(
                        (
                            f"09_specialist_called_{decision.scope}",
                            "10_specialist_proposal_received",
                        )
                    )
        coordinated = self.components.coordinator.coordinate(
            initial,
            decision,
            candidates,
            verification,
        )
        if not isinstance(coordinated, CoordinationResult):
            raise PipelineContractError(
                "Coordinator must return a CoordinationResult"
            )
        if (
            coordinated.verification_status
            in {
                "NOT_REQUESTED",
                "VERIFIED_KEEP",
                "VERIFIED_PENDING",
                "VERIFIED_REJECT",
                "FALLBACK_KEEP",
            }
            and coordinated.prediction is not initial
        ):
            raise PipelineContractError(
                "KEEP coordination must preserve prediction object identity"
            )
        if self.components.gate_observer is not None:
            self.components.gate_observer.observe(
                context=context,
                perception_result=perception_result,
                evidence=evidence,
                decision=decision,
                coordinated=coordinated,
            )
        trace.append(
            f"11_coordinator_{coordinated.verification_status.lower()}"
        )
        trace.append(f"11_reason_{coordinated.reason.lower()}")
        finalized = self.components.finalizer.finalize(
            run_id=run_id,
            sample=sample,
            coordinated=coordinated,
            decision=decision,
            trace=(*trace, "12_prediction_finalized"),
        )
        if not isinstance(finalized, tuple) or len(finalized) != 2:
            raise PipelineContractError(
                "finalizer output must be a prediction/event pair"
            )
        record, event = _validate_finalizer_output(
            finalized[0],
            finalized[1],
            run_id=run_id,
            sample=sample,
        )
        trace.append("12_prediction_finalized")
        self.components.result_sink.write(record, evidence)
        trace.append("13_result_pair_persisted")
        try:
            self.components.workflow_store.update(event)
            self.components.event_memory.update(event)
        except Exception:  # noqa: BLE001 - every store failure is fail-stop
            self._prior_finalized_prediction = record
            self._faulted = True
            raise PipelineContractError(
                "Post-persistence causal state update failed; "
                "pipeline is permanently faulted"
            ) from None
        self._prior_finalized_prediction = record
        trace.append("14_prior_state_committed")
        return PipelineRunResult(
            prediction=record,
            evidence=evidence,
            event=event,
            runtime_trace=tuple(trace),
            selected_image_frame_ids=context.selected_image_frame_ids,
            temporal_evidence=context.temporal_evidence,
        )


def _same_backend_identity(
    perception: ApiCallProvenance,
    verification: ApiCallProvenance,
    *,
    backbone_policy: str = "shared",
) -> bool:
    """Compare safe provider identity fields when both roles expose them."""

    if perception.provider != verification.provider:
        return False
    if perception.endpoint_identifier != verification.endpoint_identifier:
        return False
    if backbone_policy == "cascade_efficiency":
        return (
            perception.requested_model_identifier == "openai/gpt-5.6-luna"
            and verification.requested_model_identifier == "openai/gpt-5.6-sol"
        )
    if backbone_policy != "shared":
        return False
    for field_name in (
        "requested_model_identifier",
        "returned_model_identifier",
    ):
        first = getattr(perception, field_name)
        second = getattr(verification, field_name)
        if first is not None and second is not None and first != second:
            return False
    return True
