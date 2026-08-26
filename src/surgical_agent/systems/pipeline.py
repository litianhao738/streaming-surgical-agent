"""The single canonical streaming pipeline used by every experiment system."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

import torch
from torch import Tensor

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
from surgical_agent.research.signals.contracts import EvidenceProfile


class PipelineContractError(RuntimeError):
    """Raised when a component violates the frozen runner contract."""


@dataclass(frozen=True)
class ContextBundle:
    """Causal visual context and immutable snapshots; no GT is representable."""

    sample: InferenceSample
    frames: Tensor
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

    def __post_init__(self) -> None:
        if self.action not in {"ACCEPT", "VERIFY"}:
            raise ValueError(f"Unsupported Gate action: {self.action}")
        if (self.action == "VERIFY") != (self.scope is not None):
            raise ValueError("VERIFY requires one scope and ACCEPT forbids scope")


@dataclass(frozen=True)
class FinalizedEvent:
    video_id: str
    frame_id: int
    phase_id: int
    backend: str


@dataclass(frozen=True)
class PipelineRunResult:
    prediction: PredictionRecord
    evidence: EvidenceProfile
    event: FinalizedEvent
    runtime_trace: tuple[str, ...]


class ContextBuilder(Protocol):
    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
    ) -> PerceptionContext:
        raise NotImplementedError


class CandidateGenerator(Protocol):
    def build(self, prediction: InitialPrediction) -> CandidateTrace:
        raise NotImplementedError


class EvidenceSignalExtractor(Protocol):
    def extract(
        self,
        context: PerceptionContext,
        result: JointPerceptionResult,
    ) -> EvidenceProfile:
        raise NotImplementedError


class GatePolicy(Protocol):
    def decide(self, signals: EvidenceProfile) -> GateDecision:
        raise NotImplementedError


class SpecialistRegistry(Protocol):
    enabled_scopes: tuple[str, ...]

    def verify(
        self,
        scope: str,
        prediction: InitialPrediction,
    ) -> InitialPrediction:
        raise NotImplementedError


class Coordinator(Protocol):
    def coordinate(
        self,
        prediction: InitialPrediction,
        decision: GateDecision,
    ) -> InitialPrediction:
        raise NotImplementedError


class Finalizer(Protocol):
    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        prediction: InitialPrediction,
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


class FramesOnlyContextBuilder:
    """P2 context implementation: causal frames and prior snapshots only."""

    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> ContextBundle:
        if frames.ndim not in {4, 5}:
            raise PipelineContractError("Context frames must be 4D or 5D")
        return ContextBundle(
            sample=sample,
            frames=frames,
            workflow_snapshot=dict(workflow_snapshot),
            memory_snapshot=dict(memory_snapshot),
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
    def build(self, prediction: InitialPrediction) -> CandidateTrace:
        del prediction
        return CandidateTrace(status="DISABLED_TRACEABLE")


class NeverVerify:
    def decide(self, signals: EvidenceProfile) -> GateDecision:
        del signals
        return GateDecision(
            action="ACCEPT",
            scope=None,
            reason="P2_NEVER_VERIFY_POLICY",
        )


class DisabledSpecialistRegistry:
    enabled_scopes: tuple[str, ...] = ()

    def verify(self, scope: str, prediction: InitialPrediction) -> InitialPrediction:
        del scope, prediction
        raise PipelineContractError("Specialists are disabled in P2")


class NoOpCoordinator:
    """KEEP-only coordinator; it cannot generate or alter semantic content."""

    def coordinate(
        self,
        prediction: InitialPrediction,
        decision: GateDecision,
    ) -> InitialPrediction:
        if decision.action != "ACCEPT":
            raise PipelineContractError("P2 coordinator received an illegal VERIFY route")
        return prediction


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
        prediction: InitialPrediction,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
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
            verification_status="NOT_REQUESTED",
            alignment_version=sample.alignment_version,
            probabilities=prediction.probabilities,
            trace=trace,
            score_semantics=prediction.score_semantics,
        )
        event = FinalizedEvent(
            video_id=sample.video_id,
            frame_id=sample.target_frame_id,
            phase_id=prediction.phase_id,
            backend=prediction.backend,
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
    )


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
        trace.append("03_prior_state_snapshotted")
        context = self.components.context_builder.build(
            sample,
            frames,
            workflow_snapshot=workflow_snapshot,
            memory_snapshot=memory_snapshot,
            prior_finalized_prediction=self._prior_finalized_prediction,
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
        candidates = self.components.candidate_generator.build(initial)
        del candidates
        trace.append("07_candidates_built")
        decision = self.components.gate_policy.decide(evidence)
        if not isinstance(decision, GateDecision):
            raise PipelineContractError("Gate policy must return a GateDecision")
        trace.append(f"08_gate_{decision.action.lower()}")
        if decision.action != "ACCEPT":
            raise PipelineContractError(
                "This canonical pipeline slice permits only ACCEPT decisions"
            )
        trace.extend(("09_specialist_skipped", "10_keep_initial_prediction"))
        coordinated = self.components.coordinator.coordinate(initial, decision)
        if coordinated is not initial:
            raise PipelineContractError(
                "KEEP coordination must preserve prediction object identity"
            )
        trace.append("11_coordinator_keep")
        finalized = self.components.finalizer.finalize(
            run_id=run_id,
            sample=sample,
            prediction=coordinated,
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
        )
