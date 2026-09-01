"""Canonical Joint Perception, evidence, and durable-history integration tests."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.frame_result_writer import (
    FrameResultWriter,
    prediction_record_sha256,
)
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.perception.context_builder import (
    CausalPerceptionContextBuilder,
    PerceptionContext,
)
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
)
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    EvidenceRecord,
)
from surgical_agent.research.signals.frame_evidence import (
    FrameEvidenceSignalExtractor,
)
from surgical_agent.research.verification.contracts import CoordinationResult
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    DisabledCandidateGenerator,
    DisabledSpecialistRegistry,
    FinalizedEvent,
    GateDecision,
    LocalSmokePerception,
    NeverVerify,
    NoOpCausalStore,
    NoOpCoordinator,
    PipelineComponents,
    PipelineContractError,
    PredictionFinalizer,
)

_INVALID_STATE_ACTION_PAIRS = (
    ("Candidate", "WRITE_RELIABLE"),
    ("Candidate", "WRITE_SHORT_TERM"),
    ("Candidate", "BUFFER_PENDING"),
    ("Rejected", "WRITE_RELIABLE"),
    ("Rejected", "WRITE_SHORT_TERM"),
    ("Rejected", "BUFFER_PENDING"),
    ("Pending", "WRITE_RELIABLE"),
    ("Pending", "WRITE_SHORT_TERM"),
    ("Pending", "SKIP"),
    ("Verified", "WRITE_SHORT_TERM"),
    ("Verified", "BUFFER_PENDING"),
    ("Verified", "SKIP"),
    ("Accepted", "BUFFER_PENDING"),
    ("Accepted", "SKIP"),
)


def _sample(
    *,
    video_id: str = "VID02",
    frame_id: int = 12,
    frame_count: int = 3,
) -> InferenceSample:
    causal_frame_ids = tuple(range(frame_id - frame_count + 1, frame_id + 1))
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_id,
        causal_frame_ids=causal_frame_ids,
        media_refs=tuple(f"{value}.png" for value in causal_frame_ids),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="integration-test",
    )


def _frames(frame_count: int = 3) -> torch.Tensor:
    return torch.full((frame_count, 3, 8, 8), 0.25)


def _initial_prediction() -> InitialPrediction:
    return InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(),
        phase_id=0,
        probabilities={
            task: (0.0,) * class_count
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
        backend="integration-test",
        score_semantics="probability_v1",
    )


class RecordingPerception:
    def __init__(self, prediction: InitialPrediction | None = None) -> None:
        self.prediction = prediction or _initial_prediction()
        self.contexts: list[PerceptionContext] = []
        self.results: list[JointPerceptionResult] = []

    def predict(self, context: PerceptionContext) -> JointPerceptionResult:
        self.contexts.append(context)
        result = JointPerceptionResult(
            prediction=self.prediction,
            raw_evidence=PerceptionEvidence.local_unavailable(
                context.sample.target_frame_id
            ),
            api_provenance=ApiCallProvenance.local(),
        )
        self.results.append(result)
        return result


class RecordingSink:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.attempt_count = 0
        self.calls: list[tuple[PredictionRecord, EvidenceProfile]] = []

    def write(
        self,
        prediction: PredictionRecord,
        evidence: EvidenceProfile,
    ) -> EvidenceRecord:
        self.attempt_count += 1
        if self.attempt_count == self.fail_on_call:
            raise ArtifactWriteError("injected paired-write failure")
        self.calls.append((prediction, evidence))
        return EvidenceRecord(
            run_id=prediction.run_id,
            video_id=evidence.video_id,
            frame_id=evidence.frame_id,
            prediction_sha256=prediction_record_sha256(prediction),
            task_values=evidence.task_values,
            global_values=evidence.global_values,
            evidence_version=evidence.evidence_version,
        )


class RecordingFinalizer:
    def __init__(self) -> None:
        self._delegate = PredictionFinalizer()
        self.received_predictions: list[InitialPrediction] = []

    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        coordinated: CoordinationResult,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
        self.received_predictions.append(coordinated.prediction)
        return self._delegate.finalize(
            run_id=run_id,
            sample=sample,
            coordinated=coordinated,
            decision=decision,
            trace=trace,
        )


class FaultingStore(NoOpCausalStore):
    def __init__(
        self,
        name: str,
        *,
        fail_reset: bool = False,
        fail_update: bool = False,
    ) -> None:
        super().__init__(name)
        self.fail_reset = fail_reset
        self.fail_update = fail_update
        self.reset_attempt_count = 0
        self.update_attempt_count = 0

    def reset(self, video_id: str) -> None:
        self.reset_attempt_count += 1
        if self.fail_reset:
            raise RuntimeError("sensitive reset backend detail")
        super().reset(video_id)

    def update(self, event: FinalizedEvent) -> None:
        self.update_attempt_count += 1
        if self.fail_update:
            raise RuntimeError("sensitive update backend detail")
        super().update(event)


class VerifyGate:
    def decide(self, signals: EvidenceProfile) -> GateDecision:
        del signals
        return GateDecision(action="VERIFY", scope="forbidden", reason="test")


class ContextAwareGate:
    def __init__(self) -> None:
        self.context: PerceptionContext | None = None
        self.result: JointPerceptionResult | None = None

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
    ) -> GateDecision:
        del signals
        self.context = context
        self.result = perception_result
        return GateDecision(action="ACCEPT", scope=None, reason="context-aware")


class RecordingSpecialist:
    enabled_scopes: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.call_count = 0

    def verify(
        self,
        scope: str,
        context: object,
        prediction: InitialPrediction,
        candidates: object,
        evidence: object,
    ) -> object:
        del scope, context, candidates, evidence
        self.call_count += 1
        return prediction


class AlteringFinalizer:
    def __init__(self, alteration: str) -> None:
        self.alteration = alteration
        self._delegate = PredictionFinalizer()

    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        coordinated: CoordinationResult,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[object, object]:
        record, event = self._delegate.finalize(
            run_id=run_id,
            sample=sample,
            coordinated=coordinated,
            decision=decision,
            trace=trace,
        )
        if self.alteration == "record_type":
            return object(), event
        if self.alteration == "event_type":
            return record, object()
        if self.alteration == "record_run":
            return replace(record, run_id="other"), event
        if self.alteration == "record_video":
            return replace(record, video_id="VID30"), event
        if self.alteration == "record_frame":
            return replace(
                record,
                frame_id=13,
                causal_frame_ids=(11, 12, 13),
            ), event
        if self.alteration == "event_video":
            return record, replace(event, video_id="VID30")
        if self.alteration == "event_frame":
            return record, replace(event, frame_id=13)
        if self.alteration == "event_state":
            return record, replace(
                event,
                final_status="Rejected",
                memory_action="SKIP",
            )
        raise AssertionError(f"unknown alteration: {self.alteration}")


class InconsistentStateFinalizer:
    def __init__(self, final_status: str, memory_action: str) -> None:
        self.final_status = final_status
        self.memory_action = memory_action
        self._delegate = PredictionFinalizer()

    def finalize(
        self,
        *,
        run_id: str,
        sample: InferenceSample,
        coordinated: CoordinationResult,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
        record, event = self._delegate.finalize(
            run_id=run_id,
            sample=sample,
            coordinated=coordinated,
            decision=decision,
            trace=trace,
        )
        object.__setattr__(record, "final_status", self.final_status)
        object.__setattr__(record, "memory_action", self.memory_action)
        object.__setattr__(event, "final_status", self.final_status)
        object.__setattr__(event, "memory_action", self.memory_action)
        return record, event


def _pipeline(
    *,
    result_sink: RecordingSink | FrameResultWriter | None = None,
    perception: RecordingPerception | None = None,
    finalizer: object | None = None,
    workflow_store: NoOpCausalStore | None = None,
    event_memory: NoOpCausalStore | None = None,
    gate_policy: object | None = None,
    specialist_registry: object | None = None,
) -> CanonicalStreamingPipeline:
    return CanonicalStreamingPipeline(
        PipelineComponents(
            context_builder=CausalPerceptionContextBuilder(),
            perception=perception or RecordingPerception(),
            candidate_generator=DisabledCandidateGenerator(),
            signal_extractor=FrameEvidenceSignalExtractor(),
            gate_policy=gate_policy or NeverVerify(),  # type: ignore[arg-type]
            specialist_registry=(
                specialist_registry or DisabledSpecialistRegistry()
            ),  # type: ignore[arg-type]
            coordinator=NoOpCoordinator(),
            finalizer=finalizer or PredictionFinalizer(),  # type: ignore[arg-type]
            workflow_store=workflow_store or NoOpCausalStore("workflow"),
            event_memory=event_memory or NoOpCausalStore("memory"),
            result_sink=result_sink or RecordingSink(),
        )
    )


def test_pipeline_extracts_evidence_before_never_verify_and_persists_pair() -> None:
    sink = RecordingSink()
    pipeline = _pipeline(result_sink=sink)

    result = pipeline.run(_sample(), _frames(), run_id="run")

    assert result.prediction.gate_action == "ACCEPT"
    assert result.evidence.evidence_version == "evidence_frame_v1"
    assert result.runtime_trace[4:8] == (
        "05_perception_validated",
        "06_evidence_profile_built",
        "07_candidates_built",
        "08_gate_accept",
    )
    assert sink.calls[0][0] is result.prediction
    assert sink.calls[0][1] is result.evidence


def test_pipeline_passes_context_and_result_to_new_gate_contract() -> None:
    gate = ContextAwareGate()
    perception = RecordingPerception()
    pipeline = _pipeline(gate_policy=gate, perception=perception)

    pipeline.run(_sample(), _frames(), run_id="run")

    assert gate.context is perception.contexts[0]
    assert gate.result is perception.results[0]


def test_pipeline_keeps_legacy_one_argument_custom_gate_compatible() -> None:
    pipeline = _pipeline(gate_policy=VerifyGate())

    result = pipeline.run(_sample(), _frames(), run_id="run")

    assert result.prediction.gate_action == "VERIFY"


def test_failed_first_pair_write_does_not_commit_any_causal_state() -> None:
    pipeline = _pipeline(result_sink=RecordingSink(fail_on_call=1))

    with pytest.raises(ArtifactWriteError, match="paired-write failure"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert pipeline.prior_finalized_prediction is None
    assert pipeline.components.workflow_store.committed_frames == []
    assert pipeline.components.event_memory.committed_frames == []


def test_failed_later_pair_write_keeps_the_last_durable_prior() -> None:
    sink = RecordingSink(fail_on_call=2)
    pipeline = _pipeline(result_sink=sink)
    first = pipeline.run(_sample(frame_id=12), _frames(), run_id="run")

    with pytest.raises(ArtifactWriteError, match="paired-write failure"):
        pipeline.run(_sample(frame_id=13), _frames(), run_id="run")

    assert pipeline.prior_finalized_prediction is first.prediction
    assert pipeline.components.workflow_store.committed_frames == [12]
    assert pipeline.components.event_memory.committed_frames == [12]


def test_durable_prior_is_visible_only_on_next_same_video_call_and_resets() -> None:
    perception = RecordingPerception()
    pipeline = _pipeline(perception=perception)

    first = pipeline.run(_sample(frame_id=12), _frames(), run_id="run")
    second = pipeline.run(_sample(frame_id=13), _frames(), run_id="run")
    pipeline.run(
        _sample(video_id="VID30", frame_id=12),
        _frames(),
        run_id="run",
    )

    assert perception.contexts[0].prior_finalized_prediction is None
    assert perception.contexts[1].prior_finalized_prediction is first.prediction
    assert perception.contexts[2].prior_finalized_prediction is None
    temporal = second.evidence.task_values["instrument"]["temporal_set_change"]
    assert temporal.available is True
    assert pipeline.components.workflow_store.reset_history == ["VID02", "VID30"]
    assert pipeline.components.event_memory.reset_history == ["VID02", "VID30"]


def test_never_verify_keep_preserves_initial_prediction_object_identity() -> None:
    perception = RecordingPerception()
    finalizer = RecordingFinalizer()
    pipeline = _pipeline(perception=perception, finalizer=finalizer)

    pipeline.run(_sample(), _frames(), run_id="run")

    assert finalizer.received_predictions[0] is perception.results[0].prediction


def test_local_backend_wraps_probability_prediction_with_unavailable_evidence() -> None:
    context = CausalPerceptionContextBuilder().build(
        _sample(),
        _frames(),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )
    backend = LocalSmokePerception(LocalSmokeModel(), device=torch.device("cpu"))

    result = backend.predict(context)

    assert result.prediction.score_semantics == "probability_v1"
    assert result.raw_evidence.source == "local_smoke"
    assert result.raw_evidence.source_max_frame_id == 12
    assert all(not ranked for ranked in result.raw_evidence.ranked_candidates.values())
    assert all(
        confidence is None
        for confidence in result.raw_evidence.self_reported_confidence.values()
    )
    assert result.api_provenance == ApiCallProvenance.local()


def test_canonical_run_signature_cannot_accept_ground_truth() -> None:
    parameters: Mapping[str, inspect.Parameter] = inspect.signature(
        CanonicalStreamingPipeline.run
    ).parameters

    assert tuple(parameters) == ("self", "sample", "frames", "run_id")
    assert parameters["run_id"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("failing_store_name", ["workflow", "event"])
def test_post_persistence_store_failure_is_permanently_fail_stop(
    failing_store_name: str,
) -> None:
    workflow = FaultingStore(
        "workflow",
        fail_update=failing_store_name == "workflow",
    )
    event = FaultingStore(
        "event",
        fail_update=failing_store_name == "event",
    )
    sink = RecordingSink()
    perception = RecordingPerception()
    pipeline = _pipeline(
        result_sink=sink,
        perception=perception,
        workflow_store=workflow,
        event_memory=event,
    )

    with pytest.raises(PipelineContractError) as first_error:
        pipeline.run(_sample(), _frames(), run_id="run")

    assert "sensitive" not in str(first_error.value)
    assert pipeline.prior_finalized_prediction is sink.calls[0][0]
    assert len(sink.calls) == 1
    assert len(perception.contexts) == 1
    expected_update_attempts = (
        (1, 0) if failing_store_name == "workflow" else (1, 1)
    )
    assert (
        workflow.update_attempt_count,
        event.update_attempt_count,
    ) == expected_update_attempts
    reset_attempts = (
        workflow.reset_attempt_count,
        event.reset_attempt_count,
    )

    with pytest.raises(PipelineContractError, match="permanently faulted"):
        pipeline.run(
            _sample(video_id="VID30"),
            _frames(),
            run_id="run",
        )

    assert len(sink.calls) == 1
    assert len(perception.contexts) == 1
    assert (
        workflow.reset_attempt_count,
        event.reset_attempt_count,
    ) == reset_attempts
    assert pipeline.prior_finalized_prediction is sink.calls[0][0]


@pytest.mark.parametrize("failing_store_name", ["workflow", "event"])
def test_video_reset_failure_is_permanently_fail_stop(
    failing_store_name: str,
) -> None:
    workflow = FaultingStore(
        "workflow",
        fail_reset=failing_store_name == "workflow",
    )
    event = FaultingStore(
        "event",
        fail_reset=failing_store_name == "event",
    )
    sink = RecordingSink()
    perception = RecordingPerception()
    pipeline = _pipeline(
        result_sink=sink,
        perception=perception,
        workflow_store=workflow,
        event_memory=event,
    )

    with pytest.raises(PipelineContractError) as first_error:
        pipeline.run(_sample(), _frames(), run_id="run")

    assert "sensitive" not in str(first_error.value)
    assert pipeline.prior_finalized_prediction is None
    assert sink.attempt_count == 0
    assert perception.contexts == []
    expected_reset_attempts = (
        (1, 0) if failing_store_name == "workflow" else (1, 1)
    )
    assert (
        workflow.reset_attempt_count,
        event.reset_attempt_count,
    ) == expected_reset_attempts

    with pytest.raises(PipelineContractError, match="permanently faulted"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert (
        workflow.reset_attempt_count,
        event.reset_attempt_count,
    ) == expected_reset_attempts
    assert sink.attempt_count == 0
    assert perception.contexts == []


def test_finalized_record_is_detached_and_frozen_before_sink_and_prior(
    tmp_path: Path,
) -> None:
    mutable_probabilities = {
        task: [0.0] * class_count
        for task, class_count in TASK_CLASS_COUNTS.items()
    }
    prediction = InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(),
        phase_id=0,
        probabilities=mutable_probabilities,  # type: ignore[arg-type]
        backend="integration-test",
        score_semantics="probability_v1",
    )
    perception = RecordingPerception(prediction)
    writer = FrameResultWriter(tmp_path, run_id="run")
    pipeline = _pipeline(result_sink=writer, perception=perception)

    first = pipeline.run(_sample(frame_id=12), _frames(), run_id="run")
    prediction_path = tmp_path / "predictions/VID02.jsonl"
    first_persisted_line = prediction_path.read_bytes().splitlines()[0]
    mutable_probabilities["phase"] = [0.75] * TASK_CLASS_COUNTS["phase"]
    pipeline.run(_sample(frame_id=13), _frames(), run_id="run")

    assert first.prediction.probabilities["phase"] == (0.0,) * 7
    assert type(first.prediction.probabilities) is MappingProxyType
    assert all(
        isinstance(values, tuple)
        for values in first.prediction.probabilities.values()
    )
    assert prediction_path.read_bytes().splitlines()[0] == first_persisted_line
    assert perception.contexts[1].prior_finalized_prediction is first.prediction


def test_verify_decision_falls_back_without_enabled_specialist() -> None:
    specialist = RecordingSpecialist()
    sink = RecordingSink()
    pipeline = _pipeline(
        result_sink=sink,
        gate_policy=VerifyGate(),
        specialist_registry=specialist,
    )

    result = pipeline.run(_sample(), _frames(), run_id="run")

    assert specialist.call_count == 0
    assert result.prediction.verification_status == "FALLBACK_KEEP"
    assert sink.attempt_count == 1
    assert pipeline.prior_finalized_prediction is result.prediction
    assert pipeline.components.workflow_store.committed_frames == [12]
    assert pipeline.components.event_memory.committed_frames == [12]


@pytest.mark.parametrize("alteration", ["record_type", "event_type"])
def test_finalizer_output_types_are_validated_before_persistence(
    alteration: str,
) -> None:
    sink = RecordingSink()
    pipeline = _pipeline(
        result_sink=sink,
        finalizer=AlteringFinalizer(alteration),
    )

    with pytest.raises(PipelineContractError, match="finalizer"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert sink.attempt_count == 0
    assert pipeline.prior_finalized_prediction is None
    assert pipeline.components.workflow_store.committed_frames == []
    assert pipeline.components.event_memory.committed_frames == []


@pytest.mark.parametrize(
    "alteration",
    [
        "record_run",
        "record_video",
        "record_frame",
        "event_video",
        "event_frame",
    ],
)
def test_finalizer_output_identity_is_validated_before_persistence(
    alteration: str,
) -> None:
    sink = RecordingSink()
    pipeline = _pipeline(
        result_sink=sink,
        finalizer=AlteringFinalizer(alteration),
    )

    with pytest.raises(PipelineContractError, match="identity"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert sink.attempt_count == 0
    assert pipeline.prior_finalized_prediction is None
    assert pipeline.components.workflow_store.committed_frames == []
    assert pipeline.components.event_memory.committed_frames == []


def test_finalizer_record_and_event_reliability_state_must_match() -> None:
    sink = RecordingSink()
    pipeline = _pipeline(
        result_sink=sink,
        finalizer=AlteringFinalizer("event_state"),
    )

    with pytest.raises(PipelineContractError, match="reliability state"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert sink.attempt_count == 0


@pytest.mark.parametrize(
    ("final_status", "memory_action"),
    _INVALID_STATE_ACTION_PAIRS,
)
def test_pipeline_rejects_matching_but_incoherent_finalizer_state_action(
    final_status: str,
    memory_action: str,
) -> None:
    sink = RecordingSink()
    pipeline = _pipeline(
        result_sink=sink,
        finalizer=InconsistentStateFinalizer(final_status, memory_action),
    )

    with pytest.raises(PipelineContractError, match="reliability state"):
        pipeline.run(_sample(), _frames(), run_id="run")

    assert sink.attempt_count == 0
