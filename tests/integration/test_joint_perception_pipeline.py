"""Canonical Joint Perception, evidence, and durable-history integration tests."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import pytest
import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.frame_result_writer import prediction_record_sha256
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
    PredictionFinalizer,
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
        prediction: InitialPrediction,
        decision: GateDecision,
        trace: tuple[str, ...],
    ) -> tuple[PredictionRecord, FinalizedEvent]:
        self.received_predictions.append(prediction)
        return self._delegate.finalize(
            run_id=run_id,
            sample=sample,
            prediction=prediction,
            decision=decision,
            trace=trace,
        )


def _pipeline(
    *,
    result_sink: RecordingSink | None = None,
    perception: RecordingPerception | None = None,
    finalizer: RecordingFinalizer | PredictionFinalizer | None = None,
) -> CanonicalStreamingPipeline:
    return CanonicalStreamingPipeline(
        PipelineComponents(
            context_builder=CausalPerceptionContextBuilder(),
            perception=perception or RecordingPerception(),
            candidate_generator=DisabledCandidateGenerator(),
            signal_extractor=FrameEvidenceSignalExtractor(),
            gate_policy=NeverVerify(),
            specialist_registry=DisabledSpecialistRegistry(),
            coordinator=NoOpCoordinator(),
            finalizer=finalizer or PredictionFinalizer(),
            workflow_store=NoOpCausalStore("workflow"),
            event_memory=NoOpCausalStore("memory"),
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
