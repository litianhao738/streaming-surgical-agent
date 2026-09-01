"""End-to-end canonical VERIFY behavior without network or training."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from surgical_agent.api.errors import ApiProviderCallBudgetError
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.gate.policy import AlwaysVerifyPolicy
from surgical_agent.research.reliability.state import ALL_TASK_FIELDS
from surgical_agent.research.signals.frame_evidence import FrameEvidenceSignalExtractor
from surgical_agent.research.verification.contracts import VerificationResult
from surgical_agent.research.verification.coordinator import DeterministicCoordinator
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    NoOpCausalStore,
    PipelineComponents,
    PredictionFinalizer,
)


def _prediction(**changes: object) -> InitialPrediction:
    base = InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        probabilities={
            task: tuple(1.0 - index / (count + 1) for index in range(count))
            for task, count in TASK_CLASS_COUNTS.items()
        },
        backend="joint_mock",
        score_semantics="uncalibrated_rank_v1",
    )
    return replace(base, **changes)


class Perception:
    def __init__(self) -> None:
        self.prediction = _prediction()

    def predict(self, context: object) -> JointPerceptionResult:
        frame_id = context.sample.target_frame_id  # type: ignore[attr-defined]
        return JointPerceptionResult(
            prediction=self.prediction,
            raw_evidence=PerceptionEvidence(
                source="mock",
                ranked_candidates={
                    task: tuple(
                        RankedCandidate(
                            class_id=index,
                            score=1.0 - index / (count + 1),
                        )
                        for index in range(count)
                    )
                    for task, count in TASK_CLASS_COUNTS.items()
                },
                self_reported_confidence={task: 0.8 for task in TASK_CLASS_COUNTS},
                evidence_refs=(),
                source_max_frame_id=frame_id,
            ),
            api_provenance=ApiCallProvenance.local(),
        )


class VerifierRegistry:
    enabled_scopes = ("joint",)

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.call_count = 0

    def verify(
        self,
        scope: str,
        context: object,
        prediction: InitialPrediction,
        candidates: object,
        evidence: object,
    ) -> VerificationResult:
        del context, candidates, evidence
        self.call_count += 1
        if self.fail:
            raise RuntimeError("private provider detail")
        return VerificationResult(
            scope=scope,
            prediction=replace(prediction, instrument_ids=(1,)),
            provenance=ApiCallProvenance.local(),
        )


class BudgetFailingRegistry(VerifierRegistry):
    def verify(self, *args: object, **kwargs: object) -> VerificationResult:
        del args, kwargs
        self.call_count += 1
        raise ApiProviderCallBudgetError("provider call budget exhausted")


class MismatchedBackendRegistry(VerifierRegistry):
    def verify(
        self,
        scope: str,
        context: object,
        prediction: InitialPrediction,
        candidates: object,
        evidence: object,
    ) -> VerificationResult:
        del context, candidates, evidence
        self.call_count += 1
        return VerificationResult(
            scope=scope,
            prediction=prediction,
            provenance=ApiCallProvenance(
                source="mock",
                provider="different-provider",
                endpoint_identifier="mock://different",
                request_hash="a" * 64,
                requested_model_identifier="different-model",
                returned_model_identifier="different-model",
                cache_hit=False,
                provider_call_count=1,
            ),
        )


class FieldAwareRegistry(VerifierRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.requested_fields: tuple[str, ...] | None = None

    def verify(
        self,
        scope: str,
        context: object,
        prediction: InitialPrediction,
        candidates: object,
        evidence: object,
        *,
        requested_fields: tuple[str, ...],
    ) -> VerificationResult:
        self.requested_fields = requested_fields
        return super().verify(scope, context, prediction, candidates, evidence)


class Sink:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def write(self, prediction: object, evidence: object) -> None:
        self.calls.append((prediction, evidence))


def _sample() -> InferenceSample:
    return InferenceSample(
        video_id="VID02",
        target_frame_id=12,
        causal_frame_ids=(10, 11, 12),
        media_refs=("10.png", "11.png", "12.png"),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="verification-integration",
    )


def _pipeline(registry: VerifierRegistry) -> CanonicalStreamingPipeline:
    return CanonicalStreamingPipeline(
        PipelineComponents(
            context_builder=CausalPerceptionContextBuilder(),
            perception=Perception(),
            candidate_generator=FactorizedCandidateGenerator(
                max_candidates_per_task=2
            ),
            signal_extractor=FrameEvidenceSignalExtractor(),
            gate_policy=AlwaysVerifyPolicy(scope="joint"),
            specialist_registry=registry,
            coordinator=DeterministicCoordinator(),
            finalizer=PredictionFinalizer(),
            workflow_store=NoOpCausalStore("workflow"),
            event_memory=NoOpCausalStore("memory"),
            result_sink=Sink(),
        )
    )


def test_verify_calls_one_verifier_and_commits_valid_repair() -> None:
    registry = VerifierRegistry()
    pipeline = _pipeline(registry)

    result = pipeline.run(
        _sample(),
        torch.full((3, 3, 8, 8), 0.25),
        run_id="verify-run",
    )

    assert registry.call_count == 1
    assert result.prediction.instrument_ids == (1,)
    assert result.prediction.gate_action == "VERIFY"
    assert result.prediction.verification_status == "VERIFIED_REPAIR"
    assert "09_specialist_called_joint" in result.runtime_trace
    assert "11_coordinator_verified_repair" in result.runtime_trace
    assert pipeline.components.event_memory.committed_frames == [12]


def test_pipeline_passes_canonical_flagged_fields_to_field_aware_verifier() -> None:
    registry = FieldAwareRegistry()

    result = _pipeline(registry).run(
        _sample(),
        torch.full((3, 3, 8, 8), 0.25),
        run_id="verify-run",
    )

    assert registry.requested_fields == ALL_TASK_FIELDS
    assert result.prediction.verification_status == "VERIFIED_REPAIR"


def test_verifier_failure_is_reason_coded_keep_without_leaking_exception() -> None:
    registry = VerifierRegistry(fail=True)
    pipeline = _pipeline(registry)

    result = pipeline.run(
        _sample(),
        torch.full((3, 3, 8, 8), 0.25),
        run_id="verify-run",
    )

    assert registry.call_count == 1
    assert result.prediction.instrument_ids == (0,)
    assert result.prediction.verification_status == "FALLBACK_KEEP"
    assert "private provider detail" not in " ".join(result.runtime_trace)
    assert "09_specialist_call_failed" in result.runtime_trace
    assert pipeline.components.event_memory.committed_frames == [12]


def test_verifier_budget_exhaustion_fails_the_rollout() -> None:
    pipeline = _pipeline(BudgetFailingRegistry())

    with pytest.raises(ApiProviderCallBudgetError):
        pipeline.run(
            _sample(),
            torch.full((3, 3, 8, 8), 0.25),
            run_id="verify-run",
        )


def test_verifier_backend_mismatch_is_an_explicit_fallback_keep() -> None:
    pipeline = _pipeline(MismatchedBackendRegistry())

    result = pipeline.run(
        _sample(),
        torch.full((3, 3, 8, 8), 0.25),
        run_id="verify-run",
    )

    assert result.prediction.verification_status == "FALLBACK_KEEP"
    assert "09_specialist_backend_mismatch" in result.runtime_trace
