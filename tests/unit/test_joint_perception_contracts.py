"""Frozen contracts for joint perception predictions, evidence, and provenance."""

from __future__ import annotations

from dataclasses import fields

import pytest

from surgical_agent.api.contracts import ApiResponseRecord
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.perception.contracts import (
    EVIDENCE_REF_CODES,
    ApiCallProvenance,
    EvidenceReference,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.systems.pipeline import GateDecision, PredictionFinalizer


def valid_ranked_candidates() -> dict[str, tuple[RankedCandidate, ...]]:
    return {
        task: (RankedCandidate(class_id=lower, score=0.9),)
        for task, (lower, _upper) in {
            "instrument": (0, 6),
            "verb": (0, 9),
            "target": (0, 14),
            "ivt": (0, 99),
            "phase": (0, 6),
        }.items()
    }


def valid_confidences() -> dict[str, float | None]:
    return {task: 0.8 for task in TASK_CLASS_COUNTS}


def perception_evidence() -> PerceptionEvidence:
    return PerceptionEvidence(
        source="joint_openrouter_gpt56sol",
        ranked_candidates=valid_ranked_candidates(),
        self_reported_confidence=valid_confidences(),
        evidence_refs=(EvidenceReference(frame_id=2, code="CURRENT_VISUAL_SUPPORT"),),
        source_max_frame_id=2,
    )


def prediction(*, score_semantics: str = "probability_v1") -> InitialPrediction:
    return InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(1,),
        target_ids=(2,),
        triplet_ids=(3,),
        phase_id=4,
        probabilities={
            task: (0.5,) * class_count
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
        score_semantics=score_semantics,
    )


def api_record() -> ApiResponseRecord:
    return ApiResponseRecord(
        provider="mock",
        endpoint_identifier="mock://joint",
        request_hash="a" * 64,
        requested_model_identifier="requested-model",
        returned_model_identifier="returned-model",
        parsed_payload={"credential": "must-not-copy"},
        cache_hit=False,
        provider_call_count=1,
    )


def test_joint_result_carries_prediction_evidence_and_safe_api_provenance() -> None:
    result = JointPerceptionResult(
        prediction=prediction(score_semantics="uncalibrated_rank_v1"),
        raw_evidence=perception_evidence(),
        api_provenance=ApiCallProvenance.from_response(api_record()),
    )

    assert result.prediction.score_semantics == "uncalibrated_rank_v1"
    assert result.api_provenance.source == "mock"
    assert result.api_provenance.request_hash == "a" * 64
    assert result.api_provenance.provider_call_count == 1


def test_evidence_contract_rejects_missing_task_head() -> None:
    values = valid_ranked_candidates()
    values.pop("phase")

    with pytest.raises(ValueError, match="five task heads"):
        PerceptionEvidence(
            source="joint_openrouter_gpt56sol",
            ranked_candidates=values,
            self_reported_confidence=valid_confidences(),
            evidence_refs=(),
            source_max_frame_id=2,
        )


def test_evidence_contract_rejects_bad_scores_duplicate_ids_and_bad_refs() -> None:
    values = valid_ranked_candidates()
    values["instrument"] = (
        RankedCandidate(class_id=0, score=0.8),
        RankedCandidate(class_id=0, score=0.7),
    )
    with pytest.raises(ValueError, match="unique"):
        PerceptionEvidence(
            source="joint",
            ranked_candidates=values,
            self_reported_confidence=valid_confidences(),
            evidence_refs=(),
            source_max_frame_id=2,
        )

    with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
        RankedCandidate(class_id=0, score=float("nan"))

    with pytest.raises(ValueError, match="causal"):
        PerceptionEvidence(
            source="joint",
            ranked_candidates=valid_ranked_candidates(),
            self_reported_confidence=valid_confidences(),
            evidence_refs=(EvidenceReference(frame_id=3, code="CURRENT_VISUAL_SUPPORT"),),
            source_max_frame_id=2,
        )

    with pytest.raises(ValueError, match="closed"):
        EvidenceReference(frame_id=2, code="FREE_FORM_REASONING")


def test_evidence_contract_requires_exact_confidence_heads_and_valid_ranges() -> None:
    confidences = valid_confidences()
    confidences.pop("phase")
    with pytest.raises(ValueError, match="five task heads"):
        PerceptionEvidence(
            source="joint",
            ranked_candidates=valid_ranked_candidates(),
            self_reported_confidence=confidences,
            evidence_refs=(),
            source_max_frame_id=2,
        )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PerceptionEvidence(
            source="joint",
            ranked_candidates=valid_ranked_candidates(),
            self_reported_confidence={**valid_confidences(), "phase": 1.1},
            evidence_refs=(),
            source_max_frame_id=2,
        )


def test_local_unavailable_evidence_is_explicitly_empty() -> None:
    evidence = PerceptionEvidence.local_unavailable(4)

    assert evidence.source == "local_smoke"
    assert evidence.source_max_frame_id == 4
    assert evidence.evidence_refs == ()
    assert set(evidence.ranked_candidates) == set(TASK_CLASS_COUNTS)
    assert all(not values for values in evidence.ranked_candidates.values())
    assert evidence.self_reported_confidence == {
        task: None for task in TASK_CLASS_COUNTS
    }


def test_score_semantics_is_restricted_and_finalizer_copies_it() -> None:
    with pytest.raises(ValueError, match="score_semantics"):
        prediction(score_semantics="calibrated_probability")

    assert {field.name for field in fields(InitialPrediction)} >= {"score_semantics"}
    assert {field.name for field in fields(PredictionRecord)} >= {"score_semantics"}

    sample = InferenceSample(
        video_id="VID01",
        target_frame_id=2,
        causal_frame_ids=(1, 2),
        media_refs=("1.png", "2.png"),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit-test",
    )
    record, _event = PredictionFinalizer().finalize(
        run_id="run-1",
        sample=sample,
        prediction=prediction(score_semantics="uncalibrated_rank_v1"),
        decision=GateDecision(action="ACCEPT", scope=None, reason="test"),
        trace=(),
    )
    assert record.score_semantics == "uncalibrated_rank_v1"


def test_contract_dataclasses_have_no_sensitive_or_reasoning_fields() -> None:
    contract_types = (
        RankedCandidate,
        EvidenceReference,
        PerceptionEvidence,
        ApiCallProvenance,
        JointPerceptionResult,
        InitialPrediction,
        PredictionRecord,
    )
    forbidden = {
        "credential",
        "credentials",
        "api_key",
        "authorization",
        "raw_response",
        "free_form_reasoning",
        "reasoning",
        "chain_of_thought",
        "ground_truth",
    }
    assert not forbidden & {
        field.name
        for contract_type in contract_types
        for field in fields(contract_type)
    }
    assert EVIDENCE_REF_CODES == frozenset(
        {
            "CURRENT_VISUAL_SUPPORT",
            "CAUSAL_VISUAL_TREND",
            "PRIOR_STATE_SUPPORT",
            "AMBIGUOUS_VISUAL_SUPPORT",
        }
    )
