"""Behavioral tests for deterministic, causal frame evidence signals."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    EvidenceValue,
    PhaseTransitionGraph,
)
from surgical_agent.research.signals.frame_evidence import (
    FrameEvidenceSignalExtractor,
    load_ivt_components,
)


def _prediction(
    *,
    instrument_ids: tuple[int, ...] = (0,),
    verb_ids: tuple[int, ...] = (2,),
    target_ids: tuple[int, ...] = (1,),
    triplet_ids: tuple[int, ...] = (0,),
    phase_id: int = 1,
    score_semantics: str = "uncalibrated_rank_v1",
) -> InitialPrediction:
    return InitialPrediction(
        instrument_ids=instrument_ids,
        verb_ids=verb_ids,
        target_ids=target_ids,
        triplet_ids=triplet_ids,
        phase_id=phase_id,
        probabilities={
            task: (0.0,) * class_count
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
        score_semantics=score_semantics,
    )


def _prior(
    *,
    instrument_ids: tuple[int, ...] = (0,),
    verb_ids: tuple[int, ...] = (2,),
    target_ids: tuple[int, ...] = (1,),
    triplet_ids: tuple[int, ...] = (0,),
    phase_id: int = 1,
) -> PredictionRecord:
    current = _prediction(
        instrument_ids=instrument_ids,
        verb_ids=verb_ids,
        target_ids=target_ids,
        triplet_ids=triplet_ids,
        phase_id=phase_id,
    )
    return PredictionRecord(
        run_id="unit-run",
        video_id="VID01",
        frame_id=11,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(11,),
        instrument_ids=current.instrument_ids,
        verb_ids=current.verb_ids,
        target_ids=current.target_ids,
        triplet_ids=current.triplet_ids,
        phase_id=current.phase_id,
        granularity=current.granularity,
        backend="unit-test",
        gate_action="ACCEPT",
        verification_status="SKIPPED",
        alignment_version="unit-test",
        probabilities=current.probabilities,
    )


def _context(*, prior: PredictionRecord | None = None) -> PerceptionContext:
    sample = InferenceSample(
        video_id="VID01",
        target_frame_id=12,
        causal_frame_ids=(10, 11, 12),
        media_refs=("synthetic:10", "synthetic:11", "synthetic:12"),
        source_split=DatasetSplit.TESTING,
        alignment_version="unit-test",
    )
    return PerceptionContext(
        sample=sample,
        frames=torch.zeros((3, 3, 1, 1)),
        images=(
            ApiImageInput(
                identifier="synthetic:12", content=b"png", mime_type="image/png"
            ),
        ),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=prior,
    )


def _result(
    *,
    prediction: InitialPrediction | None = None,
    scores: tuple[float, ...] = (0.9, 0.2),
    confidence: float | None = 0.8,
    source: str = "joint_test",
) -> JointPerceptionResult:
    ranked = {
        task: tuple(
            RankedCandidate(class_id=index, score=score)
            for index, score in enumerate(scores)
        )
        for task in TASK_CLASS_COUNTS
    }
    return JointPerceptionResult(
        prediction=_prediction() if prediction is None else prediction,
        raw_evidence=PerceptionEvidence(
            source=source,
            ranked_candidates=ranked,
            self_reported_confidence={task: confidence for task in TASK_CLASS_COUNTS},
            evidence_refs=(),
            source_max_frame_id=12,
        ),
        api_provenance=ApiCallProvenance.local(),
    )


def _extractor(
    graph: PhaseTransitionGraph | None = None,
) -> FrameEvidenceSignalExtractor:
    return FrameEvidenceSignalExtractor(phase_transition_graph=graph)


def test_candidate_ambiguity_is_one_minus_top_two_margin() -> None:
    profile = _extractor().extract(
        _context(), _result(scores=(0.9, 0.6666666666666667))
    )

    assert profile.task_values["instrument"]["candidate_ambiguity"] == EvidenceValue(
        0.7666666666666667, True, "joint_rank_margin", 12
    )


def test_missing_ranking_evidence_keeps_ambiguity_unavailable() -> None:
    profile = _extractor().extract(
        _context(), _result(scores=(), source="local_smoke")
    )

    assert profile.task_values["instrument"]["candidate_ambiguity"].value is None
    assert profile.task_values["instrument"]["candidate_ambiguity"].available is False


def test_ranked_evidence_requires_uncalibrated_rank_score_semantics() -> None:
    result = _result(prediction=_prediction(score_semantics="probability_v1"))

    with pytest.raises(ValueError, match="uncalibrated_rank_v1"):
        _extractor().extract(_context(), result)


def test_non_null_confidence_requires_uncalibrated_rank_score_semantics() -> None:
    result = _result(
        prediction=_prediction(score_semantics="probability_v1"),
        scores=(),
        confidence=0.8,
    )

    with pytest.raises(ValueError, match="uncalibrated_rank_v1"):
        _extractor().extract(_context(), result)


def test_local_unavailable_probability_semantics_remain_supported() -> None:
    profile = _extractor().extract(
        _context(),
        _result(
            prediction=_prediction(score_semantics="probability_v1"),
            scores=(),
            confidence=None,
            source="local_smoke",
        ),
    )

    assert profile.task_values["instrument"]["candidate_ambiguity"].available is False
    assert profile.task_values["phase"]["self_reported_uncertainty"].available is False


def test_final_only_hard_labels_leave_score_evidence_unavailable() -> None:
    profile = _extractor().extract(
        _context(),
        _result(
            prediction=_prediction(score_semantics="hard_label_v1"),
            scores=(),
            confidence=None,
            source="joint_final_only",
        ),
    )

    for task in TASK_CLASS_COUNTS:
        assert profile.task_values[task]["candidate_ambiguity"].available is False
        assert profile.task_values[task]["self_reported_uncertainty"].available is False
    assert profile.global_values["ivt_internal_conflict"].available is True


@pytest.mark.parametrize(
    ("scores", "confidence"), (((0.8,), None), ((), 0.8))
)
def test_hard_label_semantics_cannot_smuggle_confidence_evidence(scores, confidence):
    with pytest.raises(ValueError, match="cannot carry ranking or confidence"):
        _extractor().extract(
            _context(),
            _result(
                prediction=_prediction(score_semantics="hard_label_v1"),
                scores=scores,
                confidence=confidence,
            ),
        )


def test_ivt_internal_conflict_is_fraction_of_selected_inconsistent_triplets() -> None:
    profile = _extractor().extract(
        _context(),
        _result(
            prediction=_prediction(
                instrument_ids=(0,),
                verb_ids=(2,),
                target_ids=(1,),
                triplet_ids=(0, 1),
            )
        ),
    )

    assert profile.global_values["ivt_internal_conflict"].value == 0.5
    assert all(
        profile.task_values[task]["ivt_internal_conflict"]
        == profile.global_values["ivt_internal_conflict"]
        for task in ("instrument", "verb", "target", "ivt")
    )


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [((0,), (0,), 0.0), ((), (), 0.0), ((0,), (), 1.0)],
)
def test_temporal_set_change_is_jaccard_distance(
    current: tuple[int, ...], previous: tuple[int, ...], expected: float
) -> None:
    profile = _extractor().extract(
        _context(prior=_prior(instrument_ids=previous)),
        _result(prediction=_prediction(instrument_ids=current)),
    )

    assert profile.task_values["instrument"]["temporal_set_change"].value == expected


def test_missing_prior_makes_temporal_and_phase_signals_unavailable() -> None:
    profile = _extractor().extract(_context(prior=None), _result())

    assert profile.task_values["ivt"]["temporal_set_change"].available is False
    assert profile.task_values["phase"]["phase_change_anomaly"].value is None


def test_phase_anomaly_uses_only_the_configured_frozen_graph() -> None:
    graph = PhaseTransitionGraph(
        transitions=((1, 2),),
        source_video_ids=("VID30",),
        version="phase_transition_graph_v1",
        sha256="a" * 64,
    )
    allowed = _extractor(graph).extract(
        _context(prior=_prior(phase_id=1)), _result(prediction=_prediction(phase_id=2))
    )
    disallowed = _extractor(graph).extract(
        _context(prior=_prior(phase_id=1)), _result(prediction=_prediction(phase_id=3))
    )

    assert allowed.task_values["phase"]["phase_change_anomaly"].value == 0.0
    assert disallowed.task_values["phase"]["phase_change_anomaly"].value == 1.0


def test_self_reported_uncertainty_is_one_minus_confidence_or_unavailable() -> None:
    reported = _extractor().extract(
        _context(), _result(confidence=0.12345678901234567)
    )
    absent = _extractor().extract(_context(), _result(confidence=None))

    assert (
        reported.task_values["phase"]["self_reported_uncertainty"].value
        == 0.8765432109876543
    )
    assert absent.task_values["phase"]["self_reported_uncertainty"].available is False


def test_values_are_immutable_validated_and_causally_bounded() -> None:
    with pytest.raises(ValueError, match="finite"):
        EvidenceValue(float("nan"), True, "test", 0)
    with pytest.raises(ValueError, match="unavailable"):
        EvidenceValue(0.0, False, "test", 0)

    profile = _extractor().extract(_context(), _result())
    with pytest.raises(TypeError):
        profile.task_values["instrument"]["injected"] = EvidenceValue(0.0, True, "x", 0)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        profile.frame_id = 99  # type: ignore[misc]
    assert all(
        value.source_max_frame_id <= profile.frame_id
        for values in (*profile.task_values.values(), profile.global_values)
        for value in values.values()
    )


def test_official_ivt_map_is_complete_immutable_and_uses_expected_components() -> None:
    components = load_ivt_components()

    assert len(components) == 100
    assert components[0] == (0, 2, 1)
    assert components[99] == (5, 9, 14)
    with pytest.raises(TypeError):
        components[0] = (1, 1, 1)  # type: ignore[index]


def test_profile_exposes_only_approved_evidence_frame_v1_signal_names() -> None:
    profile: EvidenceProfile = _extractor().extract(_context(), _result())

    assert set(profile.global_values) == {"ivt_internal_conflict"}
    assert set(profile.task_values["instrument"]) == {
        "candidate_ambiguity",
        "ivt_internal_conflict",
        "temporal_set_change",
        "self_reported_uncertainty",
    }
    assert set(profile.task_values["phase"]) == {
        "candidate_ambiguity",
        "phase_change_anomaly",
        "self_reported_uncertainty",
    }
