"""Deterministic, field-scoped checks for reliability-aware routing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    FieldUncertainty,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.gate.features import NO_TRACKER_FEATURE_ORDER
from surgical_agent.research.gate.policy import (
    ForcedTargetedVerificationPolicy,
    FrozenLinearBenefitGate,
)
from surgical_agent.research.gate.reliability import ReliabilityGatePolicy
from surgical_agent.research.reliability.state import (
    ALL_TASK_FIELDS,
    GateFinding,
)
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.frame_evidence import FrameEvidenceSignalExtractor
from surgical_agent.research.verification.contracts import CoordinationResult
from surgical_agent.systems.pipeline import GateDecision, PredictionFinalizer


def _probabilities() -> dict[str, tuple[float, ...]]:
    return {
        task: (0.0,) * class_count
        for task, class_count in TASK_CLASS_COUNTS.items()
    }


def _prediction(**changes: object) -> InitialPrediction:
    values: dict[str, object] = {
        "instrument_ids": (0,),
        "verb_ids": (2,),
        "target_ids": (1,),
        "triplet_ids": (0,),
        "phase_id": 1,
        "probabilities": _probabilities(),
        "backend": "joint_mock",
        "score_semantics": "uncalibrated_rank_v1",
    }
    values.update(changes)
    return InitialPrediction(**values)  # type: ignore[arg-type]


def _prior(**changes: object) -> PredictionRecord:
    values: dict[str, object] = {
        "run_id": "prior",
        "video_id": "VID30",
        "frame_id": 10,
        "source_split": DatasetSplit.VALIDATION,
        "causal_frame_ids": (10,),
        "instrument_ids": (0,),
        "verb_ids": (2,),
        "target_ids": (1,),
        "triplet_ids": (0,),
        "phase_id": 1,
        "granularity": "frame_multilabel",
        "backend": "joint_mock",
        "gate_action": "ACCEPT",
        "verification_status": "NOT_REQUESTED",
        "alignment_version": "unit-test",
        "probabilities": _probabilities(),
        "score_semantics": "uncalibrated_rank_v1",
    }
    values.update(changes)
    return PredictionRecord(**values)  # type: ignore[arg-type]


def _context(*, prior: PredictionRecord | None = None) -> PerceptionContext:
    sample = InferenceSample(
        video_id="VID30",
        target_frame_id=12,
        causal_frame_ids=(12,),
        media_refs=("synthetic:12",),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="unit-test",
    )
    return PerceptionContext(
        sample=sample,
        frames=torch.zeros((1, 3, 1, 1)),
        images=(),
        workflow_snapshot=MappingProxyType({}),
        memory_snapshot=MappingProxyType({}),
        prior_finalized_prediction=prior,
    )


def _ranking(
    selected_id: int,
    *,
    selected_confidence: float = 0.90,
    alternative_id: int | None = None,
) -> tuple[RankedCandidate, ...]:
    candidates = [RankedCandidate(selected_id, selected_confidence)]
    if alternative_id is not None:
        candidates.append(RankedCandidate(alternative_id, min(0.50, selected_confidence)))
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def _result(
    *,
    prediction: InitialPrediction | None = None,
    low_task: str | None = None,
    uncertainties: tuple[FieldUncertainty, ...] = (),
) -> JointPerceptionResult:
    prediction = prediction or _prediction()
    selected = {
        "instrument": prediction.instrument_ids[0] if prediction.instrument_ids else 0,
        "verb": prediction.verb_ids[0] if prediction.verb_ids else 0,
        "target": prediction.target_ids[0] if prediction.target_ids else 0,
        "ivt": prediction.triplet_ids[0] if prediction.triplet_ids else 0,
        "phase": prediction.phase_id,
    }
    alternatives = {
        task: (selected[task] + 1) % TASK_CLASS_COUNTS[task]
        for task in TASK_CLASS_COUNTS
    }
    evidence = PerceptionEvidence(
        source="joint_mock",
        ranked_candidates={
            task: _ranking(
                selected[task],
                selected_confidence=0.40 if task == low_task else 0.90,
                alternative_id=alternatives[task],
            )
            for task in TASK_CLASS_COUNTS
        },
        self_reported_confidence={task: None for task in TASK_CLASS_COUNTS},
        evidence_refs=(),
        source_max_frame_id=12,
        field_uncertainties=uncertainties,
    )
    return JointPerceptionResult(
        prediction=prediction,
        raw_evidence=evidence,
        api_provenance=ApiCallProvenance.local(),
    )


def _decide(
    result: JointPerceptionResult,
    *,
    context: PerceptionContext | None = None,
    policy: ReliabilityGatePolicy | None = None,
) -> GateDecision:
    context = context or _context()
    evidence = FrameEvidenceSignalExtractor().extract(context, result)
    return (policy or ReliabilityGatePolicy()).decide(
        evidence,
        context=context,
        perception_result=result,
    )


def test_reliability_gate_accepts_coherent_high_confidence_prediction() -> None:
    decision = _decide(_result())

    assert decision.action == "ACCEPT"
    assert decision.scope is None
    assert decision.findings == ()
    assert decision.flagged_fields == ()
    assert decision.selected_confidence_floor == pytest.approx(0.90)


def test_counterfactual_teacher_routes_only_lowest_confidence_flagged_field() -> None:
    result = _result(low_task="target")
    context = _context()
    evidence = FrameEvidenceSignalExtractor().extract(context, result)

    decision = ForcedTargetedVerificationPolicy(ReliabilityGatePolicy()).decide(
        evidence,
        context=context,
        perception_result=result,
    )

    assert decision.action == "VERIFY"
    assert decision.flagged_fields == ("target",)
    assert all(finding.path == "/target/selected_ids" for finding in decision.findings)


def test_learned_verify_always_routes_one_field_when_rule_gate_accepts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "gate.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "benefit_gate_linear_v1",
                "gate_stage": "final_g1",
                "source_split": "training",
                "feature_order": list(NO_TRACKER_FEATURE_ORDER),
                "weights_by_scope": {"joint": [0.0] * len(NO_TRACKER_FEATURE_ORDER)},
                "bias_by_scope": {"joint": 1.0},
                "scope_order": ["joint"],
                "threshold": 0.0,
                "training_dataset_ids": ["training-demo"],
                "training_recipe_id": "unit",
                "normalization_version": "demo_gate_features_v1_raw",
                "rollout_policy_id": "counterfactual_verify_v1",
            }
        ),
        encoding="utf-8",
    )
    result = _result()
    context = _context()
    evidence = FrameEvidenceSignalExtractor().extract(context, result)

    decision = FrozenLinearBenefitGate.from_json(artifact).decide(
        evidence,
        context=context,
        perception_result=result,
    )

    assert decision.action == "VERIFY"
    assert len(decision.flagged_fields) == 1


def test_low_confidence_uses_selected_ranked_candidate_not_dense_probabilities() -> None:
    result = _result(low_task="verb")

    decision = _decide(result)

    assert decision.selected_confidence_floor == pytest.approx(0.40)
    assert decision.flagged_fields == ("verb",)
    assert decision.findings == (
        GateFinding(
            path="/verb/selected_ids",
            reason="LOW_CONFIDENCE",
            alternative_ids=(3,),
        ),
    )


def test_low_confidence_checks_every_selected_label_not_only_the_first() -> None:
    prediction = _prediction(instrument_ids=(0, 1))
    base = _result(prediction=prediction)
    rankings = dict(base.raw_evidence.ranked_candidates)
    rankings["instrument"] = (
        RankedCandidate(0, 0.90),
        RankedCandidate(1, 0.40),
    )
    result = replace(
        base,
        raw_evidence=replace(base.raw_evidence, ranked_candidates=rankings),
    )

    decision = _decide(result)

    assert decision.selected_confidence_floor == pytest.approx(0.40)
    assert any(
        finding.path == "/instrument/selected_ids"
        and finding.reason == "LOW_CONFIDENCE"
        for finding in decision.findings
    )


def test_vlm_uncertainty_is_preserved_as_a_field_scoped_gate_finding() -> None:
    uncertainty = FieldUncertainty(
        path="/target/selected_ids",
        reason="OCCLUSION",
        alternative_ids=(2,),
    )

    decision = _decide(_result(uncertainties=(uncertainty,)))

    assert decision.flagged_fields == ("target",)
    assert decision.findings == (
        GateFinding(
            path="/target/selected_ids",
            reason="OCCLUSION",
            alternative_ids=(2,),
        ),
    )


def test_ivt_closure_and_component_compatibility_flag_only_involved_fields() -> None:
    # IVT 0 requires I=0, V=2, T=1. The prediction instead selects V=3 and T=2.
    result = _result(prediction=_prediction(verb_ids=(3,), target_ids=(2,)))

    decision = _decide(result)

    assert decision.action == "VERIFY"
    assert decision.scope == "targeted"
    assert decision.flagged_fields == ("verb", "target", "ivt")
    assert {finding.reason for finding in decision.findings} == {
        "IVT_CLOSURE",
        "TRIPLET_COMPATIBILITY",
    }


def test_phase_triplet_consistency_is_skipped_without_mapping_and_checked_when_supplied() -> None:
    result = _result()
    without_mapping = _decide(result)
    policy = ReliabilityGatePolicy(phase_allowed_ivt={1: (1,), 2: (0,)})

    with_mapping = _decide(result, policy=policy)

    assert without_mapping.action == "ACCEPT"
    assert with_mapping.flagged_fields == ("ivt", "phase")
    assert all(
        finding.reason == "PHASE_TRIPLET_CONFLICT"
        for finding in with_mapping.findings
    )


@pytest.mark.parametrize(
    "phase_allowed_ivt",
    [
        {7: (0,)},
        {1: (100,)},
        {1: (1, 1)},
    ],
)
def test_phase_triplet_mapping_rejects_out_of_ontology_or_duplicate_ids(
    phase_allowed_ivt: dict[int, tuple[int, ...]],
) -> None:
    with pytest.raises(ValueError, match="phase_allowed_ivt"):
        ReliabilityGatePolicy(phase_allowed_ivt=phase_allowed_ivt)


def test_phase_triplet_mapping_is_copied_and_immutable() -> None:
    source = {1: (0, 1)}
    policy = ReliabilityGatePolicy(phase_allowed_ivt=source)
    source[1] = (2,)

    assert policy.phase_allowed_ivt[1] == (0, 1)  # type: ignore[index]
    with pytest.raises(TypeError):
        policy.phase_allowed_ivt[1] = (2,)  # type: ignore[index]


@pytest.mark.parametrize(
    "ivt_components",
    [
        {0: (0, 2, 1)},
        {index: (7, 0, 0) for index in range(100)},
        {index: (0, 0) for index in range(100)},
    ],
)
def test_injected_ivt_component_map_rejects_incomplete_or_malformed_values(
    ivt_components: dict[int, tuple[int, ...]],
) -> None:
    with pytest.raises((TypeError, ValueError), match="ivt_components"):
        ReliabilityGatePolicy(ivt_components=ivt_components)  # type: ignore[arg-type]


def test_temporal_set_jump_and_invalid_phase_edge_are_field_scoped() -> None:
    prior = _prior(
        instrument_ids=(1,),
        verb_ids=(3,),
        target_ids=(2,),
        triplet_ids=(1,),
        phase_id=0,
    )
    graph = PhaseTransitionGraph(
        transitions=((0, 0), (1, 1)),
        source_video_ids=("VID01",),
        version="unit-test",
        sha256="0" * 64,
    )
    policy = ReliabilityGatePolicy(
        temporal_jaccard_threshold=0.80,
        phase_transition_graph=graph,
    )

    decision = _decide(_result(), context=_context(prior=prior), policy=policy)

    assert decision.flagged_fields == ALL_TASK_FIELDS
    assert {finding.reason for finding in decision.findings} == {"TEMPORAL_JUMP"}


def test_defensive_ontology_check_returns_findings_instead_of_crashing() -> None:
    prediction = _prediction()
    result = _result(prediction=prediction)
    object.__setattr__(prediction, "phase_id", 99)

    decision = _decide(result)

    assert decision.flagged_fields == ("phase",)
    assert any(
        finding.reason == "ONTOLOGY_VIOLATION"
        for finding in decision.findings
    )


def test_gate_decision_normalizes_legacy_verify_and_rejects_accept_findings() -> None:
    legacy = GateDecision(action="VERIFY", scope="joint", reason="legacy")

    assert legacy.flagged_fields == ALL_TASK_FIELDS
    with pytest.raises(ValueError, match="ACCEPT forbids"):
        GateDecision(
            action="ACCEPT",
            scope=None,
            reason="bad",
            flagged_fields=("ivt",),
        )


@pytest.mark.parametrize(
    (
        "verification_status",
        "decision",
        "expected_status",
        "expected_memory_action",
    ),
    [
        (
            "NOT_REQUESTED",
            GateDecision(
                action="ACCEPT",
                scope=None,
                reason="clean",
                selected_confidence_floor=0.90,
            ),
            "Accepted",
            "WRITE_RELIABLE",
        ),
        (
            "NOT_REQUESTED",
            GateDecision(
                action="ACCEPT",
                scope=None,
                reason="clean",
                selected_confidence_floor=0.80,
            ),
            "Accepted",
            "WRITE_SHORT_TERM",
        ),
        (
            "VERIFIED_KEEP",
            GateDecision(action="VERIFY", scope="joint", reason="verify"),
            "Verified",
            "WRITE_RELIABLE",
        ),
        (
            "FALLBACK_KEEP",
            GateDecision(action="VERIFY", scope="joint", reason="verify"),
            "Pending",
            "BUFFER_PENDING",
        ),
    ],
)
def test_finalizer_assigns_candidate_and_routes_final_state_to_memory(
    verification_status: str,
    decision: GateDecision,
    expected_status: str,
    expected_memory_action: str,
) -> None:
    result = CoordinationResult(
        prediction=_prediction(),
        verification_status=verification_status,
        reason="unit-test",
    )

    record, event = PredictionFinalizer().finalize(
        run_id="run",
        sample=_context().sample,
        coordinated=result,
        decision=decision,
        trace=("test",),
    )

    assert record.initial_state == "Candidate"
    assert record.final_status == expected_status
    assert record.memory_action == expected_memory_action
    assert event.initial_state == "Candidate"
    assert event.final_status == expected_status
    assert event.memory_action == expected_memory_action


def test_finalizer_persists_canonical_gate_and_repaired_fields() -> None:
    finding = GateFinding(
        path="/verb/selected_ids",
        reason="LOW_CONFIDENCE",
        alternative_ids=(3,),
    )
    decision = GateDecision(
        action="VERIFY",
        scope="targeted",
        reason="verify",
        findings=(finding,),
        flagged_fields=("verb",),
        selected_confidence_floor=0.40,
    )
    repaired = _prediction(verb_ids=(3,))
    coordinated = CoordinationResult(
        prediction=repaired,
        verification_status="VERIFIED_REPAIR",
        reason="repair",
        selected_candidate_id="H_unit",
        touched_tasks=("verb",),
    )

    record, event = PredictionFinalizer().finalize(
        run_id="run",
        sample=_context().sample,
        coordinated=coordinated,
        decision=decision,
        trace=("test",),
    )

    assert record.gate_reasons == ("LOW_CONFIDENCE",)
    assert record.flagged_fields == ("verb",)
    assert record.repaired_fields == ("verb",)
    assert event.gate_reasons == record.gate_reasons
    assert event.repaired_fields == record.repaired_fields
