"""Behavioral contracts for the no-training verification runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.gate.policy import (
    AlwaysVerifyPolicy,
    EvidenceThresholdPolicy,
    FrozenLinearBenefitGate,
)
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.verification.contracts import VerificationResult
from surgical_agent.research.verification.coordinator import DeterministicCoordinator
from surgical_agent.research.verification.hypotheses import (
    FactorizedCandidateGenerator,
)
from surgical_agent.systems.pipeline import GateDecision


def _probabilities() -> dict[str, tuple[float, ...]]:
    return {
        task: tuple(1.0 - index / (count + 1) for index in range(count))
        for task, count in TASK_CLASS_COUNTS.items()
    }


def _prediction(**changes: object) -> InitialPrediction:
    base = InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        probabilities=_probabilities(),
        backend="joint_mock",
        score_semantics="uncalibrated_rank_v1",
    )
    return replace(base, **changes)


def _perception_result() -> JointPerceptionResult:
    ranked = {
        task: tuple(
            RankedCandidate(class_id=index, score=1.0 - index / (count + 1))
            for index in range(count)
        )
        for task, count in TASK_CLASS_COUNTS.items()
    }
    return JointPerceptionResult(
        prediction=_prediction(),
        raw_evidence=PerceptionEvidence(
            source="mock",
            ranked_candidates=ranked,
            self_reported_confidence={task: 0.8 for task in TASK_CLASS_COUNTS},
            evidence_refs=(),
            source_max_frame_id=12,
        ),
        api_provenance=ApiCallProvenance.local(),
    )


def _evidence() -> EvidenceProfile:
    values = {
        "candidate_ambiguity": EvidenceValue(0.2, True, "joint_rank_margin", 12),
        "ivt_internal_conflict": EvidenceValue(0.0, True, "ivt_component_map_v1", 12),
        "self_reported_uncertainty": EvidenceValue(
            0.2, True, "joint_self_reported_confidence", 12
        ),
        "temporal_set_change": EvidenceValue(
            None, False, "finalized_prior_jaccard", 12
        ),
    }
    return EvidenceProfile(
        video_id="VID02",
        frame_id=12,
        task_values={
            "instrument": values,
            "verb": values,
            "target": values,
            "ivt": values,
            "phase": {
                "candidate_ambiguity": values["candidate_ambiguity"],
                "phase_change_anomaly": EvidenceValue(
                    None, False, "frozen_phase_transition_graph", 12
                ),
                "self_reported_uncertainty": values[
                    "self_reported_uncertainty"
                ],
            },
        },
        global_values={"ivt_internal_conflict": values["ivt_internal_conflict"]},
    )


def test_always_verify_policy_routes_exactly_one_configured_scope() -> None:
    policy = AlwaysVerifyPolicy(scope="joint")

    decision = policy.decide(_evidence())

    assert decision == GateDecision(
        action="VERIFY",
        scope="joint",
        reason="ALWAYS_VERIFY_JOINT",
    )


def test_evidence_threshold_policy_is_a_deterministic_no_training_baseline() -> None:
    verify = EvidenceThresholdPolicy(threshold=0.15, scope="joint").decide(
        _evidence()
    )
    accept = EvidenceThresholdPolicy(threshold=0.95, scope="joint").decide(
        _evidence()
    )

    assert verify.action == "VERIFY"
    assert verify.scope == "joint"
    assert verify.reason == "EVIDENCE_THRESHOLD_TRIGGERED"
    assert accept == GateDecision(
        action="ACCEPT",
        scope=None,
        reason="EVIDENCE_THRESHOLD_NOT_TRIGGERED",
    )


def _gate_artifact(path: Path, *, gate_stage: str = "final_g1") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "benefit_gate_linear_v1",
                "gate_stage": gate_stage,
                "source_split": "training",
                "feature_order": ["instrument.candidate_ambiguity.value"],
                "weights_by_scope": {"joint": [1.0]},
                "bias_by_scope": {"joint": 0.0},
                "scope_order": ["joint"],
                "threshold": 0.15,
                "training_dataset_ids": ["cholectrack20_train_v1"],
                "training_recipe_id": "gate_train_v1",
                "normalization_version": "evidence_frame_v1_raw",
                "rollout_policy_id": "g0_policy_v1",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_frozen_linear_gate_loads_train_derived_final_artifact(tmp_path: Path) -> None:
    gate = FrozenLinearBenefitGate.from_json(_gate_artifact(tmp_path / "gate.json"))

    decision = gate.decide(_evidence())

    assert decision.action == "VERIFY"
    assert decision.scope == "joint"
    assert decision.reason == "FINAL_G1_BENEFIT_ABOVE_THRESHOLD"


@pytest.mark.parametrize("gate_stage", ["bootstrap_g0", "unknown"])
def test_frozen_linear_gate_rejects_non_deployable_artifacts(
    tmp_path: Path,
    gate_stage: str,
) -> None:
    with pytest.raises(ValueError, match="final_g1"):
        FrozenLinearBenefitGate.from_json(
            _gate_artifact(tmp_path / "gate.json", gate_stage=gate_stage)
        )


@pytest.mark.parametrize(
    ("field", "value", "message", "error_type"),
    [
        (
            "normalization_version",
            "zscore_v1",
            "raw normalization",
            ValueError,
        ),
        (
            "feature_order",
            ["instrument.unknown.value"],
            "unknown",
            ValueError,
        ),
        (
            "feature_order",
            "instrument.candidate_ambiguity.value",
            "array",
            TypeError,
        ),
    ],
)
def test_frozen_linear_gate_rejects_incompatible_feature_artifacts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
    error_type: type[Exception],
) -> None:
    artifact = _gate_artifact(tmp_path / "gate.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload[field] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error_type, match=message):
        FrozenLinearBenefitGate.from_json(artifact)


def test_factorized_candidates_are_bounded_and_include_current_selection() -> None:
    perception = _perception_result()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception)

    assert candidates.initial_prediction is perception.prediction
    assert candidates.allowed_ids["instrument"] == (0, 1)
    assert candidates.allowed_ids["ivt"] == (0, 1)
    assert candidates.source_frame_id == 12


def test_coordinator_accepts_candidate_bounded_joint_repair() -> None:
    perception = _perception_result()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    proposed = _prediction(instrument_ids=(1,))
    result = VerificationResult(
        scope="joint",
        prediction=proposed,
        provenance=ApiCallProvenance.local(),
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(action="VERIFY", scope="joint", reason="test"),
        candidates,
        result,
    )

    assert coordinated.prediction.instrument_ids == (1,)
    assert coordinated.verification_status == "VERIFIED_REPAIR"
    assert coordinated.selected_candidate_id is not None
    assert coordinated.reason == "REPAIR_WITHIN_CANDIDATE_POOL"


def test_coordinator_falls_back_to_keep_for_out_of_pool_repair() -> None:
    perception = _perception_result()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    proposed = _prediction(instrument_ids=(6,))
    result = VerificationResult(
        scope="joint",
        prediction=proposed,
        provenance=ApiCallProvenance.local(),
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(action="VERIFY", scope="joint", reason="test"),
        candidates,
        result,
    )

    assert coordinated.prediction is perception.prediction
    assert coordinated.verification_status == "FALLBACK_KEEP"
    assert coordinated.selected_candidate_id is None
    assert coordinated.reason == "CANDIDATE_OUT_OF_POOL"
