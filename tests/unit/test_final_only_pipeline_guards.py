"""Keep label-only H0 out of research policies trained on ranking evidence."""

from dataclasses import replace
from pathlib import Path

import pytest

from surgical_agent.config.final_experiment import load_tracker_gate_cell
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
)
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.research.safety import DecisionSupportBuilder
from surgical_agent.research.verification.contracts import CandidateSet
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem
from surgical_agent.systems.final_pipeline_factory import build_final_api_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config():
    return replace(
        load_api_config(PROJECT_ROOT / "configs/perception/joint_mock_final_fixed3.yaml"),
        response_schema_version=FINAL_ONLY_SCHEMA_VERSION,
    )


@pytest.mark.parametrize(
    "profile",
    (
        "always_verify",
        "rule_gate",
        "selective_verify",
        "cascade_verify",
        "learned_gate",
        "counterfactual_verify",
    ),
)
def test_final_only_rejects_legacy_gate_profiles_before_client_use(profile):
    with pytest.raises(ValueError, match="requires pipeline_profile=single_pass"):
        DatasetApiPipelineSystem(
            client=None,
            config=_config(),
            writer=None,
            media_loader=None,
            pipeline_profile=profile,
        )


@pytest.mark.parametrize("context", ("workflow", "track_only", "track_workflow"))
def test_final_only_rejects_explicit_stateful_context(context):
    with pytest.raises(ValueError, match="context_profile=frames_only"):
        DatasetApiPipelineSystem(
            client=None,
            config=_config(),
            writer=None,
            media_loader=None,
            context_profile=context,
        )


def test_final_only_rejects_explicit_event_memory():
    with pytest.raises(ValueError, match="event_memory_enabled=False"):
        DatasetApiPipelineSystem(
            client=None,
            config=_config(),
            writer=None,
            media_loader=None,
            event_memory_enabled=True,
        )


@pytest.mark.parametrize("cell_name", ("a_base", "c_gate"))
def test_formal_factory_rejects_hard_labels_for_rule_and_learned_gates(cell_name):
    with pytest.raises(ValueError, match="has no confidence rankings"):
        build_final_api_pipeline(
            client=None,
            api_config=_config(),
            verification_api_config=load_api_config(
                PROJECT_ROOT / "configs/perception/targeted_mock_final_fixed3.yaml"
            ),
            cell=load_tracker_gate_cell(
                PROJECT_ROOT / f"configs/ablations/{cell_name}.yaml"
            ),
        )


def test_direct_gate_support_cannot_treat_hard_labels_as_zero_confidence():
    prediction = InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(2,),
        target_ids=(1,),
        triplet_ids=(0,),
        phase_id=0,
        probabilities={task: (0.0,) * count for task, count in TASK_CLASS_COUNTS.items()},
        score_semantics="hard_label_v1",
    )
    perception = JointPerceptionResult(
        prediction=prediction,
        raw_evidence=PerceptionEvidence(
            source="joint_final_only",
            ranked_candidates={task: () for task in TASK_CLASS_COUNTS},
            self_reported_confidence={task: None for task in TASK_CLASS_COUNTS},
            evidence_refs=(),
            source_max_frame_id=51,
        ),
        api_provenance=ApiCallProvenance.local(),
    )
    candidates = CandidateSet(
        initial_prediction=prediction,
        allowed_ids={
            "instrument": (0,),
            "verb": (2,),
            "target": (1,),
            "ivt": (0,),
            "phase": (0,),
        },
        source_frame_id=51,
    )

    with pytest.raises(ValueError, match="requires confidence rankings"):
        DecisionSupportBuilder().build(
            perception=perception,
            candidates=candidates,
            violations=(),
            tracker_snapshot={},
            previous=None,
        )
