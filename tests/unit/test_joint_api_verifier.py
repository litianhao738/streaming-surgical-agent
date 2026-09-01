"""Same-backbone joint verifier request and response behavior."""

from __future__ import annotations

import json

from surgical_agent.api.contracts import ApiImageInput, ApiResponseRecord
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.perception.ontology_prompt import ACADEMIC_MEDICAL_CONTEXT
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    COMPACT_TASK_LAYOUT,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    TASK_LAYOUT,
)
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.research.verification.joint_api import (
    JointApiVerifier,
    JointVerificationRequestBuilder,
    load_joint_verification_prompt_text,
)


def _config(
    *,
    schema_version: str = JOINT_PERCEPTION_SCHEMA_VERSION,
) -> ApiConfig:
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock",
            "provider": "mock",
            "endpoint_identifier": "mock://local/p3",
            "requested_model_identifier": "mock-joint-v1",
            "prompt_version": schema_version,
            "response_schema_version": schema_version,
            "generation_parameters": {"max_output_tokens": 256},
        }
    )


def _prediction() -> InitialPrediction:
    return InitialPrediction(
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


def _perception() -> JointPerceptionResult:
    return JointPerceptionResult(
        prediction=_prediction(),
        raw_evidence=PerceptionEvidence(
            source="mock",
            ranked_candidates={
                task: tuple(
                    RankedCandidate(index, 1.0 - index / (count + 1))
                    for index in range(count)
                )
                for task, count in TASK_CLASS_COUNTS.items()
            },
            self_reported_confidence={task: 0.8 for task in TASK_CLASS_COUNTS},
            evidence_refs=(),
            source_max_frame_id=12,
        ),
        api_provenance=ApiCallProvenance.local(),
    )


def _context() -> PerceptionContext:
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
        frames=None,
        images=tuple(
            ApiImageInput(identifier, "image/png", identifier.encode())
            for identifier in sample.media_refs
        ),
        workflow_snapshot={
            "source_max_frame_id": 11,
            "recent_finalized_phases": ("0",),
        },
        memory_snapshot={
            "source_max_frame_id": 11,
            "events": ({"frame_id": 11, "phase_id": 0},),
        },
        prior_finalized_prediction=None,
    )


def _evidence() -> EvidenceProfile:
    common = {
        "candidate_ambiguity": EvidenceValue(0.2, True, "joint_rank_margin", 12),
        "ivt_internal_conflict": EvidenceValue(0.0, True, "ivt_component_map_v1", 12),
        "self_reported_uncertainty": EvidenceValue(
            0.2, True, "joint_self_reported_confidence", 12
        ),
        "temporal_set_change": EvidenceValue(
            0.0, True, "finalized_prior_jaccard", 11
        ),
    }
    return EvidenceProfile(
        video_id="VID01",
        frame_id=12,
        task_values={
            "instrument": common,
            "verb": common,
            "target": common,
            "ivt": common,
            "phase": {
                "candidate_ambiguity": common["candidate_ambiguity"],
                "phase_change_anomaly": EvidenceValue(
                    0.0, True, "frozen_phase_transition_graph", 11
                ),
                "self_reported_uncertainty": common[
                    "self_reported_uncertainty"
                ],
            },
        },
        global_values={"ivt_internal_conflict": common["ivt_internal_conflict"]},
    )


def _payload(
    *,
    schema_version: str = JOINT_PERCEPTION_SCHEMA_VERSION,
    task_layout: tuple[tuple[str, int], ...] = TASK_LAYOUT,
) -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": schema_version}
    for task, count in task_layout:
        topk = [
            {"id": index, "score": 1.0 - index / (count + 1)}
            for index in range(count)
        ]
        payload[task] = (
            {"selected_id": 0, "topk": topk}
            if task == "phase"
            else {"selected_ids": [0], "topk": topk}
        )
    payload["evidence_refs"] = [
        {"frame_id": 12, "code": "CURRENT_VISUAL_SUPPORT"}
    ]
    payload["self_reported_confidence"] = {
        task: 0.8 for task, _count in task_layout
    }
    return payload


class Client:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.requests: list[object] = []
        self.payload = _payload() if payload is None else payload

    def call(self, request: object) -> ApiResponseRecord:
        self.requests.append(request)
        return ApiResponseRecord(
            provider="mock",
            endpoint_identifier="mock://local/p3",
            request_hash="a" * 64,
            requested_model_identifier="mock-joint-v1",
            returned_model_identifier="mock-joint-v1",
            parsed_payload=self.payload,
            image_count=3,
            provider_call_count=1,
        )


def test_verification_request_has_distinct_prompt_and_bounded_candidate_payload() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )

    request = JointVerificationRequestBuilder(config=_config()).build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
    )
    body = json.loads(request.payload["input_text"])

    assert request.prompt_version == "joint_verification_v1"
    assert request.response_schema_version == JOINT_PERCEPTION_SCHEMA_VERSION
    assert "supplied strict response schema" in request.payload["system_text"]
    assert "joint_perception_frame_v1" not in request.payload["system_text"]
    assert body["scope"] == "joint"
    assert body["candidate_pool"]["instrument"] == [0, 1]
    assert body["memory_snapshot"]["source_max_frame_id"] == 11
    assert "ground_truth" not in request.payload["input_text"]


def test_openrouter_joint_verification_context_preserves_canonical_terms() -> None:
    prompt = load_joint_verification_prompt_text(academic_context=True)

    assert prompt.startswith(ACADEMIC_MEDICAL_CONTEXT)
    assert "surgical recognition hypothesis" in prompt
    assert "5=cut" in prompt
    assert "5=blood_vessel" in prompt


def test_joint_api_verifier_returns_auditable_verification_result() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    client = Client()
    verifier = JointApiVerifier(
        client=client,
        request_builder=JointVerificationRequestBuilder(config=_config()),
        data_upload_authorized=False,
    )

    result = verifier.verify(
        "joint",
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
    )

    assert len(client.requests) == 1
    assert result.scope == "joint"
    assert result.prediction.instrument_ids == (0,)
    assert result.provenance.request_hash == "a" * 64


def test_joint_api_verifier_accepts_compact_schema_and_restores_dense_scores() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    client = Client(
        _payload(
            schema_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
            task_layout=COMPACT_TASK_LAYOUT,
        )
    )
    verifier = JointApiVerifier(
        client=client,
        request_builder=JointVerificationRequestBuilder(
            config=_config(
                schema_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
            )
        ),
        data_upload_authorized=False,
    )

    result = verifier.verify(
        "joint",
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
    )

    request = client.requests[0]
    assert request.response_schema_version == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert len(result.prediction.probabilities["ivt"]) == 100
    assert sum(score > 0 for score in result.prediction.probabilities["ivt"]) == 8
