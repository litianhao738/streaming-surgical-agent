"""Strict field-targeted verification request, response, and admission behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiImageInput, ApiResponseRecord
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.openrouter_routing import OPENROUTER_ROUTING_PAYLOAD_KEY
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.schema import (
    TARGETED_VERIFICATION_SCHEMA_VERSION,
    schema_for,
    validate_targeted_verification_payload,
)
from surgical_agent.api.usage import UsageLedger
from surgical_agent.config.loader import load_api_config
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
    validate_reliability_compact_joint_perception_payload,
)
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.verification.coordinator import DeterministicCoordinator
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.research.verification.targeted_api import (
    TARGETED_VERIFICATION_PROMPT_V6,
    TARGETED_VERIFICATION_PROMPT_V7,
    TargetedApiVerifier,
    TargetedVerificationRequestBuilder,
    load_targeted_verification_prompt_text,
    parse_targeted_verification_response,
)
from surgical_agent.research.verification.verifier import BoundTargetedSpecialist
from surgical_agent.systems.pipeline import GateDecision

PATHS = {
    "instrument": "/instrument/selected_ids",
    "verb": "/verb/selected_ids",
    "target": "/target/selected_ids",
    "ivt": "/ivt/selected_ids",
    "phase": "/phase/selected_id",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config() -> ApiConfig:
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock",
            "provider": "mock",
            "endpoint_identifier": "mock://local/p3",
            "requested_model_identifier": "mock-joint-v1",
            "prompt_version": "joint_perception_reliability_compact_v2",
            "response_schema_version": "joint_perception_reliability_compact_v2",
            "generation_parameters": {"max_output_tokens": 256},
        }
    )


def _fixed_visual_config() -> ApiConfig:
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock",
            "provider": "mock",
            "endpoint_identifier": "mock://local/p3",
            "requested_model_identifier": "mock-joint-v1",
            "prompt_version": "joint_perception_reliability_compact_v2",
            "response_schema_version": "joint_perception_reliability_compact_v2",
            "generation_parameters": {"max_output_tokens": 256},
            "provider_options": {
                "verification_prompt_profile": "fixed_visual_only"
            },
            "max_causal_frames": 6,
            "max_api_images": 6,
        }
    )


def _probabilities() -> dict[str, tuple[float, ...]]:
    return {
        task: tuple(1.0 - index / (count + 1) for index in range(count))
        for task, count in TASK_CLASS_COUNTS.items()
    }


def _prediction(**changes: object) -> InitialPrediction:
    return replace(
        InitialPrediction(
            instrument_ids=(0,),
            verb_ids=(0,),
            target_ids=(0,),
            triplet_ids=(0,),
            phase_id=0,
            probabilities=_probabilities(),
            backend="joint_mock",
            score_semantics="uncalibrated_rank_v1",
        ),
        **changes,
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
        track_snapshot={
            "status": "UNAVAILABLE",
            "source_max_frame_id": None,
            "frames": (),
        },
    )


def _six_frame_context() -> PerceptionContext:
    base = _context()
    frame_ids = tuple(range(7, 13))
    refs = tuple(f"synthetic:{frame_id}" for frame_id in frame_ids)
    sample = InferenceSample(
        video_id="VID01",
        target_frame_id=12,
        causal_frame_ids=frame_ids,
        media_refs=refs,
        source_split=DatasetSplit.TESTING,
        alignment_version="unit-test",
    )
    return replace(
        base,
        sample=sample,
        images=tuple(
            ApiImageInput(identifier, "image/png", identifier.encode())
            for identifier in refs
        ),
        selected_image_frame_ids=frame_ids,
        image_details=("low", "low", "low", "low", "low", "auto"),
        temporal_evidence={
            "selection_strategy": "fixed_all",
            "selected_image_frame_ids": frame_ids,
        },
    )
def _evidence() -> EvidenceProfile:
    common = {
        "candidate_ambiguity": EvidenceValue(0.2, True, "joint_rank_margin", 12),
        "ivt_internal_conflict": EvidenceValue(
            0.0, True, "ivt_component_map_v1", 12
        ),
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


def _field(
    task: str,
    *,
    selected_ids: list[int] | None = None,
    status: str = "Verified",
) -> dict[str, object]:
    selected = [0] if selected_ids is None else selected_ids
    return {
        "path": PATHS[task],
        "selected_ids": selected,
        "topk": [
            {"id": 0, "confidence": 0.9},
            {"id": 1, "confidence": 0.8},
        ],
        "status": status,
        "uncertainty": None,
    }


def _payload(*fields: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": TARGETED_VERIFICATION_SCHEMA_VERSION,
        "fields": list(fields),
    }


def _response(payload: dict[str, object]) -> ApiResponseRecord:
    return ApiResponseRecord(
        provider="mock",
        endpoint_identifier="mock://local/p3",
        request_hash="a" * 64,
        requested_model_identifier="mock-joint-v1",
        returned_model_identifier="mock-joint-v1",
        parsed_payload=payload,
        image_count=3,
        provider_call_count=1,
    )


def test_targeted_schema_is_registered_and_semantically_strict() -> None:
    payload = _payload(_field("instrument"), _field("phase"))

    validate_targeted_verification_payload(payload)
    schema = schema_for(TARGETED_VERIFICATION_SCHEMA_VERSION)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == (
        TARGETED_VERIFICATION_SCHEMA_VERSION
    )

    def mappings(value: object):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from mappings(child)
        elif isinstance(value, list):
            for child in value:
                yield from mappings(child)

    # OpenAI strict structured outputs reject the JSON Schema uniqueItems
    # keyword. Runtime semantic validation below still enforces uniqueness.
    assert all("uniqueItems" not in mapping for mapping in mappings(schema))


def test_targeted_schema_rejects_verified_with_explicit_uncertainty() -> None:
    field = _field("phase")
    field["uncertainty"] = {
        "reason": "CLOSE_ALTERNATIVES",
        "alternative_ids": [1],
    }

    with pytest.raises(ApiSchemaError):
        validate_targeted_verification_payload(_payload(field))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=True),
        lambda payload: payload["fields"][0].update(extra=True),
        lambda payload: payload["fields"][0].update(path="/unknown"),
        lambda payload: payload["fields"][0]["topk"].append(
            {"id": 0, "confidence": 0.7}
        ),
        lambda payload: payload["fields"][0].update(selected_ids=[2]),
        lambda payload: payload["fields"][0]["topk"].reverse(),
        lambda payload: payload["fields"].append(_field("instrument")),
        lambda payload: payload["fields"].append(
            _field("phase", selected_ids=[0, 1])
        ),
    ],
)
def test_targeted_schema_rejects_malformed_or_semantically_invalid_fields(
    mutation: object,
) -> None:
    payload = _payload(_field("instrument"))
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ApiSchemaError):
        validate_targeted_verification_payload(payload)


def test_candidates_carry_deterministic_confidence_records_from_perception() -> None:
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        _perception()
    )

    assert candidates.allowed_ids["instrument"] == (0, 1)
    assert tuple(
        (record.class_id, record.confidence)
        for record in candidates.candidate_records["instrument"]
    ) == ((0, 1.0), (1, 0.875))


def test_candidates_complete_the_seven_class_phase_ontology() -> None:
    perception = _perception()
    ranked = dict(perception.raw_evidence.ranked_candidates)
    ranked["phase"] = ranked["phase"][:3]
    compact = replace(
        perception,
        raw_evidence=replace(perception.raw_evidence, ranked_candidates=ranked),
    )

    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(compact)

    assert candidates.candidate_version == "factorized_closure_topk_phase_complete_v3"
    assert candidates.allowed_ids["phase"] == tuple(range(7))
    assert tuple(
        record.class_id for record in candidates.candidate_records["phase"]
    ) == tuple(range(7))
    assert tuple(
        record.confidence for record in candidates.candidate_records["phase"][3:]
    ) == (0.0, 0.0, 0.0, 0.0)


def test_default_candidates_preserve_all_eight_compact_ivt_ranks() -> None:
    candidates = FactorizedCandidateGenerator().build(_perception())

    assert len(candidates.allowed_ids["ivt"]) == 8
    assert candidates.allowed_ids["ivt"] == tuple(range(8))


def test_request_blinds_current_values_and_upstream_candidate_scores() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )

    request = TargetedVerificationRequestBuilder(config=_config()).build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("verb", "phase"),
    )
    body = json.loads(request.payload["input_text"])
    system_text = request.payload["system_text"]

    assert request.prompt_version == TARGETED_VERIFICATION_PROMPT_V6
    assert request.response_schema_version == TARGETED_VERIFICATION_SCHEMA_VERSION
    assert body["verification_protocol"] == "blind_candidate_selection_v1"
    assert body["flagged_fields"] == [PATHS["verb"], PATHS["phase"]]
    assert "current_fields" not in body
    assert set(body["candidate_fields"]) == {PATHS["verb"], PATHS["phase"]}
    assert body["candidate_fields"][PATHS["verb"]] == [0, 2]
    assert body["candidate_fields"][PATHS["phase"]] == list(range(7))
    assert body["required_fields"] == [
        {
            "path": PATHS["verb"],
            "required_ids": [2],
            "reason": "FROZEN_UNREQUESTED_IVT_CLOSURE",
        }
    ]
    assert set(body["evidence_profile"]["task_values"]) == {"verb", "phase"}
    assert "ground_truth" not in request.payload["input_text"]
    for forbidden in (
        "unflagged fields",
        "full-task recomputation",
        "chain of thought",
        "Report",
        "clinical significance",
        "next step",
    ):
        assert forbidden in system_text
    assert "current prediction and its upstream ranking scores are intentionally hidden" in system_text
    assert "Candidate order has no semantic meaning" in system_text
    assert "compare all seven workflow phases" in system_text
    assert "Every returned label describes target_frame_id only" in system_text
    assert "never union instruments" in system_text
    assert "hard closure constraints" in system_text


def test_checked_in_gemini_verifier_remains_blind_and_provider_pinned() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    config = load_api_config(
        PROJECT_ROOT
        / "configs/perception/targeted_openrouter_gemini31pro_fixed6.yaml"
    )

    request = TargetedVerificationRequestBuilder(config=config).build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("instrument",),
    )
    body = json.loads(request.payload["input_text"])

    assert request.model_identifier == "google/gemini-3.1-pro-preview"
    assert (
        request.payload[OPENROUTER_ROUTING_PAYLOAD_KEY]
        == "strict_google_ai_studio"
    )
    assert "current_fields" not in body
    assert "current prediction" not in request.payload["input_text"]


def test_conservative_same_model_request_exposes_h0_without_upstream_scores() -> None:
    perception = _perception()
    config = load_api_config(
        PROJECT_ROOT
        / "configs/perception/targeted_openrouter_gpt56sol_constrained_fixed6.yaml"
    )
    request = TargetedVerificationRequestBuilder(config=config).build(
        _context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("phase",),
        scope="workflow",
    )
    body = json.loads(request.payload["input_text"])

    assert request.prompt_version == TARGETED_VERIFICATION_PROMPT_V7
    assert body["verification_protocol"] == "conservative_h0_comparison_v1"
    assert body["current_fields"] == {PATHS["phase"]: [0]}
    assert body["admission_policy"]["default"] == "KEEP_H0"
    assert "upstream" not in request.payload["input_text"]
    assert "Verified requires" in request.payload["system_text"]


def test_interaction_request_exposes_only_closure_safe_ivt_selection() -> None:
    perception = _perception()
    config = load_api_config(
        PROJECT_ROOT
        / "configs/perception/targeted_openrouter_gpt56sol_constrained_fixed6.yaml"
    )
    request = TargetedVerificationRequestBuilder(config=config).build(
        _context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("ivt",),
        scope="interaction",
    )
    body = json.loads(request.payload["input_text"])

    assert body["flagged_fields"] == [PATHS["ivt"]]
    assert set(body["candidate_fields"]) == {PATHS["ivt"]}
    assert body["candidate_fields"][PATHS["ivt"]] == [0, 1]
    assert "deterministically derives instrument" in request.payload["system_text"]


def test_fixed_visual_verifier_keeps_all_six_frames_and_omits_tracker_state() -> None:
    perception = _perception()
    request = TargetedVerificationRequestBuilder(
        config=_fixed_visual_config()
    ).build(
        _six_frame_context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("verb", "phase"),
    )
    body = json.loads(request.payload["input_text"])

    assert len(request.images) == 6
    assert body["selected_image_frame_ids"] == [7, 8, 9, 10, 11, 12]
    assert body["temporal_evidence"]["selection_strategy"] == "fixed_all"
    assert "track_summary" not in body
    assert "workflow_summary" not in body
    assert "memory_snapshot" not in body


def test_builder_accepts_second_attempt_hypothesis_inside_frozen_pool() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=3).build(
        perception
    )
    current = replace(perception.prediction, verb_ids=(1,))

    builder = TargetedVerificationRequestBuilder(config=_config())
    baseline = builder.build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("verb",),
    )
    request = builder.build(
        _context(),
        current,
        candidates,
        _evidence(),
        requested_fields=("verb",),
    )

    assert request.payload["input_text"] == baseline.payload["input_text"]
    assert "current_fields" not in json.loads(request.payload["input_text"])


def test_instrument_request_exposes_only_frozen_ivt_closure_requirement() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )

    request = TargetedVerificationRequestBuilder(config=_config()).build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("instrument",),
    )
    body = json.loads(request.payload["input_text"])

    assert "current_fields" not in body
    assert body["required_fields"] == [
        {
            "path": PATHS["instrument"],
            "required_ids": [0],
            "reason": "FROZEN_UNREQUESTED_IVT_CLOSURE",
        }
    ]


def test_openrouter_targeted_context_preserves_canonical_terms() -> None:
    prompt = load_targeted_verification_prompt_text(academic_context=True)

    assert prompt.startswith(ACADEMIC_MEDICAL_CONTEXT)
    assert "clinical significance" in prompt
    assert "5=cut" in prompt
    assert "5=blood_vessel" in prompt


def test_builder_uses_v6_scoped_blind_prompt_without_changing_wire_schema() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    builder = TargetedVerificationRequestBuilder(config=_config())

    request = builder.build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("verb", "ivt"),
    )
    prompt = " ".join(request.payload["system_text"].split())

    assert request.prompt_version == TARGETED_VERIFICATION_PROMPT_V6
    assert request.response_schema_version == TARGETED_VERIFICATION_SCHEMA_VERSION
    assert "tool motion and tissue interaction" in prompt
    assert "ontology closure" in prompt
    assert "independent bounded classification" in prompt
    assert "label only the action active at the target frame" in prompt
    assert "satisfies required_fields" in prompt
    assert "Do not expose this cross-check as reasoning" in prompt


def test_builder_sends_only_scope_relevant_ontology() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    builder = TargetedVerificationRequestBuilder(config=_config())

    instrument = builder.build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("instrument",),
    ).payload["system_text"]
    workflow = builder.build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("phase",),
    ).payload["system_text"]
    interaction = builder.build(
        _context(),
        perception.prediction,
        candidates,
        _evidence(),
        requested_fields=("instrument", "verb", "target", "ivt"),
    ).payload["system_text"]
    instrument_ontology = instrument.split(
        "Operational CholecTrack20 ontology subset", maxsplit=1
    )[1]
    workflow_ontology = workflow.split(
        "Operational CholecTrack20 ontology subset", maxsplit=1
    )[1]
    interaction_ontology = interaction.split(
        "Operational CholecTrack20 ontology subset", maxsplit=1
    )[1]

    assert "instrument:0=grasper" in instrument_ontology
    assert "verb:" not in instrument_ontology
    assert "phase:" not in instrument_ontology
    assert "IVT candidate id=" not in instrument_ontology
    assert "phase:0=preparation" in workflow_ontology
    assert "instrument:" not in workflow_ontology
    assert "IVT candidate id=" not in workflow_ontology
    assert "IVT candidate id=" in interaction_ontology
    assert "99=(" not in interaction_ontology


def test_parser_patches_only_requested_verified_field_and_preserves_identity() -> None:
    initial = _prediction()
    result = parse_targeted_verification_response(
        _response(_payload(_field("verb", selected_ids=[1]))),
        initial=initial,
        requested_fields=("verb",),
    )

    assert result.prediction is not initial
    assert result.prediction.verb_ids == (1,)
    assert result.requested_fields == ("verb",)
    assert result.repaired_fields == ("verb",)
    assert tuple(outcome.status for outcome in result.field_outcomes) == ("Verified",)
    for task, attribute in (
        ("instrument", "instrument_ids"),
        ("target", "target_ids"),
        ("ivt", "triplet_ids"),
    ):
        assert getattr(result.prediction, attribute) is getattr(initial, attribute)
        assert result.prediction.probabilities[task] is initial.probabilities[task]
    assert result.prediction.phase_id == initial.phase_id
    assert result.prediction.probabilities["phase"] is initial.probabilities["phase"]


def test_parser_canonicalizes_confidence_ordered_multilabel_selection() -> None:
    initial = _prediction(target_ids=(0,))
    result = parse_targeted_verification_response(
        _response(_payload(_field("target", selected_ids=[1, 0]))),
        initial=initial,
        requested_fields=("target",),
    )

    assert result.prediction.target_ids == (0, 1)
    assert result.field_outcomes[0].selected_ids == (0, 1)
    assert result.repaired_fields == ("target",)


@pytest.mark.parametrize("status", ["Pending", "Rejected"])
def test_parser_keeps_entire_initial_object_for_unresolved_field(status: str) -> None:
    initial = _prediction()
    field = _field("verb", selected_ids=[1], status=status)
    field["uncertainty"] = {
        "reason": "CLOSE_ALTERNATIVES",
        "alternative_ids": [0],
    }

    result = parse_targeted_verification_response(
        _response(_payload(field)),
        initial=initial,
        requested_fields=("verb",),
    )

    assert result.prediction is initial
    assert result.repaired_fields == ()


def test_parser_rejects_missing_or_out_of_scope_returned_paths() -> None:
    with pytest.raises(ApiSchemaError, match="requested paths"):
        parse_targeted_verification_response(
            _response(_payload(_field("instrument"))),
            initial=_prediction(),
            requested_fields=("instrument", "phase"),
        )


@pytest.mark.parametrize(
    ("status", "expected_status"),
    [("Pending", "VERIFIED_PENDING"), ("Rejected", "VERIFIED_REJECT")],
)
def test_coordinator_keeps_initial_for_pending_and_rejected_outcomes(
    status: str,
    expected_status: str,
) -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    verification = parse_targeted_verification_response(
        _response(_payload(_field("verb", status=status))),
        initial=perception.prediction,
        requested_fields=("verb",),
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(
            action="VERIFY",
            scope="joint",
            reason="test",
            flagged_fields=("verb",),
        ),
        candidates,
        verification,
    )

    assert coordinated.prediction is perception.prediction
    assert coordinated.verification_status == expected_status


def test_coordinator_rejects_changes_outside_requested_fields() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    invalid = replace(
        parse_targeted_verification_response(
            _response(_payload(_field("verb"))),
            initial=perception.prediction,
            requested_fields=("verb",),
        ),
        prediction=_prediction(instrument_ids=(1,)),
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(
            action="VERIFY",
            scope="joint",
            reason="test",
            flagged_fields=("verb",),
        ),
        candidates,
        invalid,
    )

    assert coordinated.prediction is perception.prediction
    assert coordinated.verification_status == "FALLBACK_KEEP"
    assert coordinated.reason == "REQUESTED_TASK_VIOLATION"


def test_coordinator_admits_requested_repair_for_targeted_scope() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=3).build(
        perception
    )
    verification = parse_targeted_verification_response(
        _response(_payload(_field("verb", selected_ids=[1]))),
        initial=perception.prediction,
        requested_fields=("verb",),
        scope="targeted",
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(
            action="VERIFY",
            scope="targeted",
            reason="RELIABILITY_FINDINGS",
            flagged_fields=("verb",),
        ),
        candidates,
        verification,
    )

    assert coordinated.verification_status == "VERIFIED_REPAIR"
    assert coordinated.prediction.verb_ids == (1,)
    assert coordinated.touched_tasks == ("verb",)


def test_coordinator_admits_verified_keep_for_targeted_scope() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    verification = parse_targeted_verification_response(
        _response(_payload(_field("verb"))),
        initial=perception.prediction,
        requested_fields=("verb",),
        scope="targeted",
    )

    coordinated = DeterministicCoordinator().coordinate(
        perception.prediction,
        GateDecision(
            action="VERIFY",
            scope="targeted",
            reason="RELIABILITY_FINDINGS",
            flagged_fields=("verb",),
        ),
        candidates,
        verification,
    )

    assert coordinated.verification_status == "VERIFIED_KEEP"
    assert coordinated.prediction is perception.prediction


class Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[object] = []

    def call(self, request: object) -> ApiResponseRecord:
        self.requests.append(request)
        return _response(self.payload)


def test_interaction_specialist_derives_closed_components_from_ivt_only() -> None:
    perception = _perception()
    candidates = FactorizedCandidateGenerator(max_candidates_per_task=2).build(
        perception
    )
    client = Client(_payload(_field("ivt", selected_ids=[0, 1])))
    specialist = BoundTargetedSpecialist(
        verifier=TargetedApiVerifier(
            client=client,
            request_builder=TargetedVerificationRequestBuilder(config=_config()),
        ),
        context=_context(),
        candidates=candidates,
        evidence=_evidence(),
    )

    result = specialist.verify(
        hypothesis=perception.prediction,
        scope="interaction",
        attempt=1,
    )

    assert result.status == "REPAIR"
    assert result.hypothesis is not None
    assert result.hypothesis.triplet_ids == (0, 1)
    assert result.hypothesis.instrument_ids == (0,)
    assert result.hypothesis.verb_ids == (2,)
    assert result.hypothesis.target_ids == (0, 1)
    body = json.loads(client.requests[0].payload["input_text"])
    assert body["flagged_fields"] == [PATHS["ivt"]]


def test_targeted_verifier_returns_ordered_outcomes_for_one_request() -> None:
    perception = _perception()
    client = Client(_payload(_field("verb"), _field("phase")))
    verifier = TargetedApiVerifier(
        client=client,
        request_builder=TargetedVerificationRequestBuilder(config=_config()),
    )

    result = verifier.verify(
        "targeted",
        _context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("verb", "phase"),
    )

    assert len(client.requests) == 1
    assert result.scope == "targeted"
    assert result.requested_fields == ("verb", "phase")
    assert tuple(outcome.path for outcome in result.field_outcomes) == (
        PATHS["verb"],
        PATHS["phase"],
    )


def test_mock_provider_returns_only_the_requested_targeted_fields() -> None:
    perception = _perception()
    request = TargetedVerificationRequestBuilder(config=_config()).build(
        _context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("verb", "phase"),
    )

    response = MockProviderTransport().send(request)

    validate_targeted_verification_payload(response.parsed_payload)
    assert tuple(
        field["path"] for field in response.parsed_payload["fields"]
    ) == (PATHS["verb"], PATHS["phase"])


def test_client_uses_request_schema_for_targeted_live_and_cache(
    tmp_path: Path,
) -> None:
    perception = _perception()
    request = TargetedVerificationRequestBuilder(config=_config()).build(
        _context(),
        perception.prediction,
        FactorizedCandidateGenerator(max_candidates_per_task=2).build(perception),
        _evidence(),
        requested_fields=("verb",),
    )
    transport = MockProviderTransport()
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(tmp_path / "cache"),
        usage=UsageLedger(tmp_path / "usage.jsonl"),
        validator=validate_reliability_compact_joint_perception_payload,
    )

    live = client.call(request)
    replay = client.call(request)

    assert live.cache_hit is False
    assert replay.cache_hit is True
    assert transport.provider_call_count == 1
