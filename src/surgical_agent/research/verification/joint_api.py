"""Same-backbone joint verification over a bounded, gold-free candidate pool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Protocol

from surgical_agent.api.contracts import ApiRequest, ApiResponseRecord, thaw_json
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.openrouter_routing import request_routing_payload
from surgical_agent.config.schema import ApiConfig
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import (
    PerceptionContext,
    require_gold_free,
)
from surgical_agent.perception.contracts import TASK_NAMES, ApiCallProvenance
from surgical_agent.perception.joint_api_vlm import (
    JointPerceptionRequestBuilder,
    safe_track_summary,
    safe_workflow_summary,
)
from surgical_agent.perception.ontology_prompt import (
    add_academic_medical_context,
    load_prompt_ontology_text,
)
from surgical_agent.perception.parser import parse_joint_perception_response
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    VerificationResult,
)

JOINT_VERIFICATION_PROMPT_VERSION = "joint_verification_v1"


class _ApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponseRecord: ...


def load_joint_verification_prompt_text(*, academic_context: bool = False) -> str:
    prompt = (
        files("surgical_agent.research.verification.prompts")
        .joinpath("joint_verification_prompt.txt")
        .read_text(encoding="utf-8")
    )
    if academic_context:
        prompt = add_academic_medical_context(prompt)
    ontology = load_prompt_ontology_text()
    return f"{prompt.rstrip()}\n\n{ontology}\n"


def _prediction_mapping(prediction: InitialPrediction) -> dict[str, object]:
    return {
        "instrument_ids": list(prediction.instrument_ids),
        "verb_ids": list(prediction.verb_ids),
        "target_ids": list(prediction.target_ids),
        "ivt_ids": list(prediction.triplet_ids),
        "phase_id": prediction.phase_id,
    }


def _evidence_value_mapping(value: EvidenceValue) -> dict[str, object]:
    return {
        "value": value.value,
        "available": value.available,
        "source": value.source,
        "source_max_frame_id": value.source_max_frame_id,
    }


def _evidence_mapping(profile: EvidenceProfile) -> dict[str, object]:
    return {
        "task_values": {
            task: {
                name: _evidence_value_mapping(value)
                for name, value in profile.task_values[task].items()
            }
            for task in TASK_NAMES
        },
        "global_values": {
            name: _evidence_value_mapping(value)
            for name, value in profile.global_values.items()
        },
        "evidence_version": profile.evidence_version,
    }


class JointVerificationRequestBuilder:
    """Build a distinct cached request while reusing the strict perception schema."""

    def __init__(self, *, config: ApiConfig) -> None:
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be ApiConfig")
        self._perception_builder = JointPerceptionRequestBuilder(config=config)
        self.config = config
        self.backend_name = f"{self._perception_builder.backend_name}_joint_verifier"

    def build(
        self,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: CandidateSet,
        evidence: EvidenceProfile,
    ) -> ApiRequest:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be a PerceptionContext")
        if not isinstance(prediction, InitialPrediction):
            raise TypeError("prediction must be an InitialPrediction")
        if not isinstance(candidates, CandidateSet):
            raise TypeError("candidates must be a CandidateSet")
        if not isinstance(evidence, EvidenceProfile):
            raise TypeError("evidence must be an EvidenceProfile")
        if candidates.initial_prediction is not prediction:
            raise ApiContractError(
                "candidate pool must belong to the current prediction"
            )
        if (
            evidence.video_id != context.sample.video_id
            or evidence.frame_id != context.sample.target_frame_id
            or candidates.source_frame_id > context.sample.target_frame_id
        ):
            raise ApiContractError("verification inputs do not match the current frame")
        if not 1 <= len(context.images) <= self.config.max_api_images:
            raise ApiContractError("verification exceeds the configured image budget")
        self._perception_builder._validate_prior(context)
        require_gold_free(context.memory_snapshot)
        workflow_summary = safe_workflow_summary(context.workflow_snapshot)
        source_max_frame_id = workflow_summary["source_max_frame_id"]
        if (
            source_max_frame_id is not None
            and source_max_frame_id >= context.sample.target_frame_id
        ):
            raise ApiContractError(
                "workflow source_max_frame_id must be strictly earlier than target"
            )
        track_summary = safe_track_summary(context.track_snapshot)
        track_source_max = track_summary["source_max_frame_id"]
        if (
            track_source_max is not None
            and track_source_max > context.sample.target_frame_id
        ):
            raise ApiContractError("track source_max_frame_id must not exceed target")
        memory_snapshot = thaw_json(context.memory_snapshot)
        if not isinstance(memory_snapshot, Mapping):
            raise ApiContractError("memory snapshot must be a mapping")
        payload = {
            "system_text": load_joint_verification_prompt_text(
                academic_context=self.config.provider
                in {"openai", "openrouter", "openai_compatible"}
            ),
            "input_text": json.dumps(
                {
                    "video_id": context.sample.video_id,
                    "target_frame_id": context.sample.target_frame_id,
                    "causal_frame_ids": list(context.sample.causal_frame_ids),
                    "selected_image_frame_ids": list(context.selected_image_frame_ids),
                    "temporal_evidence": thaw_json(context.temporal_evidence),
                    "scope": "joint",
                    "current_prediction": _prediction_mapping(prediction),
                    "candidate_pool": {
                        task: list(candidates.allowed_ids[task]) for task in TASK_NAMES
                    },
                    "evidence_profile": _evidence_mapping(evidence),
                    "workflow_summary": workflow_summary,
                    "track_summary": track_summary,
                    "memory_snapshot": memory_snapshot,
                    "ontology_version": "cholectrack20_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            **request_routing_payload(
                provider=self.config.provider,
                provider_options=self.config.provider_options,
            ),
        }
        return ApiRequest(
            provider=self.config.provider,
            model_identifier=self.config.requested_model_identifier,
            endpoint_identifier=self.config.endpoint_identifier,
            prompt_version=JOINT_VERIFICATION_PROMPT_VERSION,
            response_schema_version=self.config.response_schema_version,
            payload=payload,
            images=context.images,
            generation_parameters=self.config.generation_parameters,
        )


class JointApiVerifier:
    """Invoke one evidence-conditioned joint verification call."""

    enabled_scopes = ("joint",)

    def __init__(
        self,
        *,
        client: _ApiClient,
        request_builder: JointVerificationRequestBuilder,
        data_upload_authorized: bool = False,
    ) -> None:
        if not hasattr(client, "call") or not callable(client.call):
            raise TypeError("client must provide call(request)")
        if not isinstance(request_builder, JointVerificationRequestBuilder):
            raise TypeError("request_builder must be JointVerificationRequestBuilder")
        if type(data_upload_authorized) is not bool:
            raise TypeError("data_upload_authorized must be boolean")
        self.client = client
        self.request_builder = request_builder
        self.data_upload_authorized = data_upload_authorized

    def verify(
        self,
        scope: str,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: object,
        evidence: EvidenceProfile,
    ) -> VerificationResult:
        if scope != "joint":
            raise ApiContractError("joint verifier only supports the joint scope")
        if not isinstance(candidates, CandidateSet):
            raise TypeError("joint verifier requires a CandidateSet")
        if not self.data_upload_authorized and any(
            not image.identifier.startswith("synthetic:") for image in context.images
        ):
            raise ApiContractError(
                "only synthetic: image identifiers are permitted without upload authorization"
            )
        request = self.request_builder.build(
            context,
            prediction,
            candidates,
            evidence,
        )
        response = self.client.call(request)
        result = parse_joint_perception_response(
            response,
            video_id=context.sample.video_id,
            frame_id=context.sample.target_frame_id,
            backend=self.request_builder.backend_name,
        )
        return VerificationResult(
            scope=scope,
            prediction=result.prediction,
            provenance=ApiCallProvenance.from_response(
                response,
                source="joint_verifier",
            ),
        )
