"""Gold-free API verification limited to Gate-requested semantic fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from importlib.resources import files
from typing import Protocol

from surgical_agent.api.contracts import ApiRequest, ApiResponseRecord, thaw_json
from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.openrouter_routing import request_routing_payload
from surgical_agent.api.schema import (
    TARGETED_VERIFICATION_SCHEMA_VERSION,
    validate_targeted_verification_payload,
)
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import (
    PerceptionContext,
    require_gold_free,
)
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    RankedCandidate,
)
from surgical_agent.perception.joint_api_vlm import (
    safe_track_summary,
    safe_workflow_summary,
)
from surgical_agent.perception.ontology_prompt import (
    add_academic_medical_context,
    load_prompt_ontology_text,
)
from surgical_agent.research.reliability.state import TASK_PATHS, canonical_tasks
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    FieldVerificationOutcome,
    VerificationResult,
    VerificationUncertainty,
)


class _ApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponseRecord: ...


def load_targeted_verification_prompt_text(*, academic_context: bool = False) -> str:
    prompt = (
        files("surgical_agent.research.verification.prompts")
        .joinpath("targeted_verification_prompt.txt")
        .read_text(encoding="utf-8")
    )
    if academic_context:
        prompt = add_academic_medical_context(prompt)
    ontology = load_prompt_ontology_text()
    return f"{prompt.rstrip()}\n\n{ontology}\n"


def _selected_ids(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    if task == "instrument":
        return prediction.instrument_ids
    if task == "verb":
        return prediction.verb_ids
    if task == "target":
        return prediction.target_ids
    if task == "ivt":
        return prediction.triplet_ids
    if task == "phase":
        return (prediction.phase_id,)
    raise KeyError(task)


def _evidence_value(value: EvidenceValue) -> dict[str, object]:
    return {
        "value": value.value,
        "available": value.available,
        "source": value.source,
        "source_max_frame_id": value.source_max_frame_id,
    }


def _evidence_mapping(
    profile: EvidenceProfile, requested_fields: tuple[str, ...]
) -> dict[str, object]:
    return {
        "task_values": {
            task: {
                name: _evidence_value(value)
                for name, value in profile.task_values[task].items()
            }
            for task in requested_fields
        },
        "global_values": {
            name: _evidence_value(value)
            for name, value in profile.global_values.items()
        },
        "evidence_version": profile.evidence_version,
    }


class TargetedVerificationRequestBuilder:
    """Build one strict request containing only the flagged field hypotheses."""

    def __init__(self, *, config: ApiConfig) -> None:
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be ApiConfig")
        self.config = config
        self.backend_name = "joint_api_vlm_targeted_verifier"

    def build(
        self,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: CandidateSet,
        evidence: EvidenceProfile,
        *,
        requested_fields: tuple[str, ...],
    ) -> ApiRequest:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be a PerceptionContext")
        if not isinstance(prediction, InitialPrediction):
            raise TypeError("prediction must be an InitialPrediction")
        if not isinstance(candidates, CandidateSet):
            raise TypeError("candidates must be a CandidateSet")
        if not isinstance(evidence, EvidenceProfile):
            raise TypeError("evidence must be an EvidenceProfile")
        requested = canonical_tasks(tuple(requested_fields))
        if not requested:
            raise ApiContractError("targeted verification requires requested fields")
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
        require_gold_free(context.memory_snapshot)
        workflow_summary = safe_workflow_summary(context.workflow_snapshot)
        track_summary = safe_track_summary(context.track_snapshot)
        memory_snapshot = thaw_json(context.memory_snapshot)
        if not isinstance(memory_snapshot, Mapping):
            raise ApiContractError("memory snapshot must be a mapping")
        candidate_fields: dict[str, list[dict[str, object]]] = {}
        for task in requested:
            records = candidates.candidate_records[task]
            if not records or len(records) > 8:
                raise ApiContractError(
                    "targeted candidate records must contain one to eight items"
                )
            candidate_fields[TASK_PATHS[task]] = [
                {"id": record.class_id, "confidence": record.confidence}
                for record in records
            ]
        input_payload = {
            "video_id": context.sample.video_id,
            "target_frame_id": context.sample.target_frame_id,
            "causal_frame_ids": list(context.sample.causal_frame_ids),
            "selected_image_frame_ids": list(context.selected_image_frame_ids),
            "temporal_evidence": thaw_json(context.temporal_evidence),
            "flagged_fields": [TASK_PATHS[task] for task in requested],
            "current_fields": [
                {
                    "path": TASK_PATHS[task],
                    "selected_ids": list(_selected_ids(prediction, task)),
                }
                for task in requested
            ],
            "candidate_fields": candidate_fields,
            "evidence_profile": _evidence_mapping(evidence, requested),
            "workflow_summary": workflow_summary,
            "track_summary": track_summary,
            "memory_snapshot": memory_snapshot,
            "ontology_version": "cholectrack20_v1",
        }
        prompt_profile = self.config.provider_options.get(
            "verification_prompt_profile", "full_context"
        )
        if prompt_profile not in {"full_context", "delta_visual_only"}:
            raise ApiContractError("verification_prompt_profile is unsupported")
        request_images = context.images
        image_details = context.image_details or tuple("auto" for _ in context.images)
        if prompt_profile == "delta_visual_only":
            input_payload = {
                "video_id": context.sample.video_id,
                "target_frame_id": context.sample.target_frame_id,
                "causal_frame_ids": list(context.sample.causal_frame_ids),
                "flagged_fields": [TASK_PATHS[task] for task in requested],
                "current_fields": input_payload["current_fields"],
                "candidate_fields": candidate_fields,
                "evidence_profile": _evidence_mapping(evidence, requested),
                "ontology_version": "cholectrack20_v1",
            }
            request_images = (context.images[-1],)
            image_details = (image_details[-1],)
        return ApiRequest(
            provider=self.config.provider,
            model_identifier=self.config.requested_model_identifier,
            endpoint_identifier=self.config.endpoint_identifier,
            prompt_version=TARGETED_VERIFICATION_SCHEMA_VERSION,
            response_schema_version=TARGETED_VERIFICATION_SCHEMA_VERSION,
            payload={
                "system_text": load_targeted_verification_prompt_text(
                    academic_context=self.config.provider in {"openai", "openrouter"}
                ),
                "image_details": list(image_details),
                "input_text": json.dumps(
                    input_payload, sort_keys=True, separators=(",", ":")
                ),
                **request_routing_payload(
                    provider=self.config.provider,
                    provider_options=self.config.provider_options,
                ),
            },
            images=request_images,
            generation_parameters=self.config.generation_parameters,
        )


def parse_targeted_verification_response(
    response: ApiResponseRecord,
    *,
    initial: InitialPrediction,
    requested_fields: tuple[str, ...],
    scope: str = "joint",
) -> VerificationResult:
    """Parse exact returned paths and patch only fully Verified requested fields."""

    if not isinstance(response, ApiResponseRecord):
        raise TypeError("response must be an ApiResponseRecord")
    if not isinstance(initial, InitialPrediction):
        raise TypeError("initial must be an InitialPrediction")
    requested = canonical_tasks(tuple(requested_fields))
    if not requested:
        raise ApiSchemaError("Targeted verification requires requested paths")
    validate_targeted_verification_payload(response.parsed_payload)
    raw_fields = response.parsed_payload["fields"]
    returned_paths = tuple(field["path"] for field in raw_fields)
    expected_paths = tuple(TASK_PATHS[task] for task in requested)
    if returned_paths != expected_paths:
        raise ApiSchemaError(
            "Targeted verification returned paths must equal requested paths"
        )
    outcomes: list[FieldVerificationOutcome] = []
    for field in raw_fields:
        uncertainty_value = field["uncertainty"]
        uncertainty = None
        if uncertainty_value is not None:
            uncertainty = VerificationUncertainty(
                reason=uncertainty_value["reason"],
                alternative_ids=tuple(uncertainty_value["alternative_ids"]),
            )
        outcomes.append(
            FieldVerificationOutcome(
                path=field["path"],
                selected_ids=tuple(field["selected_ids"]),
                candidate_records=tuple(
                    RankedCandidate(record["id"], record["confidence"])
                    for record in field["topk"]
                ),
                status=field["status"],
                uncertainty=uncertainty,
            )
        )
    repaired = tuple(
        outcome.task
        for outcome in outcomes
        if outcome.status == "Verified"
        and outcome.selected_ids != _selected_ids(initial, outcome.task)
    )
    proposed = initial
    if all(outcome.status == "Verified" for outcome in outcomes) and repaired:
        updates: dict[str, object] = {}
        probabilities = dict(initial.probabilities)
        for outcome in outcomes:
            if outcome.task not in repaired:
                continue
            dense = [0.0] * TASK_CLASS_COUNTS[outcome.task]
            for record in outcome.candidate_records:
                dense[record.class_id] = record.confidence
            probabilities[outcome.task] = tuple(dense)
            if outcome.task == "instrument":
                updates["instrument_ids"] = outcome.selected_ids
            elif outcome.task == "verb":
                updates["verb_ids"] = outcome.selected_ids
            elif outcome.task == "target":
                updates["target_ids"] = outcome.selected_ids
            elif outcome.task == "ivt":
                updates["triplet_ids"] = outcome.selected_ids
            else:
                updates["phase_id"] = outcome.selected_ids[0]
        updates["probabilities"] = probabilities
        proposed = replace(initial, **updates)
    return VerificationResult(
        scope=scope,
        prediction=proposed,
        provenance=ApiCallProvenance.from_response(
            response, source="targeted_verifier"
        ),
        requested_fields=requested,
        field_outcomes=tuple(outcomes),
        repaired_fields=repaired if proposed is not initial else (),
    )


class TargetedApiVerifier:
    """Invoke exactly one targeted verification call for the requested fields."""

    enabled_scopes = (
        "targeted",
        "joint",
        "spatial_track",
        "interaction",
        "workflow",
    )

    def __init__(
        self,
        *,
        client: _ApiClient,
        request_builder: TargetedVerificationRequestBuilder,
        data_upload_authorized: bool = False,
    ) -> None:
        if not hasattr(client, "call") or not callable(client.call):
            raise TypeError("client must provide call(request)")
        if not isinstance(request_builder, TargetedVerificationRequestBuilder):
            raise TypeError(
                "request_builder must be TargetedVerificationRequestBuilder"
            )
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
        *,
        requested_fields: tuple[str, ...],
    ) -> VerificationResult:
        if scope not in self.enabled_scopes:
            raise ApiContractError("targeted verifier received an unsupported scope")
        if not isinstance(candidates, CandidateSet):
            raise TypeError("targeted verifier requires a CandidateSet")
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
            requested_fields=requested_fields,
        )
        response = self.client.call(request)
        return parse_targeted_verification_response(
            response,
            initial=prediction,
            requested_fields=requested_fields,
            scope=scope,
        )
