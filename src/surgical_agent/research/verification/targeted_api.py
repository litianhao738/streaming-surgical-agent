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
    load_scoped_prompt_ontology_text,
)
from surgical_agent.research.reliability.state import TASK_PATHS, canonical_tasks
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    FieldVerificationOutcome,
    VerificationResult,
    VerificationUncertainty,
)


class _ApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponseRecord: ...


TARGETED_VERIFICATION_PROMPT_V2 = "targeted_verification_prompt_v2"
TARGETED_VERIFICATION_PROMPT_V5 = "targeted_verification_prompt_v5"
TARGETED_VERIFICATION_PROMPT_V6 = "targeted_verification_prompt_v6"
TARGETED_VERIFICATION_PROMPT_V7 = "targeted_verification_prompt_v7"
TARGETED_VERIFICATION_PROMPT_V8 = "targeted_verification_prompt_v8"
TARGETED_VERIFICATION_PROMPT_V9 = "targeted_verification_prompt_v9"


def load_targeted_verification_prompt_text(
    prompt_version: str = TARGETED_VERIFICATION_PROMPT_V6,
    *,
    academic_context: bool = False,
    ontology_text: str | None = None,
) -> str:
    resources = {
        TARGETED_VERIFICATION_PROMPT_V2: "targeted_verification_prompt_v2.txt",
        TARGETED_VERIFICATION_PROMPT_V5: "targeted_verification_prompt_v5.txt",
        TARGETED_VERIFICATION_PROMPT_V6: "targeted_verification_prompt_v6.txt",
        TARGETED_VERIFICATION_PROMPT_V7: "targeted_verification_prompt_v7.txt",
        TARGETED_VERIFICATION_PROMPT_V8: "targeted_verification_prompt_v8.txt",
        TARGETED_VERIFICATION_PROMPT_V9: "targeted_verification_prompt_v9.txt",
    }
    try:
        resource_name = resources[prompt_version]
    except KeyError as exc:
        raise ApiContractError(
            "targeted verification prompt version is unsupported"
        ) from exc
    prompt = (
        files("surgical_agent.research.verification.prompts")
        .joinpath(resource_name)
        .read_text(encoding="utf-8")
    )
    if academic_context:
        prompt = add_academic_medical_context(prompt)
    ontology = load_prompt_ontology_text() if ontology_text is None else ontology_text
    if not isinstance(ontology, str) or not ontology.strip():
        raise ApiContractError("targeted verification ontology must be non-empty")
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


def _fixed_dependency_constraints(
    prediction: InitialPrediction,
    requested_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """Expose only IDs required by frozen, unrequested IVT components."""

    if "ivt" in requested_fields or not prediction.triplet_ids:
        return []
    components = load_ivt_components()
    required: dict[str, set[int]] = {
        "instrument": set(),
        "verb": set(),
        "target": set(),
    }
    for triplet_id in prediction.triplet_ids:
        instrument_id, verb_id, target_id = components[triplet_id]
        required["instrument"].add(instrument_id)
        required["verb"].add(verb_id)
        required["target"].add(target_id)
    return [
        {
            "path": TASK_PATHS[task],
            "required_ids": sorted(required[task]),
            "reason": "FROZEN_UNREQUESTED_IVT_CLOSURE",
        }
        for task in ("instrument", "verb", "target")
        if task in requested_fields and required[task]
    ]


class TargetedVerificationRequestBuilder:
    """Build one strict request containing only the flagged field hypotheses."""

    def __init__(self, *, config: ApiConfig) -> None:
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be ApiConfig")
        self.config = config
        self.backend_name = "joint_api_vlm_targeted_verifier"
        self.prompt_version = (
            config.prompt_version
            if config.prompt_version
            in {
                TARGETED_VERIFICATION_PROMPT_V6,
                TARGETED_VERIFICATION_PROMPT_V7,
                TARGETED_VERIFICATION_PROMPT_V8,
                TARGETED_VERIFICATION_PROMPT_V9,
            }
            else TARGETED_VERIFICATION_PROMPT_V6
        )

    def build(
        self,
        context: PerceptionContext,
        prediction: InitialPrediction,
        candidates: CandidateSet,
        evidence: EvidenceProfile,
        *,
        requested_fields: tuple[str, ...],
        scope: str | None = None,
        review_focus: str | None = None,
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
        if not candidates.admits(prediction):
            raise ApiContractError(
                "current prediction must remain inside the frozen candidate pool"
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
        request_candidate_ids: dict[str, tuple[int, ...]] = {}
        candidate_fields: dict[str, list[int]] = {}
        for task in requested:
            records = candidates.candidate_records[task]
            limit = (
                TASK_CLASS_COUNTS[task]
                if self.prompt_version == TARGETED_VERIFICATION_PROMPT_V9
                else 20
            )
            if not records or len(records) > limit:
                raise ApiContractError(
                    "targeted candidate records must contain one to twenty items"
                )
            allowed_ids = candidates.allowed_ids[task]
            if scope == "interaction" and task == "ivt":
                components = load_ivt_components()
                allowed_ids = tuple(
                    ivt_id
                    for ivt_id in allowed_ids
                    if all(
                        component_id in candidates.allowed_ids[component_task]
                        for component_task, component_id in zip(
                            ("instrument", "verb", "target"),
                            components[ivt_id],
                        )
                    )
                )
                if not allowed_ids:
                    raise ApiContractError(
                        "interaction has no closure-safe IVT candidates"
                    )
            request_candidate_ids[task] = allowed_ids
            candidate_fields[TASK_PATHS[task]] = list(allowed_ids)
        current_fields = {
            TASK_PATHS[task]: list(_selected_ids(prediction, task))
            for task in requested
        }
        conservative = self.prompt_version == TARGETED_VERIFICATION_PROMPT_V7
        evidence_first = self.prompt_version in {
            TARGETED_VERIFICATION_PROMPT_V8,
            TARGETED_VERIFICATION_PROMPT_V9,
        }
        verification_protocol = (
            "conservative_h0_comparison_v1"
            if conservative
            else (
                "blind_positive_evidence_v1"
                if evidence_first
                else "blind_candidate_selection_v1"
            )
        )
        required_fields = _fixed_dependency_constraints(prediction, requested)
        if evidence_first:
            # V8 must not reveal H0 indirectly through dependency constraints.
            # The bound specialist performs deterministic IVT closure after the
            # visual decision instead of forcing a possibly wrong H0 component.
            required_fields = []
        input_payload = {
            "verification_protocol": verification_protocol,
            "video_id": context.sample.video_id,
            "target_frame_id": context.sample.target_frame_id,
            "causal_frame_ids": list(context.sample.causal_frame_ids),
            "selected_image_frame_ids": list(context.selected_image_frame_ids),
            "temporal_evidence": thaw_json(context.temporal_evidence),
            "flagged_fields": [TASK_PATHS[task] for task in requested],
            "candidate_fields": candidate_fields,
            "required_fields": required_fields,
            "workflow_summary": workflow_summary,
            "track_summary": track_summary,
            "memory_snapshot": memory_snapshot,
            "ontology_version": "cholectrack20_v1",
        }
        if not evidence_first:
            input_payload["evidence_profile"] = _evidence_mapping(evidence, requested)
        if conservative:
            input_payload["current_fields"] = current_fields
            input_payload["admission_policy"] = {
                "default": "KEEP_H0",
                "repair_requires": "CLEAR_VISUAL_CONTRADICTION",
                "uncertainty_action": "Pending",
            }
        prompt_profile = self.config.provider_options.get(
            "verification_prompt_profile", "full_context"
        )
        if prompt_profile not in {"full_context", "fixed_visual_only"}:
            raise ApiContractError("verification_prompt_profile is unsupported")
        request_images = context.images
        image_details = context.image_details or tuple("auto" for _ in context.images)
        if prompt_profile == "fixed_visual_only":
            input_payload = {
                "verification_protocol": verification_protocol,
                "video_id": context.sample.video_id,
                "target_frame_id": context.sample.target_frame_id,
                "causal_frame_ids": list(context.sample.causal_frame_ids),
                "selected_image_frame_ids": list(context.selected_image_frame_ids),
                "temporal_evidence": thaw_json(context.temporal_evidence),
                "flagged_fields": [TASK_PATHS[task] for task in requested],
                "candidate_fields": candidate_fields,
                "required_fields": input_payload["required_fields"],
                "ontology_version": "cholectrack20_v1",
            }
            if not evidence_first:
                input_payload["evidence_profile"] = _evidence_mapping(
                    evidence, requested
                )
            if conservative:
                input_payload["current_fields"] = current_fields
                input_payload["admission_policy"] = {
                    "default": "KEEP_H0",
                    "repair_requires": "CLEAR_VISUAL_CONTRADICTION",
                    "uncertainty_action": "Pending",
                }
        if self.prompt_version == TARGETED_VERIFICATION_PROMPT_V9:
            if review_focus not in {"scene_association", "motion_and_counterevidence"}:
                raise ApiContractError("V9 requires a declared visual review focus")
            input_payload["review_focus"] = review_focus
            input_payload["verification_protocol"] = "blind_association_review_v2"
        return ApiRequest(
            provider=self.config.provider,
            model_identifier=self.config.requested_model_identifier,
            endpoint_identifier=self.config.endpoint_identifier,
            prompt_version=self.prompt_version,
            response_schema_version=TARGETED_VERIFICATION_SCHEMA_VERSION,
            payload={
                "system_text": load_targeted_verification_prompt_text(
                    self.prompt_version,
                    academic_context=self.config.provider
                    in {"openai", "openrouter", "openai_compatible"},
                    ontology_text=load_scoped_prompt_ontology_text(
                        requested,
                        request_candidate_ids,
                        max_ivt_candidates=100
                        if self.prompt_version == TARGETED_VERIFICATION_PROMPT_V9
                        else 20,
                    ),
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
        "instrument_presence",
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
        review_focus: str | None = None,
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
            scope=scope,
            review_focus=review_focus,
        )
        response = self.client.call(request)
        return parse_targeted_verification_response(
            response,
            initial=prediction,
            requested_fields=requested_fields,
            scope=scope,
        )
