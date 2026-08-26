"""Cached API implementation of the causal joint-perception backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, Protocol

from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiRequest, ApiResponseRecord
from surgical_agent.api.errors import ApiContractError
from surgical_agent.config.schema import ApiConfig
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.context_builder import (
    PerceptionContext,
    require_gold_free,
)
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.perception.parser import parse_joint_perception_response
from surgical_agent.perception.schema import JOINT_PERCEPTION_SCHEMA_VERSION

_BACKEND_NAMES = {
    "openrouter": "joint_openrouter_gpt56sol",
    "mock": "joint_mock",
}
_OPENROUTER_JOINT_MODEL_IDENTIFIER = "openai/gpt-5.6-sol"


def load_prompt_text() -> str:
    """Load the versioned strict-output instruction without embedding it in code."""

    return (
        files("surgical_agent.perception.prompts")
        .joinpath("perception_prompt.txt")
        .read_text(encoding="utf-8")
    )


def safe_prior_mapping(prior: PredictionRecord | None) -> dict[str, object] | None:
    """Serialize only compact finalized prediction fields needed by the prompt."""

    if prior is None:
        return None
    if not isinstance(prior, PredictionRecord):
        raise ApiContractError("prior finalized prediction must be a PredictionRecord")
    return {
        "source_frame_id": prior.frame_id,
        "instrument_ids": list(prior.instrument_ids),
        "verb_ids": list(prior.verb_ids),
        "target_ids": list(prior.target_ids),
        "ivt_ids": list(prior.triplet_ids),
        "phase_id": prior.phase_id,
    }


def safe_workflow_summary(snapshot: Mapping[str, Any]) -> dict[str, object]:
    """Select the compact workflow fields allowed in a joint API payload."""

    require_gold_free(snapshot)
    source_max_frame_id = snapshot.get("source_max_frame_id")
    if source_max_frame_id is not None and (
        not isinstance(source_max_frame_id, int)
        or isinstance(source_max_frame_id, bool)
        or source_max_frame_id < 0
    ):
        raise ApiContractError("workflow source_max_frame_id must be non-negative")
    phases = snapshot.get("recent_finalized_phases", ())
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or any(
        not isinstance(phase, str) for phase in phases
    ):
        raise ApiContractError("workflow recent_finalized_phases must be text")
    return {
        "source_max_frame_id": source_max_frame_id,
        "recent_finalized_phases": list(phases),
    }


class _ApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponseRecord: ...


class JointPerceptionRequestBuilder:
    """Turn frozen causal context into one canonical multimodal API request."""

    def __init__(self, *, config: ApiConfig) -> None:
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be ApiConfig")
        config.validate()
        try:
            self.backend_name = _BACKEND_NAMES[config.provider]
        except KeyError as exc:
            raise ApiContractError("joint API backend provider is unsupported") from exc
        self.config = config

    def build(self, context: PerceptionContext) -> ApiRequest:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be PerceptionContext")
        if self.config.provider == "openrouter":
            if self.config.requested_model_identifier != (
                _OPENROUTER_JOINT_MODEL_IDENTIFIER
            ):
                raise ApiContractError(
                    "joint OpenRouter requests require the exact GPT-5.6-Sol model"
                )
            if self.config.response_schema_version != JOINT_PERCEPTION_SCHEMA_VERSION:
                raise ApiContractError(
                    "joint OpenRouter requests require the joint perception schema"
                )
        if not 1 <= len(context.images) <= self.config.max_causal_frames:
            raise ApiContractError("joint requests require one to three causal images")
        self._validate_prior(context)
        workflow_summary = safe_workflow_summary(context.workflow_snapshot)
        source_max_frame_id = workflow_summary["source_max_frame_id"]
        if (
            source_max_frame_id is not None
            and source_max_frame_id >= context.sample.target_frame_id
        ):
            raise ApiContractError(
                "workflow source_max_frame_id must be strictly earlier than target"
            )
        payload = {
            "system_text": load_prompt_text(),
            "input_text": json.dumps(
                {
                    "video_id": context.sample.video_id,
                    "target_frame_id": context.sample.target_frame_id,
                    "causal_frame_ids": list(context.sample.causal_frame_ids),
                    "prior_finalized_prediction": safe_prior_mapping(
                        context.prior_finalized_prediction
                    ),
                    "workflow_summary": workflow_summary,
                    "ontology_version": "cholectrack20_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        return ApiRequest(
            provider=self.config.provider,
            model_identifier=self.config.requested_model_identifier,
            endpoint_identifier=self.config.endpoint_identifier,
            prompt_version=self.config.prompt_version,
            response_schema_version=self.config.response_schema_version,
            payload=payload,
            images=context.images,
            generation_parameters=self.config.generation_parameters,
        )

    @staticmethod
    def _validate_prior(context: PerceptionContext) -> None:
        """Recheck manually assembled context before it reaches an API boundary."""

        prior = context.prior_finalized_prediction
        if prior is None:
            return
        if not isinstance(prior, PredictionRecord):
            raise ApiContractError("prior finalized prediction must be a PredictionRecord")
        if prior.video_id != context.sample.video_id:
            raise ApiContractError(
                "prior finalized prediction must be from the same video"
            )
        if prior.frame_id >= context.sample.target_frame_id:
            raise ApiContractError(
                "prior finalized prediction must be strictly earlier than target"
            )
        prior_causal_frame_ids = prior.causal_frame_ids
        if any(
            frame_id >= context.sample.target_frame_id
            for frame_id in prior_causal_frame_ids
        ):
            raise ApiContractError(
                "prior causal frame IDs must be strictly earlier than target"
            )
        if tuple(sorted(set(prior_causal_frame_ids))) != prior_causal_frame_ids:
            raise ApiContractError(
                "prior causal frame IDs must be unique and increasing"
            )


class JointApiVlm:
    """Call the cache-aware client once and parse its strict joint response."""

    def __init__(
        self,
        *,
        client: CachedMultimodalApiClient | _ApiClient,
        request_builder: JointPerceptionRequestBuilder,
        data_upload_authorized: bool = False,
    ) -> None:
        if not hasattr(client, "call") or not callable(client.call):
            raise TypeError("client must provide call(request)")
        if not isinstance(request_builder, JointPerceptionRequestBuilder):
            raise TypeError("request_builder must be JointPerceptionRequestBuilder")
        if type(data_upload_authorized) is not bool:
            raise TypeError("data_upload_authorized must be boolean")
        self.client = client
        self.request_builder = request_builder
        self.data_upload_authorized = data_upload_authorized

    def predict(self, context: PerceptionContext) -> JointPerceptionResult:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be PerceptionContext")
        if not self.data_upload_authorized and any(
            not image.identifier.startswith("synthetic:") for image in context.images
        ):
            raise ApiContractError(
                "only synthetic: image identifiers are permitted without upload authorization"
            )
        request = self.request_builder.build(context)
        response = self.client.call(request)
        return parse_joint_perception_response(
            response,
            video_id=context.sample.video_id,
            frame_id=context.sample.target_frame_id,
            backend=self.request_builder.backend_name,
        )
