"""Cached API implementation of the causal joint-perception backend."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, Protocol

from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiRequest, ApiResponseRecord, thaw_json
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.openrouter_routing import request_routing_payload
from surgical_agent.config.schema import ApiConfig
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.context_builder import (
    PerceptionContext,
    require_gold_free,
)
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.perception.main_h0 import (
    is_main_h0_config,
    load_main_h0_prompt,
    main_h0_input,
    validate_main_h0_config,
)
from surgical_agent.perception.ontology_prompt import (
    add_academic_medical_context,
    load_prompt_ontology_text,
)
from surgical_agent.perception.parser import parse_joint_perception_response
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSIONS,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
)
from surgical_agent.research.reliability.taskwise import TASK_RELIABILITY_STATES
from surgical_agent.tracking.contracts import PredictedTrack

_BACKEND_NAMES = {
    "openai": "joint_openai_gpt56sol",
    "openai_compatible": "joint_openai_compatible",
    "openrouter": "joint_openrouter_gpt56sol",
    "mock": "joint_mock",
}
_OPENROUTER_JOINT_MODEL_IDENTIFIERS = frozenset(
    {
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-luna",
        "openai/gpt-6-astra",
        "x-ai/grok-4.6",
        "google/gemini-3.8-flash",
        "qwen/qwen3.8-max-0902",
    }
)
_OPENAI_JOINT_MODEL_IDENTIFIERS = frozenset({"gpt-5.6-sol"})


def load_prompt_text(
    schema_version: str = JOINT_PERCEPTION_SCHEMA_VERSION,
    *,
    academic_context: bool = False,
) -> str:
    """Load the versioned strict-output instruction without embedding it in code."""

    resources = {
        JOINT_PERCEPTION_SCHEMA_VERSION: "perception_prompt.txt",
        COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: "perception_prompt_compact.txt",
        RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            "perception_prompt_reliability_compact.txt"
        ),
        GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            "perception_prompt_gate_owned_compact.txt"
        ),
    }
    try:
        resource_name = resources[schema_version]
    except KeyError as exc:
        raise ApiContractError(
            "joint perception prompt version is unsupported"
        ) from exc
    prompt = (
        files("surgical_agent.perception.prompts")
        .joinpath(resource_name)
        .read_text(encoding="utf-8")
    )
    if academic_context:
        prompt = add_academic_medical_context(prompt)
    ontology = load_prompt_ontology_text()
    return f"{prompt.rstrip()}\n\n{ontology}\n"


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
    if (
        not isinstance(phases, Sequence)
        or isinstance(phases, (str, bytes))
        or any(not isinstance(phase, str) for phase in phases)
    ):
        raise ApiContractError("workflow recent_finalized_phases must be text")
    phase_states = snapshot.get("recent_phase_states", ())
    if (
        not isinstance(phase_states, Sequence)
        or isinstance(phase_states, (str, bytes))
        or (phase_states and len(phase_states) != len(phases))
        or any(state not in TASK_RELIABILITY_STATES for state in phase_states)
    ):
        raise ApiContractError("workflow recent_phase_states are invalid")
    phase_stability = snapshot.get("phase_stability")
    if phase_stability is not None and (
        not isinstance(phase_stability, (int, float))
        or isinstance(phase_stability, bool)
        or not math.isfinite(float(phase_stability))
        or not 0.0 <= float(phase_stability) <= 1.0
    ):
        raise ApiContractError("workflow phase_stability must be finite in [0, 1]")
    transitions = snapshot.get("observed_transitions", ())
    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
        raise ApiContractError("workflow observed_transitions must be a sequence")
    normalized_transitions: list[list[str]] = []
    for transition in transitions:
        if (
            not isinstance(transition, Sequence)
            or isinstance(transition, (str, bytes))
            or len(transition) != 2
            or any(not isinstance(phase, str) for phase in transition)
        ):
            raise ApiContractError("workflow transitions must contain phase pairs")
        normalized_transitions.append([transition[0], transition[1]])
    normalized = {
        "source_max_frame_id": source_max_frame_id,
        "recent_finalized_phases": list(phases),
        "phase_stability": (
            None if phase_stability is None else float(phase_stability)
        ),
        "observed_transitions": normalized_transitions,
    }
    if phase_states:
        normalized["recent_phase_states"] = list(phase_states)
    return normalized


def safe_track_summary(snapshot: Mapping[str, Any]) -> dict[str, object]:
    """Select bounded predicted-track fields and reject hidden annotation payloads."""

    require_gold_free(snapshot)
    status = snapshot.get("status", "UNAVAILABLE")
    if status not in {"UNAVAILABLE", "AVAILABLE"}:
        raise ApiContractError("track status is unsupported")
    source_max_frame_id = snapshot.get("source_max_frame_id")
    if source_max_frame_id is not None and (
        not isinstance(source_max_frame_id, int)
        or isinstance(source_max_frame_id, bool)
        or source_max_frame_id < 0
    ):
        raise ApiContractError("track source_max_frame_id must be non-negative")
    frames = snapshot.get("frames", ())
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ApiContractError("track frames must be a sequence")
    normalized_frames: list[dict[str, object]] = []
    prior_frame_id: int | None = None
    for frame in frames:
        if not isinstance(frame, Mapping) or set(frame) != {"frame_id", "tracks"}:
            raise ApiContractError("track frame has unexpected fields")
        frame_id = frame["frame_id"]
        if (
            not isinstance(frame_id, int)
            or isinstance(frame_id, bool)
            or frame_id < 0
            or (prior_frame_id is not None and frame_id <= prior_frame_id)
        ):
            raise ApiContractError("track frame IDs must be increasing")
        tracks = frame["tracks"]
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            raise ApiContractError("track values must be a sequence")
        normalized_tracks: list[dict[str, object]] = []
        seen_track_ids: set[str] = set()
        for track in tracks:
            fields = {"track_id", "instrument_id", "bbox_tlwh", "score", "age"}
            if not isinstance(track, Mapping) or set(track) != fields:
                raise ApiContractError("track value has unexpected fields")
            track_id = track["track_id"]
            if (
                not isinstance(track_id, str)
                or not track_id
                or track_id in seen_track_ids
            ):
                raise ApiContractError("track IDs must be unique non-empty text")
            seen_track_ids.add(track_id)
            bbox = track["bbox_tlwh"]
            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes))
                or len(bbox) != 4
            ):
                raise ApiContractError("track bbox_tlwh must contain four values")
            try:
                validated_track = PredictedTrack(
                    track_id=track_id,
                    instrument_id=track["instrument_id"],
                    bbox_tlwh=tuple(bbox),
                    score=track["score"],
                    age=track["age"],
                )
            except (TypeError, ValueError) as exc:
                raise ApiContractError("track value is invalid") from exc
            normalized_tracks.append(
                {
                    "track_id": validated_track.track_id,
                    "instrument_id": validated_track.instrument_id,
                    "bbox_tlwh": list(validated_track.bbox_tlwh),
                    "score": validated_track.score,
                    "age": validated_track.age,
                }
            )
        normalized_frames.append({"frame_id": frame_id, "tracks": normalized_tracks})
        prior_frame_id = frame_id
    if normalized_frames:
        if source_max_frame_id != normalized_frames[-1]["frame_id"]:
            raise ApiContractError(
                "track source_max_frame_id must match its last frame"
            )
    elif source_max_frame_id is not None:
        raise ApiContractError("empty track context cannot declare a source frame")
    if status == "AVAILABLE" and not normalized_frames:
        raise ApiContractError("available track context requires frame records")
    if status == "UNAVAILABLE" and normalized_frames:
        raise ApiContractError("unavailable track context forbids frame records")
    return {
        "status": status,
        "source_max_frame_id": source_max_frame_id,
        "frames": normalized_frames,
    }


class _ApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponseRecord: ...


class JointPerceptionRequestBuilder:
    """Turn frozen causal context into one canonical multimodal API request."""

    def __init__(self, *, config: ApiConfig) -> None:
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be ApiConfig")
        config.validate()
        if is_main_h0_config(config):
            validate_main_h0_config(config)
        try:
            self.backend_name = _BACKEND_NAMES[config.provider]
        except KeyError as exc:
            raise ApiContractError("joint API backend provider is unsupported") from exc
        self.config = config

        if (
            config.provider == "openrouter"
            and config.requested_model_identifier == "openai/gpt-6-astra"
        ):
            self.backend_name = "joint_openrouter_gpt6astra"
        if config.provider == "openrouter" and config.requested_model_identifier in {
            "x-ai/grok-4.6",
            "google/gemini-3.8-flash",
            "qwen/qwen3.8-max-0902",
        }:
            self.backend_name = (
                "joint_openrouter_"
                + config.requested_model_identifier.replace("/", "_")
            )

    def build(self, context: PerceptionContext) -> ApiRequest:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be PerceptionContext")
        if self.config.provider in {"openai", "openrouter", "openai_compatible"}:
            if self.config.provider == "openai_compatible":
                approved_models = None
            else:
                approved_models = (
                    _OPENAI_JOINT_MODEL_IDENTIFIERS
                    if self.config.provider == "openai"
                    else _OPENROUTER_JOINT_MODEL_IDENTIFIERS
                )
            if (
                approved_models is not None
                and self.config.requested_model_identifier not in approved_models
            ):
                raise ApiContractError(
                    "joint OpenRouter requests require an approved model"
                    if self.config.provider == "openrouter"
                    else "joint OpenAI requests require gpt-5.6-sol"
                )
            if not is_main_h0_config(self.config) and (
                self.config.response_schema_version
                not in JOINT_PERCEPTION_SCHEMA_VERSIONS
            ):
                raise ApiContractError(
                    "joint OpenRouter requests require the joint perception schema"
                )
            if (
                not is_main_h0_config(self.config)
                and self.config.prompt_version != self.config.response_schema_version
            ):
                raise ApiContractError(
                    "joint OpenRouter requests require a matching prompt version"
                )
        causal_frame_ids = context.sample.causal_frame_ids
        if not 1 <= len(causal_frame_ids) <= self.config.max_causal_frames:
            raise ApiContractError("joint requests exceed the configured causal window")
        if not 1 <= len(context.images) <= self.config.max_api_images:
            raise ApiContractError("joint requests exceed the configured image budget")
        selected_image_frame_ids = context.selected_image_frame_ids
        if not selected_image_frame_ids:
            selected_image_frame_ids = causal_frame_ids[-len(context.images) :]
        if (
            len(selected_image_frame_ids) != len(context.images)
            or tuple(sorted(set(selected_image_frame_ids))) != selected_image_frame_ids
            or set(selected_image_frame_ids) - set(causal_frame_ids)
            or selected_image_frame_ids[-1] != context.sample.target_frame_id
        ):
            raise ApiContractError(
                "selected API images must be causal and include target"
            )
        require_gold_free(context.temporal_evidence)
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
        track_summary = safe_track_summary(context.track_snapshot)
        track_source_max = track_summary["source_max_frame_id"]
        if (
            track_source_max is not None
            and track_source_max > context.sample.target_frame_id
        ):
            raise ApiContractError("track source_max_frame_id must not exceed target")
        image_details = context.image_details or tuple("auto" for _ in context.images)
        if len(image_details) != len(context.images):
            raise ApiContractError("image detail count must match selected images")
        if is_main_h0_config(self.config):
            if selected_image_frame_ids != causal_frame_ids or tuple(
                image.identifier for image in context.images
            ) != context.sample.media_refs:
                raise ApiContractError("main H0 requires every causal image in source order")
            expected_details = ("low",) * (len(context.images) - 1) + ("high",)
            if image_details != expected_details:
                raise ApiContractError("main H0 requires low history and high target detail")
            return ApiRequest(
                provider=self.config.provider,
                model_identifier=self.config.requested_model_identifier,
                endpoint_identifier=self.config.endpoint_identifier,
                prompt_version=self.config.prompt_version,
                response_schema_version=self.config.response_schema_version,
                payload={
                    "system_text": load_main_h0_prompt(),
                    "image_details": list(image_details),
                    "openrouter_image_detail_mode": "explicit_v1",
                    "input_text": json.dumps(
                        main_h0_input(
                            video_id=context.sample.video_id,
                            target_frame_id=context.sample.target_frame_id,
                            frame_ids=causal_frame_ids,
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    **request_routing_payload(
                        provider=self.config.provider,
                        provider_options=self.config.provider_options,
                    ),
                },
                images=context.images,
                generation_parameters=self.config.generation_parameters,
            )
        prompt_profile = self.config.provider_options.get(
            "initial_prompt_profile", "full_context"
        )
        if prompt_profile not in {"full_context", "fixed_visual_only"}:
            raise ApiContractError("initial_prompt_profile is unsupported")
        if prompt_profile == "fixed_visual_only":
            prior_mapping = None
            workflow_mapping = {
                "source_max_frame_id": None,
                "recent_finalized_phases": [],
                "phase_stability": None,
                "observed_transitions": [],
            }
            track_mapping = {
                "status": "UNAVAILABLE",
                "source_max_frame_id": None,
                "frames": [],
            }
            temporal_mapping = {
                "schema_version": "fixed_causal_window_v1",
                "selection_strategy": "fixed_all",
            }
        else:
            prior_mapping = safe_prior_mapping(context.prior_finalized_prediction)
            workflow_mapping = workflow_summary
            track_mapping = track_summary
            temporal_mapping = thaw_json(context.temporal_evidence)
        payload = {
            "system_text": load_prompt_text(
                self.config.response_schema_version,
                academic_context=self.config.provider
                in {"openai", "openrouter", "openai_compatible"},
            ),
            "image_details": list(image_details),
            "input_text": json.dumps(
                {
                    "video_id": context.sample.video_id,
                    "target_frame_id": context.sample.target_frame_id,
                    "causal_frame_ids": list(causal_frame_ids),
                    "selected_image_frame_ids": list(selected_image_frame_ids),
                    "temporal_evidence": temporal_mapping,
                    "prior_finalized_prediction": prior_mapping,
                    "workflow_summary": workflow_mapping,
                    "track_summary": track_mapping,
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
            raise ApiContractError(
                "prior finalized prediction must be a PredictionRecord"
            )
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
