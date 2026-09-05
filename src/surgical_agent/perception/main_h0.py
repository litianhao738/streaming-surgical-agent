"""Production contract for the evaluated causal, final-label joint H0 baseline.

This module owns the production request, without importing research runners or
reading experiment artifacts. The full response schema remains in both the
system message and the provider's strict response_format, as evaluated.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING

from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.errors import ApiContractError
from surgical_agent.perception.final_only import (
    FINAL_ONLY_SCHEMA_VERSION,
    final_only_schema,
)
from surgical_agent.perception.ontology_prompt import load_prompt_ontology_text

if TYPE_CHECKING:
    from surgical_agent.config.schema import ApiConfig

MAIN_H0_PROMPT_VERSION = "joint_perception_main_h0_v1"
MAIN_H0_SCHEMA_VERSION = FINAL_ONLY_SCHEMA_VERSION
MAIN_H0_MODEL = "qwen/qwen3.8-max-0902"
MAIN_H0_SOURCE_FPS = 25

_LABEL_BOUNDARY = (
    "\nDataset label boundary: A visible instrument does not by itself establish a named "
    "verb or anatomical target. Classify the actual tool-contact relationship, not nearby "
    "anatomy alone. The 100 IVT classes are a closed vocabulary, not a list covering every "
    "possible surgical interaction. For a valid instrument whose actual verb, target, or "
    "joint combination is outside these classes, retain its instrument identity and use "
    "the corresponding null-verb/null-target IVT. Null does not necessarily mean no "
    "physical motion or contact. Do not replace an out-of-vocabulary interaction with "
    "the nearest familiar valid tuple. Conversely, do not use null merely because "
    "prediction is difficult; use a non-null tuple when that defined interaction is "
    "visually supported. These definitions apply per instrument; a frame may contain "
    "both null and non-null interactions.\n"
)


def is_main_h0_config(config: ApiConfig) -> bool:
    """Identify the named main protocol; validation reports any configuration drift."""
    return config.prompt_version == MAIN_H0_PROMPT_VERSION


def validate_main_h0_config(config: ApiConfig) -> None:
    """Reject settings that silently change the evaluated main H0 contract."""
    config.validate()
    if (
        not is_main_h0_config(config)
        or config.response_schema_version != MAIN_H0_SCHEMA_VERSION
    ):
        raise ApiContractError("main H0 requires its versioned prompt and final-only schema")
    if config.provider not in {"openrouter", "mock"}:
        raise ApiContractError("main H0 supports OpenRouter and offline mock only")
    expected_mode, expected_endpoint = (
        ("real", "https://openrouter.ai/api/v1/chat/completions")
        if config.provider == "openrouter"
        else ("mock", "mock://local/p3")
    )
    if config.mode != expected_mode or config.endpoint_identifier != expected_endpoint:
        raise ApiContractError("main H0 provider, mode and endpoint must match")
    if (
        not config.enabled
        or not config.cache_required
        or config.synthetic_input_required
        or not config.data_upload_authorized
    ):
        raise ApiContractError("main H0 requires enabled, cached and authorized dataset input")
    if config.provider == "openrouter" and (
        config.requested_model_identifier != MAIN_H0_MODEL
        or config.provider_options.get("routing_profile") != "strict_alibaba"
    ):
        raise ApiContractError("main H0 requires evaluated Qwen and strict Alibaba routing")
    if config.max_causal_frames != 3 or config.max_api_images != 3:
        raise ApiContractError("main H0 requires a maximum of three causal images")
    expected_parameters = {
        "max_output_tokens": 4096,
        "temperature": 0,
        "reasoning": {"effort": "low"},
    }
    if thaw_json(config.generation_parameters) != expected_parameters:
        raise ApiContractError("main H0 requires temperature 0, low reasoning and 4096 tokens")
    expected_options = {
        "frame_selection_strategy": "fixed_all",
        "history_image_detail": "low",
        "target_image_detail": "high",
        "initial_prompt_profile": "fixed_visual_only",
    }
    if config.provider == "openrouter":
        expected_options.update(timeout_seconds=180.0, routing_profile="strict_alibaba")
    if set(config.provider_options) != set(expected_options):
        raise ApiContractError("main H0 provider_options must contain only its fixed settings")
    for key, expected in expected_options.items():
        if config.provider_options.get(key) != expected:
            raise ApiContractError(f"main H0 requires {key}={expected}")


def load_main_h0_prompt() -> str:
    """Render the exact evaluated baseline instructions, ontology and JSON schema."""
    prompt = (
        files("surgical_agent.perception.prompts")
        .joinpath("perception_prompt_main_h0.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )
    return (
        f"{prompt}\n\n{load_prompt_ontology_text()}\n"
        f'\nThe output JSON schema_version must be exactly "{MAIN_H0_SCHEMA_VERSION}". '
        "Do not use the input ontology_version as the output schema_version.\n"
        + _LABEL_BOUNDARY
        + "\nImages are ordered from past to present; their true relative seconds are supplied. "
        "Predict ONLY the final target image. Earlier images supply context, not additional output labels. "
        "Do not output the union of actions across the window. All six root keys are required. "
        "Return exactly this JSON schema:\n"
        + json.dumps(final_only_schema(), sort_keys=True)
    )


def main_h0_input(
    *, video_id: str, target_frame_id: int, frame_ids: tuple[int, ...]
) -> dict[str, object]:
    """Return timestamped pure-visual input, preserving genuine short histories."""
    expected = tuple(
        target_frame_id + MAIN_H0_SOURCE_FPS * offset
        for offset in range(1 - len(frame_ids), 1)
    )
    if not 1 <= len(frame_ids) <= 3 or frame_ids != expected:
        raise ApiContractError("main H0 images must be contiguous causal 25-frame samples")
    payload: dict[str, object] = {
        "video_id": video_id,
        "target_frame_id": target_frame_id,
        "causal_frame_ids": list(frame_ids),
        "selected_image_frame_ids": list(frame_ids),
        "relative_seconds": list(range(1 - len(frame_ids), 1)),
        "source_fps": MAIN_H0_SOURCE_FPS,
        "predict_target_only": True,
        "temporal_evidence": {
            "schema_version": "fixed_causal_window_v1",
            "selection_strategy": "fixed_all",
        },
        "prior_finalized_prediction": None,
        "workflow_summary": {
            "source_max_frame_id": None,
            "recent_finalized_phases": [],
            "phase_stability": None,
            "observed_transitions": [],
        },
        "track_summary": {
            "status": "UNAVAILABLE",
            "source_max_frame_id": None,
            "frames": [],
        },
        "ontology_version": "cholectrack20_v1",
    }
    if len(frame_ids) < 3:
        # The early-video and gap cases have not been accuracy-benchmarked with
        # the complete three-image baseline; preserve their actual history.
        payload["window_state"] = "partial_history"
        payload["actual_image_count"] = len(frame_ids)
    return payload
