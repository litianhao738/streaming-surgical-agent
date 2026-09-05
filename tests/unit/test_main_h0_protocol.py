"""Production H0 reproduces the evaluated request without importing experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiResponseRecord, thaw_json
from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.perception.main_h0 import (
    MAIN_H0_MODEL,
    MAIN_H0_PROMPT_VERSION,
    MAIN_H0_SCHEMA_VERSION,
    is_main_h0_config,
    load_main_h0_prompt,
    validate_main_h0_config,
)
from surgical_agent.perception.parser import parse_joint_perception_response


def _config(*, provider: str = "mock") -> ApiConfig:
    options = {
        "frame_selection_strategy": "fixed_all",
        "history_image_detail": "low",
        "target_image_detail": "high",
        "initial_prompt_profile": "fixed_visual_only",
    }
    if provider == "openrouter":
        options["routing_profile"] = "strict_alibaba"
        options["timeout_seconds"] = 180.0
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock" if provider == "mock" else "real",
            "provider": provider,
            "endpoint_identifier": (
                "mock://local/p3"
                if provider == "mock"
                else "https://openrouter.ai/api/v1/chat/completions"
            ),
            "requested_model_identifier": (
                "mock-joint-v1" if provider == "mock" else MAIN_H0_MODEL
            ),
            "prompt_version": MAIN_H0_PROMPT_VERSION,
            "response_schema_version": MAIN_H0_SCHEMA_VERSION,
            "generation_parameters": {
                "max_output_tokens": 4096,
                "temperature": 0,
                "reasoning": {"effort": "low"},
            },
            "provider_options": options,
            "max_causal_frames": 3,
            "max_api_images": 3,
            "data_upload_authorized": True,
            "synthetic_input_required": False,
            "cache_required": True,
        }
    )


def _context(ids: tuple[int, ...] = (25051, 25076, 25101)) -> PerceptionContext:
    identifiers = tuple(f"synthetic:{frame_id}" for frame_id in ids)
    return PerceptionContext(
        sample=InferenceSample(
            video_id="VID103",
            target_frame_id=ids[-1],
            causal_frame_ids=ids,
            media_refs=identifiers,
            source_split=DatasetSplit.TESTING,
            alignment_version="test",
        ),
        frames=None,
        images=tuple(ApiImageInput(name, "image/png", b"png") for name in identifiers),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
        selected_image_frame_ids=ids,
        image_details=("low",) * (len(ids) - 1) + ("high",),
    )


def test_main_request_matches_evaluated_prompt_and_input() -> None:
    request = JointPerceptionRequestBuilder(config=_config()).build(_context())
    # Frozen baseline_system.txt digest, independently checked when promoting.
    # The unit test remains runnable when local experiment artifacts are absent.
    assert hashlib.sha256(request.payload["system_text"].encode()).hexdigest() == (
        "c1009de8429158eab6b9c0f4ee39b12f51e5e970335f0c554866198191b6e3e5"
    )
    expected_input = {
        "causal_frame_ids": [25051, 25076, 25101],
        "ontology_version": "cholectrack20_v1",
        "predict_target_only": True,
        "prior_finalized_prediction": None,
        "relative_seconds": [-2, -1, 0],
        "selected_image_frame_ids": [25051, 25076, 25101],
        "source_fps": 25,
        "target_frame_id": 25101,
        "temporal_evidence": {
            "schema_version": "fixed_causal_window_v1",
            "selection_strategy": "fixed_all",
        },
        "track_summary": {"frames": [], "source_max_frame_id": None, "status": "UNAVAILABLE"},
        "video_id": "VID103",
        "workflow_summary": {
            "observed_transitions": [],
            "phase_stability": None,
            "recent_finalized_phases": [],
            "source_max_frame_id": None,
        },
    }
    assert request.payload["input_text"] == json.dumps(
        expected_input, sort_keys=True, separators=(",", ":")
    )
    assert request.payload["image_details"] == ("low", "low", "high")
    assert request.payload["openrouter_image_detail_mode"] == "explicit_v1"


def test_openrouter_wire_has_one_joint_call_with_explicit_image_details() -> None:
    request = JointPerceptionRequestBuilder(config=_config(provider="openrouter")).build(
        _context()
    )
    # Serialize offline; no credentials, transport.send or network is needed.
    transport = object.__new__(OpenRouterTransport)
    body = json.loads(transport._request_body(request))
    assert body["model"] == MAIN_H0_MODEL
    assert body["temperature"] == 0
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_tokens"] == 4096
    assert body["provider"] == {
        "only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True
    }
    assert body["messages"][0]["content"] == load_main_h0_prompt()
    images = body["messages"][1]["content"][1:]
    assert [item["image_url"]["detail"] for item in images] == ["low", "low", "high"]
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"] == schema_for(MAIN_H0_SCHEMA_VERSION)
    assert set(schema["schema"]["required"]) == {
        "schema_version", "instrument", "verb", "target", "ivt", "phase"
    }
    assert "topk" not in json.dumps(schema)


@pytest.mark.parametrize("ids", [(25101,), (25076, 25101)])
def test_partial_real_history_is_preserved_and_identified(ids: tuple[int, ...]) -> None:
    request = JointPerceptionRequestBuilder(config=_config()).build(_context(ids))
    data = json.loads(request.payload["input_text"])
    assert data["actual_image_count"] == len(ids)
    assert data["window_state"] == "partial_history"
    assert data["causal_frame_ids"] == list(ids)
    assert data["relative_seconds"] == list(range(1 - len(ids), 1))
    assert len(request.images) == len(ids)


def test_main_request_rejects_wrong_spacing_image_order_and_detail() -> None:
    builder = JointPerceptionRequestBuilder(config=_config())
    with pytest.raises(ApiContractError, match="25-frame"):
        builder.build(_context((25099, 25100, 25101)))
    context = _context()
    with pytest.raises(ApiContractError, match="source order"):
        builder.build(replace(context, images=tuple(reversed(context.images))))
    with pytest.raises(ApiContractError, match="high target"):
        builder.build(replace(context, image_details=("auto", "auto", "auto")))


@pytest.mark.parametrize(
    "changes",
    [
        {"response_schema_version": "joint_perception_gate_owned_compact_v1"},
        {"max_causal_frames": 6},
        {"generation_parameters": {"max_output_tokens": 4096, "reasoning": {"effort": "minimal"}}},
        {"provider_options": {"initial_prompt_profile": "full_context"}},
    ],
)
def test_main_config_rejects_unbenchmarked_drift(changes: dict[str, object]) -> None:
    config = _config()
    assert is_main_h0_config(config)
    validate_main_h0_config(config)
    with pytest.raises(ApiContractError):
        JointPerceptionRequestBuilder(config=replace(config, **changes))


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


def test_joint_backend_calls_once_and_does_not_invent_confidence() -> None:
    transport = MockProviderTransport()

    class Client:
        def call(self, request):
            return _response(thaw_json(transport.send(request).parsed_payload))

    backend = JointApiVlm(
        client=Client(), request_builder=JointPerceptionRequestBuilder(config=_config())
    )
    result = backend.predict(_context())
    assert transport.provider_call_count == 1
    assert result.prediction.score_semantics == "hard_label_v1"
    assert result.prediction.instrument_ids == (0,)
    assert result.prediction.verb_ids == (2,)
    assert result.prediction.target_ids == (1,)
    assert result.prediction.triplet_ids == (0,)
    for task in ("instrument", "verb", "target", "ivt", "phase"):
        assert result.raw_evidence.ranked_candidates[task] == ()
        assert result.raw_evidence.self_reported_confidence[task] is None
        assert set(result.prediction.probabilities[task]) <= {0.0, 1.0}


def test_parser_keeps_model_labels_without_projecting_or_repairing() -> None:
    payload = {
        "schema_version": MAIN_H0_SCHEMA_VERSION,
        "instrument": {"selected_ids": [0]},
        "verb": {"selected_ids": [2]},
        "target": {"selected_ids": []},
        "ivt": {"selected_ids": [94]},
        "phase": {"selected_id": 3},
    }
    validator_for(MAIN_H0_SCHEMA_VERSION)(payload)
    result = parse_joint_perception_response(
        _response(payload), video_id="VID103", frame_id=25101, backend="mock"
    )
    assert result.prediction.verb_ids == (2,)
    assert result.prediction.target_ids == ()
    assert result.prediction.triplet_ids == (94,)
    payload["ivt"]["selected_ids"] = [94, 94]
    with pytest.raises(ApiSchemaError):
        parse_joint_perception_response(
            _response(payload), video_id="VID103", frame_id=25101, backend="mock"
        )
