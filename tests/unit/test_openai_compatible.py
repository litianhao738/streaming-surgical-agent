"""Contracts for generic OpenAI-compatible Qwen-style gateways."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts import run_dataset_api_pipeline as dataset_cli
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.providers.openai_compatible import OpenAICompatibleTransport
from surgical_agent.api.providers.openrouter import HttpResponse
from surgical_agent.api.registry import build_transport
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder

ENDPOINT = "https://gateway.example/v1/chat/completions"


def _config(**changes: object) -> ApiConfig:
    raw: dict[str, object] = {
        "enabled": True,
        "mode": "real",
        "provider": "openai_compatible",
        "endpoint_identifier": ENDPOINT,
        "requested_model_identifier": "qwen-test-model",
        "prompt_version": "p3_multimodal_smoke_v1",
        "response_schema_version": "p3_multimodal_smoke_v1",
        "generation_parameters": {"max_output_tokens": 128},
        "provider_options": {
            "timeout_seconds": 120.0,
            "response_format": "json_schema",
        },
    }
    raw.update(changes)
    return ApiConfig.from_mapping(raw)


def _request(**changes: Any) -> ApiRequest:
    values: dict[str, Any] = {
        "provider": "openai_compatible",
        "model_identifier": "qwen-test-model",
        "endpoint_identifier": ENDPOINT,
        "prompt_version": "p3_multimodal_smoke_v1",
        "response_schema_version": "p3_multimodal_smoke_v1",
        "payload": {"system_text": "Return JSON.", "input_text": "Inspect."},
        "images": (ApiImageInput("frame:1", "image/png", b"png"),),
        "generation_parameters": {
            "max_output_tokens": 128,
            "temperature": 0.0,
            "enable_thinking": False,
        },
    }
    values.update(changes)
    return ApiRequest(**values)


def test_compatible_transport_sends_multimodal_json_without_router_fields() -> None:
    captured: dict[str, object] = {}
    structured = {
        "schema_version": "p3_multimodal_smoke_v1",
        "message": "ok",
        "image_observed": True,
        "structured": True,
    }

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        response = {
            "id": "compatible-1",
            "object": "chat.completion",
            "model": "qwen-returned-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(structured)},
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        return HttpResponse(200, {}, json.dumps(response).encode(), total_latency_ms=9.0)

    response = OpenAICompatibleTransport(
        api_key=SecretValue("unit-test-secret"),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(_request())

    sent = json.loads(captured["body"])
    assert captured["url"] == ENDPOINT
    assert sent["model"] == "qwen-test-model"
    assert sent["stream"] is False
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert sent["enable_thinking"] is False
    assert "provider" not in sent
    assert len(sent["messages"][1]["content"]) == 2
    assert response.returned_model_identifier == "qwen-returned-model"
    assert response.total_latency_ms == 9.0
    assert response.prompt_tokens == 20
    assert response.visible_output_tokens == 10


def test_compatible_endpoint_and_registry_fail_closed() -> None:
    config = _config()
    transport = build_transport(config, api_key=SecretValue("unit-test-secret"))
    assert isinstance(transport, OpenAICompatibleTransport)

    with pytest.raises(ApiContractError, match="HTTPS chat/completions"):
        _config(endpoint_identifier="http://gateway.example/v1/chat/completions")
    with pytest.raises(ApiContractError, match="HTTPS chat/completions"):
        _config(endpoint_identifier="https://user:secret@gateway.example/v1/chat/completions")


def test_qwen_dataset_config_accepts_engineering_cli_overrides() -> None:
    config = load_api_config(
        "configs/perception/joint_qwen_compatible_dataset.yaml"
    )
    config = dataset_cli._apply_cli_model_override(
        config,
        mode="engineering",
        model="qwen-command-line-model",
    )
    config = dataset_cli._apply_cli_base_url_override(
        config,
        mode="engineering",
        base_url="https://gateway.example/compatible-mode/v1",
    )

    assert config.endpoint_identifier == (
        "https://gateway.example/compatible-mode/v1/chat/completions"
    )
    assert config.requested_model_identifier == "qwen-command-line-model"
    assert JointPerceptionRequestBuilder(config=config).backend_name == (
        "joint_openai_compatible"
    )
    dataset_cli._require_exact_dataset_config(
        config,
        has_credential=True,
        authorize_data_upload=True,
        selection_mode="engineering",
    )


def test_compatible_accounting_requires_tokens_but_not_provider_price() -> None:
    structured = {
        "schema_version": "p3_multimodal_smoke_v1",
        "message": "ok",
        "image_observed": True,
        "structured": True,
    }

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        del url, headers, body, timeout
        response = {
            "id": "compatible-2",
            "model": "qwen-returned-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(structured)},
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        return HttpResponse(200, {}, json.dumps(response).encode())

    transport = OpenAICompatibleTransport(
        api_key=SecretValue("unit-test-secret"),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    )

    response = CompleteAccountingTransport(transport).send(_request())

    assert response.provider_cost is None
    assert response.total_tokens == 30
