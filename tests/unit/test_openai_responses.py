"""Contract tests for the official OpenAI Responses transport."""

from __future__ import annotations

import json
from typing import Any

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiTransportError
from surgical_agent.api.providers.openai_responses import OpenAIResponsesTransport
from surgical_agent.api.providers.openrouter import HttpResponse
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION

ENDPOINT = "https://api.openai.com/v1/responses"


def _request(**changes: Any) -> ApiRequest:
    values: dict[str, Any] = {
        "provider": "openai",
        "model_identifier": "gpt-5.6-sol",
        "endpoint_identifier": ENDPOINT,
        "prompt_version": P3_SMOKE_SCHEMA_VERSION,
        "response_schema_version": P3_SMOKE_SCHEMA_VERSION,
        "payload": {
            "system_text": "Return the schema.",
            "input_text": "Inspect six causal images.",
            "image_details": ["low", "low", "low", "low", "low", "auto"],
        },
        "images": tuple(
            ApiImageInput(f"frame:{index}", "image/png", f"image-{index}".encode())
            for index in range(6)
        ),
        "generation_parameters": {
            "max_output_tokens": 128,
            "reasoning": {"effort": "none"},
        },
    }
    values.update(changes)
    return ApiRequest(**values)


def _response_payload() -> dict[str, object]:
    return {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "six images observed",
        "image_observed": True,
        "structured": True,
    }


def _completed_event() -> bytes:
    payload = json.dumps(_response_payload(), separators=(",", ":"))
    response = {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-sol",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": payload}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events = [
        {"type": "response.output_text.delta", "delta": payload},
        {"type": "response.completed", "response": response},
    ]
    return b"".join(
        b"event: message\n" + b"data: " + json.dumps(event).encode() + b"\n\n"
        for event in events
    )


def test_openai_responses_sends_fixed_six_images_and_strict_schema() -> None:
    captured: dict[str, object] = {}

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return HttpResponse(
            200,
            {"content-type": "text/event-stream"},
            _completed_event(),
            time_to_first_token_ms=11.0,
            total_latency_ms=25.0,
        )

    response = OpenAIResponsesTransport(
        api_key=SecretValue("test-only-openai-key"),
        endpoint_identifier=ENDPOINT,
        sender=sender,
        service_tier="priority",
    ).send(_request())

    body = json.loads(captured["body"])
    content = body["input"][0]["content"]
    assert captured["url"] == ENDPOINT
    assert content[0] == {"type": "input_text", "text": "Inspect six causal images."}
    assert [item["detail"] for item in content[1:]] == [
        "low", "low", "low", "low", "low", "auto"
    ]
    assert len(content[1:]) == 6
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["verbosity"] == "low"
    assert body["reasoning"] == {"effort": "none"}
    assert body["service_tier"] == "priority"
    assert body["store"] is False and body["stream"] is True
    assert response.parsed_payload == _response_payload()
    assert response.returned_model_identifier == "gpt-5.6-sol"
    assert response.exact_backend_model_identifier == "gpt-5.6-sol"
    assert response.input_tokens == 100
    assert response.output_tokens == 20
    assert response.time_to_first_token_ms == pytest.approx(11.0)
    assert response.total_latency_ms == pytest.approx(25.0)
    assert response.provider_cost is None


def test_openai_responses_rejects_mismatched_image_details() -> None:
    transport = OpenAIResponsesTransport(
        api_key=SecretValue("test-only-openai-key"),
        endpoint_identifier=ENDPOINT,
        sender=lambda *_args: HttpResponse(200, {}, b"{}"),
    )

    with pytest.raises(Exception, match="image_details"):
        transport.send(_request(payload={"input_text": "x", "image_details": ["low"]}))


def test_invalid_completed_content_is_retryable() -> None:
    transport = OpenAIResponsesTransport(
        api_key=SecretValue("test-only-openai-key"),
        endpoint_identifier=ENDPOINT,
        sender=lambda *_args: HttpResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(
                {
                    "id": "resp_invalid",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.6-sol",
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                }
            ).encode(),
        ),
    )

    with pytest.raises(ApiTransportError) as captured:
        transport.send(_request())

    assert captured.value.code == "response_content_invalid"
    assert captured.value.retryable is True
