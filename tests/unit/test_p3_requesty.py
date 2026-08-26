"""Contract tests for the injectable Requesty Responses transport."""

from __future__ import annotations

import json
import urllib.error
from copy import deepcopy
from typing import Any

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.providers.requesty import HttpResponse, RequestyTransport
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION


def _fixture_secret() -> SecretValue:
    """Build a test credential at runtime; never persist its complete value."""

    return SecretValue("".join(chr(value) for value in (116, 101, 115, 116, 45, 111, 110, 108, 121, 45, 107, 101, 121)))


def sample_request_value(**changes: Any) -> ApiRequest:
    values: dict[str, Any] = {
        "provider": "requesty",
        "model_identifier": "openai-responses/gpt-5.6-sol",
        "endpoint_identifier": "https://router.requesty.ai/v1/responses",
        "prompt_version": "p3_transport_probe_v1",
        "response_schema_version": P3_SMOKE_SCHEMA_VERSION,
        "payload": {"input_text": "Inspect the synthetic blue square."},
        "images": (ApiImageInput("synthetic:test", "image/png", b"png-bytes"),),
        "generation_parameters": {"max_output_tokens": 128},
    }
    values.update(changes)
    return ApiRequest(**values)


def success_response() -> dict[str, object]:
    payload = {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "synthetic image observed",
        "image_observed": True,
        "structured": True,
    }
    return {
        "id": "resp_1",
        "object": "response",
        "model": "openai-responses/gpt-5.6-sol",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
            "cost": 0.00125,
        },
    }


def requesty_transport_returning(status: int, body: bytes) -> RequestyTransport:
    def sender(url: str, headers: object, request_body: bytes, timeout: float) -> HttpResponse:
        return HttpResponse(status_code=status, headers={}, body=body)

    return RequestyTransport(
        api_key=_fixture_secret(),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    )


def test_requesty_sends_responses_multimodal_json() -> None:
    """Catches a request builder that drops image/schema/auth details."""

    captured: dict[str, object] = {}

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return HttpResponse(
            status_code=200,
            headers={
                "X-Requesty-Provider": "openai",
                "X-Requesty-Request-Id": "req_gateway_1",
                "X-Requesty-Latency-Ms": "45",
                "X-Requesty-Cache": "MISS",
            },
            body=json.dumps(success_response()).encode(),
        )

    response = RequestyTransport(
        api_key=_fixture_secret(),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    ).send(sample_request_value(generation_parameters={"max_output_tokens": 128, "temperature": 0.2, "top_p": 0.7, "reasoning": {"effort": "low"}, "seed": 4}))

    sent = json.loads(captured["body"])
    content = sent["input"][0]["content"]
    assert captured["url"] == "https://router.requesty.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer " + _fixture_secret().reveal()
    assert captured["headers"]["Content-Type"] == "application/json"
    assert content[0] == {"type": "input_text", "text": "Inspect the synthetic blue square."}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    assert sent["max_output_tokens"] == 128
    assert sent["reasoning"] == {"effort": "low"}
    assert "seed" not in sent
    assert response.returned_model_identifier == "openai-responses/gpt-5.6-sol"
    assert response.provider_request_id == "resp_1"
    assert response.provider_cost == 0.00125
    assert response.safe_metadata == {
        "requesty_provider": "openai",
        "requesty_request_id": "req_gateway_1",
        "requesty_latency_ms": 45.0,
        "requesty_cache_status": "miss",
    }
    assert response.exact_backend_model_identifier is None
    assert response.exact_identity_evidence_source is None


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [(401, "authentication", False), (403, "authentication", False), (429, "rate_limit", True), (500, "provider_5xx", True), (503, "provider_5xx", True), (400, "provider_4xx", False), (404, "provider_4xx", False)],
)
def test_requesty_http_taxonomy(status: int, code: str, retryable: bool) -> None:
    """Catches status categories that accidentally expose provider bodies."""

    transport = requesty_transport_returning(status, b'{"error":{"message":"hidden"}}')
    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())
    assert (caught.value.code, caught.value.retryable, caught.value.status_code) == (code, retryable, status)
    assert "hidden" not in str(caught.value)


@pytest.mark.parametrize("exception,code", [(TimeoutError("provider detail"), "timeout"), (urllib.error.URLError("provider detail"), "connection")])
def test_requesty_network_failures_are_safe_and_retryable(exception: Exception, code: str) -> None:
    """Catches leakage of injected network exception details."""

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        raise exception

    with pytest.raises(ApiTransportError) as caught:
        RequestyTransport(api_key=_fixture_secret(), endpoint_identifier="https://router.requesty.ai/v1/responses", sender=sender).send(sample_request_value())
    assert (caught.value.code, caught.value.retryable) == (code, True)
    assert "provider detail" not in str(caught.value)


@pytest.mark.parametrize("mutation", ["malformed_json", "incomplete", "missing_output", "invalid_output_json", "negative_tokens"])
def test_requesty_parse_failures_are_typed_and_nonretryable(mutation: str) -> None:
    """Catches malformed successful HTTP payloads escaping typed parse failure."""

    response = deepcopy(success_response())
    if mutation == "malformed_json":
        body = b"{"
    else:
        if mutation == "incomplete":
            response["status"] = "in_progress"
        elif mutation == "missing_output":
            response["output"] = []
        elif mutation == "invalid_output_json":
            response["output"][0]["content"][0]["text"] = "not-json"  # type: ignore[index]
        else:
            response["usage"]["input_tokens"] = -1  # type: ignore[index]
        body = json.dumps(response).encode()
    with pytest.raises(ApiTransportError) as caught:
        requesty_transport_returning(200, body).send(sample_request_value())
    assert (caught.value.code, caught.value.retryable) == ("parse_failure", False)


def test_requesty_persists_only_whitelisted_headers() -> None:
    """Catches unsafe or unknown gateway headers being persisted."""

    def sender(url: str, headers: object, body: bytes, timeout: float) -> HttpResponse:
        return HttpResponse(status_code=200, headers={"x-requesty-provider": "openai", "authorization": "must-not-persist", "x-unexpected": "must-not-persist"}, body=json.dumps(success_response()).encode())

    response = RequestyTransport(api_key=_fixture_secret(), endpoint_identifier="https://router.requesty.ai/v1/responses", sender=sender).send(sample_request_value())
    assert response.safe_metadata == {"requesty_provider": "openai"}


@pytest.mark.parametrize(
    "api_request",
    [
        pytest.param(sample_request_value(provider="mock"), id="wrong-provider"),
        pytest.param(sample_request_value(endpoint_identifier="https://other.invalid/v1/responses"), id="wrong-endpoint"),
        pytest.param(sample_request_value(images=()), id="no-image"),
        pytest.param(sample_request_value(images=(ApiImageInput("one", "image/png", b"a"), ApiImageInput("two", "image/png", b"b"))), id="two-images"),
        pytest.param(sample_request_value(payload={"input_text": ""}), id="empty-input-text"),
    ],
)
def test_requesty_rejects_invalid_transport_input(api_request: ApiRequest) -> None:
    """Catches invalid multimodal calls reaching the sender."""

    with pytest.raises(ApiContractError):
        requesty_transport_returning(200, json.dumps(success_response()).encode()).send(api_request)
