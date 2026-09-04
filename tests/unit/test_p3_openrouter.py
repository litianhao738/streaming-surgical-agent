"""Contract tests for the injectable OpenRouter Chat Completions transport."""

from __future__ import annotations

import json
import urllib.error
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any, ClassVar

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.openrouter_routing import (
    LATENCY_FALLBACK_ROUTING_PROFILE,
    OPENROUTER_ROUTING_PAYLOAD_KEY,
    STRICT_ANTHROPIC_ROUTING_PROFILE,
    STRICT_GOOGLE_AI_STUDIO_ROUTING_PROFILE,
    STRICT_OPENAI_ROUTING_PROFILE,
)
from surgical_agent.api.providers import openrouter as openrouter_module
from surgical_agent.api.providers.openrouter import HttpResponse, OpenRouterTransport
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSION,
)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _fixture_secret() -> SecretValue:
    return SecretValue("test-only-openrouter-key")


def sample_request_value(**changes: Any) -> ApiRequest:
    values: dict[str, Any] = {
        "provider": "openrouter",
        "model_identifier": "openai/gpt-5.6-sol",
        "endpoint_identifier": ENDPOINT,
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
        "id": "gen_1",
        "object": "chat.completion",
        "model": "openai/gpt-5.6-sol",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "cost": 0.00125,
        },
    }


def openrouter_transport_returning(status: int, body: bytes) -> OpenRouterTransport:
    def sender(
        url: str,
        headers: object,
        request_body: bytes,
        timeout: float,
    ) -> HttpResponse:
        return HttpResponse(status_code=status, headers={}, body=body)

    return OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    )


def test_openrouter_sends_chat_multimodal_structured_json() -> None:
    """Catches a request builder that drops image/schema/auth details."""

    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return HttpResponse(
            status_code=200,
            headers={"authorization": "must-not-persist"},
            body=json.dumps(success_response()).encode(),
        )

    response = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(
        sample_request_value(
            generation_parameters={
                "max_output_tokens": 128,
                "temperature": 0.2,
                "top_p": 0.7,
                "reasoning": {"effort": "low"},
                "seed": 4,
            }
        )
    )

    sent = json.loads(captured["body"])
    content = sent["messages"][1]["content"]
    assert captured["url"] == ENDPOINT
    assert captured["headers"]["Authorization"] == (
        "Bearer " + _fixture_secret().reveal()
    )
    assert captured["headers"]["Content-Type"] == "application/json"
    assert content[0] == {
        "type": "text",
        "text": "Inspect the synthetic blue square.",
    }
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert sent["max_tokens"] == 128
    assert sent["reasoning"] == {"effort": "low"}
    assert "max_output_tokens" not in sent
    assert "seed" not in sent
    assert response.returned_model_identifier == "openai/gpt-5.6-sol"
    assert response.provider_request_id == "gen_1"
    assert response.provider_cost == 0.00125
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 8
    assert response.completion_tokens_details.reasoning_tokens is None
    assert response.visible_output_tokens is None
    assert response.time_to_first_token_ms is None
    assert response.total_latency_ms is not None
    assert response.safe_metadata == {}
    assert response.exact_backend_model_identifier is None
    assert response.exact_identity_evidence_source is None


def test_openrouter_accepts_six_ordered_images() -> None:
    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        del url, headers, timeout
        captured["body"] = body
        return HttpResponse(
            status_code=200,
            headers={},
            body=json.dumps(success_response()).encode(),
        )

    images = tuple(
        ApiImageInput(f"synthetic:{index}", "image/png", f"png-{index}".encode())
        for index in range(6)
    )
    response = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(sample_request_value(images=images))

    sent = json.loads(captured["body"])
    image_parts = [
        item for item in sent["messages"][1]["content"] if item["type"] == "image_url"
    ]
    assert len(image_parts) == 6
    assert response.image_count == 6


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        (
            STRICT_OPENAI_ROUTING_PROFILE,
            {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        ),
        (
            STRICT_ANTHROPIC_ROUTING_PROFILE,
            {
                "only": ["anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        ),
        (
            STRICT_GOOGLE_AI_STUDIO_ROUTING_PROFILE,
            {
                "only": ["google-ai-studio"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        ),
        (
            LATENCY_FALLBACK_ROUTING_PROFILE,
            {
                "only": ["openai", "azure", "amazon-bedrock"],
                "allow_fallbacks": True,
                "require_parameters": True,
                "sort": "latency",
            },
        ),
    ),
)
def test_openrouter_maps_closed_routing_profiles(
    profile: str,
    expected: dict[str, object],
) -> None:
    transport = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
    )
    request = sample_request_value(
        payload={
            "input_text": "Inspect the synthetic blue square.",
            OPENROUTER_ROUTING_PAYLOAD_KEY: profile,
        }
    )

    body = json.loads(transport._request_body(request))

    assert body["provider"] == expected
    assert ("sort" in body["provider"]) is (
        profile == LATENCY_FALLBACK_ROUTING_PROFILE
    )


def test_openrouter_legacy_request_defaults_to_strict_routing() -> None:
    transport = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
    )

    body = json.loads(transport._request_body(sample_request_value()))

    assert body["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


@pytest.mark.parametrize("profile", ("unknown", "", True, None))
def test_openrouter_rejects_unknown_routing_before_network(profile: object) -> None:
    sender_called = False

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        nonlocal sender_called
        del url, headers, body, timeout
        sender_called = True
        return HttpResponse(status_code=500, headers={}, body=b"")

    transport = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    )
    request = sample_request_value(
        payload={
            "input_text": "Inspect the synthetic blue square.",
            OPENROUTER_ROUTING_PAYLOAD_KEY: profile,
        }
    )

    with pytest.raises(ApiContractError, match="routing profile"):
        transport.send(request)
    assert sender_called is False


def test_openrouter_streams_literal_sse_and_records_first_content_ttft() -> None:
    """Catches buffered SSE parsing or TTFT inferred from total completion time."""

    literal_sse = (
        b'data: {"id":"gen_stream","object":"chat.completion.chunk","model":"openai/gpt-5.6-sol","choices":[{"index":0,"delta":{"role":"assistant","content":"{\\"schema_version\\":\\"p3_multimodal_smoke_v1\\",\\"message\\":\\"streamed "},"finish_reason":null}]}\n\n'
        b'data: {"id":"gen_stream","object":"chat.completion.chunk","model":"openai/gpt-5.6-sol","choices":[{"index":0,"delta":{"content":"synthetic image observed\\",\\"image_observed\\":true,\\"structured\\":true}"},"finish_reason":null}]}\n\n'
        b'data: {"id":"gen_stream","object":"chat.completion.chunk","model":"openai/gpt-5.6-sol","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: {"id":"gen_stream","object":"chat.completion.chunk","model":"openai/gpt-5.6-sol","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18,"completion_tokens_details":{"reasoning_tokens":3,"accepted_prediction_tokens":7},"cost":0.00125}}\n\n'
        b"data: [DONE]\n\n"
    )
    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        del url, headers, timeout
        captured["body"] = body
        return HttpResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            body=literal_sse,
            time_to_first_token_ms=12.5,
            total_latency_ms=31.0,
        )

    response = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(
        sample_request_value(generation_parameters={"reasoning": {"effort": "none"}})
    )

    assert json.loads(captured["body"])["stream"] is True
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 8
    assert response.completion_tokens_details.reasoning_tokens == 3
    assert response.completion_tokens_details.to_mapping() == {"reasoning_tokens": 3}
    assert response.visible_output_tokens == 5
    assert response.time_to_first_token_ms == pytest.approx(12.5)
    assert response.total_latency_ms == pytest.approx(31.0)


def test_urllib_sender_measures_ttft_while_iterating_first_nonempty_sse_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches an HTTP sender that waits for the body before timing first content."""

    class StreamingResponse(AbstractContextManager["StreamingResponse"]):
        status = 200
        headers: ClassVar[dict[str, str]] = {"content-type": "text/event-stream"}

        def __exit__(self, *args: object) -> None:
            del args

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"{"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    clock = iter((100.0, 100.0125, 100.031))
    monkeypatch.setattr(
        openrouter_module.urllib.request,
        "urlopen",
        lambda *_a, **_k: StreamingResponse(),
    )
    monkeypatch.setattr(openrouter_module, "perf_counter", lambda: next(clock))

    first_content_notifications: list[float] = []
    response = openrouter_module.urllib_send_json(
        ENDPOINT,
        {"Content-Type": "application/json"},
        b"{}",
        1.0,
        on_first_content=first_content_notifications.append,
    )

    assert response.time_to_first_token_ms == pytest.approx(12.5)
    assert response.total_latency_ms == pytest.approx(31.0)
    assert first_content_notifications == [pytest.approx(12.5)]


def test_openrouter_keeps_three_images_in_causal_tuple_order() -> None:
    """Catches a transport that silently drops or reorders causal images."""

    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        captured["body"] = body
        return HttpResponse(
            status_code=200,
            headers={},
            body=json.dumps(success_response()).encode(),
        )

    response = OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(
        sample_request_value(
            response_schema_version=JOINT_PERCEPTION_SCHEMA_VERSION,
            payload={
                "system_text": "Return strict joint JSON.",
                "input_text": "causal-window",
            },
            images=(
                ApiImageInput("synthetic:first", "image/png", b"first"),
                ApiImageInput("synthetic:second", "image/jpeg", b"second"),
                ApiImageInput("synthetic:third", "image/webp", b"third"),
            ),
        )
    )

    sent = json.loads(captured["body"])
    content = sent["messages"][1]["content"]
    assert sent["messages"][0]["content"] == "Return strict joint JSON."
    assert "reasoning" not in sent
    assert content[0] == {"type": "text", "text": "causal-window"}
    assert [item["image_url"]["url"] for item in content[1:]] == [
        "data:image/png;base64,Zmlyc3Q=",
        "data:image/jpeg;base64,c2Vjb25k",
        "data:image/webp;base64,dGhpcmQ=",
    ]
    assert sent["response_format"]["json_schema"]["name"] == (
        "joint_perception_frame_v1"
    )
    assert response.image_count == 3


def test_openrouter_sends_compact_joint_schema_without_reasoning() -> None:
    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        del url, headers, timeout
        captured["body"] = body
        return HttpResponse(
            status_code=200,
            headers={},
            body=json.dumps(success_response()).encode(),
        )

    OpenRouterTransport(
        api_key=_fixture_secret(),
        endpoint_identifier=ENDPOINT,
        sender=sender,
    ).send(
        sample_request_value(
            response_schema_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
            generation_parameters={"max_output_tokens": 4096},
        )
    )

    sent = json.loads(captured["body"])
    schema = sent["response_format"]["json_schema"]
    assert schema["name"] == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert {
        task: schema["schema"]["properties"][task]["properties"]["topk"]["maxItems"]
        for task in ("instrument", "verb", "target", "ivt", "phase")
    } == {"instrument": 3, "verb": 4, "target": 5, "ivt": 8, "phase": 3}
    assert schema["schema"]["properties"]["evidence_refs"]["maxItems"] == 6
    assert sent["max_tokens"] == 4096
    assert "reasoning" not in sent


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, "authentication", False),
        (403, "authentication", False),
        (402, "payment_required", False),
        (408, "timeout", True),
        (429, "rate_limit", True),
        (500, "provider_5xx", True),
        (503, "provider_5xx", True),
        (400, "provider_4xx", False),
        (404, "provider_4xx", False),
    ],
)
def test_openrouter_http_taxonomy(status: int, code: str, retryable: bool) -> None:
    transport = openrouter_transport_returning(
        status, b'{"error":{"message":"hidden"}}'
    )

    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())

    assert (
        caught.value.code,
        caught.value.retryable,
        caught.value.status_code,
    ) == (code, retryable, status)
    assert "hidden" not in str(caught.value)


def test_openrouter_region_unavailable_403_is_safe_and_retryable() -> None:
    """A regional backend rejection must not masquerade as a bad API key."""

    provider_detail = "This model is not available in your region."
    transport = openrouter_transport_returning(
        403,
        json.dumps({"error": {"code": 403, "message": provider_detail}}).encode(
            "utf-8"
        ),
    )

    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())

    assert (
        caught.value.code,
        caught.value.retryable,
        caught.value.status_code,
    ) == ("region_unavailable", True, 403)
    assert provider_detail not in str(caught.value)


def test_openrouter_content_moderation_403_is_not_authentication() -> None:
    """A flagged surgical image must not be reported as an invalid API key."""

    provider_detail = (
        "requested model requires moderation. Your input was flagged for "
        '"violence/graphic".'
    )
    transport = openrouter_transport_returning(
        403,
        json.dumps({"error": {"code": 403, "message": provider_detail}}).encode(
            "utf-8"
        ),
    )

    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())

    assert (
        caught.value.code,
        caught.value.retryable,
        caught.value.status_code,
    ) == ("content_moderation", False, 403)
    assert provider_detail not in str(caught.value)


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (TimeoutError("provider detail"), "timeout"),
        (urllib.error.URLError("provider detail"), "connection"),
    ],
)
def test_openrouter_network_failures_are_safe_and_retryable(
    exception: Exception, code: str
) -> None:
    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        raise exception

    with pytest.raises(ApiTransportError) as caught:
        OpenRouterTransport(
            api_key=_fixture_secret(),
            endpoint_identifier=ENDPOINT,
            sender=sender,
        ).send(sample_request_value())

    assert (caught.value.code, caught.value.retryable) == (code, True)
    assert "provider detail" not in str(caught.value)


@pytest.mark.parametrize(
    ("mutation", "expected_code", "untrusted_value"),
    [
        pytest.param(
            "malformed_json",
            "response_envelope_invalid",
            "malformed-envelope-detail",
            id="malformed-json-envelope",
        ),
        pytest.param(
            "wrong_object",
            "response_envelope_invalid",
            "untrusted-object-kind",
            id="wrong-object-envelope",
        ),
        pytest.param(
            "missing_choice",
            "response_envelope_invalid",
            "untrusted-choice-detail",
            id="missing-choice-envelope",
        ),
        pytest.param(
            "completion_length",
            "completion_length",
            "untrusted-length-detail",
            id="length-completion",
        ),
        pytest.param(
            "completion_nonstop",
            "completion_nonstop",
            "untrusted-finish-value",
            id="other-nonstop-completion",
        ),
        pytest.param(
            "null_content",
            "response_content_invalid",
            "untrusted-refusal-detail",
            id="null-content",
        ),
        pytest.param(
            "list_content",
            "response_content_invalid",
            "untrusted-list-content",
            id="list-content",
        ),
        pytest.param(
            "invalid_output_json",
            "response_content_invalid",
            "untrusted-invalid-json-content",
            id="invalid-json-content",
        ),
        pytest.param(
            "missing_usage",
            "response_usage_invalid",
            "untrusted-usage-detail",
            id="missing-usage",
        ),
        pytest.param(
            "negative_tokens",
            "response_usage_invalid",
            "untrusted-token-detail",
            id="bad-usage",
        ),
    ],
)
def test_openrouter_parse_failures_use_safe_stage_taxonomy(
    mutation: str,
    expected_code: str,
    untrusted_value: str,
) -> None:
    response = deepcopy(success_response())
    if mutation == "malformed_json":
        body = f"{{{untrusted_value}".encode()
    else:
        if mutation == "wrong_object":
            response["object"] = untrusted_value
        elif mutation == "missing_choice":
            response["choices"] = []
            response["untrusted"] = untrusted_value
        elif mutation == "completion_length":
            response["choices"][0]["finish_reason"] = "length"  # type: ignore[index]
            response["choices"][0]["untrusted"] = untrusted_value  # type: ignore[index]
        elif mutation == "completion_nonstop":
            response["choices"][0]["finish_reason"] = untrusted_value  # type: ignore[index]
        elif mutation == "null_content":
            response["choices"][0]["message"] = {  # type: ignore[index]
                "content": None,
                "refusal": untrusted_value,
            }
        elif mutation == "list_content":
            response["choices"][0]["message"]["content"] = [  # type: ignore[index]
                {"type": "text", "text": untrusted_value}
            ]
        elif mutation == "invalid_output_json":
            response["choices"][0]["message"]["content"] = untrusted_value  # type: ignore[index]
        elif mutation == "missing_usage":
            response.pop("usage")
            response["untrusted"] = untrusted_value
        elif mutation == "negative_tokens":
            response["usage"]["prompt_tokens"] = -1  # type: ignore[index]
            response["usage"]["untrusted"] = untrusted_value  # type: ignore[index]
        else:
            raise AssertionError(f"unknown mutation: {mutation}")
        body = json.dumps(response).encode()

    with pytest.raises(ApiTransportError) as caught:
        openrouter_transport_returning(200, body).send(sample_request_value())

    assert (caught.value.code, caught.value.retryable) == (expected_code, False)
    rendered = f"{caught.value!s} {caught.value!r}"
    assert untrusted_value not in rendered


@pytest.mark.parametrize(
    "api_request",
    [
        pytest.param(sample_request_value(provider="mock"), id="wrong-provider"),
        pytest.param(
            sample_request_value(
                endpoint_identifier="https://other.invalid/v1/chat/completions"
            ),
            id="wrong-endpoint",
        ),
        pytest.param(sample_request_value(images=()), id="no-image"),
        pytest.param(
            sample_request_value(
                    images=tuple(
                        ApiImageInput(str(index), "image/png", bytes([index]))
                        for index in range(7)
                    )
                ),
                id="seven-images",
        ),
        pytest.param(
            sample_request_value(payload={"input_text": ""}),
            id="empty-input-text",
        ),
    ],
)
def test_openrouter_rejects_invalid_transport_input(
    api_request: ApiRequest,
) -> None:
    with pytest.raises(ApiContractError):
        openrouter_transport_returning(
            200, json.dumps(success_response()).encode()
        ).send(api_request)
