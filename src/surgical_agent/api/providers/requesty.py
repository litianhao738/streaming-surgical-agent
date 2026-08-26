"""Requesty Responses API transport with an injected HTTP boundary."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from surgical_agent.api.contracts import ApiRequest, ProviderResponse, thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.schema import schema_for


@dataclass(frozen=True)
class HttpResponse:
    """The small HTTP result needed by this provider transport."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


class HttpSender(Protocol):
    """Send one JSON request and return its raw response."""

    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        """Return a response or raise timeout/connection exceptions."""


def urllib_send_json(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> HttpResponse:
    """Use urllib at the sole real-network boundary."""

    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return HttpResponse(
                status_code=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )


def http_failure(status_code: int) -> ApiTransportError:
    """Map HTTP statuses to the P3 retry taxonomy without body detail."""

    if status_code in {401, 403}:
        return ApiTransportError(
            "Requesty authentication failed",
            code="authentication",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 429:
        return ApiTransportError(
            "Requesty rate limit", code="rate_limit", retryable=True, status_code=status_code
        )
    if 500 <= status_code <= 599:
        return ApiTransportError(
            "Requesty provider failure",
            code="provider_5xx",
            retryable=True,
            status_code=status_code,
        )
    if 400 <= status_code <= 499:
        return ApiTransportError(
            "Requesty request failed",
            code="provider_4xx",
            retryable=False,
            status_code=status_code,
        )
    return ApiTransportError(
        "Requesty returned an unexpected HTTP status",
        code="unexpected_http_status",
        retryable=False,
        status_code=status_code,
    )


def extract_output_text(response: Mapping[str, Any]) -> str:
    """Require the one output text value from a completed Responses result."""

    output = response.get("output")
    if not isinstance(output, list):
        raise TypeError("output must be a list")
    texts = [
        item["text"]
        for message in output
        if isinstance(message, Mapping) and message.get("type") == "message"
        for item in message.get("content", [])
        if isinstance(item, Mapping)
        and item.get("type") == "output_text"
        and isinstance(item.get("text"), str)
    ]
    if len(texts) != 1:
        raise ValueError("response must contain one output_text")
    return texts[0]


def _integer_usage(usage: Mapping[str, Any], name: str) -> int:
    value = usage.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("usage is invalid")
    return value


def _safe_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    """Normalize only documented, contract-safe Requesty response headers."""

    lowered = {
        key.lower(): value
        for key, value in headers.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    metadata: dict[str, Any] = {}
    if lowered.get("x-requesty-provider") == "openai":
        metadata["requesty_provider"] = "openai"
    request_id = lowered.get("x-requesty-request-id")
    if request_id is not None:
        metadata["requesty_request_id"] = request_id
    cache = lowered.get("x-requesty-cache")
    if cache is not None:
        metadata["requesty_cache_status"] = cache.lower()
    latency = lowered.get("x-requesty-latency-ms")
    if latency is not None:
        try:
            parsed_latency = float(latency)
        except ValueError:
            pass
        else:
            metadata["requesty_latency_ms"] = parsed_latency
    return metadata


class RequestyTransport:
    """P3 Requesty Responses transport for exactly one synthetic image."""

    provider = "requesty"

    def __init__(
        self,
        *,
        api_key: SecretValue,
        endpoint_identifier: str,
        sender: HttpSender = urllib_send_json,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(api_key, SecretValue):
            raise TypeError("api_key must be SecretValue")
        if not isinstance(endpoint_identifier, str) or not endpoint_identifier:
            raise ValueError("endpoint_identifier must be non-empty")
        if not callable(sender):
            raise TypeError("sender must be callable")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.api_key = api_key
        self.endpoint_identifier = endpoint_identifier
        self.sender = sender
        self.timeout_seconds = float(timeout_seconds)

    def _validate_request(self, request: ApiRequest) -> None:
        if request.provider != self.provider:
            raise ApiContractError("Request provider does not match Requesty transport")
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError("Request endpoint does not match Requesty transport")
        if len(request.images) != 1:
            raise ApiContractError("Requesty requires exactly one synthetic image")
        input_text = request.payload.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ApiContractError("Requesty requires non-empty payload input_text")

    def _request_body(self, request: ApiRequest) -> bytes:
        image = request.images[0]
        input_text = request.payload["input_text"]
        assert isinstance(input_text, str)
        permitted_parameters = {
            key: thaw_json(value)
            for key, value in request.generation_parameters.items()
            if key in {"max_output_tokens", "temperature", "top_p", "reasoning"}
        }
        body = {
            "model": request.model_identifier,
            "instructions": (
                "Return only the requested P3 transport-probe JSON. "
                "Do not emit surgical predictions or P4 instance fields."
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": input_text},
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{image.mime_type};base64,"
                                f"{base64.b64encode(image.content).decode('ascii')}"
                            ),
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "p3_multimodal_smoke",
                    "strict": True,
                    "schema": schema_for(request.response_schema_version),
                }
            },
            **permitted_parameters,
        }
        return json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def send(self, request: ApiRequest) -> ProviderResponse:
        self._validate_request(request)
        body = self._request_body(request)
        headers = {
            "Authorization": f"Bearer {self.api_key.reveal()}",
            "Content-Type": "application/json",
            "X-Title": "CholecTrack20 P3 transport probe",
        }
        try:
            raw_response = self.sender(
                self.endpoint_identifier, headers, body, self.timeout_seconds
            )
        except TimeoutError:
            raise ApiTransportError(
                "Requesty request timed out", code="timeout", retryable=True
            ) from None
        except (urllib.error.URLError, OSError):
            raise ApiTransportError(
                "Requesty connection failed", code="connection", retryable=True
            ) from None
        finally:
            del body
        if not 200 <= raw_response.status_code <= 299:
            raise http_failure(raw_response.status_code)
        try:
            decoded = json.loads(raw_response.body.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise TypeError("response must be an object")
            if decoded.get("object") != "response" or decoded.get("status") != "completed":
                raise ValueError("response is not completed")
            response_id = decoded.get("id")
            returned_model = decoded.get("model")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("response id is invalid")
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError("response model is invalid")
            parsed_payload = json.loads(extract_output_text(decoded))
            if not isinstance(parsed_payload, Mapping):
                raise TypeError("output JSON must be an object")
            usage = decoded.get("usage")
            if not isinstance(usage, Mapping):
                raise TypeError("usage must be an object")
            cost = usage.get("cost")
            if cost is not None and (
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or cost < 0
            ):
                raise ValueError("usage cost is invalid")
            return ProviderResponse(
                provider=self.provider,
                returned_model_identifier=returned_model,
                parsed_payload=parsed_payload,
                input_tokens=_integer_usage(usage, "input_tokens"),
                output_tokens=_integer_usage(usage, "output_tokens"),
                total_tokens=_integer_usage(usage, "total_tokens"),
                image_count=1,
                provider_request_id=response_id,
                provider_cost=(None if cost is None else float(cost)),
                exact_backend_model_identifier=None,
                exact_identity_evidence_source=None,
                safe_metadata=_safe_headers(raw_response.headers),
            )
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise ApiTransportError(
                "Requesty response parsing failed", code="parse_failure", retryable=False
            ) from None
