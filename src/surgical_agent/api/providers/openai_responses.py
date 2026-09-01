"""Official OpenAI Responses API transport for structured multimodal calls."""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, Protocol

from surgical_agent.api.contracts import (
    ApiRequest,
    CompletionTokenDetails,
    ProviderResponse,
    thaw_json,
)
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.providers.openrouter import HttpResponse, http_failure
from surgical_agent.api.schema import schema_for


class HttpSender(Protocol):
    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse: ...


def _responses_line_has_content(line: bytes) -> bool:
    if not line.startswith(b"data:"):
        return False
    try:
        event = json.loads(line[5:].strip().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(event, Mapping)
        and event.get("type") == "response.output_text.delta"
        and isinstance(event.get("delta"), str)
        and bool(event["delta"])
    )


def urllib_send_responses(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
    *,
    on_first_content: Callable[[float], None] | None = None,
) -> HttpResponse:
    """POST one Responses request while measuring the first output-text delta."""

    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    started = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            chunks: list[bytes] = []
            first_content_ms: float | None = None
            for line in response:
                chunks.append(line)
                if first_content_ms is None and _responses_line_has_content(line):
                    first_content_ms = (perf_counter() - started) * 1000.0
                    if on_first_content is not None:
                        on_first_content(first_content_ms)
            return HttpResponse(
                status_code=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=b"".join(chunks),
                time_to_first_token_ms=first_content_ms,
                total_latency_ms=(perf_counter() - started) * 1000.0,
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
            total_latency_ms=(perf_counter() - started) * 1000.0,
        )


def _response_error(code: str) -> ApiTransportError:
    return ApiTransportError(
        "OpenAI response failed a safe parsing stage",
        code=code,
        retryable=True,
    )


def _decode_responses_sse(body: bytes) -> tuple[Mapping[str, Any], str]:
    completed: Mapping[str, Any] | None = None
    deltas: list[str] = []
    for line in body.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            continue
        try:
            event = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Responses SSE event") from exc
        if not isinstance(event, Mapping):
            raise TypeError("Responses SSE event must be an object")
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ValueError("Responses text delta is invalid")
            deltas.append(delta)
        elif event_type == "response.completed":
            response = event.get("response")
            if not isinstance(response, Mapping):
                raise ValueError("Responses completion envelope is invalid")
            completed = response
        elif event_type in {"response.failed", "error"}:
            raise ValueError("Responses stream reported failure")
    if completed is None:
        raise ValueError("Responses stream has no completed response")
    return completed, "".join(deltas)


def _output_text(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        raise TypeError("Responses output is invalid")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            raise TypeError("Responses message content is invalid")
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("Responses output text is invalid")
                parts.append(text)
    if not parts:
        raise ValueError("Responses output has no text")
    return "".join(parts)


def _count(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Responses usage is invalid")
    return value


class OpenAIResponsesTransport:
    """Official Responses API adapter with one to six ordered causal images."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: SecretValue,
        endpoint_identifier: str,
        sender: HttpSender = urllib_send_responses,
        timeout_seconds: float = 120.0,
        service_tier: str | None = None,
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
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if service_tier not in {None, "auto", "default", "priority", "ultrafast"}:
            raise ValueError("service_tier is unsupported")
        self.api_key = api_key
        self.endpoint_identifier = endpoint_identifier
        self.sender = sender
        self.timeout_seconds = float(timeout_seconds)
        self.service_tier = service_tier
        self._on_first_content: Callable[[float], None] | None = None

    def set_first_content_callback(
        self, callback: Callable[[float], None] | None
    ) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("first content callback must be callable")
        self._on_first_content = callback

    @staticmethod
    def _schema_name(version: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", version)

    @staticmethod
    def _details(request: ApiRequest) -> tuple[str, ...]:
        raw = request.payload.get("image_details")
        if raw is None:
            return tuple("auto" for _ in request.images)
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != len(request.images)
            or any(value not in {"low", "high", "auto", "original"} for value in raw)
        ):
            raise ApiContractError("image_details must match the image tuple")
        return tuple(raw)

    def _validate_request(self, request: ApiRequest) -> None:
        if request.provider != self.provider:
            raise ApiContractError("request provider does not match OpenAI transport")
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError("request endpoint does not match OpenAI transport")
        if not 1 <= len(request.images) <= 6:
            raise ApiContractError("OpenAI Responses requires one to six images")
        input_text = request.payload.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ApiContractError("OpenAI Responses requires non-empty input_text")
        self._details(request)

    def _request_body(self, request: ApiRequest) -> bytes:
        input_text = request.payload["input_text"]
        assert isinstance(input_text, str)
        system_text = request.payload.get("system_text")
        if system_text is not None and (
            not isinstance(system_text, str) or not system_text.strip()
        ):
            raise ApiContractError("system_text must be non-empty text")
        content: list[dict[str, object]] = [
            {"type": "input_text", "text": input_text}
        ]
        for image, detail in zip(request.images, self._details(request), strict=True):
            encoded = base64.b64encode(image.content).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image.mime_type};base64,{encoded}",
                    "detail": detail,
                }
            )
        body: dict[str, object] = {
            "model": request.model_identifier,
            "instructions": system_text,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self._schema_name(request.response_schema_version),
                    "strict": True,
                    "schema": schema_for(request.response_schema_version),
                },
                "verbosity": "low",
            },
            "store": False,
            "stream": True,
        }
        generation = request.generation_parameters
        if "max_output_tokens" in generation:
            body["max_output_tokens"] = thaw_json(generation["max_output_tokens"])
        if "reasoning" in generation:
            body["reasoning"] = thaw_json(generation["reasoning"])
        for name in ("temperature", "top_p"):
            if name in generation:
                body[name] = thaw_json(generation[name])
        if self.service_tier is not None:
            body["service_tier"] = self.service_tier
        return json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def send(self, request: ApiRequest) -> ProviderResponse:
        self._validate_request(request)
        body = self._request_body(request)
        headers = {
            "Authorization": f"Bearer {self.api_key.reveal()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        started = perf_counter()
        try:
            if self.sender is urllib_send_responses:
                raw = urllib_send_responses(
                    self.endpoint_identifier,
                    headers,
                    body,
                    self.timeout_seconds,
                    on_first_content=self._on_first_content,
                )
            else:
                raw = self.sender(
                    self.endpoint_identifier,
                    headers,
                    body,
                    self.timeout_seconds,
                )
        except TimeoutError:
            raise ApiTransportError(
                "OpenAI request timed out", code="timeout", retryable=True
            ) from None
        except (urllib.error.URLError, OSError):
            raise ApiTransportError(
                "OpenAI connection failed", code="connection", retryable=True
            ) from None
        finally:
            del body
        if not 200 <= raw.status_code <= 299:
            raise http_failure(raw.status_code, raw.body)
        total_latency_ms = (
            raw.total_latency_ms
            if raw.total_latency_ms is not None
            else (perf_counter() - started) * 1000.0
        )
        try:
            if "text/event-stream" in raw.headers.get("content-type", "").casefold():
                response, streamed_text = _decode_responses_sse(raw.body)
            else:
                decoded = json.loads(raw.body.decode("utf-8"))
                if not isinstance(decoded, Mapping):
                    raise ValueError("Responses envelope is invalid")
                response, streamed_text = decoded, ""
            if response.get("object") != "response" or response.get("status") != "completed":
                raise ValueError("Responses completion is not complete")
            response_id = response.get("id")
            model = response.get("model")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("Responses ID is invalid")
            if not isinstance(model, str) or not model:
                raise ValueError("Responses model is invalid")
            content = streamed_text or _output_text(response)
            parsed = json.loads(content)
            if not isinstance(parsed, Mapping):
                raise TypeError("structured output must be an object")
            usage = response.get("usage")
            if not isinstance(usage, Mapping):
                raise TypeError("Responses usage is missing")
            input_tokens = _count(usage, "input_tokens")
            output_tokens = _count(usage, "output_tokens")
            total_tokens = _count(usage, "total_tokens")
            details = usage.get("output_tokens_details", {})
            if not isinstance(details, Mapping):
                raise TypeError("Responses output usage details are invalid")
            reasoning_tokens = details.get("reasoning_tokens", 0)
            if (
                not isinstance(reasoning_tokens, int)
                or isinstance(reasoning_tokens, bool)
                or reasoning_tokens < 0
            ):
                raise ValueError("Responses reasoning usage is invalid")
            return ProviderResponse(
                provider=self.provider,
                returned_model_identifier=model,
                parsed_payload=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                completion_tokens_details=CompletionTokenDetails(
                    reasoning_tokens=reasoning_tokens
                ),
                visible_output_tokens=output_tokens - reasoning_tokens,
                time_to_first_token_ms=raw.time_to_first_token_ms,
                total_latency_ms=float(total_latency_ms),
                image_count=len(request.images),
                provider_request_id=response_id,
                provider_cost=None,
                exact_backend_model_identifier=model,
                exact_identity_evidence_source="openai_responses_response.model",
                safe_metadata={},
            )
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise _response_error("response_content_invalid") from None


__all__ = ["OpenAIResponsesTransport", "urllib_send_responses"]
