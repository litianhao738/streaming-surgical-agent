"""OpenRouter Chat Completions transport."""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
from surgical_agent.api.openrouter_routing import (
    provider_preferences,
    routing_profile_from_request_payload,
)
from surgical_agent.api.schema import schema_for


@dataclass(frozen=True)
class HttpResponse:
    """The small HTTP result needed by this provider transport."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)
    time_to_first_token_ms: float | None = None
    total_latency_ms: float | None = None


class HttpSender(Protocol):
    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse: ...


def urllib_send_json(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
    *,
    on_first_content: Callable[[float], None] | None = None,
) -> HttpResponse:
    """Use urllib at the sole real-network boundary."""

    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    started = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            chunks: list[bytes] = []
            time_to_first_token_ms: float | None = None
            for line in response:
                chunks.append(line)
                if time_to_first_token_ms is None and _sse_line_has_content(line):
                    time_to_first_token_ms = (perf_counter() - started) * 1000.0
                    if on_first_content is not None:
                        on_first_content(time_to_first_token_ms)
            return HttpResponse(
                status_code=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=b"".join(chunks),
                time_to_first_token_ms=time_to_first_token_ms,
                total_latency_ms=(perf_counter() - started) * 1000.0,
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
            total_latency_ms=(perf_counter() - started) * 1000.0,
        )


def _safe_provider_error_message(body: bytes) -> str | None:
    """Read only the transient provider message used for safe categorization."""

    try:
        decoded = json.loads(body.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            return None
        error = decoded.get("error")
        if not isinstance(error, Mapping):
            return None
        message = error.get("message")
        return message if isinstance(message, str) else None
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _is_region_unavailable(body: bytes) -> bool:
    """Recognize the provider's regional rejection without retaining its body."""

    message = _safe_provider_error_message(body)
    return message is not None and "not available in your region" in message.casefold()


def _is_content_moderation_block(body: bytes) -> bool:
    """Separate image moderation rejections from credential failures."""

    message = _safe_provider_error_message(body)
    if message is None:
        return False
    normalized = message.casefold()
    return "requires moderation" in normalized and "input was flagged" in normalized


def http_failure(status_code: int, body: bytes = b"") -> ApiTransportError:
    """Map HTTP statuses without retaining the provider response body."""

    if status_code == 401:
        return ApiTransportError(
            "OpenRouter authentication failed",
            code="authentication",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 403 and _is_region_unavailable(body):
        return ApiTransportError(
            "OpenRouter model is temporarily unavailable in this region",
            code="region_unavailable",
            retryable=True,
            status_code=status_code,
        )
    if status_code == 403 and _is_content_moderation_block(body):
        return ApiTransportError(
            "OpenRouter rejected the input through provider moderation",
            code="content_moderation",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 403:
        return ApiTransportError(
            "OpenRouter authorization failed",
            code="authentication",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 402:
        return ApiTransportError(
            "OpenRouter payment required",
            code="payment_required",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 408:
        return ApiTransportError(
            "OpenRouter request timed out",
            code="timeout",
            retryable=True,
            status_code=status_code,
        )
    if status_code == 429:
        return ApiTransportError(
            "OpenRouter rate limit",
            code="rate_limit",
            retryable=True,
            status_code=status_code,
        )
    if 500 <= status_code <= 599:
        return ApiTransportError(
            "OpenRouter provider failure",
            code="provider_5xx",
            retryable=True,
            status_code=status_code,
        )
    if 400 <= status_code <= 499:
        return ApiTransportError(
            "OpenRouter request failed",
            code="provider_4xx",
            retryable=False,
            status_code=status_code,
        )
    return ApiTransportError(
        "OpenRouter returned an unexpected HTTP status",
        code="unexpected_http_status",
        retryable=False,
        status_code=status_code,
    )


def _integer_usage(usage: Mapping[str, Any], name: str) -> int:
    value = usage.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("usage is invalid")
    return value


def _sse_line_has_content(line: bytes) -> bool:
    """Inspect a streamed data line without retaining decoded response data."""

    if not line.startswith(b"data:"):
        return False
    try:
        value = json.loads(line[5:].strip().decode("utf-8"))
        choices = value.get("choices") if isinstance(value, Mapping) else None
        if not isinstance(choices, list):
            return False
        return any(
            isinstance(choice, Mapping)
            and isinstance(choice.get("delta"), Mapping)
            and isinstance(choice["delta"].get("content"), str)
            and bool(choice["delta"]["content"])
            for choice in choices
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _is_sse_response(response: HttpResponse) -> bool:
    content_type = response.headers.get("content-type", "")
    return (
        "text/event-stream" in content_type.casefold()
        or response.body.lstrip().startswith(b"data:")
    )


def _decode_sse_response(body: bytes) -> dict[str, object]:
    """Normalize OpenAI-compatible chunks into the legacy completion envelope."""

    response_id: str | None = None
    returned_model: str | None = None
    content_parts: list[str] = []
    finish_reason: object = None
    usage: Mapping[str, Any] | None = None
    seen_done = False
    saw_event = False
    for line in body.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            seen_done = True
            continue
        try:
            chunk = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid SSE data") from exc
        if (
            not isinstance(chunk, Mapping)
            or chunk.get("object") != "chat.completion.chunk"
        ):
            raise ValueError("invalid SSE chunk")
        chunk_id = chunk.get("id")
        chunk_model = chunk.get("model")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("invalid SSE response id")
        if not isinstance(chunk_model, str) or not chunk_model:
            raise ValueError("invalid SSE response model")
        if response_id is None:
            response_id = chunk_id
            returned_model = chunk_model
        elif response_id != chunk_id or returned_model != chunk_model:
            raise ValueError("inconsistent SSE response identity")
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("invalid SSE choices")
        if choices:
            choice = choices[0]
            if not isinstance(choice, Mapping) or choice.get("index") != 0:
                raise ValueError("invalid SSE choice")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise ValueError("invalid SSE delta")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("invalid SSE content")
                content_parts.append(content)
            candidate_finish_reason = choice.get("finish_reason")
            if candidate_finish_reason is not None:
                if finish_reason is None:
                    finish_reason = candidate_finish_reason
                elif finish_reason != candidate_finish_reason:
                    raise ValueError("conflicting SSE finish reason")
        candidate_usage = chunk.get("usage")
        if candidate_usage is not None:
            if usage is not None or not isinstance(candidate_usage, Mapping):
                raise ValueError("invalid SSE usage")
            usage = candidate_usage
        saw_event = True
    if not saw_event or not seen_done or response_id is None or returned_model is None:
        raise ValueError("incomplete SSE response")
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": returned_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": "".join(content_parts)},
            }
        ],
        "usage": usage,
    }


def _response_error(code: str) -> ApiTransportError:
    return ApiTransportError(
        "OpenRouter response failed a safe parsing stage",
        code=code,
        retryable=False,
    )


class OpenRouterTransport:
    """OpenRouter transport for one to six ordered multimodal images."""

    provider = "openrouter"

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
        self._on_first_content: Callable[[float], None] | None = None

    def set_first_content_callback(
        self,
        callback: Callable[[float], None] | None,
    ) -> None:
        """Set an ephemeral UI callback for the first streamed content token."""

        if callback is not None and not callable(callback):
            raise TypeError("first content callback must be callable")
        self._on_first_content = callback

    def _validate_request(self, request: ApiRequest) -> None:
        if request.provider != self.provider:
            raise ApiContractError(
                "Request provider does not match OpenRouter transport"
            )
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError(
                "Request endpoint does not match OpenRouter transport"
            )
        if not 1 <= len(request.images) <= 6:
            raise ApiContractError("OpenRouter requires one to six images")
        input_text = request.payload.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ApiContractError("OpenRouter requires non-empty payload input_text")
        system_text = request.payload.get("system_text")
        if system_text is not None and (
            not isinstance(system_text, str) or not system_text.strip()
        ):
            raise ApiContractError("OpenRouter system_text must be non-empty text")
        routing_profile_from_request_payload(request.payload)

    @staticmethod
    def _image_content(image: Any) -> dict[str, object]:
        encoded = base64.b64encode(image.content).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
        }

    @staticmethod
    def _schema_name(version: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", version)

    def _request_body(self, request: ApiRequest) -> bytes:
        input_text = request.payload["input_text"]
        assert isinstance(input_text, str)
        system_text = request.payload.get("system_text")
        assert system_text is None or isinstance(system_text, str)
        permitted_parameters = {
            key: thaw_json(value)
            for key, value in request.generation_parameters.items()
            if key in {"temperature", "top_p", "reasoning"}
        }
        max_output_tokens = request.generation_parameters.get("max_output_tokens")
        if max_output_tokens is not None:
            permitted_parameters["max_tokens"] = thaw_json(max_output_tokens)
        body = {
            "model": request.model_identifier,
            "messages": [
                {
                    "role": "system",
                    "content": system_text
                    if system_text is not None
                    else (
                        "Return only the requested P3 transport-probe JSON. "
                        "Do not emit surgical predictions or P4 instance fields."
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_text}]
                    + [self._image_content(image) for image in request.images],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema_name(request.response_schema_version),
                    "strict": True,
                    "schema": schema_for(request.response_schema_version),
                },
            },
            "provider": provider_preferences(
                routing_profile_from_request_payload(request.payload)
            ),
            "stream": True,
            **permitted_parameters,
        }
        return json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def send(self, request: ApiRequest) -> ProviderResponse:
        """Send one OpenRouter request."""

        self._validate_request(request)
        body = self._request_body(request)
        headers = {
            "Authorization": f"Bearer {self.api_key.reveal()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Title": "CholecTrack20 P3 transport probe",
        }
        started = perf_counter()
        try:
            if self.sender is urllib_send_json:
                raw_response = urllib_send_json(
                    self.endpoint_identifier,
                    headers,
                    body,
                    self.timeout_seconds,
                    on_first_content=self._on_first_content,
                )
            else:
                raw_response = self.sender(
                    self.endpoint_identifier,
                    headers,
                    body,
                    self.timeout_seconds,
                )
        except TimeoutError:
            raise ApiTransportError(
                "OpenRouter request timed out",
                code="timeout",
                retryable=True,
            ) from None
        except (urllib.error.URLError, OSError):
            raise ApiTransportError(
                "OpenRouter connection failed",
                code="connection",
                retryable=True,
            ) from None
        finally:
            del body
        if not 200 <= raw_response.status_code <= 299:
            raise http_failure(raw_response.status_code, raw_response.body)
        total_latency_ms = (
            raw_response.total_latency_ms
            if raw_response.total_latency_ms is not None
            else (perf_counter() - started) * 1000.0
        )
        if (
            not isinstance(total_latency_ms, (int, float))
            or isinstance(total_latency_ms, bool)
            or not math.isfinite(float(total_latency_ms))
            or total_latency_ms < 0
        ):
            raise _response_error("response_envelope_invalid")
        try:
            if _is_sse_response(raw_response):
                decoded = _decode_sse_response(raw_response.body)
            else:
                decoded = json.loads(raw_response.body.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise TypeError("response must be an object")
            if decoded.get("object") != "chat.completion":
                raise ValueError("response is not a chat completion")
            response_id = decoded.get("id")
            returned_model = decoded.get("model")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("response id is invalid")
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError("response model is invalid")
            choices = decoded.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("response must contain one choice")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError("response choice is invalid")
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise _response_error("response_envelope_invalid") from None

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise _response_error("completion_length")
        if finish_reason != "stop":
            raise _response_error("completion_nonstop")

        try:
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("response message is invalid")
            output_text = message.get("content")
            if not isinstance(output_text, str) or not output_text:
                raise ValueError("response content is invalid")
            parsed_payload = json.loads(output_text)
            if not isinstance(parsed_payload, Mapping):
                raise TypeError("output JSON must be an object")
        except (TypeError, ValueError, json.JSONDecodeError):
            raise _response_error("response_content_invalid") from None

        try:
            usage = decoded.get("usage")
            if not isinstance(usage, Mapping):
                raise TypeError("usage must be an object")
            cost = usage.get("cost")
            if cost is not None and (
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or not math.isfinite(float(cost))
                or cost < 0
            ):
                raise ValueError("usage cost is invalid")
            input_tokens = _integer_usage(usage, "prompt_tokens")
            output_tokens = _integer_usage(usage, "completion_tokens")
            total_tokens = _integer_usage(usage, "total_tokens")
            details_value = usage.get("completion_tokens_details")
            if details_value is None:
                completion_tokens_details = CompletionTokenDetails()
                visible_output_tokens = None
            else:
                details = details_value
                if (
                    not isinstance(details, Mapping)
                    or "reasoning_tokens" not in details
                ):
                    raise ValueError("usage completion token details are invalid")
                completion_tokens_details = CompletionTokenDetails(
                    reasoning_tokens=details["reasoning_tokens"]
                )
                if completion_tokens_details.reasoning_tokens is None:
                    raise ValueError("usage reasoning tokens are invalid")
                visible_output_tokens = (
                    output_tokens - completion_tokens_details.reasoning_tokens
                )
                if visible_output_tokens < 0:
                    raise ValueError("usage visible output tokens are invalid")
        except (OverflowError, TypeError, ValueError):
            raise _response_error("response_usage_invalid") from None

        try:
            return ProviderResponse(
                provider=self.provider,
                returned_model_identifier=returned_model,
                parsed_payload=parsed_payload,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                completion_tokens_details=completion_tokens_details,
                visible_output_tokens=visible_output_tokens,
                time_to_first_token_ms=raw_response.time_to_first_token_ms,
                total_latency_ms=float(total_latency_ms),
                image_count=len(request.images),
                provider_request_id=response_id,
                provider_cost=None if cost is None else float(cost),
                exact_backend_model_identifier=None,
                exact_identity_evidence_source=None,
                safe_metadata={},
            )
        except (TypeError, ValueError):
            raise _response_error("response_content_invalid") from None
