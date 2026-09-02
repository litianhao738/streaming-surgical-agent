"""Generic OpenAI-compatible Chat Completions transport.

This adapter deliberately omits OpenRouter routing fields.  It is intended for
engineering runs against an explicitly supplied HTTPS endpoint, such as an
enterprise Qwen-compatible gateway.
"""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from surgical_agent.api.contracts import (
    ApiRequest,
    CompletionTokenDetails,
    ProviderResponse,
    thaw_json,
)
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.providers.openrouter import (
    HttpSender,
    urllib_send_json,
)
from surgical_agent.api.schema import schema_for


def _failure(status_code: int) -> ApiTransportError:
    if status_code in {401, 403}:
        code, retryable = "authentication", False
    elif status_code == 402:
        code, retryable = "payment_required", False
    elif status_code == 408:
        code, retryable = "timeout", True
    elif status_code == 429:
        code, retryable = "rate_limit", True
    elif 500 <= status_code <= 599:
        code, retryable = "provider_5xx", True
    elif 400 <= status_code <= 499:
        code, retryable = "provider_4xx", False
    else:
        code, retryable = "unexpected_http_status", False
    return ApiTransportError(
        "OpenAI-compatible HTTP request failed",
        code=code,
        retryable=retryable,
        status_code=status_code,
    )


def _response_error(code: str) -> ApiTransportError:
    return ApiTransportError(
        "OpenAI-compatible response failed a safe parsing stage",
        code=code,
        retryable=False,
    )


def _optional_count(mapping: Mapping[str, Any], name: str) -> int | None:
    value = mapping.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("usage count is invalid")
    return value


def _json_content(value: str) -> Mapping[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, Mapping):
        raise TypeError("structured output must be an object")
    return parsed


class OpenAICompatibleTransport:
    """Send one-to-three-image requests through a generic compatible gateway."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: SecretValue,
        endpoint_identifier: str,
        sender: HttpSender = urllib_send_json,
        timeout_seconds: float = 120.0,
        response_format: str = "json_schema",
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
        if response_format not in {"json_schema", "json_object", "none"}:
            raise ValueError(
                "response_format must be json_schema, json_object or none"
            )
        self.api_key = api_key
        self.endpoint_identifier = endpoint_identifier
        self.sender = sender
        self.timeout_seconds = float(timeout_seconds)
        self.response_format = response_format

    @staticmethod
    def _image_content(image: Any) -> dict[str, object]:
        encoded = base64.b64encode(image.content).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
        }

    def _validate_request(self, request: ApiRequest) -> None:
        if request.provider != self.provider:
            raise ApiContractError("request provider does not match transport")
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError("request endpoint does not match transport")
        if not 1 <= len(request.images) <= 3:
            raise ApiContractError("compatible transport requires one to three images")
        for field_name in ("input_text", "system_text"):
            value = request.payload.get(field_name)
            if field_name == "input_text" and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ApiContractError("compatible request requires input_text")
            if value is not None and not isinstance(value, str):
                raise ApiContractError(f"compatible {field_name} must be text")
        if set(request.payload) - {"input_text", "system_text", "image_details"}:
            raise ApiContractError("compatible request contains provider-specific fields")

    def _request_body(self, request: ApiRequest) -> bytes:
        input_text = request.payload["input_text"]
        system_text = request.payload.get("system_text")
        assert isinstance(input_text, str)
        assert system_text is None or isinstance(system_text, str)
        parameters: dict[str, object] = {}
        for key in ("temperature", "top_p", "seed", "enable_thinking"):
            if key in request.generation_parameters:
                parameters[key] = thaw_json(request.generation_parameters[key])
        if "max_output_tokens" in request.generation_parameters:
            parameters["max_tokens"] = thaw_json(
                request.generation_parameters["max_output_tokens"]
            )
        body: dict[str, object] = {
            "model": request.model_identifier,
            "messages": [
                {
                    "role": "system",
                    "content": system_text or "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_text}]
                    + [self._image_content(image) for image in request.images],
                },
            ],
            "stream": False,
            **parameters,
        }
        if self.response_format == "json_schema":
            schema_name = re.sub(
                r"[^A-Za-z0-9_]", "_", request.response_schema_version
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_for(request.response_schema_version),
                },
            }
        elif self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        return json.dumps(body, allow_nan=False, separators=(",", ":")).encode()

    def send(self, request: ApiRequest) -> ProviderResponse:
        """Send and normalize one non-streaming Chat Completions response."""

        self._validate_request(request)
        body = self._request_body(request)
        started = perf_counter()
        try:
            raw = self.sender(
                self.endpoint_identifier,
                {
                    "Authorization": f"Bearer {self.api_key.reveal()}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body,
                self.timeout_seconds,
            )
        except TimeoutError:
            raise ApiTransportError(
                "compatible request timed out",
                code="timeout",
                retryable=True,
            ) from None
        except (urllib.error.URLError, OSError):
            raise ApiTransportError(
                "compatible connection failed",
                code="connection",
                retryable=True,
            ) from None
        finally:
            del body
        if not 200 <= raw.status_code <= 299:
            raise _failure(raw.status_code)
        latency = (
            raw.total_latency_ms
            if raw.total_latency_ms is not None
            else (perf_counter() - started) * 1000.0
        )
        try:
            decoded = json.loads(raw.body.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise TypeError("response must be an object")
            response_id = decoded.get("id")
            returned_model = decoded.get("model")
            choices = decoded.get("choices")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("response id is invalid")
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError("response model is invalid")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("response choices are invalid")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError("response choice is invalid")
            if choice.get("finish_reason") == "length":
                raise _response_error("completion_length")
            if choice.get("finish_reason") not in {"stop", None}:
                raise _response_error("completion_nonstop")
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("response message is invalid")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("response content is invalid")
            parsed_payload = _json_content(content)
            usage_value = decoded.get("usage", {})
            if not isinstance(usage_value, Mapping):
                raise TypeError("usage is invalid")
            input_tokens = _optional_count(usage_value, "prompt_tokens")
            output_tokens = _optional_count(usage_value, "completion_tokens")
            total_tokens = _optional_count(usage_value, "total_tokens")
            details_value = usage_value.get("completion_tokens_details")
            reasoning_tokens = None
            if details_value is not None:
                if not isinstance(details_value, Mapping):
                    raise TypeError("completion token details are invalid")
                reasoning_tokens = _optional_count(details_value, "reasoning_tokens")
            details = CompletionTokenDetails(reasoning_tokens=reasoning_tokens)
            visible_tokens = (
                None
                if output_tokens is None
                else output_tokens - (reasoning_tokens or 0)
            )
        except ApiTransportError:
            raise
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise _response_error("response_content_invalid") from None
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier=returned_model,
            parsed_payload=parsed_payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=details,
            visible_output_tokens=visible_tokens,
            time_to_first_token_ms=raw.time_to_first_token_ms,
            total_latency_ms=float(latency),
            image_count=len(request.images),
            provider_request_id=response_id,
            provider_cost=None,
            exact_backend_model_identifier=None,
            exact_identity_evidence_source=None,
            safe_metadata={},
        )


__all__ = ["OpenAICompatibleTransport"]
