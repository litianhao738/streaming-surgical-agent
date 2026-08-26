"""OpenRouter Chat Completions transport."""

from __future__ import annotations

import base64
import json
import re
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
    """Map HTTP statuses without retaining the provider response body."""

    if status_code in {401, 403}:
        return ApiTransportError(
            "OpenRouter authentication failed",
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


class OpenRouterTransport:
    """OpenRouter transport for one to three ordered multimodal images."""

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

    def _validate_request(self, request: ApiRequest) -> None:
        if request.provider != self.provider:
            raise ApiContractError(
                "Request provider does not match OpenRouter transport"
            )
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError(
                "Request endpoint does not match OpenRouter transport"
            )
        if not 1 <= len(request.images) <= 3:
            raise ApiContractError(
                "OpenRouter requires one to three images"
            )
        input_text = request.payload.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ApiContractError(
                "OpenRouter requires non-empty payload input_text"
            )
        system_text = request.payload.get("system_text")
        if system_text is not None and (
            not isinstance(system_text, str) or not system_text.strip()
        ):
            raise ApiContractError("OpenRouter system_text must be non-empty text")

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
            "provider": {"require_parameters": True},
            "stream": False,
            **permitted_parameters,
        }
        return json.dumps(body, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )

    def send(self, request: ApiRequest) -> ProviderResponse:
        """Send one OpenRouter request."""

        self._validate_request(request)
        body = self._request_body(request)
        headers = {
            "Authorization": f"Bearer {self.api_key.reveal()}",
            "Content-Type": "application/json",
            "X-Title": "CholecTrack20 P3 transport probe",
        }
        try:
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
            raise http_failure(raw_response.status_code)
        try:
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
            if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
                raise ValueError("response choice is incomplete")
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("response message is invalid")
            output_text = message.get("content")
            if not isinstance(output_text, str) or not output_text:
                raise ValueError("response content is invalid")
            parsed_payload = json.loads(output_text)
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
                input_tokens=_integer_usage(usage, "prompt_tokens"),
                output_tokens=_integer_usage(usage, "completion_tokens"),
                total_tokens=_integer_usage(usage, "total_tokens"),
                image_count=len(request.images),
                provider_request_id=response_id,
                provider_cost=None if cost is None else float(cost),
                exact_backend_model_identifier=None,
                exact_identity_evidence_source=None,
                safe_metadata={},
            )
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise ApiTransportError(
                "OpenRouter response parsing failed",
                code="parse_failure",
                retryable=False,
            ) from None
