"""Provider-neutral P3 API contracts with no credential-bearing fields."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


def _freeze_json(value: Any, *, path: str) -> Any:
    """Copy JSON-compatible values into immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True)
class ApiImageInput:
    """In-memory image whose bytes are bound to the request by SHA-256."""

    identifier: str
    mime_type: str
    content: bytes = field(repr=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("image identifier must not be empty")
        if not self.mime_type.startswith("image/"):
            raise ValueError("mime_type must be an image media type")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("image content must be non-empty bytes")
        object.__setattr__(self, "sha256", hashlib.sha256(self.content).hexdigest())


@dataclass(frozen=True)
class ApiRequest:
    """Canonicalizable request description without authentication material."""

    provider: str
    model_identifier: str
    endpoint_identifier: str
    prompt_version: str
    response_schema_version: str
    payload: Mapping[str, Any]
    images: tuple[ApiImageInput, ...] = ()
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "model_identifier",
            "endpoint_identifier",
            "prompt_version",
            "response_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        images = tuple(self.images)
        if any(not isinstance(image, ApiImageInput) for image in images):
            raise TypeError("images must contain only ApiImageInput values")
        identifiers = [image.identifier for image in images]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("image identifiers must be unique within one request")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "payload", _freeze_json(self.payload, path="payload"))
        object.__setattr__(
            self,
            "generation_parameters",
            _freeze_json(self.generation_parameters, path="generation_parameters"),
        )


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized transport response before cache and usage accounting."""

    provider: str
    model_identifier: str | None
    raw_response: str
    parsed_payload: Mapping[str, Any] | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    image_count: int | None = None
    provider_request_id: str | None = None
    timestamp: str | None = None
    estimated_cost: float | None = None


@dataclass(frozen=True)
class ApiResponseRecord:
    """Auditable logical-call result returned by the cache-aware client."""

    provider: str
    requested_model_identifier: str
    model_identifier: str
    endpoint_identifier: str
    request_hash: str
    raw_response: str
    parsed_payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    image_count: int | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    timestamp: str | None = None
    cache_hit: bool = False
    provider_call: bool = True
    provider_request_id: str | None = None
    estimated_cost: float | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if len(self.request_hash) != 64:
            raise ValueError("request_hash must be a SHA-256 hex digest")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        for name in ("input_tokens", "output_tokens", "image_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when available")


class ProviderTransport(Protocol):
    """Provider-specific transport boundary; implementations obtain secrets externally."""

    provider: str
    endpoint_identifier: str

    def send(self, request: ApiRequest) -> ProviderResponse:
        """Send one provider call or raise a typed transport error."""


class ResponseValidator(Protocol):
    """P3 response-schema boundary, independent of P4 task semantics."""

    def __call__(self, payload: Mapping[str, Any]) -> None:
        """Raise a schema error when the structured payload is invalid."""


class MultimodalApiClient(Protocol):
    """Cache-aware provider-neutral client implemented during P3."""

    def call(self, request: ApiRequest) -> ApiResponseRecord:
        """Execute one logical call or return a validated cache replay."""


SleepFunction = Callable[[float], None]
