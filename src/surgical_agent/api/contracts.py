"""Provider-neutral P3 API contracts with no credential-bearing fields."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
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


def thaw_json(value: Any) -> Any:
    """Return mutable JSON containers without exposing stored internal state."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and equality evidence."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_nonempty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_optional_nonempty_string(value: object, *, name: str) -> None:
    if value is not None:
        _require_nonempty_string(value, name=name)


def _require_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_optional_count(value: object, *, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer when available")


def _require_count(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_optional_number(value: object, *, name: str) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number when available")


def _require_identity_evidence_pair(
    backend: object,
    source: object,
) -> None:
    if (backend is None) != (source is None):
        raise ValueError(
            "exact backend identity and evidence source must appear together"
        )


def _require_optional_aware_iso_timestamp(value: object, *, name: str) -> None:
    if value is None:
        return
    _require_nonempty_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")


_GENERATION_PARAMETER_KEYS = frozenset(
    {
        "deterministic_mock",
        "max_output_tokens",
        "reasoning",
        "seed",
        "temperature",
        "top_p",
    }
)
_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)


def freeze_generation_parameters(value: object) -> Mapping[str, Any]:
    """Validate and freeze the explicit P3 generation-option allowlist."""

    mapping = _require_mapping(value, name="generation_parameters")
    unknown = set(mapping) - _GENERATION_PARAMETER_KEYS
    if unknown:
        raise ValueError("generation_parameters contains unknown fields")
    result: dict[str, Any] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError("generation_parameters keys must be strings")
        if key in {"temperature", "top_p"}:
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
            ):
                raise TypeError(f"generation_parameters.{key} must be a finite number")
            number = float(item)
            if key == "temperature" and not 0.0 <= number <= 2.0:
                raise ValueError("generation_parameters.temperature is out of range")
            if key == "top_p" and not 0.0 < number <= 1.0:
                raise ValueError("generation_parameters.top_p is out of range")
            result[key] = item
        elif key == "max_output_tokens":
            if (
                not isinstance(item, int)
                or isinstance(item, bool)
                or not 1 <= item <= 1_000_000
            ):
                raise ValueError(
                    "generation_parameters.max_output_tokens must be a positive integer"
                )
            result[key] = item
        elif key == "seed":
            if (
                not isinstance(item, int)
                or isinstance(item, bool)
                or not -(2**31) <= item < 2**31
            ):
                raise ValueError(
                    "generation_parameters.seed must be a signed 32-bit integer"
                )
            result[key] = item
        elif key == "deterministic_mock":
            if type(item) is not bool:
                raise TypeError(
                    "generation_parameters.deterministic_mock must be boolean"
                )
            result[key] = item
        elif key == "reasoning":
            reasoning = _require_mapping(item, name="generation_parameters.reasoning")
            if set(reasoning) != {"effort"}:
                raise ValueError("generation_parameters.reasoning has invalid fields")
            effort = reasoning["effort"]
            if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
                raise ValueError("generation_parameters.reasoning.effort is invalid")
            result[key] = MappingProxyType({"effort": effort})
    return MappingProxyType(result)


_SAFE_METADATA_KEYS = frozenset(
    {
        "finish_reason",
    }
)
_FINISH_REASONS = frozenset({"content_filter", "length", "stop", "tool_calls"})


def freeze_safe_metadata(
    value: object,
    *,
    path: str = "safe_metadata",
) -> Mapping[str, Any]:
    """Validate and freeze normalized provider metadata with no free-form values."""

    mapping = _require_mapping(value, name=path)
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} keys must be strings")
    unknown = set(mapping) - _SAFE_METADATA_KEYS
    if unknown:
        raise ValueError(f"{path} contains unsafe key or unknown field")
    result: dict[str, Any] = {}
    for key, item in mapping.items():
        if key == "finish_reason" and (
            not isinstance(item, str) or item not in _FINISH_REASONS
        ):
            raise ValueError(f"{path}.finish_reason is invalid")
        result[key] = item
    return MappingProxyType(result)


@dataclass(frozen=True)
class ApiImageInput:
    """In-memory image whose bytes are bound to the request by SHA-256."""

    identifier: str
    mime_type: str
    content: bytes = field(repr=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.identifier, name="image identifier")
        if not isinstance(self.mime_type, str) or not self.mime_type.startswith(
            "image/"
        ):
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
            _require_nonempty_string(getattr(self, name), name=name)
        images = tuple(self.images)
        if any(not isinstance(image, ApiImageInput) for image in images):
            raise TypeError("images must contain only ApiImageInput values")
        identifiers = [image.identifier for image in images]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("image identifiers must be unique within one request")
        object.__setattr__(self, "images", images)
        payload = _require_mapping(self.payload, name="payload")
        object.__setattr__(self, "payload", _freeze_json(payload, path="payload"))
        object.__setattr__(
            self,
            "generation_parameters",
            freeze_generation_parameters(self.generation_parameters),
        )


@dataclass(frozen=True)
class ImageProvenance:
    """Allowlisted image identity evidence without image bytes."""

    identifier: str
    mime_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.identifier, name="image identifier")
        _require_nonempty_string(self.mime_type, name="image mime_type")
        _require_count(self.size_bytes, name="image size_bytes")
        _require_sha256(self.sha256, name="image sha256")


@dataclass(frozen=True)
class CanonicalRequestMetadata:
    """Safe, exact provenance bound to a canonical request digest."""

    schema_version: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    prompt_version: str
    response_schema_version: str
    generation_parameters: Mapping[str, Any]
    payload_sha256: str
    images: tuple[ImageProvenance, ...]
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "api_request_metadata_v1":
            raise ValueError(
                "canonical request metadata has unsupported schema_version"
            )
        for name in (
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
        ):
            _require_nonempty_string(getattr(self, name), name=name)
        _require_sha256(self.payload_sha256, name="payload_sha256")
        _require_sha256(self.request_hash, name="request_hash")
        images = tuple(self.images)
        if any(not isinstance(image, ImageProvenance) for image in images):
            raise TypeError("canonical request images contain invalid provenance")
        object.__setattr__(self, "images", images)
        object.__setattr__(
            self,
            "generation_parameters",
            freeze_generation_parameters(self.generation_parameters),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "endpoint_identifier": self.endpoint_identifier,
            "requested_model_identifier": self.requested_model_identifier,
            "prompt_version": self.prompt_version,
            "response_schema_version": self.response_schema_version,
            "generation_parameters": thaw_json(self.generation_parameters),
            "payload_sha256": self.payload_sha256,
            "images": [asdict(image) for image in self.images],
            "request_hash": self.request_hash,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> CanonicalRequestMetadata:
        expected = {
            "schema_version",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
            "generation_parameters",
            "payload_sha256",
            "images",
            "request_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("canonical request metadata has invalid fields")
        images = value["images"]
        if not isinstance(images, list):
            raise TypeError("canonical request images must be a list")
        image_fields = {"identifier", "mime_type", "sha256", "size_bytes"}
        provenance: list[ImageProvenance] = []
        for item in images:
            if not isinstance(item, Mapping):
                raise TypeError("canonical request image must be a mapping")
            if set(item) != image_fields:
                raise ValueError("canonical request image has invalid fields")
            provenance.append(
                ImageProvenance(**{name: item[name] for name in image_fields})
            )
        return cls(
            schema_version=value["schema_version"],
            provider=value["provider"],
            endpoint_identifier=value["endpoint_identifier"],
            requested_model_identifier=value["requested_model_identifier"],
            prompt_version=value["prompt_version"],
            response_schema_version=value["response_schema_version"],
            generation_parameters=freeze_generation_parameters(
                value["generation_parameters"]
            ),
            payload_sha256=value["payload_sha256"],
            images=tuple(provenance),
            request_hash=value["request_hash"],
        )


@dataclass(frozen=True)
class CompletionTokenDetails:
    """The one allowlisted completion-token detail persisted for auditability."""

    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_optional_count(self.reasoning_tokens, name="reasoning_tokens")

    def to_mapping(self) -> dict[str, int | None]:
        return {"reasoning_tokens": self.reasoning_tokens}

    @classmethod
    def from_mapping(cls, value: object) -> CompletionTokenDetails:
        mapping = _require_mapping(value, name="completion_tokens_details")
        if set(mapping) != {"reasoning_tokens"}:
            raise ValueError("completion_tokens_details has invalid fields")
        return cls(reasoning_tokens=mapping["reasoning_tokens"])


@dataclass(frozen=True)
class ProviderResponse:
    """Allowlisted normalized transport response before persistence."""

    provider: str
    returned_model_identifier: str
    parsed_payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: CompletionTokenDetails = field(
        default_factory=CompletionTokenDetails
    )
    visible_output_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    total_latency_ms: float | None = None
    image_count: int | None = None
    provider_request_id: str | None = None
    timestamp: str | None = None
    provider_cost: float | None = None
    exact_backend_model_identifier: str | None = None
    exact_identity_evidence_source: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        _require_nonempty_string(
            self.returned_model_identifier, name="returned_model_identifier"
        )
        for name in (
            "provider_request_id",
            "exact_backend_model_identifier",
            "exact_identity_evidence_source",
        ):
            _require_optional_nonempty_string(getattr(self, name), name=name)
        _require_optional_aware_iso_timestamp(self.timestamp, name="timestamp")
        _require_identity_evidence_pair(
            self.exact_backend_model_identifier,
            self.exact_identity_evidence_source,
        )
        for name in ("input_tokens", "output_tokens", "total_tokens", "image_count"):
            _require_optional_count(getattr(self, name), name=name)
        if not isinstance(self.completion_tokens_details, CompletionTokenDetails):
            raise TypeError("completion_tokens_details must be CompletionTokenDetails")
        _require_optional_count(
            self.visible_output_tokens, name="visible_output_tokens"
        )
        _require_optional_number(self.time_to_first_token_ms, name="time_to_first_token_ms")
        _require_optional_number(self.total_latency_ms, name="total_latency_ms")
        if self.completion_tokens_details.reasoning_tokens is not None:
            if self.output_tokens is None:
                raise ValueError("reasoning tokens require completion tokens")
            visible = self.output_tokens - self.completion_tokens_details.reasoning_tokens
            if visible < 0:
                raise ValueError("reasoning tokens cannot exceed completion tokens")
            if self.visible_output_tokens is not None and self.visible_output_tokens != visible:
                raise ValueError("visible output tokens must match completion usage")
            object.__setattr__(self, "visible_output_tokens", visible)
        _require_optional_number(self.provider_cost, name="provider_cost")
        parsed_payload = _require_mapping(self.parsed_payload, name="parsed_payload")
        safe_metadata = freeze_safe_metadata(self.safe_metadata)
        object.__setattr__(
            self,
            "parsed_payload",
            _freeze_json(parsed_payload, path="parsed_payload"),
        )
        object.__setattr__(self, "safe_metadata", safe_metadata)

    @property
    def prompt_tokens(self) -> int | None:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int | None:
        return self.output_tokens

    @property
    def latency_ms(self) -> float | None:
        """Compatibility alias for the provider-attempt total latency."""

        return self.total_latency_ms


@dataclass(frozen=True)
class ApiResponseRecord:
    """Allowlisted logical-call result returned by the cache-aware client."""

    provider: str
    endpoint_identifier: str
    request_hash: str
    requested_model_identifier: str
    returned_model_identifier: str
    parsed_payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: CompletionTokenDetails = field(
        default_factory=CompletionTokenDetails
    )
    visible_output_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    image_count: int | None = None
    latency_ms: float | None = None
    total_latency_ms: float | None = None
    retry_count: int = 0
    provider_call_count: int = 1
    timestamp: str | None = None
    cache_hit: bool = False
    provider_request_id: str | None = None
    provider_cost: float | None = None
    origin_provider_cost: float | None = None
    exact_backend_model_identifier: str | None = None
    exact_identity_evidence_source: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "returned_model_identifier",
        ):
            _require_nonempty_string(getattr(self, name), name=name)
        _require_sha256(self.request_hash, name="request_hash")
        for name in (
            "provider_request_id",
            "exact_backend_model_identifier",
            "exact_identity_evidence_source",
        ):
            _require_optional_nonempty_string(getattr(self, name), name=name)
        _require_optional_aware_iso_timestamp(self.timestamp, name="timestamp")
        _require_identity_evidence_pair(
            self.exact_backend_model_identifier,
            self.exact_identity_evidence_source,
        )
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "image_count",
        ):
            _require_optional_count(getattr(self, name), name=name)
        if not isinstance(self.completion_tokens_details, CompletionTokenDetails):
            raise TypeError("completion_tokens_details must be CompletionTokenDetails")
        _require_optional_count(
            self.visible_output_tokens, name="visible_output_tokens"
        )
        _require_optional_number(self.time_to_first_token_ms, name="time_to_first_token_ms")
        if self.completion_tokens_details.reasoning_tokens is not None:
            if self.output_tokens is None:
                raise ValueError("reasoning tokens require completion tokens")
            visible = self.output_tokens - self.completion_tokens_details.reasoning_tokens
            if visible < 0:
                raise ValueError("reasoning tokens cannot exceed completion tokens")
            if self.visible_output_tokens is not None and self.visible_output_tokens != visible:
                raise ValueError("visible output tokens must match completion usage")
            object.__setattr__(self, "visible_output_tokens", visible)
        for name in ("retry_count", "provider_call_count"):
            _require_count(getattr(self, name), name=name)
        if self.provider_call_count == 0:
            if self.retry_count != 0:
                raise ValueError(
                    "response retry count is incoherent with provider calls"
                )
        elif self.retry_count != self.provider_call_count - 1:
            raise ValueError("response retry count is incoherent with provider calls")
        for name in (
            "latency_ms",
            "total_latency_ms",
            "provider_cost",
            "origin_provider_cost",
        ):
            _require_optional_number(getattr(self, name), name=name)
        if self.total_latency_ms is None:
            object.__setattr__(self, "total_latency_ms", self.latency_ms)
        elif self.latency_ms is None:
            object.__setattr__(self, "latency_ms", self.total_latency_ms)
        elif self.total_latency_ms != self.latency_ms:
            raise ValueError("total latency must match legacy latency")
        if type(self.cache_hit) is not bool:
            raise TypeError("cache_hit must be a boolean")
        if self.cache_hit and self.provider_call_count != 0:
            raise ValueError("cache-hit response cannot contain current provider calls")
        parsed_payload = _require_mapping(self.parsed_payload, name="parsed_payload")
        safe_metadata = freeze_safe_metadata(self.safe_metadata)
        object.__setattr__(
            self,
            "parsed_payload",
            _freeze_json(parsed_payload, path="parsed_payload"),
        )
        object.__setattr__(self, "safe_metadata", safe_metadata)

    @property
    def model_identifier(self) -> str:
        """Compatibility alias for the returned provider model identifier."""

        return self.returned_model_identifier

    @property
    def prompt_tokens(self) -> int | None:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int | None:
        return self.output_tokens

    @property
    def provider_call(self) -> bool:
        """Compatibility view derived from the auditable provider call count."""

        return self.provider_call_count > 0

    def to_persisted_mapping(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_identifier": self.endpoint_identifier,
            "request_hash": self.request_hash,
            "requested_model_identifier": self.requested_model_identifier,
            "returned_model_identifier": self.returned_model_identifier,
            "parsed_payload": thaw_json(self.parsed_payload),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_details": self.completion_tokens_details.to_mapping(),
            "visible_output_tokens": self.visible_output_tokens,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "image_count": self.image_count,
            "latency_ms": self.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "retry_count": self.retry_count,
            "provider_call_count": self.provider_call_count,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "provider_request_id": self.provider_request_id,
            "provider_cost": self.provider_cost,
            "origin_provider_cost": self.origin_provider_cost,
            "exact_backend_model_identifier": self.exact_backend_model_identifier,
            "exact_identity_evidence_source": self.exact_identity_evidence_source,
            "safe_metadata": thaw_json(self.safe_metadata),
        }

    @classmethod
    def from_persisted_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> ApiResponseRecord:
        expected = {
            "provider",
            "endpoint_identifier",
            "request_hash",
            "requested_model_identifier",
            "returned_model_identifier",
            "parsed_payload",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "completion_tokens_details",
            "visible_output_tokens",
            "time_to_first_token_ms",
            "image_count",
            "latency_ms",
            "total_latency_ms",
            "retry_count",
            "provider_call_count",
            "timestamp",
            "cache_hit",
            "provider_request_id",
            "provider_cost",
            "origin_provider_cost",
            "exact_backend_model_identifier",
            "exact_identity_evidence_source",
            "safe_metadata",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("persisted API response has invalid fields")
        if value["prompt_tokens"] != value["input_tokens"]:
            raise ValueError("persisted prompt token aliases disagree")
        if value["completion_tokens"] != value["output_tokens"]:
            raise ValueError("persisted completion token aliases disagree")
        values = {name: value[name] for name in sorted(expected)}
        del values["prompt_tokens"]
        del values["completion_tokens"]
        values["completion_tokens_details"] = CompletionTokenDetails.from_mapping(
            value["completion_tokens_details"]
        )
        return cls(**values)


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
