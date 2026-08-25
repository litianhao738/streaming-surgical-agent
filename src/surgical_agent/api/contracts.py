"""Provider-neutral API boundaries without transport implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ApiRequest:
    """Serializable multimodal request description.

    Authentication is intentionally absent and must be supplied by a provider
    adapter from an environment or external secret mechanism.
    """

    provider: str
    model_identifier: str
    prompt_version: str
    payload: Mapping[str, Any]
    image_refs: tuple[str, ...] = ()
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApiResponseRecord:
    """Auditable result returned by a future provider adapter."""

    provider: str
    model_identifier: str
    request_hash: str
    raw_response: str
    parsed_payload: Mapping[str, Any] | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    image_count: int | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    timestamp: str | None = None
    cache_hit: bool = False
    estimated_cost: float | None = None


class MultimodalApiClient(Protocol):
    """Transport boundary implemented in P3, not during P0."""

    def call(self, request: ApiRequest) -> ApiResponseRecord:
        """Execute one request or return a cache replay record."""
