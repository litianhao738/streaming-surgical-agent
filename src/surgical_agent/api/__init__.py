"""Provider-neutral P3 API client, cache, retry, and audit contracts."""

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    ApiResponseRecord,
    MultimodalApiClient,
    ProviderResponse,
    ProviderTransport,
)
from surgical_agent.api.request_hash import request_sha256
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger


def __getattr__(name: str):
    """Load clients after leaf contracts, avoiding the schema import cycle."""
    if name == "CachedMultimodalApiClient":
        from surgical_agent.api.client import CachedMultimodalApiClient

        return CachedMultimodalApiClient
    if name in {"build_transport", "build_validator", "determine_p3_status"}:
        from surgical_agent.api import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ApiImageInput",
    "ApiRequest",
    "ApiResponseRecord",
    "CachedMultimodalApiClient",
    "FileApiCache",
    "MultimodalApiClient",
    "ProviderResponse",
    "ProviderTransport",
    "RetryPolicy",
    "UsageLedger",
    "build_transport",
    "build_validator",
    "determine_p3_status",
    "request_sha256",
]
