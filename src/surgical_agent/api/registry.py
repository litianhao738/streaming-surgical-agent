"""Config-driven transport registry with no implicit real-provider fallback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from surgical_agent.api.contracts import ProviderTransport
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.providers.mock import MockProviderTransport


def build_transport(config: Mapping[str, Any]) -> ProviderTransport:
    """Build only explicitly approved transports from sanitized config."""

    provider = config.get("provider")
    endpoint = config.get("endpoint_identifier")
    if provider == "mock":
        if endpoint != MockProviderTransport.endpoint_identifier:
            raise ApiContractError("Mock endpoint_identifier does not match registry")
        return MockProviderTransport(
            returned_model_identifier=str(
                config.get("returned_model_identifier", "mock-model-returned-v1")
            ),
            retryable_failures_before_success=int(
                config.get("retryable_failures_before_success", 0)
            ),
        )
    if provider is None:
        raise ApiContractError("REAL_API_SMOKE_BLOCKED: provider is not configured")
    raise ApiContractError(
        f"REAL_API_SMOKE_BLOCKED: no approved adapter for provider {provider!r}"
    )
