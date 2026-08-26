"""Config-driven transport and response-validator registry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    ProviderTransport,
    ResponseValidator,
)
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.api.schema import validator_for
from surgical_agent.config.schema import ApiConfig

_MOCK_OPTION_KEYS = frozenset(
    {
        "returned_model_identifier",
        "retryable_failures_before_success",
        "malformed_payload",
        "provider_cost",
    }
)


def _as_effective_config(
    config: ApiConfig | Mapping[str, Any],
) -> tuple[ApiConfig, dict[str, object]]:
    """Accept typed configs and retain compatibility with the existing smoke script."""

    if isinstance(config, ApiConfig):
        return config, {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be ApiConfig or a mapping")
    raw = dict(config)
    effective = ApiConfig.from_mapping(raw)
    # The legacy mock YAML keeps these four values at top level. They are
    # compatibility overrides while typed callers use provider_options.
    legacy = {key: raw[key] for key in _MOCK_OPTION_KEYS if key in raw}
    return effective, legacy


def _require_timeout(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ApiContractError(
            "API timeout_seconds must be a positive finite number"
        )
    return float(value)


def _require_transport_identity(
    config: ApiConfig,
    transport: ProviderTransport,
) -> ProviderTransport:
    if transport.provider != config.provider:
        raise ApiContractError("transport provider does not match config provider")
    if transport.endpoint_identifier != config.endpoint_identifier:
        raise ApiContractError("transport endpoint does not match config endpoint")
    return transport


def build_transport(
    config: ApiConfig | Mapping[str, Any],
    *,
    api_key: SecretValue | None = None,
    mock_options: Mapping[str, object] | None = None,
) -> ProviderTransport:
    """Build the provider transport selected by the effective API config."""

    effective, legacy_options = _as_effective_config(config)
    effective.require_enabled()
    effective.validate()
    if mock_options is not None and not isinstance(mock_options, Mapping):
        raise TypeError("mock_options must be a mapping")

    if effective.provider == "mock":
        if api_key is not None:
            raise ApiContractError("mock provider must not receive a credential")
        options = {**legacy_options, **dict(mock_options or {})}
        return _require_transport_identity(
            effective,
            MockProviderTransport.from_config(effective, options),
        )

    if effective.provider == "openrouter":
        if api_key is None:
            raise ApiContractError("OpenRouter provider requires a credential")
        timeout = _require_timeout(
            effective.provider_options.get("timeout_seconds", 60.0)
        )
        return _require_transport_identity(
            effective,
            OpenRouterTransport(
                api_key=api_key,
                endpoint_identifier=effective.endpoint_identifier,
                timeout_seconds=timeout,
            ),
        )

    raise ApiContractError(f"unknown provider: {effective.provider}")


def build_validator(config: ApiConfig | Mapping[str, Any]) -> ResponseValidator:
    """Build the validator registered for the effective response schema."""

    effective, _ = _as_effective_config(config)
    effective.require_enabled()
    effective.validate()
    return validator_for(effective.response_schema_version)


def determine_p3_status(
    record: ApiResponseRecord,
    *,
    transport_gates_passed: bool,
) -> str:
    """Return PASS only with explicit exact-backend identity evidence."""

    if type(transport_gates_passed) is not bool:
        raise ApiContractError("transport_gates_passed must be a boolean")
    if not isinstance(record, ApiResponseRecord):
        raise TypeError("record must be ApiResponseRecord")
    if (
        transport_gates_passed
        and record.exact_backend_model_identifier
        and record.exact_identity_evidence_source
    ):
        return "PASS"
    return "PARTIAL"
