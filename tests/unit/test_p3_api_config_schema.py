from pathlib import Path

import pytest

from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.schema import (
    P3_SMOKE_ALLOWED_KEYS,
    P3_SMOKE_SCHEMA_VERSION,
    schema_for,
    validate_p3_smoke_payload,
    validator_for,
)
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig


def _valid_api_mapping(**changes: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "enabled": True,
        "mode": "real",
        "provider": "openrouter",
        "endpoint_identifier": "https://openrouter.ai/api/v1/chat/completions",
        "requested_model_identifier": "openai/gpt-5.6-sol",
        "prompt_version": "p3_transport_probe_v1",
        "response_schema_version": P3_SMOKE_SCHEMA_VERSION,
        "generation_parameters": {"max_output_tokens": 128},
        "provider_options": {"timeout_seconds": 60.0},
        "synthetic_input_required": True,
        "cache_required": True,
    }
    raw.update(changes)
    return raw


def test_openrouter_config_is_effective_and_non_secret() -> None:
    config = load_api_config(Path("configs/api/openrouter.yaml"))

    assert config.enabled is True
    assert config.mode == "real"
    assert config.provider == "openrouter"
    assert config.endpoint_identifier == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert config.requested_model_identifier == "openai/gpt-5.6-sol"
    assert not hasattr(config, "api_key")


def test_registry_routes_openrouter_with_secret() -> None:
    config = ApiConfig.from_mapping(_valid_api_mapping())

    transport = build_transport(config, api_key=SecretValue("test-secret"))

    assert isinstance(transport, OpenRouterTransport)
    assert transport.provider == "openrouter"
    assert transport.endpoint_identifier == config.endpoint_identifier


def test_openrouter_config_rejects_unapproved_endpoint() -> None:
    with pytest.raises(ApiContractError, match="endpoint"):
        ApiConfig.from_mapping(
            _valid_api_mapping(
                endpoint_identifier="https://example.invalid/chat/completions"
            )
        )


def test_disabled_config_fails_before_transport_construction() -> None:
    config = ApiConfig.from_mapping(_valid_api_mapping(enabled=False))

    with pytest.raises(ApiContractError, match="disabled"):
        config.require_enabled()


def test_incomplete_disabled_config_fails_contract_validation() -> None:
    with pytest.raises(ApiContractError, match="missing fields"):
        ApiConfig.from_mapping(
            {
                "enabled": False,
                "mode": "real",
                "provider": "openrouter",
            }
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("enabled", "false"),
        ("mode", None),
        ("synthetic_input_required", "false"),
        ("cache_required", 0),
    ),
)
def test_config_rejects_coerced_scalar_values(
    field_name: str, replacement: object
) -> None:
    with pytest.raises(ApiContractError) as error:
        ApiConfig.from_mapping(_valid_api_mapping(**{field_name: replacement}))

    assert "false" not in str(error.value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("generation_parameters", ["not", "a", "mapping"]),
        ("provider_options", ["not", "a", "mapping"]),
        ("generation_parameters", {"nonfinite": float("nan")}),
        ("provider_options", {1: "non-json-key"}),
    ),
)
def test_config_rejects_non_json_option_containers(
    field_name: str, replacement: object
) -> None:
    with pytest.raises(ApiContractError):
        ApiConfig.from_mapping(_valid_api_mapping(**{field_name: replacement}))


def test_config_option_mappings_are_recursively_immutable_copies() -> None:
    generation_parameters = {"nested": {"items": [1, {"value": 2}]}}
    provider_options = {"nested": {"labels": ["one", "two"]}}
    config = ApiConfig.from_mapping(
        _valid_api_mapping(
            generation_parameters=generation_parameters,
            provider_options=provider_options,
        )
    )
    generation_parameters["nested"]["items"][1]["value"] = 9  # type: ignore[index]
    provider_options["nested"]["labels"].append("three")  # type: ignore[index]

    assert config.generation_parameters["nested"]["items"] == (1, {"value": 2})
    assert config.provider_options["nested"]["labels"] == ("one", "two")
    with pytest.raises(TypeError):
        config.generation_parameters["nested"]["items"][1]["value"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        config.provider_options["nested"]["labels"] += ("three",)  # type: ignore[index]


def test_unknown_schema_fails_closed() -> None:
    with pytest.raises(ApiContractError, match="schema"):
        validator_for("unknown")


@pytest.mark.parametrize("identifier", (None, 1, [P3_SMOKE_SCHEMA_VERSION]))
@pytest.mark.parametrize("lookup", (schema_for, validator_for))
def test_schema_registry_rejects_non_string_identifiers(
    lookup: object, identifier: object
) -> None:
    with pytest.raises(ApiContractError, match="schema"):
        lookup(identifier)  # type: ignore[operator]


def test_schema_registry_returns_a_defensive_schema_copy() -> None:
    first = schema_for(P3_SMOKE_SCHEMA_VERSION)
    first["properties"]["message"]["minLength"] = 999
    first["required"].append("mutated")

    second = schema_for(P3_SMOKE_SCHEMA_VERSION)
    assert second["properties"]["message"]["minLength"] == 1
    assert second["required"] == [
        "image_observed",
        "message",
        "schema_version",
        "structured",
    ]


def test_strict_schema_declares_type_for_every_property() -> None:
    schema = schema_for(P3_SMOKE_SCHEMA_VERSION)

    assert all("type" in definition for definition in schema["properties"].values())


def test_p3_validator_rejects_extra_and_p4_fields() -> None:
    payload = {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "ok",
        "image_observed": True,
        "structured": True,
        "instances": [],
    }

    with pytest.raises(ApiSchemaError, match="exact fields"):
        validate_p3_smoke_payload(payload)
    assert schema_for(P3_SMOKE_SCHEMA_VERSION)["additionalProperties"] is False


def test_allowed_key_boundary_cannot_be_widened_at_runtime() -> None:
    extra_key = "".join(("inst", "ances"))  # noqa: FLY002

    with pytest.raises(AttributeError):
        P3_SMOKE_ALLOWED_KEYS.add(extra_key)

    payload = {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "ok",
        "image_observed": True,
        "structured": True,
    }
    payload[extra_key] = []
    with pytest.raises(ApiSchemaError, match="exact fields"):
        validate_p3_smoke_payload(payload)


@pytest.mark.parametrize(
    "payload",
    (
        None,
        [
            P3_SMOKE_SCHEMA_VERSION,
            "ok",
            True,
            True,
        ],
        {
            "schema_version": None,
            "message": "ok",
            "image_observed": True,
            "structured": True,
        },
        {
            "schema_version": P3_SMOKE_SCHEMA_VERSION,
            "message": 1,
            "image_observed": True,
            "structured": True,
        },
        {
            "schema_version": P3_SMOKE_SCHEMA_VERSION,
            "message": "ok",
            "image_observed": 1,
            "structured": True,
        },
        {
            "schema_version": P3_SMOKE_SCHEMA_VERSION,
            "message": "ok",
            "image_observed": True,
            "structured": "true",
        },
    ),
)
def test_p3_validator_raises_schema_error_for_wrong_input_types(payload: object) -> None:
    with pytest.raises(ApiSchemaError):
        validate_p3_smoke_payload(payload)  # type: ignore[arg-type]


def test_registry_routes_effective_mock_config_and_validator() -> None:
    config = ApiConfig.from_mapping(
        _valid_api_mapping(
            mode="mock",
            provider="mock",
            endpoint_identifier="mock://local/p3",
            provider_options={
                "returned_model_identifier": "configured-returned-v1",
                "retryable_failures_before_success": 2,
                "malformed_payload": False,
                "provider_cost": 0.25,
            },
        )
    )

    transport = build_transport(config, api_key=None)

    assert isinstance(transport, MockProviderTransport)
    assert transport.returned_model_identifier == "configured-returned-v1"
    assert transport.retryable_failures_before_success == 2
    assert transport.provider_cost == 0.25
    assert build_validator(config) is validate_p3_smoke_payload


def test_registry_routes_openrouter_with_configured_timeout() -> None:
    config = load_api_config(Path("configs/api/openrouter.yaml"))

    transport = build_transport(
        config,
        api_key=SecretValue("test-secret"),
    )

    assert isinstance(transport, OpenRouterTransport)
    assert transport.provider == config.provider
    assert transport.endpoint_identifier == config.endpoint_identifier
    assert transport.timeout_seconds == 60.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"retryable_failures_before_success": True},
        {"retryable_failures_before_success": "2"},
        {"provider_cost": "0.25"},
        {"provider_cost": True},
        {"malformed_payload": "false"},
    ],
)
def test_mock_config_rejects_coerced_override_values(
    overrides: dict[str, object],
) -> None:
    config = ApiConfig.from_mapping(
        _valid_api_mapping(
            mode="mock",
            provider="mock",
            endpoint_identifier="mock://local/p3",
        )
    )

    with pytest.raises((ApiContractError, TypeError, ValueError)):
        build_transport(config, api_key=None, mock_options=overrides)


@pytest.mark.parametrize("timeout", [True, "60", 0, -1, float("nan")])
def test_openrouter_config_rejects_coerced_or_invalid_timeout(
    timeout: object,
) -> None:
    with pytest.raises((ApiContractError, TypeError, ValueError)):
        config = ApiConfig.from_mapping(
            _valid_api_mapping(provider_options={"timeout_seconds": timeout})
        )
        build_transport(config, api_key=SecretValue("test-secret"))


def test_registry_rejects_mock_credential_and_openrouter_missing_credential() -> None:
    mock = ApiConfig.from_mapping(
        _valid_api_mapping(
            mode="mock",
            provider="mock",
            endpoint_identifier="mock://local/p3",
        )
    )
    openrouter = load_api_config(Path("configs/api/openrouter.yaml"))

    with pytest.raises(ApiContractError, match="credential"):
        build_transport(mock, api_key=SecretValue("not-for-mock"))
    with pytest.raises(ApiContractError, match="credential"):
        build_transport(openrouter, api_key=None)


def test_registry_rejects_transport_endpoint_provider_mismatch() -> None:
    mock = ApiConfig.from_mapping(
        _valid_api_mapping(
            mode="mock",
            provider="mock",
            endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        )
    )

    with pytest.raises(ApiContractError, match="endpoint"):
        build_transport(mock, api_key=None)
