from pathlib import Path

import pytest

from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.schema import (
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
        "provider": "requesty",
        "endpoint_identifier": "https://router.requesty.ai/v1/responses",
        "requested_model_identifier": "openai-responses/gpt-5.6-sol",
        "prompt_version": "p3_transport_probe_v1",
        "response_schema_version": P3_SMOKE_SCHEMA_VERSION,
        "generation_parameters": {"max_output_tokens": 128},
        "provider_options": {"timeout_seconds": 60.0},
        "synthetic_input_required": True,
        "cache_required": True,
    }
    raw.update(changes)
    return raw


def test_requesty_config_is_effective_and_non_secret() -> None:
    config = load_api_config(Path("configs/api/requesty.yaml"))

    assert config.enabled is True
    assert config.mode == "real"
    assert config.provider == "requesty"
    assert config.endpoint_identifier == "https://router.requesty.ai/v1/responses"
    assert config.requested_model_identifier == "openai-responses/gpt-5.6-sol"
    assert not hasattr(config, "api_key")


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
                "provider": "requesty",
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
