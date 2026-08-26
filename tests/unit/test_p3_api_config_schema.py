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


def test_requesty_config_is_effective_and_non_secret() -> None:
    config = load_api_config(Path("configs/api/requesty.yaml"))

    assert config.enabled is True
    assert config.mode == "real"
    assert config.provider == "requesty"
    assert config.endpoint_identifier == "https://router.requesty.ai/v1/responses"
    assert config.requested_model_identifier == "openai-responses/gpt-5.6-sol"
    assert not hasattr(config, "api_key")


def test_disabled_config_fails_before_transport_construction() -> None:
    with pytest.raises(ApiContractError, match="disabled"):
        ApiConfig.from_mapping(
            {
                "enabled": False,
                "mode": "real",
                "provider": "requesty",
            }
        ).require_enabled()


def test_unknown_schema_fails_closed() -> None:
    with pytest.raises(ApiContractError, match="schema"):
        validator_for("unknown")


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
