"""P3-only structured smoke schema; P4 surgical semantics remain deferred."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable

from surgical_agent.api.errors import ApiContractError, ApiSchemaError

P3_SMOKE_SCHEMA_VERSION = "p3_multimodal_smoke_v1"
P3_SMOKE_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "message",
        "image_observed",
        "structured",
    }
)
_P3_SMOKE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": P3_SMOKE_SCHEMA_VERSION},
        "message": {"type": "string", "minLength": 1},
        "image_observed": {"type": "boolean"},
        "structured": {"const": True},
    },
    "required": sorted(P3_SMOKE_ALLOWED_KEYS),
}


def _freeze_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_schema(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_schema(item) for item in value)
    return value


def _copy_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _copy_schema(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_schema(item) for item in value]
    return value


_FROZEN_P3_SMOKE_JSON_SCHEMA = _freeze_schema(_P3_SMOKE_JSON_SCHEMA)
P3_SMOKE_JSON_SCHEMA = _copy_schema(_FROZEN_P3_SMOKE_JSON_SCHEMA)


def validate_p3_smoke_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ApiSchemaError("P3 smoke response must be a mapping")
    if payload.get("schema_version") != P3_SMOKE_SCHEMA_VERSION:
        raise ApiSchemaError("P3 smoke response has an unsupported schema_version")
    if set(payload) != P3_SMOKE_ALLOWED_KEYS:
        raise ApiSchemaError("P3 smoke response must contain exact fields")
    if not isinstance(payload["message"], str) or not payload["message"].strip():
        raise ApiSchemaError("P3 smoke response requires a non-empty message")
    if type(payload["image_observed"]) is not bool:
        raise ApiSchemaError("P3 smoke response requires boolean image_observed")
    if payload["structured"] is not True:
        raise ApiSchemaError("P3 smoke response must declare structured=true")


SCHEMAS: Mapping[
    str, tuple[Mapping[str, Any], Callable[[Mapping[str, Any]], None]]
] = MappingProxyType(
    {
        P3_SMOKE_SCHEMA_VERSION: (
            _FROZEN_P3_SMOKE_JSON_SCHEMA,
            validate_p3_smoke_payload,
        )
    }
)


def schema_for(version: str) -> dict[str, Any]:
    """Return the one registered response schema or fail closed."""

    if not isinstance(version, str):
        raise ApiContractError("Unsupported API response schema identifier")
    try:
        schema = SCHEMAS[version][0]
    except KeyError as exc:
        raise ApiContractError(f"Unsupported API response schema: {version}") from exc
    copied = _copy_schema(schema)
    assert isinstance(copied, dict)
    return copied


def validator_for(version: str) -> Callable[[Mapping[str, Any]], None]:
    """Return the one registered response validator or fail closed."""

    if not isinstance(version, str):
        raise ApiContractError("Unsupported API response schema identifier")
    try:
        return SCHEMAS[version][1]
    except KeyError as exc:
        raise ApiContractError(f"Unsupported API response schema: {version}") from exc
