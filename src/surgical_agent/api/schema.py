"""Registered structured response schemas with strict validators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.expert_ablation import EXPERT_CONTRACTS
from surgical_agent.research.verification.grounded_repair import GROUNDED_CONTRACTS
from surgical_agent.perception.final_only import (
    FINAL_ONLY_SCHEMA_VERSION, final_only_schema, validate_final_only,
)
from surgical_agent.perception.contracts import (
    FIELD_UNCERTAINTY_PATHS,
    FIELD_UNCERTAINTY_REASONS,
)
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    joint_perception_schema,
    validate_compact_joint_perception_payload,
    validate_gate_owned_compact_joint_perception_payload,
    validate_joint_perception_payload,
    validate_reliability_compact_joint_perception_payload,
)

P3_SMOKE_SCHEMA_VERSION = "p3_multimodal_smoke_v1"
TARGETED_VERIFICATION_SCHEMA_VERSION = "targeted_verification_v1"
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
        "schema_version": {
            "type": "string",
            "const": P3_SMOKE_SCHEMA_VERSION,
        },
        "message": {"type": "string", "minLength": 1},
        "image_observed": {"type": "boolean"},
        "structured": {"type": "boolean", "const": True},
    },
    "required": sorted(P3_SMOKE_ALLOWED_KEYS),
}
_TARGETED_PATHS = tuple(FIELD_UNCERTAINTY_PATHS)
_TARGETED_VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {
            "type": "string",
            "const": TARGETED_VERIFICATION_SCHEMA_VERSION,
        },
        "fields": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "enum": list(_TARGETED_PATHS)},
                    "selected_ids": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "topk": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer", "minimum": 0},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["id", "confidence"],
                        },
                    },
                    "status": {
                        "type": "string",
                        "enum": ["Verified", "Pending", "Rejected"],
                    },
                    "uncertainty": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "reason": {
                                        "type": "string",
                                        "enum": sorted(FIELD_UNCERTAINTY_REASONS),
                                    },
                                    "alternative_ids": {
                                        "type": "array",
                                        "maxItems": 8,
                                        "items": {"type": "integer", "minimum": 0},
                                    },
                                },
                                "required": ["reason", "alternative_ids"],
                            },
                        ]
                    },
                },
                "required": [
                    "path",
                    "selected_ids",
                    "topk",
                    "status",
                    "uncertainty",
                ],
            },
        },
    },
    "required": ["schema_version", "fields"],
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
_FROZEN_TARGETED_VERIFICATION_JSON_SCHEMA = _freeze_schema(
    _TARGETED_VERIFICATION_JSON_SCHEMA
)
P3_SMOKE_JSON_SCHEMA = _copy_schema(_FROZEN_P3_SMOKE_JSON_SCHEMA)
_FROZEN_JOINT_PERCEPTION_JSON_SCHEMA = _freeze_schema(
    joint_perception_schema(JOINT_PERCEPTION_SCHEMA_VERSION)
)
_FROZEN_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA = _freeze_schema(
    joint_perception_schema(COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)
)
_FROZEN_RELIABILITY_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA = _freeze_schema(
    joint_perception_schema(RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)
)
_FROZEN_GATE_OWNED_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA = _freeze_schema(
    joint_perception_schema(GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)
)


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


def _targeted_invalid() -> None:
    raise ApiSchemaError("Targeted verification response violates the strict schema")


def _targeted_ids(value: object, *, task: str, maximum: int = 8) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) > maximum
        or len(set(value)) != len(value)
    ):
        _targeted_invalid()
    lower, upper = TASK_ID_BOUNDS[task]
    values = tuple(value)
    if any(
        not isinstance(item, int)
        or isinstance(item, bool)
        or not lower <= item <= upper
        for item in values
    ):
        _targeted_invalid()
    return values


def validate_targeted_verification_payload(payload: Mapping[str, Any]) -> None:
    """Validate exact targeted fields and their ontology-scoped semantics."""

    try:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "fields",
        }:
            _targeted_invalid()
        if payload["schema_version"] != TARGETED_VERIFICATION_SCHEMA_VERSION:
            _targeted_invalid()
        fields = payload["fields"]
        if not isinstance(fields, (list, tuple)) or not 1 <= len(fields) <= 5:
            _targeted_invalid()
        paths: set[str] = set()
        for field in fields:
            if not isinstance(field, Mapping) or set(field) != {
                "path",
                "selected_ids",
                "topk",
                "status",
                "uncertainty",
            }:
                _targeted_invalid()
            path = field["path"]
            if path not in FIELD_UNCERTAINTY_PATHS or path in paths:
                _targeted_invalid()
            paths.add(path)
            task = FIELD_UNCERTAINTY_PATHS[path]
            selected = _targeted_ids(field["selected_ids"], task=task)
            if task == "phase" and len(selected) != 1:
                _targeted_invalid()
            topk = field["topk"]
            if not isinstance(topk, (list, tuple)) or not 1 <= len(topk) <= 8:
                _targeted_invalid()
            topk_ids: list[int] = []
            confidences: list[float] = []
            for record in topk:
                if not isinstance(record, Mapping) or set(record) != {
                    "id",
                    "confidence",
                }:
                    _targeted_invalid()
                topk_ids.extend(_targeted_ids([record["id"]], task=task))
                confidence = record["confidence"]
                if (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    _targeted_invalid()
                confidences.append(float(confidence))
            if len(set(topk_ids)) != len(topk_ids):
                _targeted_invalid()
            if any(
                confidences[index] < confidences[index + 1]
                for index in range(len(confidences) - 1)
            ):
                _targeted_invalid()
            if not set(selected).issubset(topk_ids):
                _targeted_invalid()
            if field["status"] not in {"Verified", "Pending", "Rejected"}:
                _targeted_invalid()
            uncertainty = field["uncertainty"]
            # A decisive selection and an explicit ambiguity finding are
            # mutually exclusive.  Keeping this invariant at the wire
            # boundary prevents a provider from labelling a close alternative
            # as Verified and having it silently admitted as a repair.
            if field["status"] == "Verified" and uncertainty is not None:
                _targeted_invalid()
            if uncertainty is not None:
                if not isinstance(uncertainty, Mapping) or set(uncertainty) != {
                    "reason",
                    "alternative_ids",
                }:
                    _targeted_invalid()
                if uncertainty["reason"] not in FIELD_UNCERTAINTY_REASONS:
                    _targeted_invalid()
                alternatives = _targeted_ids(
                    uncertainty["alternative_ids"], task=task
                )
                if not set(alternatives).issubset(topk_ids):
                    _targeted_invalid()
    except (KeyError, OverflowError, TypeError, ValueError):
        _targeted_invalid()


SCHEMAS: Mapping[
    str, tuple[Mapping[str, Any], Callable[[Mapping[str, Any]], None]]
] = MappingProxyType(
    {
        P3_SMOKE_SCHEMA_VERSION: (
            _FROZEN_P3_SMOKE_JSON_SCHEMA,
            validate_p3_smoke_payload,
        ),
        **{version: (_freeze_schema(schema), validator)
           for version, (schema, validator) in EXPERT_CONTRACTS.items()},
        **{version: (_freeze_schema(schema), validator)
           for version, (schema, validator) in GROUNDED_CONTRACTS.items()},
        JOINT_PERCEPTION_SCHEMA_VERSION: (
            _FROZEN_JOINT_PERCEPTION_JSON_SCHEMA,
            validate_joint_perception_payload,
        ),
        TARGETED_VERIFICATION_SCHEMA_VERSION: (
            _FROZEN_TARGETED_VERIFICATION_JSON_SCHEMA,
            validate_targeted_verification_payload,
        ),
        COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            _FROZEN_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA,
            validate_compact_joint_perception_payload,
        ),
        RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            _FROZEN_RELIABILITY_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA,
            validate_reliability_compact_joint_perception_payload,
        ),
        GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            _FROZEN_GATE_OWNED_COMPACT_JOINT_PERCEPTION_JSON_SCHEMA,
            validate_gate_owned_compact_joint_perception_payload,
        ),
        FINAL_ONLY_SCHEMA_VERSION: (
            _freeze_schema(final_only_schema()), validate_final_only,
        ),
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
