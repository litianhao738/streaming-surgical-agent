"""P3-only structured smoke schema; P4 surgical semantics remain deferred."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from surgical_agent.api.errors import ApiSchemaError

P3_SMOKE_SCHEMA_VERSION = "p3_multimodal_smoke_v1"


def validate_p3_smoke_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != P3_SMOKE_SCHEMA_VERSION:
        raise ApiSchemaError("P3 smoke response has an unsupported schema_version")
    if not isinstance(payload.get("message"), str) or not payload["message"].strip():
        raise ApiSchemaError("P3 smoke response requires a non-empty message")
    if not isinstance(payload.get("image_observed"), bool):
        raise ApiSchemaError("P3 smoke response requires boolean image_observed")
    if payload.get("structured") is not True:
        raise ApiSchemaError("P3 smoke response must declare structured=true")
