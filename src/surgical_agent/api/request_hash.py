"""Canonical multimodal request hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from surgical_agent.api.contracts import ApiRequest

REQUEST_HASH_VERSION = "api_request_sha256_v1"


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def canonical_request_payload(request: ApiRequest) -> dict[str, Any]:
    """Return every non-secret field that can affect a provider response."""

    return {
        "hash_version": REQUEST_HASH_VERSION,
        "provider": request.provider,
        "model_identifier": request.model_identifier,
        "endpoint_identifier": request.endpoint_identifier,
        "prompt_version": request.prompt_version,
        "response_schema_version": request.response_schema_version,
        "payload": _plain_json(request.payload),
        "images": [
            {
                "identifier": image.identifier,
                "mime_type": image.mime_type,
                "sha256": image.sha256,
                "byte_count": len(image.content),
            }
            for image in request.images
        ],
        "generation_parameters": _plain_json(request.generation_parameters),
    }


def canonical_request_bytes(request: ApiRequest) -> bytes:
    return json.dumps(
        canonical_request_payload(request),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def request_sha256(request: ApiRequest) -> str:
    return hashlib.sha256(canonical_request_bytes(request)).hexdigest()
