"""Canonical multimodal request provenance and hashing."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from surgical_agent.api.contracts import (
    ApiRequest,
    CanonicalRequestMetadata,
    ImageProvenance,
    _freeze_json,
    canonical_json_bytes,
    thaw_json,
)


def canonical_request_metadata(request: ApiRequest) -> CanonicalRequestMetadata:
    """Build the safe provenance and request digest from one canonical body."""

    payload = thaw_json(request.payload)
    generation = thaw_json(request.generation_parameters)
    image_rows = tuple(
        ImageProvenance(
            identifier=image.identifier,
            mime_type=image.mime_type,
            size_bytes=len(image.content),
            sha256=image.sha256,
        )
        for image in request.images
    )
    payload_bytes = canonical_json_bytes(payload)
    hash_body = {
        "provider": request.provider,
        "endpoint_identifier": request.endpoint_identifier,
        "requested_model_identifier": request.model_identifier,
        "prompt_version": request.prompt_version,
        "response_schema_version": request.response_schema_version,
        "generation_parameters": generation,
        "payload": payload,
        "images": [asdict(image) for image in image_rows],
    }
    request_hash = hashlib.sha256(canonical_json_bytes(hash_body)).hexdigest()
    return CanonicalRequestMetadata(
        schema_version="api_request_metadata_v1",
        provider=request.provider,
        endpoint_identifier=request.endpoint_identifier,
        requested_model_identifier=request.model_identifier,
        prompt_version=request.prompt_version,
        response_schema_version=request.response_schema_version,
        generation_parameters=_freeze_json(generation, path="generation_parameters"),
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        images=image_rows,
        request_hash=request_hash,
    )


def canonical_request_hash(request: ApiRequest) -> str:
    """Return the single canonical request digest."""

    return canonical_request_metadata(request).request_hash


def request_sha256(request: ApiRequest) -> str:
    """Compatibility alias retained for existing P3 callers and public tests."""

    return canonical_request_hash(request)
