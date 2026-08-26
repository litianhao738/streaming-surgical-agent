"""P3 canonical request/image hashing contracts."""

from __future__ import annotations

from dataclasses import fields

import pytest

from surgical_agent.api import contracts, request_hash
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    ApiResponseRecord,
    ImageProvenance,
    ProviderResponse,
)
from surgical_agent.api.request_hash import request_sha256


def _request(*, image_bytes: bytes = b"image-a", **changes: object) -> ApiRequest:
    values: dict[str, object] = {
        "provider": "mock",
        "model_identifier": "requested-model",
        "endpoint_identifier": "mock://local/p3",
        "prompt_version": "prompt-v1",
        "response_schema_version": "schema-v1",
        "payload": {"b": [2, 3], "a": {"value": 1}},
        "images": (ApiImageInput("synthetic:test", "image/png", image_bytes),),
        "generation_parameters": {"temperature": 0.0},
    }
    values.update(changes)
    return ApiRequest(**values)


def test_request_hash_is_order_stable_and_binds_image_content() -> None:
    first = _request(payload={"b": [2, 3], "a": {"value": 1}})
    reordered = _request(payload={"a": {"value": 1}, "b": [2, 3]})
    changed_image = _request(image_bytes=b"image-b")

    assert request_sha256(first) == request_sha256(reordered)
    assert request_sha256(first) != request_sha256(changed_image)


def test_canonical_metadata_reconstructs_the_safe_request() -> None:
    request = _request()

    assert hasattr(request_hash, "canonical_request_metadata")
    metadata = request_hash.canonical_request_metadata(request)

    assert metadata.schema_version == "api_request_metadata_v1"
    assert metadata.provider == request.provider
    assert metadata.endpoint_identifier == request.endpoint_identifier
    assert metadata.requested_model_identifier == request.model_identifier
    assert metadata.prompt_version == request.prompt_version
    assert metadata.response_schema_version == request.response_schema_version
    assert metadata.generation_parameters == {"temperature": 0.0}
    assert metadata.images[0].identifier == request.images[0].identifier
    assert metadata.images[0].mime_type == request.images[0].mime_type
    assert metadata.images[0].sha256 == request.images[0].sha256
    assert metadata.images[0].size_bytes == len(request.images[0].content)
    assert len(metadata.payload_sha256) == 64
    assert len(metadata.request_hash) == 64
    assert request_hash.canonical_request_hash(request) == metadata.request_hash
    assert request_sha256(request) == metadata.request_hash


def test_canonical_metadata_mapping_round_trip_is_exact_and_immutable() -> None:
    assert hasattr(contracts, "CanonicalRequestMetadata")
    metadata = request_hash.canonical_request_metadata(_request())
    persisted = metadata.to_mapping()

    restored = contracts.CanonicalRequestMetadata.from_mapping(persisted)
    persisted["generation_parameters"]["temperature"] = 1.0

    assert restored == metadata
    assert restored.generation_parameters == {"temperature": 0.0}
    with pytest.raises(TypeError):
        restored.generation_parameters["temperature"] = 1.0


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_canonical_metadata_rejects_nonexact_persisted_fields(mutation: str) -> None:
    assert hasattr(contracts, "CanonicalRequestMetadata")
    persisted = request_hash.canonical_request_metadata(_request()).to_mapping()
    if mutation == "missing":
        persisted.pop("provider")
    else:
        persisted["raw_response"] = "unsafe"

    with pytest.raises(ValueError, match="invalid fields"):
        contracts.CanonicalRequestMetadata.from_mapping(persisted)


def test_persistable_response_contracts_have_no_raw_response_field() -> None:
    assert "raw_response" not in {item.name for item in fields(ProviderResponse)}
    assert "raw_response" not in {item.name for item in fields(ApiResponseRecord)}


def test_provenance_and_response_contracts_reject_missing_required_counts() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        ImageProvenance(
            identifier="synthetic:test",
            mime_type="image/png",
            size_bytes=None,
            sha256="0" * 64,
        )

    persisted = {
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "request_hash": "0" * 64,
        "requested_model_identifier": "requested-model",
        "returned_model_identifier": "returned-model",
        "parsed_payload": {},
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "image_count": None,
        "latency_ms": None,
        "retry_count": None,
        "provider_call_count": 1,
        "timestamp": None,
        "cache_hit": False,
        "provider_request_id": None,
        "provider_cost": None,
        "origin_provider_cost": None,
        "exact_backend_model_identifier": None,
        "exact_identity_evidence_source": None,
        "safe_metadata": {},
    }
    with pytest.raises(ValueError, match="retry_count"):
        ApiResponseRecord.from_persisted_mapping(persisted)


def test_response_safe_metadata_rejects_body_and_credential_keys() -> None:
    for unsafe_key in ("raw_response", "payload", "authorization", "api_key"):
        with pytest.raises(ValueError, match="unsafe key"):
            ProviderResponse(
                provider="mock",
                returned_model_identifier="returned-model",
                parsed_payload={},
                safe_metadata={unsafe_key: "must-not-persist"},
            )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("provider", "other"),
        ("model_identifier", "other-model"),
        ("endpoint_identifier", "mock://other"),
        ("prompt_version", "prompt-v2"),
        ("response_schema_version", "schema-v2"),
        ("payload", {"a": 9}),
        ("generation_parameters", {"temperature": 0.1}),
    ),
)
def test_request_hash_changes_for_every_response_affecting_field(
    field_name: str,
    replacement: object,
) -> None:
    assert request_sha256(_request()) != request_sha256(
        _request(**{field_name: replacement})
    )


def test_request_copies_payload_and_has_no_credential_fields() -> None:
    payload = {"nested": [1, 2]}
    request = _request(payload=payload)
    before = request_sha256(request)
    payload["nested"].append(3)

    assert request_sha256(request) == before
    assert {field.name for field in fields(ApiRequest)}.isdisjoint(
        {"api_key", "token", "password", "secret", "authorization"}
    )


def test_request_rejects_non_json_and_nonfinite_payload_values() -> None:
    with pytest.raises(TypeError, match="non-JSON"):
        _request(payload={"bad": object()})
    with pytest.raises(ValueError, match="non-finite"):
        _request(payload={"bad": float("nan")})
