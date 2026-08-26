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


@pytest.mark.parametrize(
    "images",
    [
        [
            [
                ("identifier", "synthetic:test"),
                ("mime_type", "image/png"),
                ("size_bytes", 7),
                ("sha256", "0" * 64),
            ]
        ],
        [
            {
                "identifier": "synthetic:test",
                "mime_type": "image/png",
                "size_bytes": 7,
                "sha256": "0" * 64,
                "authorization": "credential",
            }
        ],
        [
            {
                "identifier": "synthetic:test",
                "mime_type": "image/png",
                "size_bytes": 7,
            }
        ],
    ],
)
def test_canonical_metadata_requires_exact_mapping_image_items(
    images: object,
) -> None:
    persisted = request_hash.canonical_request_metadata(_request()).to_mapping()
    persisted["images"] = images

    with pytest.raises((TypeError, ValueError), match="image"):
        contracts.CanonicalRequestMetadata.from_mapping(persisted)


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
    "generation_parameters",
    [
        {"authorization": "credential"},
        {"access_token": "credential"},
        {"Temperature": 0.0},
        {"temperature": True},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": float("nan")},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"max_output_tokens": False},
        {"max_output_tokens": 0},
        {"seed": True},
        {"deterministic_mock": 1},
        {"reasoning": "high"},
        {"reasoning": {}},
        {"reasoning": {"effort": "extreme"}},
        {"reasoning": {"effort": "high", "authorization": "credential"}},
        {"reasoning": {"secret": "credential"}},
    ],
)
def test_request_generation_parameters_are_an_exact_typed_allowlist(
    generation_parameters: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="generation_parameters"):
        _request(generation_parameters=generation_parameters)


def test_request_accepts_only_the_supported_generation_options() -> None:
    request = _request(
        generation_parameters={
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 128,
            "seed": 7,
            "deterministic_mock": True,
            "reasoning": {"effort": "high"},
        }
    )

    assert request.generation_parameters == {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_output_tokens": 128,
        "seed": 7,
        "deterministic_mock": True,
        "reasoning": {"effort": "high"},
    }


def test_reasoning_round_trip_and_hash_bind_reviewed_effort() -> None:
    high = _request(generation_parameters={"reasoning": {"effort": "high"}})
    low = _request(generation_parameters={"reasoning": {"effort": "low"}})
    restored = contracts.CanonicalRequestMetadata.from_mapping(
        request_hash.canonical_request_metadata(high).to_mapping()
    )

    assert restored.generation_parameters == {"reasoning": {"effort": "high"}}
    assert request_sha256(high) != request_sha256(low)


def test_canonical_metadata_revalidates_generation_parameter_allowlist() -> None:
    persisted = request_hash.canonical_request_metadata(_request()).to_mapping()
    persisted["generation_parameters"] = {"access_token": "credential"}

    with pytest.raises(ValueError, match="generation_parameters"):
        contracts.CanonicalRequestMetadata.from_mapping(persisted)


@pytest.mark.parametrize(
    ("safe_metadata", "error_match"),
    [
        ({"authorization": "credential"}, "safe_metadata"),
        ({"finish_reason": "provider error detail"}, "finish_reason"),
        ({"finish_reason": {"nested": "stop"}}, "finish_reason"),
    ],
)
def test_response_safe_metadata_is_an_exact_normalized_allowlist(
    safe_metadata: object,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        ProviderResponse(
            provider="mock",
            returned_model_identifier="returned-model",
            parsed_payload={},
            safe_metadata=safe_metadata,
        )


def test_response_accepts_normalized_safe_metadata_fields() -> None:
    response = ProviderResponse(
        provider="mock",
        returned_model_identifier="returned-model",
        parsed_payload={},
        safe_metadata={"finish_reason": "stop"},
    )

    assert response.safe_metadata["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "timestamp",
    ["not-a-time", "2026-08-26T00:00:00", "2026-08-26", ""],
)
def test_response_contracts_reject_nonaware_iso_timestamps(timestamp: str) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        ProviderResponse(
            provider="mock",
            returned_model_identifier="returned-model",
            parsed_payload={},
            timestamp=timestamp,
        )

    values = {
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "request_hash": "0" * 64,
        "requested_model_identifier": "requested-model",
        "returned_model_identifier": "returned-model",
        "parsed_payload": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "image_count": 0,
        "latency_ms": 0.0,
        "retry_count": 0,
        "provider_call_count": 1,
        "timestamp": timestamp,
        "cache_hit": False,
        "provider_cost": 0.0,
        "origin_provider_cost": 0.0,
    }
    with pytest.raises(ValueError, match="timestamp"):
        ApiResponseRecord(**values)


@pytest.mark.parametrize(
    ("contract", "changes"),
    [
        ("request_payload", {"payload": ["not", "mapping"]}),
        ("request_generation", {"generation_parameters": []}),
        ("provider_payload", {"parsed_payload": []}),
        ("provider_metadata", {"safe_metadata": []}),
    ],
)
def test_runtime_contracts_require_mappings(
    contract: str,
    changes: dict[str, object],
) -> None:
    if contract.startswith("request"):
        with pytest.raises(TypeError, match="mapping"):
            _request(**changes)
    else:
        values: dict[str, object] = {
            "provider": "mock",
            "returned_model_identifier": "returned-model",
            "parsed_payload": {},
            "safe_metadata": {},
        }
        values.update(changes)
        with pytest.raises(TypeError, match="mapping"):
            ProviderResponse(**values)


@pytest.mark.parametrize(
    ("backend", "source"),
    [("backend-model", None), (None, "response.model")],
)
def test_response_requires_coherent_exact_backend_evidence_pair(
    backend: str | None,
    source: str | None,
) -> None:
    with pytest.raises(ValueError, match="exact backend identity"):
        ProviderResponse(
            provider="mock",
            returned_model_identifier="returned-model",
            parsed_payload={},
            exact_backend_model_identifier=backend,
            exact_identity_evidence_source=source,
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
