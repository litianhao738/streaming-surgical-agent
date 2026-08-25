"""P3 canonical request/image hashing contracts."""

from __future__ import annotations

from dataclasses import fields

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.request_hash import request_sha256


def _request(*, image_bytes: bytes = b"image-a", **changes: object) -> ApiRequest:
    values: dict[str, object] = {
        "provider": "mock",
        "model_identifier": "requested-model",
        "endpoint_identifier": "mock://local/p3",
        "prompt_version": "prompt-v1",
        "response_schema_version": "schema-v1",
        "payload": {"b": [2, 3], "a": {"value": 1}},
        "images": (
            ApiImageInput("synthetic:test", "image/png", image_bytes),
        ),
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
