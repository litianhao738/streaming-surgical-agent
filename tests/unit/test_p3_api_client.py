"""P3 cache, retry, schema, and usage-accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.api import errors
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiImageInput, ApiRequest, ProviderResponse
from surgical_agent.api.errors import ApiCacheError, ApiSchemaError, ApiTransportError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import (
    P3_SMOKE_SCHEMA_VERSION,
    validate_p3_smoke_payload,
)
from surgical_agent.api.usage import UsageLedger


def _request() -> ApiRequest:
    return ApiRequest(
        provider="mock",
        model_identifier="mock-requested-alias",
        endpoint_identifier="mock://local/p3",
        prompt_version="p3-test-v1",
        response_schema_version=P3_SMOKE_SCHEMA_VERSION,
        payload={"probe": "transport"},
        images=(ApiImageInput("synthetic:test", "image/png", b"not-real-png"),),
        generation_parameters={"temperature": 0.0},
    )


def _client(
    tmp_path: Path,
    transport: object,
    *,
    max_attempts: int = 3,
    validator: object = validate_p3_smoke_payload,
) -> tuple[CachedMultimodalApiClient, UsageLedger]:
    usage = UsageLedger(tmp_path / "api_usage.jsonl")
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(tmp_path / "cache"),
        usage=usage,
        validator=validator,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _: None,
    )
    return client, usage


def _valid_provider_response(**changes: object) -> ProviderResponse:
    values: dict[str, object] = {
        "provider": "mock",
        "returned_model_identifier": "mock-model-returned-v1",
        "parsed_payload": {
            "schema_version": P3_SMOKE_SCHEMA_VERSION,
            "message": "mock multimodal response",
            "image_observed": True,
            "structured": True,
        },
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "image_count": 1,
        "provider_request_id": "mock-request-1",
        "timestamp": "2026-08-26T00:00:00+00:00",
        "provider_cost": 0.25,
        "safe_metadata": {"finish_reason": "stop"},
    }
    values.update(changes)
    return ProviderResponse(**values)


class SequenceTransport:
    provider = "mock"
    endpoint_identifier = "mock://local/p3"

    def __init__(self, outcomes: list[ApiTransportError | ProviderResponse]) -> None:
        self.outcomes = list(outcomes)
        self.provider_call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.provider_call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, ApiTransportError):
            raise outcome
        return outcome


def test_transport_error_keeps_only_a_safe_category() -> None:
    error = ApiTransportError(
        "provider body with authorization token",
        code="rate_limit",
        retryable=True,
        status_code=429,
    )

    assert str(error) == "API transport failed with category rate_limit"
    with pytest.raises(ValueError, match="safe category"):
        ApiTransportError(
            "provider body",
            code="authorization: Bearer unsafe",
            retryable=False,
        )


def test_cache_replay_is_validated_and_not_counted_as_provider_call(
    tmp_path: Path,
) -> None:
    transport = MockProviderTransport(returned_model_identifier="returned-exact-v1")
    client, usage = _client(tmp_path, transport)

    first = client.call(_request())
    replay = client.call(_request())

    assert first.model_identifier == "returned-exact-v1"
    assert first.requested_model_identifier == "mock-requested-alias"
    assert not first.cache_hit
    assert replay.cache_hit
    assert not replay.provider_call
    assert replay.retry_count == 0
    assert transport.provider_call_count == 1
    assert usage.summarize() == {
        "schema_version": "api_usage_summary_v2",
        "logical_calls": 2,
        "provider_calls": 1,
        "retries": 0,
        "cache_hits": 1,
        "successful_calls": 2,
        "failed_calls": 0,
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "provider_cost": 0.0,
    }


def test_retry_count_and_provider_attempts_are_auditable(tmp_path: Path) -> None:
    transport = MockProviderTransport(retryable_failures_before_success=2)
    client, usage = _client(tmp_path, transport)

    response = client.call(_request())

    assert response.retry_count == 2
    assert transport.provider_call_count == 3
    assert usage.summarize()["provider_calls"] == 3


def test_retry_exhaustion_logs_all_attempts_and_does_not_cache(
    tmp_path: Path,
) -> None:
    transport = MockProviderTransport(retryable_failures_before_success=9)
    client, usage = _client(tmp_path, transport, max_attempts=3)

    assert hasattr(errors, "ApiCallFailure")
    with pytest.raises(errors.ApiCallFailure) as caught:
        client.call(_request())

    assert caught.value.attempt_count == 3
    assert caught.value.provider_call_count == 3
    assert caught.value.retry_count == 2
    assert transport.provider_call_count == 3
    summary = usage.summarize()
    assert summary["logical_calls"] == 1
    assert summary["provider_calls"] == 3
    assert summary["failed_calls"] == 1
    assert not list((tmp_path / "cache").glob("*.json"))


def test_malformed_structured_response_fails_closed(tmp_path: Path) -> None:
    transport = MockProviderTransport(malformed_payload=True)
    client, usage = _client(tmp_path, transport)

    with pytest.raises(ApiSchemaError, match="schema_version"):
        client.call(_request())

    assert usage.summarize()["failed_calls"] == 1
    failure = usage.records()[0]
    assert failure["returned_model_identifier"] == "mock-model-returned-v1"
    assert failure["provider_call_count"] == 1
    assert failure["retry_count"] == 0
    assert failure["error"] == {
        "code": "schema_error",
        "retryable": False,
        "status_code": None,
    }
    assert not list((tmp_path / "cache").glob("*.json"))


def test_corrupt_cache_fails_closed_without_provider_call(tmp_path: Path) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["response"]["request_hash"] = "0" * 64
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApiCacheError, match="request hash"):
        client.call(request)
    assert transport.provider_call_count == 1
    assert client.usage.summarize()["failed_calls"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider", "other"),
        ("endpoint_identifier", "https://invalid.example/responses"),
        ("requested_model_identifier", "other/model"),
        ("prompt_version", "other_prompt"),
        ("response_schema_version", "other_schema"),
        ("generation_parameters", {"temperature": 1}),
        ("payload_sha256", "0" * 64),
        ("request_hash", "0" * 64),
        ("schema_version", "other_metadata_schema"),
        (
            "images",
            [
                {
                    "identifier": "synthetic:other",
                    "mime_type": "image/png",
                    "size_bytes": 12,
                    "sha256": "0" * 64,
                }
            ],
        ),
    ],
)
def test_cache_rejects_tampered_request_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["request"][field] = replacement
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ApiCacheError, match="request metadata"):
        client.call(request)
    assert transport.provider_call_count == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider", "other"),
        ("endpoint_identifier", "mock://other"),
        ("requested_model_identifier", "other-model"),
    ],
)
def test_cache_rejects_response_identity_mismatches(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    first = client.call(_request())
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["response"][field] = replacement
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ApiCacheError, match="response identity"):
        client.call(_request())
    assert transport.provider_call_count == 1


def test_present_nonfile_cache_entry_fails_closed(tmp_path: Path) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    first = client.call(_request())
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    cache_path.unlink()
    cache_path.mkdir()

    with pytest.raises(ApiCacheError, match="not a regular file"):
        client.call(_request())
    assert transport.provider_call_count == 1


@pytest.mark.parametrize(
    ("location", "mutation"),
    [
        ("envelope", "missing"),
        ("envelope", "unknown"),
        ("request", "missing"),
        ("request", "unknown"),
        ("response", "missing"),
        ("response", "unknown"),
    ],
)
def test_cache_rejects_nonexact_envelope_fields(
    tmp_path: Path,
    location: str,
    mutation: str,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    target = raw if location == "envelope" else raw[location]
    if mutation == "missing":
        target.pop(next(iter(target)))
    else:
        target["raw_response"] = "unsafe"
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ApiCacheError):
        client.call(request)
    assert transport.provider_call_count == 1


def test_cache_hit_has_zero_current_provider_cost(tmp_path: Path) -> None:
    transport = MockProviderTransport(provider_cost=0.25)
    client, usage = _client(tmp_path, transport)
    request = _request()

    first = client.call(request)
    second = client.call(request)

    assert first.provider_cost == 0.25
    assert first.origin_provider_cost == 0.25
    assert second.cache_hit is True
    assert second.provider_call_count == 0
    assert second.retry_count == 0
    assert second.provider_cost == 0.0
    assert second.origin_provider_cost == 0.25
    assert usage.summarize()["provider_cost"] == 0.25


def test_retryable_then_nonretryable_preserves_all_counts(tmp_path: Path) -> None:
    transport = SequenceTransport(
        [
            ApiTransportError("provider body one", code="rate_limit", retryable=True),
            ApiTransportError(
                "provider body two", code="authentication", retryable=False
            ),
        ]
    )
    client, _ = _client(tmp_path, transport)
    usage_path = tmp_path / "api_usage.jsonl"

    assert hasattr(errors, "ApiCallFailure")
    with pytest.raises(errors.ApiCallFailure) as caught:
        client.call(_request())

    assert caught.value.attempt_count == 2
    assert caught.value.retry_count == 1
    assert str(caught.value) == "API call failed with category authentication"
    row = json.loads(usage_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["provider_call_count"] == 2
    assert row["retry_count"] == 1
    assert row["error"] == {
        "code": "authentication",
        "retryable": False,
        "status_code": None,
    }


def test_schema_failure_after_retry_preserves_transport_counts(tmp_path: Path) -> None:
    transport = SequenceTransport(
        [
            ApiTransportError("provider body", code="busy", retryable=True),
            _valid_provider_response(parsed_payload={"schema_version": "malformed"}),
        ]
    )
    client, _ = _client(tmp_path, transport)

    with pytest.raises(ApiSchemaError):
        client.call(_request())

    row = json.loads(
        (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert row["provider_call_count"] == 2
    assert row["retry_count"] == 1
    assert row["error"]["code"] == "schema_error"


def test_failure_ledger_has_no_raw_response_or_payload(tmp_path: Path) -> None:
    transport = MockProviderTransport(malformed_payload=True)
    client, _ = _client(tmp_path, transport)
    usage_path = tmp_path / "api_usage.jsonl"

    with pytest.raises(ApiSchemaError):
        client.call(_request())

    rendered = usage_path.read_text(encoding="utf-8")
    assert "raw_response" not in rendered
    assert "parsed_payload" not in rendered
    assert "authorization" not in rendered.lower()
    assert '"probe"' not in rendered


def test_unexpected_validator_failure_is_sanitized_and_counted(tmp_path: Path) -> None:
    def broken_validator(payload: object) -> None:
        del payload
        raise RuntimeError("sensitive validator detail")

    transport = MockProviderTransport(retryable_failures_before_success=1)
    client, _ = _client(tmp_path, transport, validator=broken_validator)

    with pytest.raises(ApiSchemaError, match="P3 response validator failed"):
        client.call(_request())

    rendered = (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8")
    row = json.loads(rendered)
    assert row["provider_call_count"] == 2
    assert row["retry_count"] == 1
    assert row["error"]["code"] == "schema_error"
    assert "sensitive validator detail" not in rendered


def test_success_cache_and_usage_persist_only_allowlisted_safe_fields(
    tmp_path: Path,
) -> None:
    transport = MockProviderTransport(provider_cost=0.25)
    client, _ = _client(tmp_path, transport)
    response = client.call(_request())

    cache_text = (tmp_path / "cache" / f"{response.request_hash}.json").read_text(
        encoding="utf-8"
    )
    usage_text = (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8")
    for persisted in (cache_text, usage_text):
        assert "raw_response" not in persisted
        assert "authorization" not in persisted.lower()
    assert "parsed_payload" in cache_text
    assert "parsed_payload" not in usage_text
    assert '"probe"' not in usage_text
