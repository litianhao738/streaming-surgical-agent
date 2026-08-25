"""P3 cache, retry, schema, and usage-accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.errors import ApiCacheError, ApiRetryExhausted, ApiSchemaError
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
    transport: MockProviderTransport,
    *,
    max_attempts: int = 3,
) -> tuple[CachedMultimodalApiClient, UsageLedger]:
    usage = UsageLedger(tmp_path / "api_usage.jsonl")
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(tmp_path / "cache"),
        usage=usage,
        validator=validate_p3_smoke_payload,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _: None,
    )
    return client, usage


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
        "schema_version": "api_usage_summary_v1",
        "logical_calls": 2,
        "provider_calls": 1,
        "cache_hits": 1,
        "successful_calls": 2,
        "failed_calls": 0,
        "input_tokens": 12,
        "output_tokens": 7,
    }


def test_retry_count_and_provider_attempts_are_auditable(tmp_path: Path) -> None:
    transport = MockProviderTransport(retryable_failures_before_success=2)
    client, usage = _client(tmp_path, transport)

    response = client.call(_request())

    assert response.retry_count == 2
    assert transport.provider_call_count == 3
    assert usage.summarize()["provider_calls"] == 3


def test_retry_exhaustion_logs_failure_and_does_not_cache(tmp_path: Path) -> None:
    transport = MockProviderTransport(retryable_failures_before_success=9)
    client, usage = _client(tmp_path, transport, max_attempts=3)

    with pytest.raises(ApiRetryExhausted):
        client.call(_request())

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
    assert failure["error_raw_response"] is not None
    assert failure["error_parsed_payload"] == {"schema_version": "malformed"}
    assert not list((tmp_path / "cache").glob("*.json"))


def test_corrupt_cache_fails_closed_without_provider_call(tmp_path: Path) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["request_hash"] = "0" * 64
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApiCacheError, match="filename/content mismatch"):
        client.call(request)
    assert transport.provider_call_count == 1
    assert client.usage.summarize()["failed_calls"] == 1
