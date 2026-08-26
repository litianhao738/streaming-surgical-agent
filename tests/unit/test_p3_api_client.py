"""P3 cache, retry, schema, and usage-accounting contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from surgical_agent.api import cache as cache_module
from surgical_agent.api import errors
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    ApiResponseRecord,
    ProviderResponse,
)
from surgical_agent.api.errors import (
    ApiCacheError,
    ApiContractError,
    ApiSchemaError,
    ApiTransportError,
)
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy, RetryResult
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


def _origin_record(**changes: object) -> tuple[object, ApiResponseRecord]:
    metadata = canonical_request_metadata(_request())
    values: dict[str, object] = {
        "provider": metadata.provider,
        "endpoint_identifier": metadata.endpoint_identifier,
        "request_hash": metadata.request_hash,
        "requested_model_identifier": metadata.requested_model_identifier,
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
        "latency_ms": 1.0,
        "retry_count": 0,
        "provider_call_count": 1,
        "timestamp": "2026-08-26T00:00:00+00:00",
        "cache_hit": False,
        "provider_request_id": "mock-request-1",
        "provider_cost": 0.25,
        "origin_provider_cost": 0.25,
        "safe_metadata": {"finish_reason": "stop"},
    }
    values.update(changes)
    return metadata, ApiResponseRecord(**values)


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


def test_failed_cache_validation_never_recounts_origin_usage(tmp_path: Path) -> None:
    transport = MockProviderTransport(provider_cost=0.25)
    client, usage = _client(tmp_path, transport)
    first = client.call(_request())
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["response"]["parsed_payload"] = {"schema_version": "malformed"}
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    for expected_calls in (2, 3):
        with pytest.raises(ApiSchemaError):
            client.call(_request())
        rows = usage.records()
        failed = rows[-1]
        assert len(rows) == expected_calls
        assert failed["cache_hit"] is True
        assert failed["provider_call_count"] == 0
        assert failed["retry_count"] == 0
        assert failed["latency_ms"] == 0.0
        assert failed["provider_cost"] == 0.0
        assert failed["origin_provider_cost"] == 0.25

    assert transport.provider_call_count == 1
    assert usage.summarize() == {
        "schema_version": "api_usage_summary_v2",
        "logical_calls": 3,
        "provider_calls": 1,
        "retries": 0,
        "cache_hits": 2,
        "successful_calls": 1,
        "failed_calls": 2,
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "provider_cost": 0.25,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "other"},
        {"endpoint_identifier": "mock://other"},
        {"requested_model_identifier": "other-model"},
    ],
)
def test_cache_put_rejects_identity_mismatch_before_io(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    metadata, record = _origin_record(**changes)
    cache = FileApiCache(tmp_path / "cache")

    with pytest.raises(ApiCacheError, match="identity"):
        cache.put(metadata, record)
    assert not cache.root.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "cache_hit": True,
            "provider_call_count": 0,
            "retry_count": 0,
            "provider_cost": 0.0,
        },
        {"provider_call_count": 0},
        {"latency_ms": None},
        {"provider_cost": 0.0, "origin_provider_cost": 0.25},
    ],
)
def test_cache_put_rejects_incoherent_origin_record(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    metadata, record = _origin_record(**changes)

    with pytest.raises(ApiCacheError, match="origin"):
        FileApiCache(tmp_path / "cache").put(metadata, record)


def test_cache_get_rejects_path_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    first = client.call(_request())
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    replacement = tmp_path / "replacement.json"
    real_open = os.open
    raced = False

    def raced_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal raced
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == cache_path and not raced:
            raced = True
            replacement.write_text("{}", encoding="utf-8")
            os.replace(replacement, cache_path)
        return descriptor

    monkeypatch.setattr(os, "open", raced_open)

    with pytest.raises(ApiCacheError, match="raced"):
        client.call(_request())
    assert raced is True
    assert transport.provider_call_count == 1


def test_cache_get_wraps_first_filesystem_failure_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    cache.put(metadata, record)
    real_lstat = cache_module.os.lstat

    def denied_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if Path(path) == cache.path_for(metadata.request_hash):
            raise PermissionError("sensitive-path-detail")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(cache_module.os, "lstat", denied_lstat)

    with pytest.raises(ApiCacheError) as caught:
        cache.get(metadata)
    assert "sensitive-path-detail" not in str(caught.value)
    assert str(cache.root) not in str(caught.value)


def test_cache_get_rejects_stable_symlink_without_reading_target(
    tmp_path: Path,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    first = client.call(_request())
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    target = tmp_path / "same-directory-target.json"
    cache_path.replace(target)
    try:
        os.symlink(target, cache_path)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ApiCacheError, match="regular file"):
        client.call(_request())
    assert transport.provider_call_count == 1


def test_cache_put_accepts_identical_concurrent_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    real_link = os.link
    raced = False

    def raced_link(source: object, destination: object, **kwargs: object) -> None:
        nonlocal raced
        if not raced:
            raced = True
            real_link(source, destination, **kwargs)
            raise FileExistsError
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", raced_link)

    path = cache.put(metadata, record)

    assert raced is True
    assert path.is_file()
    assert cache.get(metadata) == record
    assert not list(cache.root.glob("*.tmp"))


def test_cache_put_rejects_divergent_concurrent_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    raced = False

    def raced_link(source: object, destination: object, **kwargs: object) -> None:
        nonlocal raced
        del source, kwargs
        raced = True
        Path(destination).write_text("{}", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(os, "link", raced_link)

    with pytest.raises(ApiCacheError):
        cache.put(metadata, record)
    assert raced is True
    assert not list(cache.root.glob("*.tmp"))


def test_cache_put_verifies_destination_after_successful_link_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    real_link = os.link
    replacement = tmp_path / "replacement.json"

    def raced_link(source: object, destination: object, **kwargs: object) -> None:
        real_link(source, destination, **kwargs)
        replacement.write_text("{}", encoding="utf-8")
        os.replace(replacement, destination)

    monkeypatch.setattr(os, "link", raced_link)

    with pytest.raises(ApiCacheError):
        cache.put(metadata, record)
    assert not list(cache.root.glob("*.tmp"))


def test_cache_put_rejects_late_replacement_after_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    cache_path = cache.path_for(metadata.request_hash)
    replacement = tmp_path / "post-parse-replacement.json"
    real_parse = cache._parse_envelope
    raced = False

    def raced_parse(content: str, expected: object) -> ApiResponseRecord:
        nonlocal raced
        parsed = real_parse(content, expected)
        replacement.write_text("{}", encoding="utf-8")
        os.replace(replacement, cache_path)
        raced = True
        return parsed

    monkeypatch.setattr(cache, "_parse_envelope", raced_parse)

    with pytest.raises(ApiCacheError, match="raced"):
        cache.put(metadata, record)
    assert raced is True
    assert cache_path.read_text(encoding="utf-8") == "{}"


def test_cache_reconstruction_rejects_list_pair_image_provenance(
    tmp_path: Path,
) -> None:
    metadata, record = _origin_record()
    cache = FileApiCache(tmp_path / "cache")
    path = cache.put(metadata, record)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    image = envelope["request"]["images"][0]
    envelope["request"]["images"] = [list(image.items())]
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ApiCacheError):
        cache.get(metadata)


def test_invalid_transport_timestamp_is_logged_safely_before_cache_put(
    tmp_path: Path,
) -> None:
    poisoned = object.__new__(ProviderResponse)
    valid = _valid_provider_response()
    for field_name in valid.__dataclass_fields__:
        object.__setattr__(poisoned, field_name, getattr(valid, field_name))
    object.__setattr__(poisoned, "timestamp", "provider body: Bearer SECRET")
    transport = SequenceTransport([poisoned])
    client, _ = _client(tmp_path, transport)

    with pytest.raises(ApiContractError, match="contract"):
        client.call(_request())

    assert not list((tmp_path / "cache").glob("*.json"))
    rendered = (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8")
    row = json.loads(rendered)
    assert row["error"] == {
        "code": "contract_error",
        "retryable": False,
        "status_code": None,
    }
    assert "Bearer" not in rendered


def test_unsafe_request_metadata_is_rejected_before_any_persistence(
    tmp_path: Path,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)

    with pytest.raises(ValueError, match="generation_parameters"):
        unsafe = ApiRequest(
            provider="mock",
            model_identifier="mock-requested-alias",
            endpoint_identifier="mock://local/p3",
            prompt_version="p3-test-v1",
            response_schema_version=P3_SMOKE_SCHEMA_VERSION,
            payload={"probe": "transport"},
            generation_parameters={"access_token": "credential"},
        )
        client.call(unsafe)

    assert transport.provider_call_count == 0
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "api_usage.jsonl").exists()


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": True},
        {"max_attempts": 0},
        {"base_delay_seconds": float("nan")},
        {"base_delay_seconds": float("inf")},
        {"base_delay_seconds": True},
        {"max_delay_seconds": float("nan")},
        {"max_delay_seconds": -1.0},
    ],
)
def test_retry_policy_rejects_invalid_numeric_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize(
    ("attempt_count", "retry_count"),
    [(True, 0), (0, 0), (1, True), (1, -1), (3, 1)],
)
def test_retry_result_requires_coherent_exact_counts(
    attempt_count: object,
    retry_count: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="count"):
        RetryResult(
            value="unused", attempt_count=attempt_count, retry_count=retry_count
        )


@pytest.mark.parametrize(
    ("attempt_count", "retry_count"),
    [(True, 0), (0, 0), (1, True), (1, -1), (3, 1)],
)
def test_api_call_failure_requires_coherent_exact_counts(
    attempt_count: object,
    retry_count: object,
) -> None:
    cause = ApiTransportError("safe", code="busy", retryable=True)
    with pytest.raises((TypeError, ValueError), match="count"):
        errors.ApiCallFailure(
            cause,
            attempt_count=attempt_count,
            retry_count=retry_count,
        )


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
