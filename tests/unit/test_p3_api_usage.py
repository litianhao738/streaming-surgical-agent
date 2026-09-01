"""Strict P3 usage-record parsing, validation, and summary contracts."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    CanonicalRequestMetadata,
)
from surgical_agent.api.usage import UsageLedger, UsageRecord


def _valid_request_mapping() -> dict[str, Any]:
    return {
        "schema_version": "api_request_metadata_v1",
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "requested_model_identifier": "requested-model",
        "prompt_version": "prompt-v1",
        "response_schema_version": "p3_multimodal_smoke_v1",
        "generation_parameters": {"temperature": 0.0},
        "payload_sha256": "a" * 64,
        "images": [],
        "request_hash": "b" * 64,
    }


def _valid_usage_mapping() -> dict[str, Any]:
    return {
        "schema_version": "api_usage_record_v3",
        "request": _valid_request_mapping(),
        "request_hash": "b" * 64,
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "requested_model_identifier": "requested-model",
        "returned_model_identifier": "returned-model",
        "exact_backend_model_identifier": None,
        "exact_identity_evidence_source": None,
        "provider_request_id": "request-1",
        "safe_provider_metadata": {"finish_reason": "stop"},
        "cache_hit": False,
        "logical_call_count": 1,
        "provider_call_count": 1,
        "retry_count": 0,
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "visible_output_tokens": 7,
        "time_to_first_token_ms": 2.5,
        "latency_ms": 10.5,
        "total_latency_ms": 10.5,
        "provider_cost": 0.25,
        "origin_provider_cost": 0.25,
        "timestamp": "2026-08-26T00:00:00+00:00",
        "error": None,
    }


def _set_path(value: dict[str, Any], path: str, replacement: object) -> None:
    parts = path.split(".")
    target: dict[str, Any] = value
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


def test_usage_record_round_trip_preserves_the_exact_valid_mapping() -> None:
    mapping = _valid_usage_mapping()

    record = UsageRecord.from_mapping(mapping)

    assert record.to_mapping() == mapping


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_usage_record_requires_exact_fields(mutation: str) -> None:
    mapping = _valid_usage_mapping()
    if mutation == "missing":
        mapping.pop("provider")
    else:
        mapping["raw_response"] = "credential"

    with pytest.raises(ValueError, match="usage record"):
        UsageRecord.from_mapping(mapping)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("schema_version", "api_usage_record_v1"),
        ("request_hash", "c" * 64),
        ("provider", "other"),
        ("endpoint_identifier", "mock://other"),
        ("requested_model_identifier", "other-model"),
        ("provider_request_id", ""),
        ("cache_hit", 1),
        ("logical_call_count", 2),
        ("provider_call_count", True),
        ("provider_call_count", -1),
        ("retry_count", True),
        ("retry_count", 1),
        ("input_tokens", True),
        ("output_tokens", -1),
        ("total_tokens", 1.5),
        ("latency_ms", math.nan),
        ("provider_cost", math.inf),
        ("origin_provider_cost", -0.1),
        ("timestamp", "2026-08-26T00:00:00"),
        ("timestamp", "not-a-time"),
        ("request.generation_parameters", {"access_token": "credential"}),
        ("safe_provider_metadata", {"finish_reason": "provider detail"}),
    ],
)
def test_usage_record_rejects_corrupt_types_versions_and_bindings(
    path: str,
    replacement: object,
) -> None:
    mapping = _valid_usage_mapping()
    _set_path(mapping, path, replacement)

    with pytest.raises((TypeError, ValueError)):
        UsageRecord.from_mapping(mapping)


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
        [{"identifier": "synthetic:test", "mime_type": "image/png"}],
        [
            {
                "identifier": "synthetic:test",
                "mime_type": "image/png",
                "size_bytes": 7,
                "sha256": "0" * 64,
                "secret": "credential",
            }
        ],
    ],
)
def test_usage_reconstruction_rejects_nonexact_image_provenance(
    images: object,
) -> None:
    mapping = _valid_usage_mapping()
    mapping["request"]["images"] = images

    with pytest.raises((TypeError, ValueError), match="image"):
        UsageRecord.from_mapping(mapping)


@pytest.mark.parametrize(
    ("backend", "source"),
    [("backend-model", None), (None, "response.model")],
)
def test_usage_record_requires_coherent_exact_backend_evidence(
    backend: str | None,
    source: str | None,
) -> None:
    mapping = _valid_usage_mapping()
    mapping["exact_backend_model_identifier"] = backend
    mapping["exact_identity_evidence_source"] = source

    with pytest.raises(ValueError, match="exact backend identity"):
        UsageRecord.from_mapping(mapping)


@pytest.mark.parametrize(
    "error",
    [
        [],
        {"code": "busy", "retryable": True},
        {"code": "busy", "retryable": 1, "status_code": 429},
        {"code": "Busy detail", "retryable": True, "status_code": 429},
        {"code": "busy", "retryable": True, "status_code": True},
        {"code": "busy", "retryable": True, "status_code": 99},
    ],
)
def test_usage_record_error_is_an_exact_typed_category_mapping(error: object) -> None:
    mapping = _valid_usage_mapping()
    mapping["error"] = error

    with pytest.raises((TypeError, ValueError), match="error"):
        UsageRecord.from_mapping(mapping)


def test_usage_record_accepts_a_strict_failure_category() -> None:
    mapping = _valid_usage_mapping()
    mapping.update(
        {
            "returned_model_identifier": None,
            "provider_request_id": None,
            "safe_provider_metadata": {},
            "provider_call_count": 2,
            "retry_count": 1,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "completion_tokens_details": {"reasoning_tokens": None},
            "visible_output_tokens": None,
            "time_to_first_token_ms": None,
            "provider_cost": None,
            "origin_provider_cost": None,
            "error": {
                "code": "rate_limit",
                "retryable": True,
                "status_code": 429,
            },
        }
    )

    assert UsageRecord.from_mapping(mapping).to_mapping() == mapping


def test_usage_record_rejects_current_cost_without_provider_calls() -> None:
    mapping = _valid_usage_mapping()
    mapping["provider_call_count"] = 0

    with pytest.raises(ValueError, match="current provider cost"):
        UsageRecord.from_mapping(mapping)


def test_usage_record_rejects_nonzero_cache_latency() -> None:
    mapping = _valid_usage_mapping()
    mapping.update(
        {
            "cache_hit": True,
            "provider_call_count": 0,
            "latency_ms": 0.1,
            "provider_cost": 0.0,
        }
    )

    with pytest.raises(ValueError, match="latency"):
        UsageRecord.from_mapping(mapping)


def test_usage_record_rejects_origin_cost_mismatch_for_provider_call() -> None:
    mapping = _valid_usage_mapping()
    mapping["origin_provider_cost"] = 0.5

    with pytest.raises(ValueError, match="origin provider cost"):
        UsageRecord.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("request", []),
        ("safe_provider_metadata", []),
        ("error", []),
    ],
)
def test_usage_record_requires_mapping_values(
    field: str,
    replacement: object,
) -> None:
    mapping = _valid_usage_mapping()
    mapping[field] = replacement

    with pytest.raises(TypeError, match="mapping"):
        UsageRecord.from_mapping(mapping)


def test_usage_ledger_records_rejects_invalid_json_and_mixed_versions(
    tmp_path: Path,
) -> None:
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid usage record at line 1"):
        ledger.records()

    first = _valid_usage_mapping()
    second = copy.deepcopy(first)
    second["schema_version"] = "api_usage_record_v1"
    ledger.path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid usage record at line 2"):
        ledger.records()


def test_usage_ledger_records_rejects_unknown_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    row = _valid_usage_mapping()
    row["unknown"] = "credential"
    ledger.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid usage record at line 1") as caught:
        ledger.records()
    assert "credential" not in str(caught.value)

    row = _valid_usage_mapping()
    row["provider_cost"] = math.nan
    ledger.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid usage record at line 1"):
        ledger.records()


def test_usage_summary_counts_only_current_provider_usage(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    origin = _valid_usage_mapping()
    replay = copy.deepcopy(origin)
    replay.update(
        {
            "cache_hit": True,
            "provider_call_count": 0,
            "retry_count": 0,
            "latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "time_to_first_token_ms": None,
            "provider_cost": 0.0,
            "origin_provider_cost": 0.25,
            "timestamp": "2026-08-26T00:01:00+00:00",
        }
    )
    ledger.path.write_text(
        json.dumps(origin) + "\n" + json.dumps(replay) + "\n",
        encoding="utf-8",
    )

    assert ledger.summarize() == {
        "schema_version": "api_usage_summary_v3",
        "logical_calls": 2,
        "provider_calls": 1,
        "retries": 0,
        "cache_hits": 1,
        "successful_calls": 2,
        "failed_calls": 0,
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "reasoning_tokens": 0,
        "visible_output_tokens": 7,
        "provider_cost": 0.25,
    }


def test_usage_success_without_response_timestamp_emits_current_utc(
    tmp_path: Path,
) -> None:
    metadata = CanonicalRequestMetadata.from_mapping(_valid_request_mapping())
    response_values = _valid_usage_mapping()
    response = ApiResponseRecord(
        **{
            field: response_values[field]
            for field in (
                "provider",
                "endpoint_identifier",
                "request_hash",
                "requested_model_identifier",
                "returned_model_identifier",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "retry_count",
                "provider_call_count",
                "provider_request_id",
                "provider_cost",
                "origin_provider_cost",
            )
        },
        parsed_payload={},
        timestamp=None,
        safe_metadata={"finish_reason": "stop"},
    )
    ledger = UsageLedger(tmp_path / "usage.jsonl")

    ledger.log_success(metadata, response)

    timestamp = ledger.records()[0]["timestamp"]
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    assert timestamp != "unknown"
