"""Atomic allowlisted usage/error accounting for logical and provider calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    CanonicalRequestMetadata,
    ProviderResponse,
    _freeze_json,
    thaw_json,
)
from surgical_agent.artifacts.manifest import atomic_write_text

USAGE_SCHEMA_VERSION = "api_usage_record_v2"


@dataclass(frozen=True)
class UsageRecord:
    schema_version: str
    request: Mapping[str, Any]
    request_hash: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    returned_model_identifier: str | None
    exact_backend_model_identifier: str | None
    exact_identity_evidence_source: str | None
    provider_request_id: str | None
    safe_provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    logical_call_count: int = 1
    provider_call_count: int = 0
    retry_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    provider_cost: float | None = None
    origin_provider_cost: float | None = None
    timestamp: str = ""
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != USAGE_SCHEMA_VERSION:
            raise ValueError("usage record has unsupported schema_version")
        object.__setattr__(self, "request", _freeze_json(self.request, path="request"))
        object.__setattr__(
            self,
            "safe_provider_metadata",
            _freeze_json(self.safe_provider_metadata, path="safe_provider_metadata"),
        )
        if self.error is not None:
            if set(self.error) != {"code", "retryable", "status_code"}:
                raise ValueError("usage error metadata has invalid fields")
            object.__setattr__(self, "error", _freeze_json(self.error, path="error"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": thaw_json(self.request),
            "request_hash": self.request_hash,
            "provider": self.provider,
            "endpoint_identifier": self.endpoint_identifier,
            "requested_model_identifier": self.requested_model_identifier,
            "returned_model_identifier": self.returned_model_identifier,
            "exact_backend_model_identifier": self.exact_backend_model_identifier,
            "exact_identity_evidence_source": self.exact_identity_evidence_source,
            "provider_request_id": self.provider_request_id,
            "safe_provider_metadata": thaw_json(self.safe_provider_metadata),
            "cache_hit": self.cache_hit,
            "logical_call_count": self.logical_call_count,
            "provider_call_count": self.provider_call_count,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "provider_cost": self.provider_cost,
            "origin_provider_cost": self.origin_provider_cost,
            "timestamp": self.timestamp,
            "error": thaw_json(self.error),
        }


class UsageLedger:
    """Small auditable ledger rewritten atomically after every logical call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            return ()
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed usage JSONL at {self.path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"Usage record {line_number} is not an object")
            records.append(value)
        return tuple(records)

    def _append(self, record: UsageRecord) -> None:
        records = [*self.records(), record.to_mapping()]
        content = "".join(
            json.dumps(item, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
            for item in records
        )
        atomic_write_text(self.path, content)

    def log_success(
        self,
        metadata: CanonicalRequestMetadata,
        response: ApiResponseRecord,
    ) -> None:
        self._append(
            UsageRecord(
                schema_version=USAGE_SCHEMA_VERSION,
                request=metadata.to_mapping(),
                request_hash=metadata.request_hash,
                provider=metadata.provider,
                endpoint_identifier=metadata.endpoint_identifier,
                requested_model_identifier=metadata.requested_model_identifier,
                returned_model_identifier=response.returned_model_identifier,
                exact_backend_model_identifier=response.exact_backend_model_identifier,
                exact_identity_evidence_source=response.exact_identity_evidence_source,
                provider_request_id=response.provider_request_id,
                safe_provider_metadata=response.safe_metadata,
                cache_hit=response.cache_hit,
                logical_call_count=1,
                provider_call_count=response.provider_call_count,
                retry_count=response.retry_count,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                provider_cost=response.provider_cost,
                origin_provider_cost=response.origin_provider_cost,
                timestamp=response.timestamp or "unknown",
                error=None,
            )
        )

    def log_failure(
        self,
        metadata: CanonicalRequestMetadata,
        *,
        error_code: str,
        retryable: bool,
        status_code: int | None,
        retry_count: int,
        provider_call_count: int,
        latency_ms: float,
        timestamp: str,
        response: ProviderResponse | ApiResponseRecord | None = None,
    ) -> None:
        self._append(
            UsageRecord(
                schema_version=USAGE_SCHEMA_VERSION,
                request=metadata.to_mapping(),
                request_hash=metadata.request_hash,
                provider=metadata.provider,
                endpoint_identifier=metadata.endpoint_identifier,
                requested_model_identifier=metadata.requested_model_identifier,
                returned_model_identifier=(
                    response.returned_model_identifier if response is not None else None
                ),
                exact_backend_model_identifier=(
                    response.exact_backend_model_identifier
                    if response is not None
                    else None
                ),
                exact_identity_evidence_source=(
                    response.exact_identity_evidence_source
                    if response is not None
                    else None
                ),
                provider_request_id=(
                    response.provider_request_id if response is not None else None
                ),
                safe_provider_metadata=(
                    response.safe_metadata if response is not None else {}
                ),
                cache_hit=False,
                logical_call_count=1,
                provider_call_count=provider_call_count,
                retry_count=retry_count,
                input_tokens=response.input_tokens if response is not None else None,
                output_tokens=response.output_tokens if response is not None else None,
                total_tokens=response.total_tokens if response is not None else None,
                latency_ms=latency_ms,
                provider_cost=response.provider_cost if response is not None else None,
                origin_provider_cost=(
                    response.provider_cost if response is not None else None
                ),
                timestamp=timestamp,
                error={
                    "code": error_code,
                    "retryable": retryable,
                    "status_code": status_code,
                },
            )
        )

    def summarize(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "api_usage_summary_v2",
            "logical_calls": sum(int(item["logical_call_count"]) for item in records),
            "provider_calls": sum(int(item["provider_call_count"]) for item in records),
            "retries": sum(int(item["retry_count"]) for item in records),
            "cache_hits": sum(bool(item["cache_hit"]) for item in records),
            "successful_calls": sum(item["error"] is None for item in records),
            "failed_calls": sum(item["error"] is not None for item in records),
            "input_tokens": sum(
                int(item["input_tokens"] or 0)
                for item in records
                if not item["cache_hit"]
            ),
            "output_tokens": sum(
                int(item["output_tokens"] or 0)
                for item in records
                if not item["cache_hit"]
            ),
            "total_tokens": sum(
                int(item["total_tokens"] or 0)
                for item in records
                if not item["cache_hit"]
            ),
            "provider_cost": sum(
                float(item["provider_cost"] or 0.0) for item in records
            ),
        }
