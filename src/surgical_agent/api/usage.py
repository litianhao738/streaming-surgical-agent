"""Atomic, strictly validated usage/error accounting for P3 calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    CanonicalRequestMetadata,
    ProviderResponse,
    _freeze_json,
    _require_count,
    _require_identity_evidence_pair,
    _require_mapping,
    _require_nonempty_string,
    _require_optional_count,
    _require_optional_nonempty_string,
    _require_optional_number,
    _require_sha256,
    freeze_safe_metadata,
    thaw_json,
)
from surgical_agent.artifacts.manifest import atomic_write_text

USAGE_SCHEMA_VERSION = "api_usage_record_v2"
_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_USAGE_FIELDS = frozenset(
    {
        "schema_version",
        "request",
        "request_hash",
        "provider",
        "endpoint_identifier",
        "requested_model_identifier",
        "returned_model_identifier",
        "exact_backend_model_identifier",
        "exact_identity_evidence_source",
        "provider_request_id",
        "safe_provider_metadata",
        "cache_hit",
        "logical_call_count",
        "provider_call_count",
        "retry_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "provider_cost",
        "origin_provider_cost",
        "timestamp",
        "error",
    }
)


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("usage timestamp must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "usage timestamp must be a timezone-aware ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("usage timestamp must be a timezone-aware ISO timestamp")


def _freeze_error(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, name="error")
    if set(mapping) != {"code", "retryable", "status_code"}:
        raise ValueError("usage error metadata has invalid fields")
    code = mapping["code"]
    if not isinstance(code, str) or _ERROR_CODE_PATTERN.fullmatch(code) is None:
        raise ValueError("usage error code must be a safe category")
    if type(mapping["retryable"]) is not bool:
        raise TypeError("usage error retryable must be boolean")
    status_code = mapping["status_code"]
    if status_code is not None and (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 100 <= status_code <= 599
    ):
        raise ValueError("usage error status_code must be an HTTP status")
    return MappingProxyType(dict(mapping))


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
        request_mapping = _require_mapping(self.request, name="request")
        metadata = CanonicalRequestMetadata.from_mapping(request_mapping)
        _require_sha256(self.request_hash, name="request_hash")
        for name in ("provider", "endpoint_identifier", "requested_model_identifier"):
            _require_nonempty_string(getattr(self, name), name=name)
        if (
            self.request_hash != metadata.request_hash
            or self.provider != metadata.provider
            or self.endpoint_identifier != metadata.endpoint_identifier
            or self.requested_model_identifier != metadata.requested_model_identifier
        ):
            raise ValueError("usage record request metadata binding mismatch")
        for name in (
            "returned_model_identifier",
            "exact_backend_model_identifier",
            "exact_identity_evidence_source",
            "provider_request_id",
        ):
            _require_optional_nonempty_string(getattr(self, name), name=name)
        _require_identity_evidence_pair(
            self.exact_backend_model_identifier,
            self.exact_identity_evidence_source,
        )
        if type(self.cache_hit) is not bool:
            raise TypeError("usage cache_hit must be boolean")
        _require_count(self.logical_call_count, name="logical_call_count")
        if self.logical_call_count != 1:
            raise ValueError("logical_call_count must equal one")
        _require_count(self.provider_call_count, name="provider_call_count")
        _require_count(self.retry_count, name="retry_count")
        if self.provider_call_count == 0:
            if self.retry_count != 0:
                raise ValueError("usage retry count is incoherent with provider calls")
        elif self.retry_count != self.provider_call_count - 1:
            raise ValueError("usage retry count is incoherent with provider calls")
        if self.cache_hit and self.provider_call_count != 0:
            raise ValueError("cache-hit usage cannot contain current provider calls")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            _require_optional_count(getattr(self, name), name=name)
        for name in ("latency_ms", "provider_cost", "origin_provider_cost"):
            _require_optional_number(getattr(self, name), name=name)
        if self.provider_call_count == 0 and self.provider_cost not in (None, 0.0):
            raise ValueError(
                "usage with no provider calls cannot have current provider cost"
            )
        if (
            self.provider_call_count > 0
            and self.origin_provider_cost != self.provider_cost
        ):
            raise ValueError("origin provider cost must match the current origin call")
        if self.cache_hit and self.provider_cost != 0.0:
            raise ValueError("cache-hit usage must have zero current provider cost")
        if self.cache_hit and self.latency_ms != 0.0:
            raise ValueError("cache-hit usage must have zero current latency")
        _validate_timestamp(self.timestamp)
        safe_metadata = freeze_safe_metadata(
            self.safe_provider_metadata,
            path="safe_provider_metadata",
        )
        error = _freeze_error(self.error)
        object.__setattr__(
            self,
            "request",
            _freeze_json(metadata.to_mapping(), path="request"),
        )
        object.__setattr__(self, "safe_provider_metadata", safe_metadata)
        object.__setattr__(self, "error", error)

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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UsageRecord:
        mapping = _require_mapping(value, name="usage record")
        if set(mapping) != _USAGE_FIELDS:
            raise ValueError("usage record has invalid fields")
        return cls(**{name: mapping[name] for name in sorted(_USAGE_FIELDS)})


class UsageLedger:
    """Small auditable ledger rewritten atomically after every logical call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def _typed_records(self) -> tuple[UsageRecord, ...]:
        if not self.path.is_file():
            return ()
        records: list[UsageRecord] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
                record = UsageRecord.from_mapping(value)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid usage record at line {line_number}") from exc
            records.append(record)
        return tuple(records)

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_mapping() for record in self._typed_records())

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
                timestamp=response.timestamp or datetime.now(UTC).isoformat(),
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
        is_record = isinstance(response, ApiResponseRecord)
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
                cache_hit=response.cache_hit if is_record else False,
                logical_call_count=1,
                provider_call_count=provider_call_count,
                retry_count=retry_count,
                input_tokens=response.input_tokens if response is not None else None,
                output_tokens=response.output_tokens if response is not None else None,
                total_tokens=response.total_tokens if response is not None else None,
                latency_ms=latency_ms,
                provider_cost=response.provider_cost if response is not None else None,
                origin_provider_cost=(
                    response.origin_provider_cost
                    if is_record
                    else response.provider_cost
                    if response is not None
                    else None
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
        records = self._typed_records()
        return {
            "schema_version": "api_usage_summary_v2",
            "logical_calls": sum(item.logical_call_count for item in records),
            "provider_calls": sum(item.provider_call_count for item in records),
            "retries": sum(item.retry_count for item in records),
            "cache_hits": sum(item.cache_hit for item in records),
            "successful_calls": sum(item.error is None for item in records),
            "failed_calls": sum(item.error is not None for item in records),
            "input_tokens": sum(
                item.input_tokens or 0
                for item in records
                if item.provider_call_count > 0
            ),
            "output_tokens": sum(
                item.output_tokens or 0
                for item in records
                if item.provider_call_count > 0
            ),
            "total_tokens": sum(
                item.total_tokens or 0
                for item in records
                if item.provider_call_count > 0
            ),
            "provider_cost": sum(
                ((item.provider_cost or 0.0) for item in records),
                0.0,
            ),
        }
