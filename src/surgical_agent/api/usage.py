"""Atomic JSONL usage/error accounting for logical and provider calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from surgical_agent.api.contracts import ApiRequest, ApiResponseRecord
from surgical_agent.artifacts.manifest import atomic_write_text

USAGE_SCHEMA_VERSION = "api_usage_record_v1"


@dataclass(frozen=True)
class UsageRecord:
    schema_version: str
    request_hash: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    returned_model_identifier: str | None
    successful: bool
    cache_hit: bool
    logical_call_count: int
    provider_call_count: int
    input_tokens: int | None
    output_tokens: int | None
    image_count: int
    latency_ms: float | None
    retry_count: int
    estimated_cost: float | None
    error_code: str | None
    error_raw_response: str | None
    error_parsed_payload: Mapping[str, Any] | None
    timestamp: str | None


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
        records = [*self.records(), asdict(record)]
        content = "".join(
            json.dumps(item, sort_keys=True, ensure_ascii=True) + "\n"
            for item in records
        )
        atomic_write_text(self.path, content)

    def log_success(self, request: ApiRequest, response: ApiResponseRecord) -> None:
        provider_call_count = 0 if response.cache_hit else response.retry_count + 1
        self._append(
            UsageRecord(
                schema_version=USAGE_SCHEMA_VERSION,
                request_hash=response.request_hash,
                provider=request.provider,
                endpoint_identifier=request.endpoint_identifier,
                requested_model_identifier=request.model_identifier,
                returned_model_identifier=response.model_identifier,
                successful=True,
                cache_hit=response.cache_hit,
                logical_call_count=1,
                provider_call_count=provider_call_count,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                image_count=len(request.images),
                latency_ms=response.latency_ms,
                retry_count=response.retry_count,
                estimated_cost=response.estimated_cost,
                error_code=None,
                error_raw_response=None,
                error_parsed_payload=None,
                timestamp=response.timestamp,
            )
        )

    def log_failure(
        self,
        request: ApiRequest,
        *,
        request_hash: str,
        error_code: str,
        retry_count: int,
        provider_call_count: int,
        latency_ms: float,
        timestamp: str,
        returned_model_identifier: str | None = None,
        raw_response: str | None = None,
        parsed_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._append(
            UsageRecord(
                schema_version=USAGE_SCHEMA_VERSION,
                request_hash=request_hash,
                provider=request.provider,
                endpoint_identifier=request.endpoint_identifier,
                requested_model_identifier=request.model_identifier,
                returned_model_identifier=returned_model_identifier,
                successful=False,
                cache_hit=False,
                logical_call_count=1,
                provider_call_count=provider_call_count,
                input_tokens=None,
                output_tokens=None,
                image_count=len(request.images),
                latency_ms=latency_ms,
                retry_count=retry_count,
                estimated_cost=None,
                error_code=error_code,
                error_raw_response=raw_response,
                error_parsed_payload=parsed_payload,
                timestamp=timestamp,
            )
        )

    def summarize(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "api_usage_summary_v1",
            "logical_calls": sum(int(item["logical_call_count"]) for item in records),
            "provider_calls": sum(int(item["provider_call_count"]) for item in records),
            "cache_hits": sum(bool(item["cache_hit"]) for item in records),
            "successful_calls": sum(bool(item["successful"]) for item in records),
            "failed_calls": sum(not bool(item["successful"]) for item in records),
            "input_tokens": sum(
                int(item["input_tokens"] or 0) for item in records if not item["cache_hit"]
            ),
            "output_tokens": sum(
                int(item["output_tokens"] or 0) for item in records if not item["cache_hit"]
            ),
        }
