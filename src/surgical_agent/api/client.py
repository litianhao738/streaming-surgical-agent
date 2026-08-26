"""Cache-aware provider-neutral P3 multimodal API client."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.contracts import (
    ApiRequest,
    ApiResponseRecord,
    CanonicalRequestMetadata,
    ProviderResponse,
    ProviderTransport,
    ResponseValidator,
    SleepFunction,
)
from surgical_agent.api.errors import (
    ApiCallFailure,
    ApiContractError,
    ApiError,
    ApiSchemaError,
)
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger


class CachedMultimodalApiClient:
    """One client shared by future perception and verification roles."""

    def __init__(
        self,
        *,
        transport: ProviderTransport,
        cache: FileApiCache,
        usage: UsageLedger,
        validator: ResponseValidator,
        retry_policy: RetryPolicy | None = None,
        sleep: SleepFunction | None = None,
    ) -> None:
        self.transport = transport
        self.cache = cache
        self.usage = usage
        self.validator = validator
        self.retry_policy = retry_policy or RetryPolicy()
        self.sleep = sleep

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _validate_transport_identity(self, request: ApiRequest) -> None:
        if request.provider != self.transport.provider:
            raise ApiContractError("Request provider does not match selected transport")
        if request.endpoint_identifier != self.transport.endpoint_identifier:
            raise ApiContractError("Request endpoint does not match selected transport")

    def _validate_payload(self, response: ProviderResponse | ApiResponseRecord) -> None:
        try:
            self.validator(response.parsed_payload)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiSchemaError("P3 response validator failed") from exc

    def _record(
        self,
        request: ApiRequest,
        response: ProviderResponse,
        *,
        metadata: CanonicalRequestMetadata,
        retry_count: int,
        provider_call_count: int,
        latency_ms: float,
    ) -> ApiResponseRecord:
        if response.provider != request.provider:
            raise ApiContractError("Provider response identity mismatch")
        self._validate_payload(response)
        return ApiResponseRecord(
            provider=response.provider,
            endpoint_identifier=request.endpoint_identifier,
            request_hash=metadata.request_hash,
            requested_model_identifier=request.model_identifier,
            returned_model_identifier=response.returned_model_identifier,
            parsed_payload=response.parsed_payload,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            image_count=response.image_count,
            latency_ms=latency_ms,
            retry_count=retry_count,
            provider_call_count=provider_call_count,
            timestamp=response.timestamp or self._now(),
            cache_hit=False,
            provider_request_id=response.provider_request_id,
            provider_cost=response.provider_cost,
            origin_provider_cost=response.provider_cost,
            exact_backend_model_identifier=response.exact_backend_model_identifier,
            exact_identity_evidence_source=response.exact_identity_evidence_source,
            safe_metadata=response.safe_metadata,
        )

    def _log_failure(
        self,
        metadata: CanonicalRequestMetadata,
        *,
        error_code: str,
        retryable: bool = False,
        status_code: int | None = None,
        retry_count: int,
        provider_call_count: int,
        latency_ms: float,
        response: ProviderResponse | ApiResponseRecord | None = None,
    ) -> None:
        self.usage.log_failure(
            metadata,
            error_code=error_code,
            retryable=retryable,
            status_code=status_code,
            retry_count=retry_count,
            provider_call_count=provider_call_count,
            latency_ms=latency_ms,
            timestamp=self._now(),
            response=response,
        )

    def call(self, request: ApiRequest) -> ApiResponseRecord:
        self._validate_transport_identity(request)
        metadata = canonical_request_metadata(request)
        cache_start = perf_counter()
        try:
            cached = self.cache.get(metadata)
        except ApiError as exc:
            self._log_failure(
                metadata,
                error_code=exc.code,
                retry_count=0,
                provider_call_count=0,
                latency_ms=(perf_counter() - cache_start) * 1000.0,
            )
            raise
        if cached is not None:
            try:
                self._validate_payload(cached)
            except ApiError as exc:
                failed_replay = replace(
                    cached,
                    cache_hit=True,
                    provider_call_count=0,
                    retry_count=0,
                    latency_ms=0.0,
                    provider_cost=0.0,
                    origin_provider_cost=(
                        cached.origin_provider_cost
                        if cached.origin_provider_cost is not None
                        else cached.provider_cost
                    ),
                )
                self._log_failure(
                    metadata,
                    error_code=exc.code,
                    retry_count=0,
                    provider_call_count=0,
                    latency_ms=0.0,
                    response=failed_replay,
                )
                raise
            replay = replace(
                cached,
                cache_hit=True,
                provider_call_count=0,
                retry_count=0,
                latency_ms=0.0,
                provider_cost=0.0,
                origin_provider_cost=(
                    cached.origin_provider_cost
                    if cached.origin_provider_cost is not None
                    else cached.provider_cost
                ),
            )
            self.usage.log_success(metadata, replay)
            return replay

        start = perf_counter()
        try:
            if self.sleep is None:
                retried = self.retry_policy.execute(
                    lambda: self.transport.send(request)
                )
            else:
                retried = self.retry_policy.execute(
                    lambda: self.transport.send(request), sleep=self.sleep
                )
        except ApiCallFailure as exc:
            self._log_failure(
                metadata,
                error_code=exc.cause.code,
                retryable=exc.cause.retryable,
                status_code=exc.cause.status_code,
                retry_count=exc.retry_count,
                provider_call_count=exc.attempt_count,
                latency_ms=(perf_counter() - start) * 1000.0,
            )
            raise

        latency_ms = (perf_counter() - start) * 1000.0
        try:
            record = self._record(
                request,
                retried.value,
                metadata=metadata,
                retry_count=retried.retry_count,
                provider_call_count=retried.attempt_count,
                latency_ms=latency_ms,
            )
            self.cache.put(metadata, record)
        except ApiError as exc:
            self._log_failure(
                metadata,
                error_code=exc.code,
                retry_count=retried.retry_count,
                provider_call_count=retried.attempt_count,
                latency_ms=latency_ms,
                response=retried.value,
            )
            raise
        self.usage.log_success(metadata, record)
        return record
