"""Cache-aware provider-neutral P3 multimodal API client."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.contracts import (
    ApiRequest,
    ApiResponseRecord,
    ProviderResponse,
    ProviderTransport,
    ResponseValidator,
    SleepFunction,
)
from surgical_agent.api.errors import (
    ApiContractError,
    ApiError,
    ApiRetryExhausted,
    ApiSchemaError,
    ApiTransportError,
)
from surgical_agent.api.request_hash import request_sha256
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

    def _record(
        self,
        request: ApiRequest,
        response: ProviderResponse,
        *,
        request_hash: str,
        retry_count: int,
        latency_ms: float,
    ) -> ApiResponseRecord:
        if response.provider != request.provider:
            raise ApiContractError("Provider response identity mismatch")
        if not response.model_identifier:
            raise ApiSchemaError("Provider response omitted the returned model field")
        if response.parsed_payload is None:
            raise ApiSchemaError("Provider response omitted a structured payload")
        self.validator(response.parsed_payload)
        return ApiResponseRecord(
            provider=response.provider,
            requested_model_identifier=request.model_identifier,
            model_identifier=response.model_identifier,
            endpoint_identifier=request.endpoint_identifier,
            request_hash=request_hash,
            raw_response=response.raw_response,
            parsed_payload=response.parsed_payload,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            image_count=response.image_count,
            latency_ms=latency_ms,
            retry_count=retry_count,
            timestamp=response.timestamp or self._now(),
            cache_hit=False,
            provider_call=True,
            provider_request_id=response.provider_request_id,
            estimated_cost=response.estimated_cost,
        )

    def call(self, request: ApiRequest) -> ApiResponseRecord:
        self._validate_transport_identity(request)
        digest = request_sha256(request)
        cache_start = perf_counter()
        try:
            cached = self.cache.get(digest)
        except ApiError as exc:
            self.usage.log_failure(
                request,
                request_hash=digest,
                error_code=exc.code,
                retry_count=0,
                provider_call_count=0,
                latency_ms=(perf_counter() - cache_start) * 1000.0,
                timestamp=self._now(),
            )
            raise
        if cached is not None:
            try:
                self.validator(cached.parsed_payload)
            except ApiError as exc:
                self.usage.log_failure(
                    request,
                    request_hash=digest,
                    error_code=exc.code,
                    retry_count=0,
                    provider_call_count=0,
                    latency_ms=(perf_counter() - cache_start) * 1000.0,
                    timestamp=self._now(),
                    returned_model_identifier=cached.model_identifier,
                    raw_response=cached.raw_response,
                    parsed_payload=cached.parsed_payload,
                )
                raise
            replay = replace(
                cached,
                cache_hit=True,
                provider_call=False,
                latency_ms=0.0,
                retry_count=0,
            )
            self.usage.log_success(request, replay)
            return replay

        start = perf_counter()
        sleep = self.sleep
        try:
            if sleep is None:
                retried = self.retry_policy.execute(lambda: self.transport.send(request))
            else:
                retried = self.retry_policy.execute(
                    lambda: self.transport.send(request), sleep=sleep
                )
        except ApiRetryExhausted as exc:
            latency_ms = (perf_counter() - start) * 1000.0
            self.usage.log_failure(
                request,
                request_hash=digest,
                error_code=exc.cause.code,
                retry_count=exc.retry_count,
                provider_call_count=exc.retry_count + 1,
                latency_ms=latency_ms,
                timestamp=self._now(),
            )
            raise
        except ApiTransportError as exc:
            latency_ms = (perf_counter() - start) * 1000.0
            self.usage.log_failure(
                request,
                request_hash=digest,
                error_code=exc.code,
                retry_count=0,
                provider_call_count=1,
                latency_ms=latency_ms,
                timestamp=self._now(),
            )
            raise

        latency_ms = (perf_counter() - start) * 1000.0
        try:
            record = self._record(
                request,
                retried.value,
                request_hash=digest,
                retry_count=retried.retry_count,
                latency_ms=latency_ms,
            )
            self.cache.put(record)
        except ApiError as exc:
            self.usage.log_failure(
                request,
                request_hash=digest,
                error_code=exc.code,
                retry_count=retried.retry_count,
                provider_call_count=retried.retry_count + 1,
                latency_ms=latency_ms,
                timestamp=self._now(),
                returned_model_identifier=retried.value.model_identifier,
                raw_response=retried.value.raw_response,
                parsed_payload=retried.value.parsed_payload,
            )
            raise
        self.usage.log_success(request, record)
        return record
