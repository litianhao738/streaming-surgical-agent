"""Deterministic non-network transport for P3 tests and smoke artifacts."""

from __future__ import annotations

from datetime import UTC, datetime

from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION


class MockProviderTransport:
    provider = "mock"
    endpoint_identifier = "mock://local/p3"

    def __init__(
        self,
        *,
        returned_model_identifier: str = "mock-model-returned-v1",
        retryable_failures_before_success: int = 0,
        malformed_payload: bool = False,
        provider_cost: float | None = 0.0,
    ) -> None:
        if retryable_failures_before_success < 0:
            raise ValueError("retryable_failures_before_success must be non-negative")
        self.returned_model_identifier = returned_model_identifier
        self.retryable_failures_before_success = retryable_failures_before_success
        self.malformed_payload = malformed_payload
        self.provider_cost = provider_cost
        self.provider_call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        if request.provider != self.provider:
            raise ApiContractError("Request provider does not match mock transport")
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError("Request endpoint does not match mock transport")
        self.provider_call_count += 1
        if self.provider_call_count <= self.retryable_failures_before_success:
            raise ApiTransportError(
                "Injected transient mock failure",
                code="mock_transient",
                retryable=True,
            )
        payload = (
            {"schema_version": "malformed"}
            if self.malformed_payload
            else {
                "schema_version": P3_SMOKE_SCHEMA_VERSION,
                "message": "deterministic mock multimodal response",
                "image_observed": bool(request.images),
                "structured": True,
            }
        )
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier=self.returned_model_identifier,
            parsed_payload=payload,
            input_tokens=12,
            output_tokens=7,
            total_tokens=19,
            image_count=len(request.images),
            provider_request_id=f"mock-request-{self.provider_call_count}",
            timestamp=datetime.now(UTC).isoformat(),
            provider_cost=self.provider_cost,
            safe_metadata={"finish_reason": "stop"},
        )
