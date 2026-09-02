"""Provider transport wrappers that enforce real-response accounting."""

from __future__ import annotations

import math

from surgical_agent.api.contracts import ApiRequest, ProviderResponse, ProviderTransport
from surgical_agent.api.errors import ApiTransportError


def _require_complete_real_accounting(response: ProviderResponse) -> None:
    token_counts = (
        response.input_tokens,
        response.output_tokens,
        response.total_tokens,
    )
    if any(type(value) is not int or value < 0 for value in token_counts):
        raise ApiTransportError(
            "real single-pass accounting is incomplete",
            code="response_usage_invalid",
            retryable=False,
        )
    # Direct OpenAI-compatible APIs commonly return token usage but no monetary
    # charge. OpenRouter does return origin cost, so retain the stronger
    # requirement there without fabricating a price for direct gateways.
    if response.provider in {"openai", "openai_compatible"}:
        return
    cost = response.provider_cost
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or cost < 0
    ):
        raise ApiTransportError(
            "real single-pass accounting is incomplete",
            code="response_usage_invalid",
            retryable=False,
        )


class CompleteAccountingTransport:
    """Fail before parsing/persistence when an origin response lacks accounting."""

    def __init__(self, transport: ProviderTransport) -> None:
        self._transport = transport
        self.provider = transport.provider
        self.endpoint_identifier = transport.endpoint_identifier

    def send(self, request: ApiRequest) -> ProviderResponse:
        response = self._transport.send(request)
        _require_complete_real_accounting(response)
        return response
