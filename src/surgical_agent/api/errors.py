"""Typed P3 API failure taxonomy."""

from __future__ import annotations


class ApiError(RuntimeError):
    """Base class for explicit API-layer failures."""

    code = "api_error"


class ApiContractError(ApiError):
    code = "contract_error"


class ApiSchemaError(ApiError):
    code = "schema_error"


class ApiCacheError(ApiError):
    code = "cache_error"


class ApiTransportError(ApiError):
    """Provider/transport failure with an explicit retry decision."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ApiRetryExhausted(ApiError):
    code = "retry_exhausted"

    def __init__(self, cause: ApiTransportError, *, retry_count: int) -> None:
        super().__init__(
            f"API retry budget exhausted after {retry_count} retries: {cause}"
        )
        self.cause = cause
        self.retry_count = retry_count
