"""Typed P3 API failure taxonomy."""

from __future__ import annotations

import re


class ApiError(RuntimeError):
    """Base class for explicit API-layer failures."""

    code = "api_error"


class ApiContractError(ApiError):
    code = "contract_error"


class ApiSchemaError(ApiError):
    code = "schema_error"


class ApiCacheError(ApiError):
    code = "cache_error"


class ApiProviderCallBudgetError(ApiError):
    code = "provider_call_budget_exhausted"


class ApiTransportError(ApiError):
    """Provider/transport failure with an explicit safe category."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        del message
        if not isinstance(code, str) or re.fullmatch(r"[a-z][a-z0-9_]*", code) is None:
            raise ValueError("transport error code must be a safe category")
        if type(retryable) is not bool:
            raise TypeError("transport retryable must be a boolean")
        if status_code is not None and (
            not isinstance(status_code, int)
            or isinstance(status_code, bool)
            or not 100 <= status_code <= 599
        ):
            raise ValueError("transport status_code must be an HTTP status")
        super().__init__(f"API transport failed with category {code}")
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


class ApiCallFailure(ApiError):
    """Terminal transport outcome carrying exact attempt evidence."""

    code = "api_call_failed"

    def __init__(
        self,
        cause: ApiTransportError,
        *,
        attempt_count: int,
        retry_count: int,
    ) -> None:
        if not isinstance(cause, ApiTransportError):
            raise TypeError("API call failure cause must be a transport error")
        if (
            not isinstance(attempt_count, int)
            or isinstance(attempt_count, bool)
            or attempt_count <= 0
        ):
            raise ValueError("attempt count must be a positive integer")
        if (
            not isinstance(retry_count, int)
            or isinstance(retry_count, bool)
            or retry_count < 0
        ):
            raise ValueError("retry count must be a non-negative integer")
        if retry_count != attempt_count - 1:
            raise ValueError("retry count must equal attempt count minus one")
        super().__init__(f"API call failed with category {cause.code}")
        self.cause = cause
        self.attempt_count = attempt_count
        self.provider_call_count = attempt_count
        self.retry_count = retry_count


class ApiRetryExhausted(ApiCallFailure):
    """Deprecated compatibility type; terminal calls now raise ApiCallFailure."""
