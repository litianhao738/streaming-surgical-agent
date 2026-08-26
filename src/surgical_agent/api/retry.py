"""Bounded typed retry policy for provider transports."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from surgical_agent.api.contracts import SleepFunction
from surgical_agent.api.errors import ApiCallFailure, ApiTransportError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempt_count: int
    retry_count: int


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")

    def execute(
        self,
        operation: Callable[[], T],
        *,
        sleep: SleepFunction = time.sleep,
    ) -> RetryResult[T]:
        attempts = 0
        retries = 0
        while True:
            attempts += 1
            try:
                return RetryResult(
                    value=operation(),
                    attempt_count=attempts,
                    retry_count=retries,
                )
            except ApiTransportError as exc:
                if not exc.retryable or attempts >= self.max_attempts:
                    raise ApiCallFailure(
                        exc,
                        attempt_count=attempts,
                        retry_count=retries,
                    ) from exc
                delay = min(
                    self.base_delay_seconds * (2**retries),
                    self.max_delay_seconds,
                )
                sleep(delay)
                retries += 1
