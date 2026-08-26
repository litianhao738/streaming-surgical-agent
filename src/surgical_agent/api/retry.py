"""Bounded typed retry policy for provider transports."""

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt_count, int)
            or isinstance(self.attempt_count, bool)
            or self.attempt_count <= 0
        ):
            raise ValueError("attempt count must be a positive integer")
        if (
            not isinstance(self.retry_count, int)
            or isinstance(self.retry_count, bool)
            or self.retry_count < 0
        ):
            raise ValueError("retry count must be a non-negative integer")
        if self.retry_count != self.attempt_count - 1:
            raise ValueError("retry count must equal attempt count minus one")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts <= 0
        ):
            raise ValueError("max_attempts must be positive")
        for name in ("base_delay_seconds", "max_delay_seconds"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError("retry delays must be finite non-negative numbers")

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
