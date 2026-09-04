"""Hard per-invocation provider-call budget."""

from __future__ import annotations

from threading import RLock

from surgical_agent.api.errors import ApiProviderCallBudgetError


class ProviderCallBudget:
    """Count exact transport attempts, including retries after cache misses."""

    def __init__(self, limit: int) -> None:
        if type(limit) is not int:
            raise TypeError("provider call budget limit must be a positive integer")
        if limit <= 0:
            raise ValueError("provider call budget limit must be a positive integer")
        self._limit = limit
        self._used = 0
        self._lock = RLock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._limit - self._used

    def consume(self) -> None:
        """Authorize one transport attempt, or fail before that attempt."""
        with self._lock:
            if self._used >= self._limit:
                raise ApiProviderCallBudgetError("provider call budget exhausted")
            self._used += 1
