"""Hard per-invocation provider-call budget."""

from __future__ import annotations

from surgical_agent.api.errors import ApiProviderCallBudgetError


class ProviderCallBudget:
    """Count cache misses that are authorized to reach a provider transport."""

    def __init__(self, limit: int) -> None:
        if type(limit) is not int:
            raise TypeError("provider call budget limit must be a positive integer")
        if limit <= 0:
            raise ValueError("provider call budget limit must be a positive integer")
        self._limit = limit
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._limit - self._used

    def consume(self) -> None:
        """Authorize one cache miss, or fail before any provider call."""
        if self._used >= self._limit:
            raise ApiProviderCallBudgetError("provider call budget exhausted")
        self._used += 1
