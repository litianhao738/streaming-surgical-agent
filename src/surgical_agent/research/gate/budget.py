"""Unified bounded verification budget with mandatory priority."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock
from typing import Literal

from surgical_agent.research.gate.contracts import VerificationPriority

BudgetBucket = Literal["SAFETY_RESERVE", "OPTIONAL", "SHARED"]


@dataclass(frozen=True)
class BudgetGrant:
    granted: bool
    priority: VerificationPriority
    reason: str
    bucket: BudgetBucket | None = None


@dataclass(frozen=True)
class BudgetSnapshot:
    video_id: str
    safety_reserve_remaining: int
    optional_remaining: int
    shared_remaining: int
    charged_attempts: int


class VerificationBudgetManager:
    """Charge every logical Specialist attempt, including failed attempts."""

    def __init__(
        self,
        *,
        safety_reserve: int,
        optional_capacity: int,
        shared_capacity: int = 0,
    ) -> None:
        for name, value in (
            ("safety_reserve", safety_reserve),
            ("optional_capacity", optional_capacity),
            ("shared_capacity", shared_capacity),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._initial = (safety_reserve, optional_capacity, shared_capacity)
        self._lock = RLock()
        self._video_id: str | None = None
        self._safety = 0
        self._optional = 0
        self._shared = 0
        self._charged = 0

    def reset(self, video_id: str) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be non-empty")
        with self._lock:
            self._video_id = video_id.strip()
            self._safety, self._optional, self._shared = self._initial
            self._charged = 0

    def request(self, priority: VerificationPriority) -> BudgetGrant:
        if priority not in {"MANDATORY", "OPTIONAL"}:
            raise ValueError("priority must be MANDATORY or OPTIONAL")
        with self._lock:
            if self._video_id is None:
                raise RuntimeError("budget manager must reset before use")
            if priority == "MANDATORY":
                if self._safety > 0:
                    self._safety -= 1
                    self._charged += 1
                    return BudgetGrant(True, priority, "MANDATORY_SAFETY_RESERVE", "SAFETY_RESERVE")
                if self._shared > 0:
                    self._shared -= 1
                    self._charged += 1
                    return BudgetGrant(True, priority, "MANDATORY_SHARED_CAPACITY", "SHARED")
                return BudgetGrant(False, priority, "MANDATORY_BUDGET_EXHAUSTED")
            if self._optional > 0:
                self._optional -= 1
                self._charged += 1
                return BudgetGrant(True, priority, "OPTIONAL_CAPACITY", "OPTIONAL")
            return BudgetGrant(False, priority, "OPTIONAL_BUDGET_EXHAUSTED")

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            if self._video_id is None:
                raise RuntimeError("budget manager must reset before snapshot")
            return BudgetSnapshot(
                video_id=self._video_id,
                safety_reserve_remaining=self._safety,
                optional_remaining=self._optional,
                shared_remaining=self._shared,
                charged_attempts=self._charged,
            )

    def restore(self, snapshot: BudgetSnapshot) -> None:
        """Restore only a trusted transaction snapshot for rollback/recovery."""

        if not isinstance(snapshot, BudgetSnapshot):
            raise TypeError("snapshot must be BudgetSnapshot")
        remaining = (
            snapshot.safety_reserve_remaining,
            snapshot.optional_remaining,
            snapshot.shared_remaining,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= initial
            for value, initial in zip(remaining, self._initial, strict=True)
        ):
            raise ValueError("budget snapshot remaining counts are invalid")
        if snapshot.charged_attempts != sum(
            initial - value
            for value, initial in zip(remaining, self._initial, strict=True)
        ):
            raise ValueError("budget snapshot charged count is incoherent")
        with self._lock:
            if self._video_id not in {None, snapshot.video_id}:
                raise ValueError("budget snapshot belongs to another video")
            self._video_id = snapshot.video_id
            self._safety = snapshot.safety_reserve_remaining
            self._optional = snapshot.optional_remaining
            self._shared = snapshot.shared_remaining
            self._charged = snapshot.charged_attempts

    def restore_charged_buckets(
        self,
        video_id: str,
        buckets: Iterable[BudgetBucket],
    ) -> None:
        """Reconstruct the remaining budget from committed attempt audit rows."""

        values = tuple(buckets)
        if any(value not in {"SAFETY_RESERVE", "OPTIONAL", "SHARED"} for value in values):
            raise ValueError("committed budget audit contains an unknown bucket")
        counts = Counter(values)
        remaining = tuple(
            initial - counts[bucket]
            for initial, bucket in zip(
                self._initial,
                ("SAFETY_RESERVE", "OPTIONAL", "SHARED"),
                strict=True,
            )
        )
        self.reset(video_id)
        self.restore(
            BudgetSnapshot(
                video_id=video_id,
                safety_reserve_remaining=remaining[0],
                optional_remaining=remaining[1],
                shared_remaining=remaining[2],
                charged_attempts=len(values),
            )
        )


__all__ = [
    "BudgetGrant",
    "BudgetSnapshot",
    "VerificationBudgetManager",
]
