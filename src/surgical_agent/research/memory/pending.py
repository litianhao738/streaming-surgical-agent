"""Bounded causal resolution of separately persisted Pending items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.memory.contracts import (
    PendingEntry,
    ReliabilityMemorySnapshot,
)
from surgical_agent.research.outcome import FinalOutcome
from surgical_agent.research.safety import SafetyValidator
from surgical_agent.runtime.finalization import (
    AtomicFinalizationStore,
    FinalizationRecord,
)
from surgical_agent.runtime.state import ObservationIdentity


@dataclass(frozen=True)
class PendingResolutionResult:
    status: Literal["STILL_PENDING", "VERIFIED", "REJECTED"]
    hypothesis: InitialPrediction | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status == "VERIFIED" and not isinstance(
            self.hypothesis, InitialPrediction
        ):
            raise ValueError("VERIFIED resolution requires a hypothesis")
        if self.status != "VERIFIED" and self.hypothesis is not None:
            raise ValueError("only VERIFIED resolution may carry a hypothesis")
        if not self.reason:
            raise ValueError("Pending resolution requires an auditable reason")


class PendingResolutionBackend(Protocol):
    def resolve(
        self,
        *,
        item: PendingEntry,
        now: ObservationIdentity,
        memory: ReliabilityMemorySnapshot,
    ) -> PendingResolutionResult: ...


def has_meaningful_new_evidence(
    item: PendingEntry,
    now: ObservationIdentity,
    memory: ReliabilityMemorySnapshot,
) -> bool:
    """Conservative deterministic trigger; never retry merely because one frame arrived."""

    threshold_t = item.created_t if item.last_checked_t is None else item.last_checked_t
    newer = tuple(
        entry for entry in memory.trusted if entry.observation.observation_time > threshold_t
    )
    if not newer or now.observation_time <= threshold_t:
        return False
    if item.hypothesis is None:
        return True
    latest = newer[-1].hypothesis
    semantic_change = (
        latest.phase_id != item.hypothesis.phase_id
        or latest.triplet_ids != item.hypothesis.triplet_ids
    )
    return semantic_change or now.frame_id - item.observation.frame_id >= 2


class BoundedPendingResolver:
    def __init__(
        self,
        *,
        backend: PendingResolutionBackend,
        validator: SafetyValidator,
    ) -> None:
        if not hasattr(backend, "resolve") or not callable(backend.resolve):
            raise TypeError("Pending backend must implement resolve")
        if not isinstance(validator, SafetyValidator):
            raise TypeError("validator must be SafetyValidator")
        self.backend = backend
        self.validator = validator

    def resolve_eligible(
        self,
        *,
        now: ObservationIdentity,
        store: AtomicFinalizationStore,
    ) -> tuple[FinalizationRecord, ...]:
        snapshot = store.snapshot()
        if snapshot.video_id != now.video_id:
            raise ValueError("Pending resolution crossed a video boundary")
        resolved: list[FinalizationRecord] = []
        for item in snapshot.pending:
            if item.resolution_attempts >= item.max_resolution_attempts:
                continue
            if not has_meaningful_new_evidence(item, now, snapshot):
                continue
            try:
                result = self.backend.resolve(item=item, now=now, memory=snapshot)
            except Exception:  # noqa: BLE001 - failed external resolution is one attempt
                store.mark_pending_checked(item.observation.key, checked_t=now.observation_time)
                continue
            if not isinstance(result, PendingResolutionResult):
                store.mark_pending_checked(item.observation.key, checked_t=now.observation_time)
                continue
            if result.status == "STILL_PENDING":
                store.mark_pending_checked(item.observation.key, checked_t=now.observation_time)
                continue
            if result.status == "VERIFIED":
                assert result.hypothesis is not None
                if self.validator.validate(result.hypothesis):
                    store.mark_pending_checked(
                        item.observation.key,
                        checked_t=now.observation_time,
                    )
                    continue
                outcome = FinalOutcome(
                    "Verified",
                    result.hypothesis,
                    f"PENDING_RESOLUTION:{result.reason}",
                )
            else:
                outcome = FinalOutcome(
                    "Rejected",
                    None,
                    f"PENDING_RESOLUTION:{result.reason}",
                )
            resolved.append(
                store.resolve_pending(
                    item.observation.key,
                    checked_t=now.observation_time,
                    outcome=outcome,
                )
            )
        return tuple(resolved)


__all__ = [
    "BoundedPendingResolver",
    "PendingResolutionBackend",
    "PendingResolutionResult",
    "has_meaningful_new_evidence",
]
