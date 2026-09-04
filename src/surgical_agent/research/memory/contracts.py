"""Reliability-aware memory records kept after atomic finalization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.runtime.state import ObservationIdentity

MemoryDestination = Literal["RELIABLE_LONG_TERM", "SHORT_TERM", "PENDING_BUFFER"]


@dataclass(frozen=True)
class MemoryEntry:
    observation: ObservationIdentity
    state: Literal["Accepted", "Verified"]
    hypothesis: InitialPrediction
    provenance: str

    def __post_init__(self) -> None:
        if self.state not in {"Accepted", "Verified"}:
            raise ValueError("trusted Memory only accepts Accepted or Verified")
        if not isinstance(self.observation, ObservationIdentity):
            raise TypeError("observation must be ObservationIdentity")
        if not isinstance(self.hypothesis, InitialPrediction):
            raise TypeError("trusted Memory requires a hypothesis")

    @property
    def destination(self) -> MemoryDestination:
        return "RELIABLE_LONG_TERM" if self.state == "Verified" else "SHORT_TERM"


@dataclass(frozen=True)
class PendingEntry:
    observation: ObservationIdentity
    hypothesis: InitialPrediction | None
    scope: str | None
    created_t: float
    last_checked_t: float | None = None
    resolution_attempts: int = 0
    max_resolution_attempts: int = 1
    expiry: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationIdentity):
            raise TypeError("observation must be ObservationIdentity")
        if self.hypothesis is not None and not isinstance(
            self.hypothesis, InitialPrediction
        ):
            raise TypeError("pending hypothesis must be InitialPrediction or None")
        if self.resolution_attempts < 0:
            raise ValueError("resolution_attempts must be non-negative")
        if self.max_resolution_attempts not in {1, 2}:
            raise ValueError("max_resolution_attempts must be 1 or 2")
        if self.resolution_attempts > self.max_resolution_attempts:
            raise ValueError("resolution attempts exceeded the bound")


@dataclass(frozen=True)
class ReliabilityMemorySnapshot:
    video_id: str
    verified: tuple[MemoryEntry, ...]
    accepted: tuple[MemoryEntry, ...]
    pending: tuple[PendingEntry, ...]

    @property
    def trusted(self) -> tuple[MemoryEntry, ...]:
        return tuple(
            sorted((*self.verified, *self.accepted), key=lambda item: item.observation)
        )

    def as_mapping(self) -> MappingProxyType:
        def payload(item: MemoryEntry) -> dict[str, object]:
            hypothesis = item.hypothesis
            return {
                "observation_id": item.observation.key,
                "frame_id": item.observation.frame_id,
                "state": item.state,
                "provenance": item.provenance,
                "instrument_ids": hypothesis.instrument_ids,
                "verb_ids": hypothesis.verb_ids,
                "target_ids": hypothesis.target_ids,
                "ivt_ids": hypothesis.triplet_ids,
                "phase_id": hypothesis.phase_id,
            }

        return MappingProxyType(
            {
                "component": "reliability_aware_memory",
                "video_id": self.video_id,
                "status": "FINALIZED_ONLY",
                "verified_events": tuple(payload(item) for item in self.verified),
                "accepted_events": tuple(payload(item) for item in self.accepted),
                "pending_ids": tuple(item.observation.key for item in self.pending),
                "trusted_source_max_frame_id": (
                    self.trusted[-1].observation.frame_id if self.trusted else None
                ),
                "source_max_frame_id": (
                    self.trusted[-1].observation.frame_id if self.trusted else None
                ),
            }
        )


__all__ = [
    "MemoryDestination",
    "MemoryEntry",
    "PendingEntry",
    "ReliabilityMemorySnapshot",
]
