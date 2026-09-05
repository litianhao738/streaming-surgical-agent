"""Atomic durable finalization for frame audit, trusted Memory and Pending."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.memory.contracts import (
    MemoryEntry,
    PendingEntry,
    ReliabilityMemorySnapshot,
)
from surgical_agent.research.outcome import ExecutionOutcome, FinalOutcome
from surgical_agent.runtime.state import ObservationIdentity


def _hypothesis_payload(value: InitialPrediction | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "instrument_ids": list(value.instrument_ids),
        "verb_ids": list(value.verb_ids),
        "target_ids": list(value.target_ids),
        "triplet_ids": list(value.triplet_ids),
        "phase_id": value.phase_id,
        "probabilities": {
            task: list(scores) for task, scores in value.probabilities.items()
        },
        "granularity": value.granularity,
        "backend": value.backend,
        "score_semantics": value.score_semantics,
    }


def _parse_hypothesis(value: object) -> InitialPrediction | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("invalid persisted hypothesis")
    return InitialPrediction(
        instrument_ids=tuple(value["instrument_ids"]),  # type: ignore[arg-type]
        verb_ids=tuple(value["verb_ids"]),  # type: ignore[arg-type]
        target_ids=tuple(value["target_ids"]),  # type: ignore[arg-type]
        triplet_ids=tuple(value["triplet_ids"]),  # type: ignore[arg-type]
        phase_id=int(value["phase_id"]),
        probabilities={
            str(task): tuple(float(score) for score in scores)
            for task, scores in value["probabilities"].items()  # type: ignore[union-attr]
        },
        granularity=str(value["granularity"]),
        backend=str(value["backend"]),
        score_semantics=str(value["score_semantics"]),
    )


def _observation_payload(value: ObservationIdentity) -> dict[str, object]:
    return {
        "video_id": value.video_id,
        "segment_id": value.segment_id,
        "frame_id": value.frame_id,
        "observation_time": value.observation_time,
    }


def _parse_observation(value: object) -> ObservationIdentity:
    if not isinstance(value, dict):
        raise TypeError("invalid persisted observation")
    return ObservationIdentity(
        video_id=str(value["video_id"]),
        segment_id=str(value["segment_id"]),
        frame_id=int(value["frame_id"]),
        observation_time=float(value["observation_time"]),
    )


@dataclass(frozen=True)
class FinalizationRecord:
    observation: ObservationIdentity
    outcome: FinalOutcome | ExecutionOutcome
    audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationIdentity):
            raise TypeError("finalization record requires an observation")
        if not isinstance(self.outcome, (FinalOutcome, ExecutionOutcome)):
            raise TypeError("finalization record requires an outcome")
        if not isinstance(self.audit, Mapping):
            raise TypeError("finalization audit must be a mapping")
        try:
            normalized = json.loads(
                json.dumps(dict(self.audit), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("finalization audit must be finite JSON data") from exc
        forbidden = {"ground_truth", "label", "labels", "target_mask"}

        def reject_gold(value: object) -> None:
            if isinstance(value, dict):
                if forbidden & {str(key).lower() for key in value}:
                    raise ValueError("finalization audit cannot contain GT-bearing fields")
                for nested in value.values():
                    reject_gold(nested)
            elif isinstance(value, list):
                for nested in value:
                    reject_gold(nested)

        reject_gold(normalized)
        object.__setattr__(self, "audit", MappingProxyType(normalized))

    @property
    def semantic(self) -> bool:
        return isinstance(self.outcome, FinalOutcome)


class AtomicFinalizationStore:
    """Commit audit and all Memory destinations through one atomic state file."""

    schema_version = "streaming_finalization_v2"
    legacy_schema_version = "streaming_finalization_v1"

    def __init__(
        self,
        *,
        state_path: str | Path | None = None,
        state_dir: str | Path | None = None,
        max_verified: int = 64,
        max_accepted: int = 64,
        max_resolution_attempts: int = 1,
    ) -> None:
        if max_verified <= 0 or max_accepted <= 0:
            raise ValueError("trusted Memory bounds must be positive")
        if max_resolution_attempts not in {1, 2}:
            raise ValueError("max_resolution_attempts must be 1 or 2")
        if state_path is not None and state_dir is not None:
            raise ValueError("state_path and state_dir are mutually exclusive")
        self._configured_state_path = (
            None if state_path is None else Path(state_path).expanduser().resolve()
        )
        self._state_dir = (
            None if state_dir is None else Path(state_dir).expanduser().resolve()
        )
        self.state_path = self._configured_state_path
        self.max_verified = max_verified
        self.max_accepted = max_accepted
        self.max_resolution_attempts = max_resolution_attempts
        self._lock = RLock()
        self._video_id: str | None = None
        self._records: dict[str, FinalizationRecord] = {}
        self._verified: list[MemoryEntry] = []
        self._accepted: list[MemoryEntry] = []
        self._pending: dict[str, PendingEntry] = {}
        self._last_observation: ObservationIdentity | None = None

    def reset(self, video_id: str, *, recover: bool = True) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be non-empty")
        with self._lock:
            if self._state_dir is not None:
                safe_video_id = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id.strip())
                if not safe_video_id or safe_video_id in {".", ".."}:
                    raise ValueError("video_id cannot form a safe state filename")
                self.state_path = self._state_dir / f"{safe_video_id}.json"
            else:
                self.state_path = self._configured_state_path
            self._video_id = video_id.strip()
            self._records = {}
            self._verified = []
            self._accepted = []
            self._pending = {}
            self._last_observation = None
            if recover and self.state_path is not None and self.state_path.is_file():
                self._load(self.state_path)
                if self._video_id != video_id.strip():
                    raise ValueError("persisted finalization state belongs to another video")

    def snapshot(self) -> ReliabilityMemorySnapshot:
        with self._lock:
            if self._video_id is None:
                raise RuntimeError("finalization store must reset before snapshot")
            return ReliabilityMemorySnapshot(
                video_id=self._video_id,
                verified=tuple(self._verified),
                accepted=tuple(self._accepted),
                pending=tuple(
                    sorted(self._pending.values(), key=lambda item: item.observation)
                ),
            )

    def has_record(self, observation_key: str) -> bool:
        with self._lock:
            return observation_key in self._records

    def record_for(self, observation_key: str) -> FinalizationRecord | None:
        with self._lock:
            return self._records.get(observation_key)

    def mark_pending_checked(self, source_key: str, *, checked_t: float) -> PendingEntry:
        """Charge one bounded resolution attempt without promoting Pending to Memory."""

        with self._lock:
            item = self._require_pending_attempt(source_key, checked_t)
            updated = PendingEntry(
                observation=item.observation,
                hypothesis=item.hypothesis,
                scope=item.scope,
                created_t=item.created_t,
                last_checked_t=checked_t,
                resolution_attempts=item.resolution_attempts + 1,
                max_resolution_attempts=item.max_resolution_attempts,
                expiry=item.expiry,
            )
            pending = dict(self._pending)
            pending[source_key] = updated
            self._persist_current(pending=pending)
            self._pending = pending
            return updated

    def resolve_pending(
        self,
        source_key: str,
        *,
        checked_t: float,
        outcome: FinalOutcome,
    ) -> FinalizationRecord:
        """Atomically promote a resolved item to Verified or remove it as Rejected."""

        if not isinstance(outcome, FinalOutcome) or outcome.state not in {
            "Verified",
            "Rejected",
        }:
            raise ValueError("Pending resolution outcome must be Verified or Rejected")
        with self._lock:
            item = self._require_pending_attempt(source_key, checked_t)
            if outcome.state == "Verified" and outcome.hypothesis is None:
                raise ValueError("Verified Pending resolution requires a hypothesis")
            original = self._records[source_key]
            audit = dict(original.audit)
            audit["pending_resolution"] = {
                "original_state": "Pending",
                "resolved_t": checked_t,
                "attempt": item.resolution_attempts + 1,
                "state": outcome.state,
                "provenance": outcome.provenance,
            }
            resolved_record = FinalizationRecord(item.observation, outcome, audit)
            records = dict(self._records)
            records[source_key] = resolved_record
            pending = dict(self._pending)
            del pending[source_key]
            verified = list(self._verified)
            if outcome.state == "Verified":
                assert outcome.hypothesis is not None
                verified.append(
                    MemoryEntry(
                        observation=item.observation,
                        state="Verified",
                        hypothesis=outcome.hypothesis,
                        provenance=outcome.provenance,
                        task_states=outcome.task_states,
                    )
                )
                del verified[:-self.max_verified]
            self._persist_current(
                records=records,
                verified=verified,
                pending=pending,
            )
            self._records = records
            self._verified = verified
            self._pending = pending
            return resolved_record

    @property
    def records(self) -> tuple[FinalizationRecord, ...]:
        with self._lock:
            return tuple(
                sorted(self._records.values(), key=lambda item: item.observation)
            )

    @property
    def prior_trusted_hypothesis(self) -> InitialPrediction | None:
        trusted = self.snapshot().trusted
        return trusted[-1].hypothesis if trusted else None

    def commit(
        self,
        *,
        observation: ObservationIdentity,
        outcome: FinalOutcome | ExecutionOutcome,
        pending_scope: str | None = None,
        audit: Mapping[str, object] | None = None,
    ) -> FinalizationRecord:
        if not isinstance(observation, ObservationIdentity):
            raise TypeError("observation must be ObservationIdentity")
        if not isinstance(outcome, (FinalOutcome, ExecutionOutcome)):
            raise TypeError("outcome must be FinalOutcome or ExecutionOutcome")
        with self._lock:
            if self._video_id != observation.video_id:
                raise ValueError("finalization crossed a video boundary")
            existing = self._records.get(observation.key)
            record = FinalizationRecord(observation, outcome, audit or {})
            if existing is not None:
                if self.record_payload(existing) != self.record_payload(record):
                    raise ValueError("observation was already committed differently")
                return existing
            if self._last_observation is not None and (
                observation.observation_time <= self._last_observation.observation_time
            ):
                raise ValueError("finalization observations must be time ordered")

            records = dict(self._records)
            verified = list(self._verified)
            accepted = list(self._accepted)
            pending = dict(self._pending)
            records[observation.key] = record
            if isinstance(outcome, FinalOutcome):
                if outcome.state in {"Accepted", "Verified"}:
                    assert outcome.hypothesis is not None
                    entry = MemoryEntry(
                        observation=observation,
                        state=outcome.state,
                        hypothesis=outcome.hypothesis,
                        provenance=outcome.provenance,
                        task_states=outcome.task_states,
                    )
                    destination = (
                        verified
                        if entry.destination == "RELIABLE_LONG_TERM"
                        else accepted
                    )
                    destination.append(entry)
                    bound = (
                        self.max_verified
                        if entry.destination == "RELIABLE_LONG_TERM"
                        else self.max_accepted
                    )
                    del destination[:-bound]
                elif outcome.state == "Pending":
                    pending[observation.key] = PendingEntry(
                        observation=observation,
                        hypothesis=outcome.hypothesis,
                        scope=pending_scope,
                        created_t=observation.observation_time,
                        max_resolution_attempts=self.max_resolution_attempts,
                    )
                # Rejected and execution-only outcomes are audit records only.

            payload = self._state_payload(
                video_id=self._video_id,
                records=records,
                verified=verified,
                accepted=accepted,
                pending=pending,
                last_observation=observation,
            )
            if self.state_path is not None:
                self._atomic_write(payload)
            self._records = records
            self._verified = verified
            self._accepted = accepted
            self._pending = pending
            self._last_observation = observation
            return record

    def _atomic_write(self, payload: dict[str, object]) -> None:
        assert self.state_path is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            dir=self.state_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _require_pending_attempt(
        self,
        source_key: str,
        checked_t: float,
    ) -> PendingEntry:
        try:
            item = self._pending[source_key]
        except KeyError as exc:
            raise KeyError("unknown Pending observation") from exc
        if item.resolution_attempts >= item.max_resolution_attempts:
            raise RuntimeError("Pending resolution-attempt bound is exhausted")
        if checked_t <= item.created_t:
            raise ValueError("Pending resolution requires later causal evidence")
        if self._last_observation is None or checked_t > self._last_observation.observation_time:
            raise ValueError("Pending resolution cannot use uncommitted future evidence")
        return item

    def _persist_current(
        self,
        *,
        records: dict[str, FinalizationRecord] | None = None,
        verified: list[MemoryEntry] | None = None,
        pending: dict[str, PendingEntry] | None = None,
    ) -> None:
        if self._video_id is None or self._last_observation is None:
            raise RuntimeError("finalization store has no committed causal state")
        payload = self._state_payload(
            video_id=self._video_id,
            records=self._records if records is None else records,
            verified=self._verified if verified is None else verified,
            accepted=self._accepted,
            pending=self._pending if pending is None else pending,
            last_observation=self._last_observation,
        )
        if self.state_path is not None:
            self._atomic_write(payload)

    @classmethod
    def record_payload(cls, record: FinalizationRecord) -> dict[str, object]:
        """Return the canonical, path-free frame output/audit payload."""
        outcome = record.outcome
        if isinstance(outcome, FinalOutcome):
            assert outcome.task_states is not None
            encoded_outcome: dict[str, object] = {
                "kind": "SEMANTIC",
                "state": outcome.state,
                "hypothesis": _hypothesis_payload(outcome.hypothesis),
                "provenance": outcome.provenance,
                "lower_reliability": outcome.lower_reliability,
                "task_states": dict(outcome.task_states),
                "verified_tasks": list(outcome.verified_tasks),
                "checked_tasks": list(outcome.checked_tasks),
                "derived_tasks": list(outcome.derived_tasks),
            }
        else:
            encoded_outcome = {
                "kind": "EXECUTION",
                "status": outcome.status,
                "reason": outcome.reason,
            }
        return {
            "observation": _observation_payload(record.observation),
            "outcome": encoded_outcome,
            "audit": dict(record.audit),
        }

    @classmethod
    def _memory_payload(cls, item: MemoryEntry) -> dict[str, object]:
        assert item.task_states is not None
        return {
            "observation": _observation_payload(item.observation),
            "state": item.state,
            "hypothesis": _hypothesis_payload(item.hypothesis),
            "provenance": item.provenance,
            "task_states": dict(item.task_states),
            "verified_tasks": list(item.verified_tasks),
            "checked_tasks": list(item.checked_tasks),
            "derived_tasks": list(item.derived_tasks),
        }

    @classmethod
    def _pending_payload(cls, item: PendingEntry) -> dict[str, object]:
        return {
            "observation": _observation_payload(item.observation),
            "hypothesis": _hypothesis_payload(item.hypothesis),
            "scope": item.scope,
            "created_t": item.created_t,
            "last_checked_t": item.last_checked_t,
            "resolution_attempts": item.resolution_attempts,
            "max_resolution_attempts": item.max_resolution_attempts,
            "expiry": item.expiry,
        }

    @classmethod
    def _state_payload(
        cls,
        *,
        video_id: str,
        records: dict[str, FinalizationRecord],
        verified: list[MemoryEntry],
        accepted: list[MemoryEntry],
        pending: dict[str, PendingEntry],
        last_observation: ObservationIdentity,
    ) -> dict[str, object]:
        return {
            "schema_version": cls.schema_version,
            "video_id": video_id,
            "records": {
                key: cls.record_payload(value) for key, value in records.items()
            },
            "verified": [cls._memory_payload(item) for item in verified],
            "accepted": [cls._memory_payload(item) for item in accepted],
            "pending": {
                key: cls._pending_payload(value) for key, value in pending.items()
            },
            "last_observation": _observation_payload(last_observation),
        }

    def _load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid finalization state file") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") not in {
            self.schema_version,
            self.legacy_schema_version,
        }:
            raise ValueError("unsupported finalization state schema")
        legacy = payload.get("schema_version") == self.legacy_schema_version
        self._video_id = str(payload["video_id"])
        records: dict[str, FinalizationRecord] = {}
        for key, raw in payload["records"].items():
            observation = _parse_observation(raw["observation"])
            raw_outcome = raw["outcome"]
            if raw_outcome["kind"] == "SEMANTIC":
                raw_state = str(raw_outcome["state"])
                task_states = raw_outcome.get("task_states")
                if legacy and raw_state == "Verified":
                    # V1 stored one scope-level check as whole-frame Verified.
                    # Migrate it conservatively instead of trusting all five heads.
                    raw_state = "Accepted"
                    task_states = {
                        task: "Checked"
                        for task in ("instrument", "verb", "target", "ivt", "phase")
                    }
                outcome: FinalOutcome | ExecutionOutcome = FinalOutcome(
                    state=raw_state,  # type: ignore[arg-type]
                    hypothesis=_parse_hypothesis(raw_outcome["hypothesis"]),
                    provenance=raw_outcome["provenance"],
                    lower_reliability=bool(raw_outcome["lower_reliability"]),
                    task_states=task_states,
                )
            else:
                outcome = ExecutionOutcome(raw_outcome["status"], raw_outcome["reason"])
            records[str(key)] = FinalizationRecord(
                observation,
                outcome,
                raw.get("audit", {}),
            )
        self._records = records
        memory = [
            self._parse_memory(item, legacy=legacy)
            for item in (*payload["verified"], *payload["accepted"])
        ]
        self._verified = [
            item for item in memory if item.destination == "RELIABLE_LONG_TERM"
        ]
        self._accepted = [
            item for item in memory if item.destination == "SHORT_TERM"
        ]
        self._pending = {
            str(key): self._parse_pending(value)
            for key, value in payload["pending"].items()
        }
        self._last_observation = _parse_observation(payload["last_observation"])

    @staticmethod
    def _parse_memory(value: object, *, legacy: bool = False) -> MemoryEntry:
        if not isinstance(value, dict):
            raise TypeError("invalid trusted Memory entry")
        hypothesis = _parse_hypothesis(value["hypothesis"])
        assert hypothesis is not None
        raw_state = str(value["state"])
        task_states = value.get("task_states")
        if legacy and raw_state == "Verified":
            raw_state = "Accepted"
            task_states = {
                task: "Checked"
                for task in ("instrument", "verb", "target", "ivt", "phase")
            }
        return MemoryEntry(
            observation=_parse_observation(value["observation"]),
            state=raw_state,  # type: ignore[arg-type]
            hypothesis=hypothesis,
            provenance=str(value["provenance"]),
            task_states=task_states,
        )

    @staticmethod
    def _parse_pending(value: object) -> PendingEntry:
        if not isinstance(value, dict):
            raise TypeError("invalid Pending entry")
        return PendingEntry(
            observation=_parse_observation(value["observation"]),
            hypothesis=_parse_hypothesis(value["hypothesis"]),
            scope=value["scope"],
            created_t=float(value["created_t"]),
            last_checked_t=(
                None if value["last_checked_t"] is None else float(value["last_checked_t"])
            ),
            resolution_attempts=int(value["resolution_attempts"]),
            max_resolution_attempts=int(value["max_resolution_attempts"]),
            expiry=None if value["expiry"] is None else float(value["expiry"]),
        )


__all__ = ["AtomicFinalizationStore", "FinalizationRecord"]
