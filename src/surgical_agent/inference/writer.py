"""Atomic JSONL and completion-manifest writer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from surgical_agent.inference.schemas import PredictionRecord


class ArtifactWriteError(RuntimeError):
    """Raised when an output cannot be durably materialized."""


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ArtifactWriteError(f"Atomic write failed for {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


class PredictionWriter:
    """Rewrites a small P2 JSONL atomically and writes completion metadata last."""

    def __init__(self, output_dir: str | Path, *, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_id = run_id
        self.prediction_path = self.output_dir / "predictions.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self._records: list[PredictionRecord] = []
        self._sample_ids: set[tuple[str, int]] = set()
        self._finalized = False

    @property
    def records(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._records)

    def write(self, record: PredictionRecord) -> None:
        if self._finalized:
            raise ArtifactWriteError("Cannot append after writer finalization")
        if record.run_id != self.run_id:
            raise ArtifactWriteError("Prediction run_id does not match writer run_id")
        sample_id = (record.video_id, record.frame_id)
        if sample_id in self._sample_ids:
            raise ArtifactWriteError(f"Duplicate prediction sample: {sample_id}")
        candidate_records = [*self._records, record]
        content = "".join(
            json.dumps(
                asdict(item),
                default=_json_default,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
            for item in candidate_records
        )
        _atomic_write(self.prediction_path, content)
        self._records = candidate_records
        self._sample_ids.add(sample_id)

    def finalize(self, metadata: dict[str, Any]) -> Path:
        if self._finalized:
            raise ArtifactWriteError("Writer was already finalized")
        if not self._records or not self.prediction_path.is_file():
            raise ArtifactWriteError("Cannot finalize an empty prediction run")
        digest = hashlib.sha256(self.prediction_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "prediction_artifact_manifest_v1",
            "status": "COMPLETE",
            "run_id": self.run_id,
            "record_count": len(self._records),
            "predictions_file": self.prediction_path.name,
            "predictions_sha256": digest,
            "metadata": metadata,
        }
        _atomic_write(
            self.manifest_path,
            json.dumps(manifest, sort_keys=True, indent=2, default=_json_default) + "\n",
        )
        self._finalized = True
        return self.manifest_path
