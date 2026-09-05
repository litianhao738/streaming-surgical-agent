"""Materialize atomic finalization state into resumable research artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from surgical_agent.artifacts.manifest import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from surgical_agent.research.temporal_events import (
    ReliabilityAwareTemplateReporter,
    TemporalEventAggregator,
)
from surgical_agent.runtime.finalization import (
    AtomicFinalizationStore,
    FinalizationRecord,
)

_SAFE_VIDEO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class FinalPipelineArtifactWriter:
    """Write one canonical JSONL row per observation and a verified manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        cell: str,
        initial_model_requested: str | None = None,
        verification_model_requested: str | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if cell not in {"A_base", "B_tracker", "C_gate", "D_full"}:
            raise ValueError("unknown formal ablation cell")
        if (initial_model_requested is None) != (
            verification_model_requested is None
        ):
            raise ValueError("artifact model identities must be recorded as a pair")
        if initial_model_requested is not None and (
            not initial_model_requested.strip()
            or not verification_model_requested
            or not verification_model_requested.strip()
        ):
            raise ValueError("artifact model identities must be non-empty text")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.records_dir = self.output_dir / "final_records"
        self.status_path = self.output_dir / "run_status.json"
        self.manifest_path = self.output_dir / "manifest.json"
        self.run_id = run_id.strip()
        self.cell = cell
        self.initial_model_requested = initial_model_requested
        self.verification_model_requested = verification_model_requested
        self._records: dict[str, dict[str, object]] = {}
        self._source_records: dict[str, FinalizationRecord] = {}

    def begin(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.status_path,
            {
                "schema_version": "final_pipeline_run_status_v1",
                "status": "INCOMPLETE",
                "run_id": self.run_id,
                "cell": self.cell,
                **self._model_identity(),
            },
        )

    def mark_failed(self, category: str) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", category) is None:
            raise ValueError("failure category must be a safe identifier")
        atomic_write_json(
            self.status_path,
            {
                "schema_version": "final_pipeline_run_status_v1",
                "status": "INCOMPLETE",
                "failure_category": category,
                "run_id": self.run_id,
                "cell": self.cell,
                **self._model_identity(),
            },
        )

    def write(self, record: FinalizationRecord) -> None:
        payload = {
            "schema_version": "final_pipeline_output_v2",
            "run_id": self.run_id,
            "cell": self.cell,
            **AtomicFinalizationStore.record_payload(record),
        }
        key = record.observation.key
        existing = self._records.get(key)
        if existing is not None and existing != payload:
            existing_outcome = existing.get("outcome")
            new_outcome = payload.get("outcome")
            audit = payload.get("audit")
            is_pending_resolution = (
                isinstance(existing_outcome, dict)
                and existing_outcome.get("kind") == "SEMANTIC"
                and existing_outcome.get("state") == "Pending"
                and isinstance(new_outcome, dict)
                and new_outcome.get("kind") == "SEMANTIC"
                and new_outcome.get("state") in {"Verified", "Rejected"}
                and isinstance(audit, dict)
                and "pending_resolution" in audit
            )
            if not is_pending_resolution:
                raise ValueError(
                    "the same observation produced conflicting output records"
                )
        self._records[key] = payload
        self._source_records[key] = record
        video_id = record.observation.video_id
        if _SAFE_VIDEO_ID.fullmatch(video_id) is None:
            raise ValueError("video_id cannot form a safe artifact filename")
        rows = sorted(
            (
                item
                for item in self._records.values()
                if item["observation"]["video_id"] == video_id  # type: ignore[index]
            ),
            key=lambda item: (
                item["observation"]["observation_time"],  # type: ignore[index]
                item["observation"]["frame_id"],  # type: ignore[index]
            ),
        )
        content = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        )
        atomic_write_text(self.records_dir / f"{video_id}.jsonl", content)

    def _model_identity(self) -> dict[str, str]:
        if self.initial_model_requested is None:
            return {}
        assert self.verification_model_requested is not None
        return {
            "initial_model_requested": self.initial_model_requested,
            "verification_model_requested": self.verification_model_requested,
        }

    def finalize(self) -> Path:
        if not self._records:
            raise ValueError("cannot finalize an empty formal Pipeline run")
        files = tuple(sorted(self.records_dir.glob("*.jsonl")))
        events = TemporalEventAggregator().aggregate(tuple(self._source_records.values()))
        report_path = self.output_dir / "reliability_event_report.json"
        atomic_write_json(
            report_path,
            ReliabilityAwareTemplateReporter().render(events),
        )
        execution_failures = sum(
            row["outcome"]["kind"] == "EXECUTION"  # type: ignore[index]
            for row in self._records.values()
        )
        manifest = {
            "schema_version": "final_pipeline_artifact_manifest_v1",
            "status": (
                "COMPLETE" if execution_failures == 0 else "COMPLETE_WITH_EXECUTION_FAILURES"
            ),
            "run_id": self.run_id,
            "cell": self.cell,
            **self._model_identity(),
            "record_count": len(self._records),
            "execution_failure_count": execution_failures,
            "paper_metric_eligible": execution_failures == 0,
            "files": [
                {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in (*files, report_path)
            ],
        }
        atomic_write_json(self.manifest_path, manifest)
        atomic_write_json(
            self.status_path,
            {
                "schema_version": "final_pipeline_run_status_v1",
                "status": manifest["status"],
                "run_id": self.run_id,
                "cell": self.cell,
                **self._model_identity(),
                "manifest": self.manifest_path.name,
            },
        )
        return self.manifest_path


__all__ = ["FinalPipelineArtifactWriter"]
