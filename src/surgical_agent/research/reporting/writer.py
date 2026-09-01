"""Atomic JSONL writer for event report artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from surgical_agent.inference.writer import ArtifactWriteError, atomic_write_text
from surgical_agent.research.reporting.contracts import ReportRecord


class EventReportWriter:
    """Rewrite the complete small report set and publish its manifest last."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.report_path = self.output_dir / "event_reports.jsonl"
        self.manifest_path = self.output_dir / "event_report_manifest.json"
        self._records: list[ReportRecord] = []
        self._report_ids: set[str] = set()
        self._finalized = False

    @property
    def records(self) -> tuple[ReportRecord, ...]:
        return tuple(self._records)

    def write(self, record: ReportRecord) -> None:
        if self._finalized:
            raise ArtifactWriteError("Cannot append after event report finalization")
        if not isinstance(record, ReportRecord):
            raise TypeError("event report writer accepts only ReportRecord values")
        if record.report_id in self._report_ids:
            raise ArtifactWriteError(f"Duplicate report ID: {record.report_id}")
        candidate = (*self._records, record)
        content = "".join(
            json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=True) + "\n"
            for item in candidate
        )
        atomic_write_text(self.report_path, content)
        self._records.append(record)
        self._report_ids.add(record.report_id)

    def write_many(self, records: tuple[ReportRecord, ...]) -> None:
        for record in records:
            self.write(record)

    def finalize(self) -> Path:
        if self._finalized:
            raise ArtifactWriteError("Event report writer was already finalized")
        if not self._records:
            atomic_write_text(self.report_path, "")
        digest = hashlib.sha256(self.report_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "event_report_manifest_v1",
            "status": "COMPLETE",
            "record_count": len(self._records),
            "event_reports_file": self.report_path.name,
            "event_reports_sha256": digest,
        }
        atomic_write_text(
            self.manifest_path,
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        )
        self._finalized = True
        return self.manifest_path


__all__ = ["EventReportWriter"]
