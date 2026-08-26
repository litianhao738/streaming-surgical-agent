"""Durable paired prediction and evidence artifact writer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.inference.writer import (
    ArtifactWriteError,
    atomic_write_text,
    json_default,
)
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    EvidenceRecord,
    EvidenceValue,
)


class FrameResultSink(Protocol):
    """Persistence boundary that returns only after both frame records are durable."""

    def write(
        self, prediction: PredictionRecord, evidence: EvidenceProfile
    ) -> EvidenceRecord:
        """Persist one prediction/evidence pair."""
        ...


def prediction_record_sha256(record: PredictionRecord) -> str:
    """Hash every serialized prediction field using canonical JSON."""

    payload = json.dumps(
        asdict(record),
        default=json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_payload(record: PredictionRecord) -> dict[str, object]:
    return asdict(record)


def _evidence_value_payload(value: EvidenceValue) -> dict[str, object]:
    return asdict(value)


def _evidence_payload(record: EvidenceRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "video_id": record.video_id,
        "frame_id": record.frame_id,
        "prediction_sha256": record.prediction_sha256,
        "task_values": {
            task: {
                name: _evidence_value_payload(value)
                for name, value in values.items()
            }
            for task, values in record.task_values.items()
        },
        "global_values": {
            name: _evidence_value_payload(value)
            for name, value in record.global_values.items()
        },
        "evidence_version": record.evidence_version,
        "schema_version": record.schema_version,
    }


def _jsonl(payloads: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(
            payload,
            default=json_default,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
        for payload in payloads
    )


def _json_document(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        default=json_default,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    ) + "\n"


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class FrameResultWriter:
    """Persist paired per-video records behind an explicit incomplete marker."""

    def __init__(self, output_dir: str | Path, *, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_id = run_id
        self.predictions_dir = self.output_dir / "predictions"
        self.evidence_dir = self.output_dir / "evidence"
        self.manifest_path = self.output_dir / "manifest.json"
        self.status_path = self.output_dir / "run_status.json"
        self._predictions: list[PredictionRecord] = []
        self._evidence_records: list[EvidenceRecord] = []
        self._sample_ids: set[tuple[str, int]] = set()
        self._write_failed = False
        self._finalized = False

    @property
    def predictions(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._predictions)

    @property
    def evidence_records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence_records)

    def write(
        self, prediction: PredictionRecord, evidence: EvidenceProfile
    ) -> EvidenceRecord:
        if self._finalized:
            raise ArtifactWriteError("Cannot append after writer finalization")
        if self._write_failed:
            raise ArtifactWriteError("Cannot append to an incomplete failed run")
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("prediction must be a PredictionRecord")
        if not isinstance(evidence, EvidenceProfile):
            raise TypeError("evidence must be an EvidenceProfile")
        if prediction.run_id != self.run_id:
            raise ArtifactWriteError("Prediction run_id does not match writer run_id")
        sample_id = (prediction.video_id, prediction.frame_id)
        if sample_id != (evidence.video_id, evidence.frame_id):
            raise ArtifactWriteError("Prediction and evidence sample identity mismatch")
        if sample_id in self._sample_ids:
            raise ArtifactWriteError(f"Duplicate frame result sample: {sample_id}")

        evidence_record = EvidenceRecord(
            run_id=self.run_id,
            video_id=evidence.video_id,
            frame_id=evidence.frame_id,
            prediction_sha256=prediction_record_sha256(prediction),
            task_values=evidence.task_values,
            global_values=evidence.global_values,
            evidence_version=evidence.evidence_version,
        )
        video_predictions = [
            *(
                item
                for item in self._predictions
                if item.video_id == prediction.video_id
            ),
            prediction,
        ]
        video_evidence = [
            *(
                item
                for item in self._evidence_records
                if item.video_id == prediction.video_id
            ),
            evidence_record,
        ]
        prediction_content = _jsonl(
            [_prediction_payload(item) for item in video_predictions]
        )
        evidence_content = _jsonl(
            [_evidence_payload(item) for item in video_evidence]
        )
        incomplete_status = {
            "schema_version": "frame_result_run_status_v1",
            "status": "INCOMPLETE",
            "run_id": self.run_id,
            "current_sample": {
                "video_id": prediction.video_id,
                "frame_id": prediction.frame_id,
            },
        }
        prediction_path = self.predictions_dir / f"{prediction.video_id}.jsonl"
        evidence_path = self.evidence_dir / f"{prediction.video_id}.jsonl"
        try:
            atomic_write_text(self.status_path, _json_document(incomplete_status))
            atomic_write_text(prediction_path, prediction_content)
            atomic_write_text(evidence_path, evidence_content)
        except (ArtifactWriteError, OSError) as error:
            self._write_failed = True
            if isinstance(error, ArtifactWriteError):
                raise
            raise ArtifactWriteError(f"Frame result write failed: {error}") from error

        self._predictions.append(prediction)
        self._evidence_records.append(evidence_record)
        self._sample_ids.add(sample_id)
        return evidence_record

    def finalize(self, metadata: dict[str, Any]) -> Path:
        if self._finalized:
            raise ArtifactWriteError("Writer was already finalized")
        if self._write_failed:
            raise ArtifactWriteError("Cannot finalize an incomplete failed run")
        if not self._predictions:
            raise ArtifactWriteError("Cannot finalize an empty frame-result run")

        try:
            videos = self._verify_and_hash_persisted_records()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._write_failed = True
            raise ArtifactWriteError(f"Persisted frame-result verification failed: {error}") from error

        manifest = {
            "schema_version": "frame_result_artifact_manifest_v1",
            "status": "COMPLETE",
            "run_id": self.run_id,
            "record_count": len(self._predictions),
            "videos": videos,
            "metadata": metadata,
        }
        complete_status = {
            "schema_version": "frame_result_run_status_v1",
            "status": "COMPLETE",
            "run_id": self.run_id,
            "manifest_file": self.manifest_path.name,
        }
        try:
            atomic_write_text(self.manifest_path, _json_document(manifest))
            atomic_write_text(self.status_path, _json_document(complete_status))
        except (ArtifactWriteError, OSError) as error:
            self._write_failed = True
            if isinstance(error, ArtifactWriteError):
                raise
            raise ArtifactWriteError(f"Frame result finalization failed: {error}") from error
        self._finalized = True
        return self.manifest_path

    def _verify_and_hash_persisted_records(self) -> dict[str, dict[str, object]]:
        prediction_samples = {
            (record.video_id, record.frame_id): record for record in self._predictions
        }
        evidence_samples = {
            (record.video_id, record.frame_id): record
            for record in self._evidence_records
        }
        if set(prediction_samples) != set(evidence_samples):
            raise ValueError("prediction and evidence sample sets differ")
        for sample_id, prediction in prediction_samples.items():
            if evidence_samples[sample_id].prediction_sha256 != prediction_record_sha256(
                prediction
            ):
                raise ValueError(f"prediction hash mismatch for {sample_id}")

        video_manifests: dict[str, dict[str, object]] = {}
        video_ids = sorted({record.video_id for record in self._predictions})
        for video_id in video_ids:
            prediction_path = self.predictions_dir / f"{video_id}.jsonl"
            evidence_path = self.evidence_dir / f"{video_id}.jsonl"
            persisted_predictions = _read_jsonl(prediction_path)
            persisted_evidence = _read_jsonl(evidence_path)
            persisted_prediction_samples = _payload_samples(persisted_predictions)
            persisted_evidence_samples = _payload_samples(persisted_evidence)
            if set(persisted_prediction_samples) != set(persisted_evidence_samples):
                raise ValueError(f"persisted sample sets differ for {video_id}")

            for sample_id, prediction_payload in persisted_prediction_samples.items():
                evidence_payload = persisted_evidence_samples[sample_id]
                if evidence_payload.get("prediction_sha256") != _canonical_mapping_sha256(
                    prediction_payload
                ):
                    raise ValueError(f"persisted prediction hash mismatch for {sample_id}")

            expected_predictions = [
                _prediction_payload(record)
                for record in self._predictions
                if record.video_id == video_id
            ]
            expected_evidence = [
                _evidence_payload(record)
                for record in self._evidence_records
                if record.video_id == video_id
            ]
            if persisted_predictions != _json_round_trip(expected_predictions):
                raise ValueError(f"persisted prediction records differ for {video_id}")
            if persisted_evidence != _json_round_trip(expected_evidence):
                raise ValueError(f"persisted evidence records differ for {video_id}")

            video_manifests[video_id] = {
                "record_count": len(persisted_predictions),
                "predictions_file": prediction_path.relative_to(
                    self.output_dir
                ).as_posix(),
                "predictions_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
                "evidence_file": evidence_path.relative_to(
                    self.output_dir
                ).as_posix(),
                "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            }
        return video_manifests


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(payload, dict) for payload in payloads):
        raise TypeError(f"JSONL records must be objects: {path}")
    return payloads


def _payload_samples(
    payloads: list[dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    samples: dict[tuple[str, int], dict[str, object]] = {}
    for payload in payloads:
        video_id = payload.get("video_id")
        frame_id = payload.get("frame_id")
        if not isinstance(video_id, str) or not isinstance(frame_id, int):
            raise TypeError("persisted sample identity has invalid types")
        sample_id = (video_id, frame_id)
        if sample_id in samples:
            raise ValueError(f"duplicate persisted sample: {sample_id}")
        samples[sample_id] = payload
    return samples


def _json_round_trip(payload: object) -> object:
    return json.loads(json.dumps(payload, default=json_default))
