"""Durable paired prediction and evidence artifact writer."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

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

_SAFE_VIDEO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_FAILURE_CATEGORY = re.compile(r"[a-z][a-z0-9_]*\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_ALLOWED_METADATA_KEYS = frozenset({"paper_metric_eligible"})


class FrameResultSink(Protocol):
    """Persistence boundary that returns only after both frame records are durable."""

    def write(
        self, prediction: PredictionRecord, evidence: EvidenceProfile
    ) -> EvidenceRecord:
        """Persist one prediction/evidence pair."""
        ...


def prediction_record_sha256(record: PredictionRecord) -> str:
    """Hash every serialized prediction field using canonical JSON."""

    return _canonical_mapping_sha256(_prediction_payload(record))


def _prediction_payload(record: PredictionRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "video_id": record.video_id,
        "frame_id": record.frame_id,
        "source_split": record.source_split,
        "causal_frame_ids": tuple(record.causal_frame_ids),
        "instrument_ids": tuple(record.instrument_ids),
        "verb_ids": tuple(record.verb_ids),
        "target_ids": tuple(record.target_ids),
        "triplet_ids": tuple(record.triplet_ids),
        "phase_id": record.phase_id,
        "granularity": record.granularity,
        "backend": record.backend,
        "gate_action": record.gate_action,
        "verification_status": record.verification_status,
        "alignment_version": record.alignment_version,
        "probabilities": {
            task: tuple(values) for task, values in record.probabilities.items()
        },
        "trace": tuple(record.trace),
        "failure_reason": record.failure_reason,
        "schema_version": record.schema_version,
        "score_semantics": record.score_semantics,
        "initial_state": record.initial_state,
        "gate_reasons": tuple(record.gate_reasons),
        "flagged_fields": tuple(record.flagged_fields),
        "repaired_fields": tuple(record.repaired_fields),
        "final_status": record.final_status,
        "memory_action": record.memory_action,
    }


def _prediction_from_payload(payload: Mapping[str, object]) -> PredictionRecord:
    values = dict(payload)
    probabilities = values["probabilities"]
    if not isinstance(probabilities, Mapping):
        raise TypeError("normalized prediction probabilities must be a mapping")
    values["probabilities"] = MappingProxyType(dict(probabilities))
    return PredictionRecord(**values)  # type: ignore[arg-type]


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
            allow_nan=False,
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
        allow_nan=False,
    ) + "\n"


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        default=json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
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
        self._require_fresh_owned_tree()
        self._prediction_payloads: list[dict[str, object]] = []
        self._evidence_records: list[EvidenceRecord] = []
        self._sample_ids: set[tuple[str, int]] = set()
        self._write_failed = False
        self._finalized = False
        self._completion_deferred = False

    @property
    def predictions(self) -> tuple[PredictionRecord, ...]:
        return tuple(
            _prediction_from_payload(payload) for payload in self._prediction_payloads
        )

    @property
    def evidence_records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence_records)

    def begin(self) -> None:
        """Persist the durable lifecycle marker before rollout work begins."""

        if self._finalized or self._prediction_payloads:
            raise ArtifactWriteError("Cannot begin an active or finalized run")
        self._write_status({"status": "INCOMPLETE"})

    def mark_failed(self, failure_category: str) -> None:
        """Persist only a sanitized category while keeping the run incomplete."""

        if _SAFE_FAILURE_CATEGORY.fullmatch(failure_category) is None:
            raise ArtifactWriteError("failure category must be a safe identifier")
        self._write_status(
            {
                "status": "INCOMPLETE",
                "failure_category": failure_category,
            }
        )

    def complete(self) -> None:
        """Publish COMPLETE after deferred rollout-level requirements succeed."""

        if not self._finalized or not self._completion_deferred:
            raise ArtifactWriteError("No deferred frame-result run is ready to complete")
        self._write_status(
            {
                "status": "COMPLETE",
                "manifest_file": self.manifest_path.name,
            }
        )
        self._completion_deferred = False

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
        prediction_path, evidence_path = self._video_paths(prediction.video_id)
        prediction_payload = _prediction_payload(prediction)

        evidence_record = EvidenceRecord(
            run_id=self.run_id,
            video_id=evidence.video_id,
            frame_id=evidence.frame_id,
            prediction_sha256=_canonical_mapping_sha256(prediction_payload),
            task_values=evidence.task_values,
            global_values=evidence.global_values,
            evidence_version=evidence.evidence_version,
        )
        video_prediction_payloads = [
            *(
                payload
                for payload in self._prediction_payloads
                if payload["video_id"] == prediction.video_id
            ),
            prediction_payload,
        ]
        video_evidence = [
            *(
                item
                for item in self._evidence_records
                if item.video_id == prediction.video_id
            ),
            evidence_record,
        ]
        prediction_content = _jsonl(video_prediction_payloads)
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
        try:
            atomic_write_text(self.status_path, _json_document(incomplete_status))
            atomic_write_text(prediction_path, prediction_content)
            atomic_write_text(evidence_path, evidence_content)
        except (ArtifactWriteError, OSError) as error:
            self._write_failed = True
            if isinstance(error, ArtifactWriteError):
                raise
            raise ArtifactWriteError(f"Frame result write failed: {error}") from error

        self._prediction_payloads.append(prediction_payload)
        self._evidence_records.append(evidence_record)
        self._sample_ids.add(sample_id)
        return evidence_record

    def finalize(
        self,
        metadata: Mapping[str, object] | None = None,
        *,
        defer_completion: bool = False,
    ) -> Path:
        if self._finalized:
            raise ArtifactWriteError("Writer was already finalized")
        if self._write_failed:
            raise ArtifactWriteError("Cannot finalize an incomplete failed run")
        if not self._prediction_payloads:
            raise ArtifactWriteError("Cannot finalize an empty frame-result run")
        normalized_metadata = _normalize_metadata(metadata)

        try:
            videos = self._verify_and_hash_persisted_records()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._write_failed = True
            raise ArtifactWriteError(f"Persisted frame-result verification failed: {error}") from error

        manifest = {
            "schema_version": "frame_result_artifact_manifest_v1",
            "status": "COMPLETE",
            "run_id": self.run_id,
            "record_count": len(self._prediction_payloads),
            "videos": videos,
            "metadata": normalized_metadata,
        }
        try:
            atomic_write_text(self.manifest_path, _json_document(manifest))
            if not defer_completion:
                self._write_status(
                    {
                        "status": "COMPLETE",
                        "manifest_file": self.manifest_path.name,
                    }
                )
        except (ArtifactWriteError, OSError) as error:
            self._write_failed = True
            if isinstance(error, ArtifactWriteError):
                raise
            raise ArtifactWriteError(f"Frame result finalization failed: {error}") from error
        self._finalized = True
        self._completion_deferred = defer_completion
        return self.manifest_path

    def _write_status(self, payload: Mapping[str, object]) -> None:
        status = {
            "schema_version": "frame_result_run_status_v1",
            **payload,
            "run_id": self.run_id,
        }
        try:
            atomic_write_text(self.status_path, _json_document(status))
        except (ArtifactWriteError, OSError) as error:
            if isinstance(error, ArtifactWriteError):
                raise
            raise ArtifactWriteError(f"Frame-result status write failed: {error}") from error

    def _require_fresh_owned_tree(self) -> None:
        stale_artifacts = [
            path
            for path in (self.status_path, self.manifest_path)
            if path.exists()
        ]
        try:
            for directory in (self.predictions_dir, self.evidence_dir):
                if directory.exists():
                    stale_artifacts.extend(directory.rglob("*.jsonl"))
        except OSError as error:
            raise ArtifactWriteError(
                f"Cannot verify fresh frame-result output tree: {error}"
            ) from error
        if stale_artifacts:
            raise ArtifactWriteError(
                "FrameResultWriter requires a fresh output tree; owned artifact "
                f"already exists: {stale_artifacts[0]}"
            )

    def _video_paths(self, video_id: str) -> tuple[Path, Path]:
        if (
            not _SAFE_VIDEO_ID.fullmatch(video_id)
            or video_id.endswith(".")
            or video_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        ):
            raise ArtifactWriteError(
                "video_id must be one safe filename component using letters, "
                "digits, dot, underscore, or hyphen"
            )
        prediction_root = self.predictions_dir.resolve()
        evidence_root = self.evidence_dir.resolve()
        if (
            prediction_root != self.output_dir / "predictions"
            or evidence_root != self.output_dir / "evidence"
            or prediction_root == evidence_root
        ):
            raise ArtifactWriteError(
                "Prediction and evidence directories must be distinct children "
                "of the output directory"
            )
        prediction_path = (prediction_root / f"{video_id}.jsonl").resolve()
        evidence_path = (evidence_root / f"{video_id}.jsonl").resolve()
        if (
            prediction_path.parent != prediction_root
            or evidence_path.parent != evidence_root
            or prediction_path == evidence_path
        ):
            raise ArtifactWriteError(
                "video_id did not resolve to distinct prediction and evidence files"
            )
        return prediction_path, evidence_path

    def _verify_and_hash_persisted_records(self) -> dict[str, dict[str, object]]:
        prediction_samples = {
            (str(payload["video_id"]), int(payload["frame_id"])): payload
            for payload in self._prediction_payloads
        }
        evidence_samples = {
            (record.video_id, record.frame_id): record
            for record in self._evidence_records
        }
        if set(prediction_samples) != set(evidence_samples):
            raise ValueError("prediction and evidence sample sets differ")
        for sample_id, prediction_payload in prediction_samples.items():
            if evidence_samples[
                sample_id
            ].prediction_sha256 != _canonical_mapping_sha256(prediction_payload):
                raise ValueError(f"prediction hash mismatch for {sample_id}")

        video_manifests: dict[str, dict[str, object]] = {}
        video_ids = sorted(
            {str(payload["video_id"]) for payload in self._prediction_payloads}
        )
        for video_id in video_ids:
            prediction_path, evidence_path = self._video_paths(video_id)
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
                payload
                for payload in self._prediction_payloads
                if payload["video_id"] == video_id
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
        json.loads(line, parse_constant=_reject_nonfinite_json)
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
    return json.loads(
        json.dumps(payload, default=json_default, allow_nan=False),
        parse_constant=_reject_nonfinite_json,
    )


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _normalize_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, bool]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ArtifactWriteError("metadata must be a mapping when provided")
    unknown_keys = set(metadata) - _ALLOWED_METADATA_KEYS
    if unknown_keys:
        raise ArtifactWriteError(
            f"metadata contains unsupported keys: {sorted(unknown_keys)!r}"
        )
    if "paper_metric_eligible" not in metadata:
        return {}
    paper_metric_eligible = metadata["paper_metric_eligible"]
    if type(paper_metric_eligible) is not bool:
        raise ArtifactWriteError("paper_metric_eligible metadata must be a real bool")
    return {"paper_metric_eligible": paper_metric_eligible}
