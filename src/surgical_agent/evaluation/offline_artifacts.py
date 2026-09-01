"""Load and verify completed frame-result artifacts for offline evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.inference.schemas import PredictionRecord

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CANONICAL_PAPER_VIDEOS = {
    DatasetSplit.VALIDATION: ("VID110", "VID30"),
    DatasetSplit.TESTING: (
        "VID01",
        "VID06",
        "VID07",
        "VID111",
        "VID12",
        "VID25",
        "VID39",
        "VID92",
    ),
}
_PREDICTION_KEYS = frozenset(
    {
        "run_id",
        "video_id",
        "frame_id",
        "source_split",
        "causal_frame_ids",
        "instrument_ids",
        "verb_ids",
        "target_ids",
        "triplet_ids",
        "phase_id",
        "granularity",
        "backend",
        "gate_action",
        "verification_status",
        "alignment_version",
        "probabilities",
        "trace",
        "failure_reason",
        "schema_version",
        "score_semantics",
    }
)
_RELIABILITY_PREDICTION_KEYS = frozenset(
    {
        "initial_state",
        "gate_reasons",
        "flagged_fields",
        "repaired_fields",
        "final_status",
        "memory_action",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "run_id",
        "video_id",
        "frame_id",
        "prediction_sha256",
        "task_values",
        "global_values",
        "evidence_version",
        "schema_version",
    }
)


class OfflineEvaluationError(RuntimeError):
    """A persisted run is incomplete, inconsistent, or unsafe to evaluate."""


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class CompletedRun:
    run_dir: Path
    run_id: str
    mode: str
    declared_split: DatasetSplit | None
    effective_split: DatasetSplit
    video_ids: tuple[str, ...]
    frame_counts: Mapping[str, int]
    predictions: tuple[PredictionRecord, ...]
    prediction_files: Mapping[str, ArtifactFile]
    evidence_files: Mapping[str, ArtifactFile]
    input_hashes: Mapping[str, str]
    provider: str
    model_requested: str
    models_returned: tuple[str, ...]
    prompt_version: str
    response_schema_version: str
    repair_manifest_sha256: str
    alignment_versions: tuple[str, ...]
    paper_metric_eligible: bool

    @property
    def prediction_identities(self) -> tuple[tuple[str, int], ...]:
        return tuple((item.video_id, item.frame_id) for item in self.predictions)


def identity_sha256(identities: Sequence[tuple[str, int]]) -> str:
    """Hash an ordered prediction identity list using canonical JSON."""

    payload = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_completed_run(run_dir: str | Path) -> CompletedRun:
    """Verify a completed rollout and reconstruct its prediction records."""

    root = Path(run_dir).expanduser().resolve()
    status_path = root / "run_status.json"
    rollout_path = root / "dataset_rollout_artifact.json"
    status = _read_object(status_path)
    _require_keys(
        status,
        {"schema_version", "status", "manifest_file", "run_id"},
        "run status",
    )
    if status["schema_version"] != "frame_result_run_status_v1":
        raise OfflineEvaluationError("unsupported run status schema_version")
    if status["status"] != "COMPLETE":
        raise OfflineEvaluationError("run status must be COMPLETE")
    run_id = _text(status["run_id"], "run_id")
    if _SAFE_ID.fullmatch(run_id) is None:
        raise OfflineEvaluationError("run_id is not a safe identifier")
    if status["manifest_file"] != "manifest.json":
        raise OfflineEvaluationError("run status manifest_file must be manifest.json")

    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    _require_keys(
        manifest,
        {"schema_version", "status", "run_id", "record_count", "videos", "metadata"},
        "frame manifest",
    )
    if manifest["schema_version"] != "frame_result_artifact_manifest_v1":
        raise OfflineEvaluationError("unsupported frame manifest schema_version")
    if manifest["status"] != "COMPLETE" or manifest["run_id"] != run_id:
        raise OfflineEvaluationError("frame manifest does not describe this completed run")

    rollout = _read_object(rollout_path)
    _require_keys(
        rollout,
        {
            "schema_version",
            "status",
            "run_id",
            "mode",
            "split",
            "video_ids",
            "expected_frame_counts",
            "completed_frame_counts",
            "provider",
            "model_requested",
            "models_returned",
            "prompt_version",
            "response_schema_version",
            "repair_manifest_sha256",
            "alignment_versions",
            "usage",
            "cache_entry_count",
            "track20_image_uploaded",
            "paper_metric_eligible",
            "manifest_file",
        },
        "rollout artifact",
    )
    if rollout["schema_version"] != "cholectrack20_api_rollout_v1":
        raise OfflineEvaluationError("unsupported rollout artifact schema_version")
    if rollout["run_id"] != run_id or rollout["manifest_file"] != "manifest.json":
        raise OfflineEvaluationError("rollout artifact does not describe this run")

    metadata = _mapping(manifest["metadata"], "manifest metadata")
    manifest_eligible = _bool(metadata.get("paper_metric_eligible"), "paper_metric_eligible")
    rollout_eligible = _bool(rollout["paper_metric_eligible"], "paper_metric_eligible")
    if manifest_eligible != rollout_eligible:
        raise OfflineEvaluationError("paper_metric_eligible disagrees between artifacts")

    mode = _text(rollout["mode"], "mode")
    if mode not in {"engineering", "paper"}:
        raise OfflineEvaluationError("mode must be engineering or paper")
    provider = _text(rollout["provider"], "provider")
    expected_status = {
        "mock": "MOCK_COMPLETE",
        "openai": "REAL_RESPONSE_RECEIVED",
        "openrouter": "REAL_RESPONSE_RECEIVED",
    }.get(
        provider
    )
    if expected_status is None or rollout["status"] != expected_status:
        raise OfflineEvaluationError("rollout status does not match provider")
    if mode == "engineering" and rollout["split"] is not None:
        raise OfflineEvaluationError("engineering rollout split must be null")
    declared_split = None
    if mode == "paper":
        declared_split = _split(rollout["split"], "rollout split")
        if declared_split not in {DatasetSplit.VALIDATION, DatasetSplit.TESTING}:
            raise OfflineEvaluationError("paper rollout split must be validation or testing")

    video_ids = _string_tuple(rollout["video_ids"], "video_ids")
    if not video_ids or tuple(sorted(set(video_ids))) != video_ids:
        raise OfflineEvaluationError("video_ids must be sorted and unique")
    if any(_SAFE_ID.fullmatch(item) is None for item in video_ids):
        raise OfflineEvaluationError("video_ids contain an unsafe identifier")
    if mode == "engineering" and len(video_ids) != 1:
        raise OfflineEvaluationError("engineering rollout must contain one video")
    if mode == "paper" and video_ids != _CANONICAL_PAPER_VIDEOS[declared_split]:
        raise OfflineEvaluationError(
            "paper rollout video_ids do not match canonical split membership"
        )
    videos = _mapping(manifest["videos"], "manifest videos")
    if set(videos) != set(video_ids):
        raise OfflineEvaluationError("manifest and rollout video sets differ")

    expected_counts = _count_mapping(rollout["expected_frame_counts"], video_ids)
    completed_counts = _count_mapping(rollout["completed_frame_counts"], video_ids)
    if expected_counts != completed_counts:
        raise OfflineEvaluationError("rollout expected and completed frame counts differ")

    predictions: list[PredictionRecord] = []
    raw_predictions: list[dict[str, Any]] = []
    prediction_files: dict[str, ArtifactFile] = {}
    evidence_files: dict[str, ArtifactFile] = {}
    for video_id in video_ids:
        video_manifest = _mapping(videos[video_id], f"manifest video {video_id}")
        _require_keys(
            video_manifest,
            {
                "record_count",
                "predictions_file",
                "predictions_sha256",
                "evidence_file",
                "evidence_sha256",
            },
            f"manifest video {video_id}",
        )
        count = _positive_int(video_manifest["record_count"], "record_count")
        if count != expected_counts[video_id]:
            raise OfflineEvaluationError(f"frame count mismatch for {video_id}")
        prediction_file = _artifact_file(
            root,
            video_manifest["predictions_file"],
            video_manifest["predictions_sha256"],
            f"predictions/{video_id}.jsonl",
        )
        evidence_file = _artifact_file(
            root,
            video_manifest["evidence_file"],
            video_manifest["evidence_sha256"],
            f"evidence/{video_id}.jsonl",
        )
        prediction_files[video_id] = prediction_file
        evidence_files[video_id] = evidence_file
        prediction_rows = _read_jsonl(root / prediction_file.relative_path)
        if len(prediction_rows) != count:
            raise OfflineEvaluationError(f"prediction record count mismatch for {video_id}")
        prior_frame_id = -1
        for row in prediction_rows:
            prediction = _prediction(row, run_id=run_id, video_id=video_id)
            if prediction.frame_id <= prior_frame_id:
                raise OfflineEvaluationError(f"frame IDs are not increasing for {video_id}")
            prior_frame_id = prediction.frame_id
            predictions.append(prediction)
            raw_predictions.append(row)

    score_semantics = {item.score_semantics for item in predictions}
    if len(score_semantics) != 1:
        raise OfflineEvaluationError("all predictions must use one score semantics")

    raw_index = {
        (str(row["video_id"]), int(row["frame_id"])): row for row in raw_predictions
    }
    seen_evidence: set[tuple[str, int]] = set()
    for video_id in video_ids:
        rows = _read_jsonl(root / evidence_files[video_id].relative_path)
        if len(rows) != completed_counts[video_id]:
            raise OfflineEvaluationError(f"evidence record count mismatch for {video_id}")
        for row in rows:
            _require_keys(row, _EVIDENCE_KEYS, "evidence record")
            frame_id = _nonnegative_int(row["frame_id"], "evidence frame_id")
            identity = (_text(row["video_id"], "evidence video_id"), frame_id)
            if row["run_id"] != run_id or identity[0] != video_id:
                raise OfflineEvaluationError("evidence identity does not match its run file")
            if identity in seen_evidence or identity not in raw_index:
                raise OfflineEvaluationError("prediction and evidence identities differ")
            seen_evidence.add(identity)
            expected_hash = _canonical_mapping_sha256(raw_index[identity])
            if row["prediction_sha256"] != expected_hash:
                raise OfflineEvaluationError("evidence prediction SHA-256 mismatch")
            if row["schema_version"] != "evidence_record_v1":
                raise OfflineEvaluationError("unsupported evidence schema_version")
    if seen_evidence != set(raw_index):
        raise OfflineEvaluationError("prediction and evidence identities differ")

    record_count = _positive_int(manifest["record_count"], "manifest record_count")
    if record_count != len(predictions):
        raise OfflineEvaluationError("manifest record count is incorrect")
    source_splits = {item.source_split for item in predictions}
    if len(source_splits) != 1:
        raise OfflineEvaluationError("predictions mix source splits")
    effective_split = next(iter(source_splits))
    if mode == "engineering" and effective_split is not DatasetSplit.VALIDATION:
        raise OfflineEvaluationError("engineering rollout must use validation predictions")
    if declared_split is not None and effective_split is not declared_split:
        raise OfflineEvaluationError("declared and prediction source splits differ")

    alignment_versions = _string_tuple(rollout["alignment_versions"], "alignment_versions")
    if tuple(sorted({item.alignment_version for item in predictions})) != alignment_versions:
        raise OfflineEvaluationError("rollout alignment_versions do not match predictions")
    repair_sha = _digest(rollout["repair_manifest_sha256"], "repair_manifest_sha256")
    returned_models = _string_tuple(rollout["models_returned"], "models_returned")
    if not returned_models:
        raise OfflineEvaluationError("models_returned must not be empty")

    return CompletedRun(
        run_dir=root,
        run_id=run_id,
        mode=mode,
        declared_split=declared_split,
        effective_split=effective_split,
        video_ids=video_ids,
        frame_counts=MappingProxyType(dict(completed_counts)),
        predictions=tuple(predictions),
        prediction_files=MappingProxyType(prediction_files),
        evidence_files=MappingProxyType(evidence_files),
        input_hashes=MappingProxyType(
            {
                "run_status": _file_sha256(status_path),
                "frame_manifest": _file_sha256(manifest_path),
                "rollout_artifact": _file_sha256(rollout_path),
            }
        ),
        provider=provider,
        model_requested=_text(rollout["model_requested"], "model_requested"),
        models_returned=returned_models,
        prompt_version=_text(rollout["prompt_version"], "prompt_version"),
        response_schema_version=_text(
            rollout["response_schema_version"], "response_schema_version"
        ),
        repair_manifest_sha256=repair_sha,
        alignment_versions=alignment_versions,
        paper_metric_eligible=rollout_eligible,
    )


def _prediction(row: dict[str, Any], *, run_id: str, video_id: str) -> PredictionRecord:
    if set(row) not in {
        _PREDICTION_KEYS,
        _PREDICTION_KEYS | _RELIABILITY_PREDICTION_KEYS,
    }:
        raise OfflineEvaluationError(
            "prediction record fields do not match the required schema"
        )
    frame_id = _nonnegative_int(row["frame_id"], "frame_id")
    if row["run_id"] != run_id or row["video_id"] != video_id:
        raise OfflineEvaluationError("prediction identity does not match its run file")
    causal = _int_tuple(row["causal_frame_ids"], "causal_frame_ids")
    if not causal or tuple(sorted(set(causal))) != causal or causal[-1] != frame_id:
        raise OfflineEvaluationError("causal_frame_ids must be increasing and end at frame_id")
    probabilities = _mapping(row["probabilities"], "probabilities")
    if set(probabilities) != set(TASK_CLASS_COUNTS):
        raise OfflineEvaluationError("probabilities must contain exactly five task heads")
    normalized_probabilities: dict[str, tuple[float, ...]] = {}
    for task, count in TASK_CLASS_COUNTS.items():
        values = probabilities[task]
        if not isinstance(values, list) or len(values) != count:
            raise OfflineEvaluationError(f"{task} probabilities have the wrong length")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise OfflineEvaluationError(f"{task} probabilities must be finite in [0, 1]")
        normalized_probabilities[task] = tuple(float(value) for value in values)
    try:
        return PredictionRecord(
            run_id=run_id,
            video_id=video_id,
            frame_id=frame_id,
            source_split=_split(row["source_split"], "source_split"),
            causal_frame_ids=causal,
            instrument_ids=_int_tuple(row["instrument_ids"], "instrument_ids"),
            verb_ids=_int_tuple(row["verb_ids"], "verb_ids"),
            target_ids=_int_tuple(row["target_ids"], "target_ids"),
            triplet_ids=_int_tuple(row["triplet_ids"], "triplet_ids"),
            phase_id=_nonnegative_int(row["phase_id"], "phase_id"),
            granularity=_text(row["granularity"], "granularity"),
            backend=_text(row["backend"], "backend"),
            gate_action=_text(row["gate_action"], "gate_action"),
            verification_status=_text(row["verification_status"], "verification_status"),
            alignment_version=_text(row["alignment_version"], "alignment_version"),
            probabilities=MappingProxyType(normalized_probabilities),
            trace=_string_tuple(row["trace"], "trace", allow_empty=True),
            failure_reason=row["failure_reason"],
            schema_version=_text(row["schema_version"], "schema_version"),
            score_semantics=_text(row["score_semantics"], "score_semantics"),
            initial_state=(
                "Candidate"
                if "initial_state" not in row
                else _text(row["initial_state"], "initial_state")
            ),
            gate_reasons=(
                ()
                if "gate_reasons" not in row
                else _string_tuple(
                    row["gate_reasons"], "gate_reasons", allow_empty=True
                )
            ),
            flagged_fields=(
                ()
                if "flagged_fields" not in row
                else _string_tuple(
                    row["flagged_fields"], "flagged_fields", allow_empty=True
                )
            ),
            repaired_fields=(
                ()
                if "repaired_fields" not in row
                else _string_tuple(
                    row["repaired_fields"], "repaired_fields", allow_empty=True
                )
            ),
            final_status=(
                "Candidate"
                if "final_status" not in row
                else _text(row["final_status"], "final_status")
            ),
            memory_action=(
                "SKIP"
                if "memory_action" not in row
                else _text(row["memory_action"], "memory_action")
            ),
        )
    except (TypeError, ValueError) as error:
        raise OfflineEvaluationError(f"invalid prediction record: {error}") from error


def _artifact_file(
    root: Path, relative: object, digest: object, expected_relative: str
) -> ArtifactFile:
    value = _text(relative, "artifact path")
    if value != expected_relative:
        raise OfflineEvaluationError(f"artifact path must be {expected_relative}")
    sha256 = _digest(digest, "artifact SHA-256")
    path = (root / value).resolve()
    if path.parent.parent != root or not path.is_file():
        raise OfflineEvaluationError(f"artifact file is missing or outside run directory: {value}")
    if _file_sha256(path) != sha256:
        raise OfflineEvaluationError(f"artifact SHA-256 mismatch: {value}")
    return ArtifactFile(value, sha256)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise OfflineEvaluationError(f"cannot read strict JSON {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise OfflineEvaluationError(f"JSON document must be an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [
            json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            for line in lines
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise OfflineEvaluationError(f"cannot read strict JSONL {path.name}: {error}") from error
    if any(not isinstance(item, dict) for item in values):
        raise OfflineEvaluationError(f"JSONL records must be objects: {path.name}")
    return values


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _require_keys(value: Mapping[str, Any], keys: set[str] | frozenset[str], name: str) -> None:
    if set(value) != set(keys):
        raise OfflineEvaluationError(f"{name} fields do not match the required schema")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OfflineEvaluationError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OfflineEvaluationError(f"{name} must be a non-empty stripped string")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise OfflineEvaluationError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OfflineEvaluationError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise OfflineEvaluationError(f"{name} must be a positive integer")
    return result


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise OfflineEvaluationError(f"{name} must be an integer array")
    return tuple(value)


def _string_tuple(
    value: object, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise OfflineEvaluationError(f"{name} must be a string array")
    result = tuple(value)
    if not allow_empty and not result:
        raise OfflineEvaluationError(f"{name} must not be empty")
    return result


def _count_mapping(value: object, video_ids: tuple[str, ...]) -> dict[str, int]:
    mapping = _mapping(value, "frame counts")
    if set(mapping) != set(video_ids):
        raise OfflineEvaluationError("frame count video keys differ from video_ids")
    return {key: _positive_int(mapping[key], f"frame count for {key}") for key in video_ids}


def _split(value: object, name: str) -> DatasetSplit:
    if not isinstance(value, str):
        raise OfflineEvaluationError(f"{name} must be an official split string")
    try:
        return DatasetSplit(value)
    except ValueError as error:
        raise OfflineEvaluationError(f"{name} must be an official split string") from error


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OfflineEvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
