"""Formal offline frame evaluator and deterministic report writer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from surgical_agent.data.schemas import FrameSupervisionTarget
from surgical_agent.evaluation.frame_ground_truth import (
    EvaluationData,
    load_evaluation_data,
)
from surgical_agent.evaluation.frame_metrics import FrameMetricAccumulator
from surgical_agent.evaluation.offline_artifacts import (
    CompletedRun,
    OfflineEvaluationError,
    identity_sha256,
    load_completed_run,
)
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.inference.writer import ArtifactWriteError, atomic_write_text

_TASKS = ("instrument", "verb", "target", "ivt", "phase")


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    status_path: Path
    report_path: Path
    manifest_path: Path


def resolve_evaluation_output(
    run_dir: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path | None,
) -> Path:
    """Resolve one fresh output directory that cannot overlap either input."""

    run = Path(run_dir).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run.parent / f"{run.name}__evaluation"
    )
    paths = (run, dataset, destination)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _overlap(left, right):
                raise OfflineEvaluationError("evaluation paths must not overlap")
    if destination.exists():
        raise OfflineEvaluationError("evaluation output directory must be fresh")
    return destination


def align_scored_pairs(
    run: CompletedRun,
    data: EvaluationData,
) -> tuple[tuple[PredictionRecord, FrameSupervisionTarget], ...]:
    """Align every authoritative target with its exact runtime prediction."""

    prediction_identities = tuple(run.prediction_identities)
    if prediction_identities != data.runtime_identities:
        raise OfflineEvaluationError(
            "predictions do not match the reconstructed runtime selection"
        )
    prediction_by_id = {
        (prediction.video_id, prediction.frame_id): prediction
        for prediction in run.predictions
    }
    if len(prediction_by_id) != len(run.predictions):
        raise OfflineEvaluationError("duplicate prediction identities")
    runtime_set = set(data.runtime_identities)
    outside = set(data.targets) - runtime_set
    if outside:
        raise OfflineEvaluationError("a GT-bearing frame is outside the runtime selection")
    missing = set(data.targets) - set(prediction_by_id)
    if missing:
        raise OfflineEvaluationError("a GT-bearing frame has no prediction")
    pairs = tuple(
        (prediction_by_id[identity], data.targets[identity])
        for identity in data.runtime_identities
        if identity in data.targets
    )
    if not pairs:
        raise OfflineEvaluationError("the aligned scored set is empty")
    return pairs


def evaluate_and_write(
    run: CompletedRun,
    data: EvaluationData,
    destination: str | Path,
    *,
    test_gt_authorized: bool,
) -> EvaluationArtifacts:
    """Compute existing formal metrics and publish a completed evaluation."""

    output_dir = Path(destination).expanduser().resolve()
    if output_dir.exists():
        raise OfflineEvaluationError("evaluation output directory must be fresh")
    status_path = output_dir / "evaluation_status.json"
    report_path = output_dir / "evaluation_report.json"
    manifest_path = output_dir / "evaluation_manifest.json"
    artifacts = EvaluationArtifacts(output_dir, status_path, report_path, manifest_path)
    try:
        _write_json(
            status_path,
            {
                "schema_version": "offline_frame_evaluation_status_v1",
                "status": "INCOMPLETE",
                "run_id": run.run_id,
            },
        )
        pairs = align_scored_pairs(run, data)
        accumulator = FrameMetricAccumulator()
        for prediction, target in pairs:
            accumulator.update(prediction, target)
        metrics = accumulator.compute()

        scored_ids = tuple(
            (prediction.video_id, prediction.frame_id) for prediction, _ in pairs
        )
        scored_set = set(scored_ids)
        unscored_ids = tuple(
            identity for identity in data.runtime_identities if identity not in scored_set
        )
        scored_counts = _identity_counts(run.video_ids, scored_ids)
        unscored_counts = _identity_counts(run.video_ids, unscored_ids)
        reasons = _eligibility_reasons(run, data)
        report = {
            "schema_version": "offline_frame_evaluation_v1",
            "run_id": run.run_id,
            "source": {
                "mode": run.mode,
                "split": run.effective_split.value,
                "provider": run.provider,
                "model_requested": run.model_requested,
                "models_returned": run.models_returned,
                "prompt_version": run.prompt_version,
                "response_schema_version": run.response_schema_version,
                "alignment_versions": run.alignment_versions,
            },
            "evaluation_scope": (
                "engineering_partial"
                if run.mode == "engineering"
                else "paper_mode_complete"
            ),
            "paper_metric_eligible": False,
            "eligibility_reasons": reasons,
            "test_gt_authorized": bool(
                data.split.value == "testing" and test_gt_authorized
            ),
            "video_ids": run.video_ids,
            "rollout_frame_counts": dict(run.frame_counts),
            "scored_frame_counts": scored_counts,
            "unscored_prediction_counts": unscored_counts,
            "unscored_prediction_identity_sha256": identity_sha256(unscored_ids),
            "gt_task_valid_frame_counts": _task_valid_counts(run.video_ids, pairs),
            "gt_provenance": dict(data.provenance_by_video),
            "metrics": metrics,
        }
        _write_json(report_path, report)
        report_sha = _sha256(report_path)

        manifest = {
            "schema_version": "offline_frame_evaluation_manifest_v1",
            "status": "COMPLETE",
            "run_id": run.run_id,
            "report_file": report_path.name,
            "report_sha256": report_sha,
            "input_hashes": {
                "run_status_sha256": run.input_hashes["run_status"],
                "frame_manifest_sha256": run.input_hashes["frame_manifest"],
                "rollout_artifact_sha256": run.input_hashes["rollout_artifact"],
                "prediction_files": run.prediction_files,
                "evidence_files": run.evidence_files,
            },
            "dataset": {
                "repair_manifest_sha256": data.repair_manifest_sha256,
                "gt_sources": data.sources,
            },
            "selection": {
                "identity_schema_version": "ordered_video_frame_identity_v1",
                "canonical_runtime_identity_sha256": identity_sha256(
                    data.runtime_identities
                ),
                "prediction_identity_sha256": identity_sha256(
                    prediction_identities(run)
                ),
                "scored_identity_sha256": identity_sha256(scored_ids),
                "unscored_prediction_identity_sha256": identity_sha256(unscored_ids),
            },
            "prediction_schema_version": run.predictions[0].schema_version,
            "metric_schema_version": metrics.schema_version,
            "score_semantics": metrics.score_semantics,
            "paper_metric_eligible": False,
        }
        _write_json(manifest_path, manifest)
        manifest_sha = _sha256(manifest_path)
        if _sha256(report_path) != report_sha or _sha256(manifest_path) != manifest_sha:
            raise OfflineEvaluationError("persisted evaluation hash verification failed")
        _write_json(
            status_path,
            {
                "schema_version": "offline_frame_evaluation_status_v1",
                "status": "COMPLETE",
                "run_id": run.run_id,
                "report_file": report_path.name,
                "report_sha256": report_sha,
                "manifest_file": manifest_path.name,
                "manifest_sha256": manifest_sha,
            },
        )
        return artifacts
    except Exception as error:
        if status_path.exists():
            _try_mark_incomplete(status_path, run.run_id, error)
        if isinstance(error, OfflineEvaluationError):
            raise
        raise OfflineEvaluationError(f"offline evaluation failed: {error}") from error


def evaluate_completed_run(
    run_dir: str | Path,
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    authorize_test_gt_evaluation: bool = False,
) -> EvaluationArtifacts:
    """Execute the artifact-only, GT, alignment, metric, and output flow."""

    run = load_completed_run(run_dir)
    destination = resolve_evaluation_output(run.run_dir, dataset_root, output_dir)
    data = load_evaluation_data(
        run,
        dataset_root,
        authorize_test_gt_evaluation=authorize_test_gt_evaluation,
    )
    return evaluate_and_write(
        run,
        data,
        destination,
        test_gt_authorized=authorize_test_gt_evaluation,
    )


def prediction_identities(run: CompletedRun) -> tuple[tuple[str, int], ...]:
    return tuple(run.prediction_identities)


def _eligibility_reasons(run: CompletedRun, data: EvaluationData) -> tuple[str, ...]:
    reasons = {"rollout_v1_uncommitted_selection"}
    if not run.paper_metric_eligible:
        reasons.add("input_not_paper_eligible")
    if run.mode == "engineering":
        reasons.add("engineering_partial")
    if any(value == "candidate_repaired_validation" for value in data.provenance_by_video.values()):
        reasons.add("vid30_candidate_repair")
    return tuple(sorted(reasons))


def _identity_counts(
    video_ids: tuple[str, ...], identities: tuple[tuple[str, int], ...]
) -> dict[str, int]:
    return {
        video_id: sum(identity[0] == video_id for identity in identities)
        for video_id in video_ids
    }


def _task_valid_counts(
    video_ids: tuple[str, ...],
    pairs: tuple[tuple[PredictionRecord, FrameSupervisionTarget], ...],
) -> dict[str, dict[str, int]]:
    return {
        task: {
            video_id: sum(
                prediction.video_id == video_id and getattr(target.mask, task)
                for prediction, target in pairs
            )
            for video_id in video_ids
        }
        for task in _TASKS
    }


def _failure_category(error: Exception) -> str:
    message = str(error).lower()
    if "authorization" in message:
        return "test_gt_locked"
    if "selection" in message or "identit" in message:
        return "alignment_mismatch"
    if isinstance(error, ArtifactWriteError):
        return "output_failure"
    return "metric_failure"


def _try_mark_incomplete(path: Path, run_id: str, error: Exception) -> None:
    try:
        _write_json(
            path,
            {
                "schema_version": "offline_frame_evaluation_status_v1",
                "status": "INCOMPLETE",
                "run_id": run_id,
                "failure_category": _failure_category(error),
            },
        )
    except OfflineEvaluationError:
        return


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        content = json.dumps(
            _json_value(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        atomic_write_text(path, content)
    except (OSError, TypeError, ValueError, ArtifactWriteError) as error:
        raise OfflineEvaluationError("could not write canonical evaluation JSON") from error


def _json_value(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
