"""Leakage-audited held-out detection evaluation for Tracker OOF artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.config import (
    TrackerTrainingConfig,
    load_tracker_training_config,
)
from surgical_agent.tracking.detector import calculate_detection_metrics
from surgical_agent.tracking.oof_index import TrackerOOFIndex, load_tracker_oof_index
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider
from surgical_agent.tracking.training_data import (
    deterministic_video_folds,
    supervision_qualified_training_video_ids,
)

TRACKER_OOF_EVALUATION_SCHEMA_VERSION = "tracker_oof_heldout_evaluation_v1"
TRACKER_TRAINING_MANIFEST_SCHEMA_VERSION = "predicted_tracker_training_manifest_v1"
VID31_UNSCORED_REASON = "NO_INSTANCE_LEVEL_BOUNDING_BOX_GROUND_TRUTH"
_FOLD_PATTERN = re.compile(r"fold_(\d+)")


@dataclass(frozen=True)
class _ScoredVideo:
    predictions: tuple[tuple[tuple[int, tuple[float, float, float, float], float], ...], ...]
    targets: tuple[tuple[tuple[int, tuple[float, float, float, float]], ...], ...]
    prediction_count: int
    ground_truth_count: int

    @property
    def frame_count(self) -> int:
        return len(self.targets)


def _json_object(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return value


def _text_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{name} must be a list of non-empty video IDs")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate video IDs")
    return result


def _provider_for_artifact(
    index: TrackerOOFIndex,
    artifact: Path,
    mapped_video_ids: tuple[str, ...],
) -> PrecomputedPredictedTrackProvider:
    if not mapped_video_ids:
        raise ValueError("an indexed Tracker artifact has no mapped videos")
    provider = index.provider_for(mapped_video_ids[0])
    if index.video_to_artifact[mapped_video_ids[0]] != artifact:
        raise RuntimeError("Tracker OOF artifact grouping changed during evaluation")
    if provider.available_video_ids != tuple(sorted(mapped_video_ids)):
        raise ValueError(
            "Tracker OOF artifact scope does not exactly match its index mapping"
        )
    return provider


def _verify_provider_provenance(
    provider: PrecomputedPredictedTrackProvider,
    *,
    config_sha256: str,
    repair_manifest_sha256: str,
) -> None:
    if provider.inference_config_sha256 != config_sha256:
        raise ValueError("Tracker OOF artifact was produced with a different config")
    if provider.dataset_repair_manifest_sha256 != repair_manifest_sha256:
        raise ValueError(
            "Tracker OOF artifact was produced from a different repair manifest"
        )


def _verify_checkpoint(
    directory: Path,
    manifest: Mapping[str, object],
    provider: PrecomputedPredictedTrackProvider,
) -> str:
    checkpoint_digest = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_digest, str) or len(checkpoint_digest) != 64:
        raise ValueError("Tracker training manifest has no valid checkpoint digest")
    if checkpoint_digest != provider.checkpoint_sha256:
        raise ValueError("prediction artifact and training manifest use different checkpoints")
    checkpoint = directory / "checkpoint.pt"
    if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_digest:
        raise ValueError("Tracker checkpoint is missing or digest-invalid")
    return checkpoint_digest


def _audit_fold(
    *,
    fold_index: int,
    artifact: Path,
    held_out: tuple[str, ...],
    qualified: tuple[str, ...],
    expected_held_out: tuple[str, ...],
    provider: PrecomputedPredictedTrackProvider,
    config: TrackerTrainingConfig,
    config_sha256: str,
    repair_manifest_sha256: str,
) -> dict[str, object]:
    if held_out != tuple(sorted(expected_held_out)):
        raise ValueError(f"fold_{fold_index} does not match deterministic video folds")
    manifest_path = artifact.parent / "training_manifest.json"
    manifest = _json_object(manifest_path, name=f"fold_{fold_index} training manifest")
    if manifest.get("schema_version") != TRACKER_TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"fold_{fold_index} has an unsupported training manifest")
    if manifest.get("mode") != "oof":
        raise ValueError(f"fold_{fold_index} was not trained in OOF mode")
    training = _text_tuple(
        manifest.get("training_video_ids"), name=f"fold_{fold_index}.training_video_ids"
    )
    excluded = _text_tuple(
        manifest.get("excluded_video_ids"), name=f"fold_{fold_index}.excluded_video_ids"
    )
    if tuple(sorted(excluded)) != held_out:
        raise ValueError(f"fold_{fold_index} manifest does not exclude its held-out videos")
    expected_training = tuple(video_id for video_id in qualified if video_id not in held_out)
    if tuple(sorted(training)) != tuple(sorted(expected_training)):
        raise ValueError(f"fold_{fold_index} training set violates the OOF partition")
    if set(training) & set(held_out):
        raise ValueError(f"fold_{fold_index} leaked a held-out video into training")
    if manifest.get("tracker_config_sha256") != config_sha256:
        raise ValueError(f"fold_{fold_index} manifest uses a different Tracker config")
    if manifest.get("dataset_repair_manifest_sha256") != repair_manifest_sha256:
        raise ValueError(f"fold_{fold_index} manifest uses a different repair manifest")
    if manifest.get("config") != asdict(config):
        raise ValueError(f"fold_{fold_index} embedded config does not match the frozen config")
    checkpoint_digest = _verify_checkpoint(artifact.parent, manifest, provider)
    return {
        "fold_index": fold_index,
        "training_video_ids": list(training),
        "held_out_video_ids": list(held_out),
        "training_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_digest,
        "held_out_excluded_from_optimizer": True,
    }


def _audit_vid31(
    *,
    index: TrackerOOFIndex,
    artifact: Path,
    provider: PrecomputedPredictedTrackProvider,
    qualified: tuple[str, ...],
    repair_manifest_sha256: str,
) -> dict[str, object]:
    if provider.available_video_ids != ("VID31",):
        raise ValueError("the VID31 prediction artifact must contain only VID31")
    full_directory = index.path.parent.parent / "full"
    manifest_path = full_directory / "training_manifest.json"
    manifest = _json_object(manifest_path, name="full Tracker training manifest")
    if manifest.get("schema_version") != TRACKER_TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("full Tracker has an unsupported training manifest")
    if manifest.get("mode") != "full":
        raise ValueError("VID31 must be predicted by the completed full Tracker")
    training = _text_tuple(
        manifest.get("training_video_ids"), name="full.training_video_ids"
    )
    excluded = _text_tuple(
        manifest.get("excluded_video_ids"), name="full.excluded_video_ids"
    )
    if tuple(sorted(training)) != tuple(sorted(qualified)) or excluded:
        raise ValueError("full Tracker training partition does not match qualified videos")
    if "VID31" in training:
        raise ValueError("full Tracker illegally consumed VID31 instance supervision")
    if manifest.get("dataset_repair_manifest_sha256") != repair_manifest_sha256:
        raise ValueError("full Tracker manifest uses a different repair manifest")
    checkpoint_digest = _verify_checkpoint(full_directory, manifest, provider)
    return {
        "source": "FULL_CHECKPOINT_WITHOUT_VID31_INSTANCE_SUPERVISION",
        "training_video_ids": list(training),
        "training_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_digest,
        "vid31_excluded_from_optimizer": True,
    }


def _runtime_coverage(
    adapter: Any,
    *,
    video_id: str,
    provider: PrecomputedPredictedTrackProvider,
) -> tuple[object, dict[int, tuple[tuple[int, tuple[float, float, float, float], float], ...]]]:
    video = provider.video(video_id)
    if video.source_split is not DatasetSplit.TRAINING:
        raise ValueError(f"{video_id} OOF predictions are not marked as Training split")
    expected = tuple(
        sample.target_frame_id for sample in adapter.iter_inference_video(video_id)
    )
    observed = tuple(frame.frame_id for frame in video.frames)
    if not expected:
        raise ValueError(f"{video_id} has no runtime frames")
    if observed != expected:
        missing = len(set(expected) - set(observed))
        extra = len(set(observed) - set(expected))
        raise ValueError(
            f"{video_id} prediction coverage is not exact: missing={missing}, extra={extra}"
        )
    predictions = {
        frame.frame_id: tuple(
            (track.instrument_id, track.bbox_tlwh, track.score)
            for track in frame.tracks
        )
        for frame in video.frames
    }
    return video, predictions


def _score_video(
    adapter: Any,
    *,
    video_id: str,
    predictions_by_frame: Mapping[
        int, tuple[tuple[int, tuple[float, float, float, float], float], ...]
    ],
) -> _ScoredVideo:
    predictions: list[
        tuple[tuple[int, tuple[float, float, float, float], float], ...]
    ] = []
    targets: list[tuple[tuple[int, tuple[float, float, float, float]], ...]] = []
    seen: set[int] = set()
    for resolved in adapter.iter_video(video_id):
        frame_id = resolved.inference.target_frame_id
        if frame_id in seen:
            raise ValueError(f"{video_id} evaluation contains duplicate frame IDs")
        seen.add(frame_id)
        evaluation = resolved.evaluation
        if evaluation is None or not evaluation.instance_supervision_available:
            raise ValueError(f"{video_id} lacks required instance-level evaluation GT")
        if evaluation.video_id != video_id or evaluation.frame_id != frame_id:
            raise ValueError(f"{video_id} evaluation GT is misaligned")
        try:
            frame_predictions = predictions_by_frame[frame_id]
        except KeyError as exc:
            raise ValueError(f"{video_id} has no prediction for scored frame {frame_id}") from exc
        frame_targets = tuple(
            (
                instance.instrument_id,
                (
                    instance.bbox.x,
                    instance.bbox.y,
                    instance.bbox.width,
                    instance.bbox.height,
                ),
            )
            for instance in evaluation.instances
            if instance.mask.instrument
            and instance.bbox.has_positive_extent
            and instance.bbox.is_inside_unit_frame
        )
        predictions.append(frame_predictions)
        targets.append(frame_targets)
    if not targets:
        raise ValueError(f"{video_id} has no instance-supervised evaluation frames")
    frozen_predictions = tuple(predictions)
    frozen_targets = tuple(targets)
    return _ScoredVideo(
        predictions=frozen_predictions,
        targets=frozen_targets,
        prediction_count=sum(len(frame) for frame in frozen_predictions),
        ground_truth_count=sum(len(frame) for frame in frozen_targets),
    )


def _metrics(
    scored: Sequence[_ScoredVideo],
    *,
    num_instrument_classes: int,
    iou_threshold: float,
) -> dict[str, object]:
    if not scored:
        raise ValueError("Tracker OOF metrics require scored videos")
    predictions = tuple(frame for video in scored for frame in video.predictions)
    targets = tuple(frame for video in scored for frame in video.targets)
    return calculate_detection_metrics(
        predictions,
        targets,
        num_classes=num_instrument_classes,
        iou_threshold=iou_threshold,
    )


def evaluate_tracker_oof(
    *,
    adapter: Any,
    index_path: str | Path,
    tracker_config_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate exact video-held-out predictions after strict leakage checks."""

    if iou_threshold != 0.5:
        raise ValueError("formal Tracker OOF evaluation freezes IoU at 0.5")
    index = load_tracker_oof_index(index_path)
    config_path = Path(tracker_config_path).expanduser().resolve()
    config = load_tracker_training_config(config_path)
    if config.oof_folds != len(
        {artifact for video_id, artifact in index.video_to_artifact.items() if video_id != "VID31"}
    ):
        raise ValueError("Tracker config fold count does not match indexed fold artifacts")
    if config.oof_folds < 2:
        raise ValueError("formal Tracker evaluation requires at least two folds")

    repair_manifest = Path(adapter.dataset_root).resolve() / "repair_manifest.json"
    if not repair_manifest.is_file():
        raise ValueError("dataset repair manifest is missing")
    config_sha256 = sha256_file(config_path)
    repair_manifest_sha256 = sha256_file(repair_manifest)

    all_training = tuple(
        video_id
        for video_id, entry in sorted(adapter.entries.items())
        if entry.split is DatasetSplit.TRAINING
    )
    indexed = tuple(sorted(index.video_to_artifact))
    if indexed != all_training:
        raise ValueError("Tracker OOF index does not exactly cover the Training split")
    qualified = supervision_qualified_training_video_ids(adapter)
    unscored = tuple(video_id for video_id in all_training if video_id not in qualified)
    if unscored != ("VID31",):
        raise ValueError("formal Tracker metric exclusion must be exactly VID31")

    artifact_groups: dict[Path, list[str]] = defaultdict(list)
    for video_id, artifact in index.video_to_artifact.items():
        artifact_groups[artifact].append(video_id)
    providers: dict[Path, PrecomputedPredictedTrackProvider] = {}
    for artifact, video_ids in artifact_groups.items():
        mapped = tuple(sorted(video_ids))
        provider = _provider_for_artifact(index, artifact, mapped)
        _verify_provider_provenance(
            provider,
            config_sha256=config_sha256,
            repair_manifest_sha256=repair_manifest_sha256,
        )
        providers[artifact] = provider

    expected_folds = deterministic_video_folds(qualified, config.oof_folds)
    fold_artifacts: dict[int, tuple[Path, tuple[str, ...]]] = {}
    vid31_artifact: Path | None = None
    for artifact, video_ids in artifact_groups.items():
        mapped = tuple(sorted(video_ids))
        if mapped == ("VID31",):
            if vid31_artifact is not None:
                raise ValueError("VID31 is mapped by multiple artifacts")
            vid31_artifact = artifact
            continue
        match = _FOLD_PATTERN.fullmatch(artifact.parent.name)
        if match is None:
            raise ValueError("instance-supervised OOF artifact is outside a fold_N directory")
        fold_index = int(match.group(1))
        if fold_index in fold_artifacts:
            raise ValueError(f"fold_{fold_index} is represented by multiple artifacts")
        fold_artifacts[fold_index] = (artifact, mapped)
    if tuple(sorted(fold_artifacts)) != tuple(range(config.oof_folds)):
        raise ValueError("Tracker OOF index does not contain every configured fold")
    if vid31_artifact is None:
        raise ValueError("Tracker OOF index has no VID31 prediction artifact")

    fold_audits: dict[int, dict[str, object]] = {}
    for fold_index, (artifact, held_out) in sorted(fold_artifacts.items()):
        fold_audits[fold_index] = _audit_fold(
            fold_index=fold_index,
            artifact=artifact,
            held_out=held_out,
            qualified=qualified,
            expected_held_out=expected_folds[fold_index],
            provider=providers[artifact],
            config=config,
            config_sha256=config_sha256,
            repair_manifest_sha256=repair_manifest_sha256,
        )
    vid31_audit = _audit_vid31(
        index=index,
        artifact=vid31_artifact,
        provider=providers[vid31_artifact],
        qualified=qualified,
        repair_manifest_sha256=repair_manifest_sha256,
    )

    scored_by_video: dict[str, _ScoredVideo] = {}
    per_video: dict[str, dict[str, object]] = {}
    for video_id in all_training:
        artifact = index.video_to_artifact[video_id]
        provider = providers[artifact]
        video, predictions_by_frame = _runtime_coverage(
            adapter,
            video_id=video_id,
            provider=provider,
        )
        runtime_prediction_count = sum(len(frame.tracks) for frame in video.frames)
        artifact_relative = artifact.relative_to(index.path.parent).as_posix()
        if video_id == "VID31":
            per_video[video_id] = {
                "metric_status": "NOT_SCORED",
                "reason": VID31_UNSCORED_REASON,
                "artifact": artifact_relative,
                "runtime_frame_count": len(video.frames),
                "runtime_prediction_count": runtime_prediction_count,
                "predicted_frame_coverage": 1.0,
                "metrics": None,
            }
            continue
        scored = _score_video(
            adapter,
            video_id=video_id,
            predictions_by_frame=predictions_by_frame,
        )
        scored_by_video[video_id] = scored
        video_metrics = _metrics(
            (scored,),
            num_instrument_classes=config.num_classes - 1,
            iou_threshold=iou_threshold,
        )
        per_video[video_id] = {
            "metric_status": "SCORED",
            "artifact": artifact_relative,
            "runtime_frame_count": len(video.frames),
            "runtime_prediction_count": runtime_prediction_count,
            "predicted_frame_coverage": 1.0,
            "scored_frame_count": scored.frame_count,
            "scored_prediction_count": scored.prediction_count,
            "scored_ground_truth_count": scored.ground_truth_count,
            "metrics": video_metrics,
        }

    per_fold: dict[str, dict[str, object]] = {}
    for fold_index, (artifact, held_out) in sorted(fold_artifacts.items()):
        fold_scored = tuple(scored_by_video[video_id] for video_id in held_out)
        per_fold[f"fold_{fold_index}"] = {
            **fold_audits[fold_index],
            "artifact": artifact.relative_to(index.path.parent).as_posix(),
            "scored_frame_count": sum(item.frame_count for item in fold_scored),
            "scored_prediction_count": sum(item.prediction_count for item in fold_scored),
            "scored_ground_truth_count": sum(
                item.ground_truth_count for item in fold_scored
            ),
            "metrics": _metrics(
                fold_scored,
                num_instrument_classes=config.num_classes - 1,
                iou_threshold=iou_threshold,
            ),
        }

    all_scored = tuple(scored_by_video[video_id] for video_id in qualified)
    overall_metrics = _metrics(
        all_scored,
        num_instrument_classes=config.num_classes - 1,
        iou_threshold=iou_threshold,
    )
    video_metric_records = [per_video[video_id]["metrics"] for video_id in qualified]
    if any(not isinstance(item, Mapping) for item in video_metric_records):
        raise RuntimeError("scored video metrics were not materialized")
    video_macro_mean = {
        metric: fmean(float(item[metric]) for item in video_metric_records)
        for metric in ("macro_ap50", "precision", "recall", "f1")
    }

    return {
        "schema_version": TRACKER_OOF_EVALUATION_SCHEMA_VERSION,
        "status": "PASS",
        "source_split": "Training",
        "metric_protocol": {
            "prediction_source": "STORED_CURRENT_FRAME_PREDICTED_TRACKS",
            "instrument_class_ids": list(range(config.num_classes - 1)),
            "score_threshold": config.score_threshold,
            "iou_threshold": iou_threshold,
            "ap_interpolation": "ALL_POINT_PRECISION_ENVELOPE",
            "matching": "ONE_TO_ONE_WITHIN_FRAME_AND_INSTRUMENT_CLASS",
            "primary_aggregation": "POOLED_SCORED_FRAMES",
            "label_based_threshold_tuning": False,
        },
        "provenance": {
            "oof_index_sha256": sha256_file(index.path),
            "tracker_config_sha256": config_sha256,
            "dataset_repair_manifest_sha256": repair_manifest_sha256,
            "artifact_sha256": {
                artifact.relative_to(index.path.parent).as_posix(): digest
                for artifact, digest in sorted(
                    index.artifact_sha256.items(), key=lambda item: item[0].as_posix()
                )
            },
        },
        "coverage": {
            "all_training_video_ids": list(all_training),
            "metric_eligible_video_ids": list(qualified),
            "metric_excluded_video_ids": list(unscored),
            "indexed_video_count": len(indexed),
            "scored_video_count": len(qualified),
            "all_runtime_frames_have_predictions": True,
        },
        "leakage_audit": {
            "status": "PASS",
            "fold_count": config.oof_folds,
            "folds": {
                f"fold_{index}": audit for index, audit in sorted(fold_audits.items())
            },
            "vid31": vid31_audit,
        },
        "overall_pooled": {
            "scored_frame_count": sum(item.frame_count for item in all_scored),
            "scored_prediction_count": sum(item.prediction_count for item in all_scored),
            "scored_ground_truth_count": sum(
                item.ground_truth_count for item in all_scored
            ),
            "metrics": overall_metrics,
        },
        "video_macro_mean": {
            "definition": "UNWEIGHTED_MEAN_OF_PER_VIDEO_METRICS",
            **video_macro_mean,
        },
        "per_fold": per_fold,
        "per_video": per_video,
    }


__all__ = [
    "TRACKER_OOF_EVALUATION_SCHEMA_VERSION",
    "VID31_UNSCORED_REASON",
    "evaluate_tracker_oof",
]
