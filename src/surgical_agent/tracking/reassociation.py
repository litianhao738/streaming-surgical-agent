"""Rebuild causal identities from stored detections without running a detector."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.tracking.associator import (
    CausalHungarianAssociator,
    InstrumentDetection,
)
from surgical_agent.tracking.config import load_tracker_training_config
from surgical_agent.tracking.contracts import PredictedTrackFrame
from surgical_agent.tracking.oof_index import (
    load_tracker_oof_index,
    write_tracker_oof_index,
)
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider

REASSOCIATION_VERSION = "causal_hungarian_gap_reset_v1"


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _detections(frame: PredictedTrackFrame) -> Counter:
    return Counter(
        (track.instrument_id, track.bbox_tlwh, track.score) for track in frame.tracks
    )


def reassociate_frames(
    video_id: str,
    frames: tuple[PredictedTrackFrame, ...],
    *,
    iou_threshold: float,
    max_age: int,
    max_frame_id_gap: int = 25,
) -> tuple[tuple[PredictedTrackFrame, ...], dict[str, int]]:
    """Preserve every stored detection and frame while rebuilding ID and age."""
    associator = CausalHungarianAssociator(
        iou_threshold=iou_threshold,
        max_age=max_age,
        max_frame_id_gap=max_frame_id_gap,
    )
    associator.reset(video_id)
    result: list[PredictedTrackFrame] = []
    gaps = 0
    old_aged_boundaries = 0
    old_shared_boundaries = 0
    seen_ids: set[str] = set()
    segment_ids: set[str] = set()
    for frame in frames:
        at_gap = bool(
            result and frame.frame_id - result[-1].frame_id > max_frame_id_gap
        )
        if at_gap:
            gaps += 1
            old_aged_boundaries += any(track.age > 1 for track in frame.tracks)
            previous = frames[len(result) - 1]
            old_shared_boundaries += bool(
                {track.track_id for track in previous.tracks}
                & {track.track_id for track in frame.tracks}
            )
            seen_ids.update(segment_ids)
            segment_ids.clear()
        detections = tuple(
            InstrumentDetection(track.instrument_id, track.bbox_tlwh, track.score)
            for track in frame.tracks
        )
        rebuilt = PredictedTrackFrame(
            frame.frame_id, associator.update(frame.frame_id, detections)
        )
        if _detections(frame) != _detections(rebuilt):
            raise RuntimeError("reassociation changed the detection multiset")
        ids = {track.track_id for track in rebuilt.tracks}
        if ids & seen_ids:
            raise RuntimeError("reassociation reused an ID across a reset boundary")
        if at_gap and any(track.age != 1 for track in rebuilt.tracks):
            raise RuntimeError("reassociation retained age after a reset boundary")
        segment_ids.update(ids)
        result.append(rebuilt)
    if tuple(frame.frame_id for frame in result) != tuple(
        frame.frame_id for frame in frames
    ):
        raise RuntimeError("reassociation changed frame coverage")
    return tuple(result), {
        "frame_count": len(frames),
        "detection_count": sum(len(frame.tracks) for frame in frames),
        "gap_count": gaps,
        "source_gap_boundaries_with_shared_adjacent_ids": old_shared_boundaries,
        "source_gap_boundaries_with_age_above_one": old_aged_boundaries,
        "output_cross_segment_id_reuse_count": 0,
        "output_gap_age_violation_count": 0,
    }


def _validate_source(
    source: Path, config_path: Path, training_manifest_path: Path
) -> PrecomputedPredictedTrackProvider:
    provider = PrecomputedPredictedTrackProvider.from_json(source)
    manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    checkpoint = training_manifest_path.parent / "checkpoint.pt"
    if manifest.get("checkpoint_sha256") != provider.checkpoint_sha256:
        raise ValueError("source prediction and training manifest checkpoints differ")
    if sha256_file(checkpoint) != provider.checkpoint_sha256:
        raise ValueError("source checkpoint digest is invalid")
    if (
        manifest.get("dataset_repair_manifest_sha256")
        != provider.dataset_repair_manifest_sha256
    ):
        raise ValueError("source prediction and training repair manifests differ")
    if sha256_file(config_path) != provider.inference_config_sha256:
        raise ValueError("source prediction and inference config differ")
    config = load_tracker_training_config(config_path)
    training_config = manifest.get("config", {})
    for key in ("association_iou_threshold", "max_age"):
        if training_config.get(key) != getattr(config, key):
            raise ValueError(f"original association parameter differs: {key}")
    return provider


def reassociate_tracker_artifact(
    *,
    source_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    training_manifest_path: str | Path,
    max_frame_id_gap: int = 25,
) -> dict[str, object]:
    """Export a new compatible artifact plus a hash-bound migration sidecar."""
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    config_file = Path(config_path).resolve()
    training_file = Path(training_manifest_path).resolve()
    sidecar = output.parent / "reassociation_manifest.json"
    if output.exists() or sidecar.exists() or output == source:
        raise FileExistsError("reassociation refuses to overwrite an existing output")
    provider = _validate_source(source, config_file, training_file)
    config = load_tracker_training_config(config_file)
    source_digest = sha256_file(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    statistics: dict[str, object] = {}
    for video_id in provider.available_video_ids:
        rebuilt, stats = reassociate_frames(
            video_id,
            provider.video(video_id).frames,
            iou_threshold=config.association_iou_threshold,
            max_age=config.max_age,
            max_frame_id_gap=max_frame_id_gap,
        )
        payload["videos"][video_id]["frames"] = [
            {
                "frame_id": frame.frame_id,
                "tracks": [track.as_mapping() for track in frame.tracks],
            }
            for frame in rebuilt
        ]
        statistics[video_id] = stats
    if sha256_file(source) != source_digest:
        raise ValueError("source artifact changed during reassociation")
    _write_new_json(output, payload)
    exported = PrecomputedPredictedTrackProvider.from_json(output)
    if exported.available_video_ids != provider.available_video_ids:
        raise RuntimeError("export changed video coverage")
    for video_id in provider.available_video_ids:
        original = provider.video(video_id)
        observed = exported.video(video_id)
        if original.source_split != observed.source_split or len(
            original.frames
        ) != len(observed.frames):
            raise RuntimeError("export changed video split or frame count")
        for old, new in zip(original.frames, observed.frames):
            if old.frame_id != new.frame_id or _detections(old) != _detections(new):
                raise RuntimeError("export changed stored detection evidence")
    manifest: dict[str, object] = {
        "schema_version": "tracker_reassociation_manifest_v1",
        "source_artifact": str(source),
        "source_artifact_sha256": source_digest,
        "output_artifact": output.name,
        "output_artifact_sha256": sha256_file(output),
        "algorithm": {
            "version": REASSOCIATION_VERSION,
            "max_frame_id_gap": max_frame_id_gap,
            "iou_threshold": config.association_iou_threshold,
            "max_age": config.max_age,
        },
        "algorithm_source_sha256": sha256_file(
            Path(__file__).with_name("associator.py")
        ),
        "migration_source_sha256": sha256_file(__file__),
        "original_training_manifest": str(training_file),
        "original_training_manifest_sha256": sha256_file(training_file),
        "checkpoint_sha256": provider.checkpoint_sha256,
        "inference_config_sha256": provider.inference_config_sha256,
        "dataset_repair_manifest_sha256": provider.dataset_repair_manifest_sha256,
        "detector_inference_performed": False,
        "training_performed": False,
        "detections_and_frame_coverage_unchanged": True,
        "per_video": statistics,
    }
    _write_new_json(sidecar, manifest)
    return manifest


def _copy_new_file(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, output.open("xb") as output_stream:
        shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
    if sha256_file(source) != sha256_file(output):
        raise RuntimeError("training provenance copy changed file contents")


def reassociate_tracker_bundle(
    *,
    oof_index_path: str | Path,
    full_artifact_path: str | Path,
    full_training_directory: str | Path,
    oof_config_path: str | Path,
    full_config_path: str | Path,
    output_root: str | Path,
    max_frame_id_gap: int = 25,
) -> dict[str, object]:
    """Copy immutable training provenance and migrate full/OOF predictions."""
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError("reassociation output root must not already exist")
    index = load_tracker_oof_index(oof_index_path)
    full_artifact = Path(full_artifact_path).resolve()
    full_training = Path(full_training_directory).resolve()
    oof_config = Path(oof_config_path).resolve()
    full_config = Path(full_config_path).resolve()
    index_raw = json.loads(index.path.read_text(encoding="utf-8"))
    jobs: list[tuple[Path, Path, Path, Path]] = []
    for artifact in sorted(index.artifact_sha256):
        relative = artifact.relative_to(index.path.parent)
        training_file = (
            full_training / "training_manifest.json"
            if relative.parent.as_posix() == "vid31"
            else artifact.parent / "training_manifest.json"
        )
        jobs.append((artifact, output / "oof" / relative, oof_config, training_file))
    jobs.append(
        (
            full_artifact,
            output / "full/predicted_tracks.json",
            full_config,
            full_training / "training_manifest.json",
        )
    )
    # Validate all source provenance before creating any output.
    for source, _, config, training_file in jobs:
        if output == source.parent or output.is_relative_to(source.parent):
            raise ValueError("output must not be inside an original artifact directory")
        _validate_source(source, config, training_file)
    output.mkdir(parents=True, exist_ok=False)
    manifests: dict[str, object] = {}
    copied: set[Path] = set()
    for source, destination, config, training_file in jobs:
        relative = destination.relative_to(output)
        provenance_output = (
            output / "full"
            if relative.parent.as_posix() == "oof/vid31"
            else destination.parent
        )
        if provenance_output not in copied:
            for filename in ("checkpoint.pt", "training_manifest.json"):
                _copy_new_file(
                    training_file.parent / filename, provenance_output / filename
                )
            copied.add(provenance_output)
        manifests[relative.as_posix()] = reassociate_tracker_artifact(
            source_path=source,
            output_path=destination,
            config_path=config,
            training_manifest_path=training_file,
            max_frame_id_gap=max_frame_id_gap,
        )
    new_index = write_tracker_oof_index(
        output / "oof/index.json",
        video_to_artifact={
            video_id: output / "oof" / artifact.relative_to(index.path.parent)
            for video_id, artifact in index.video_to_artifact.items()
        },
        fold_count=index_raw["fold_count"],
    )
    summary: dict[str, object] = {
        "schema_version": "tracker_reassociation_bundle_v1",
        "status": "PASS",
        "algorithm_version": REASSOCIATION_VERSION,
        "source_oof_index": str(index.path),
        "source_oof_index_sha256": sha256_file(index.path),
        "oof_index": "oof/index.json",
        "oof_index_sha256": sha256_file(new_index),
        "full_artifact": "full/predicted_tracks.json",
        "full_artifact_sha256": sha256_file(output / "full/predicted_tracks.json"),
        "checkpoint_copy_mode": "INDEPENDENT_BYTE_COPY_NO_OPTIMIZER_STATE",
        "training_performed": False,
        "detector_inference_performed": False,
        "artifacts": manifests,
    }
    _write_new_json(output / "migration_manifest.json", summary)
    return summary


__all__ = [
    "reassociate_frames",
    "reassociate_tracker_artifact",
    "reassociate_tracker_bundle",
]
