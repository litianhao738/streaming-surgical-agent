"""Portable index over video-held-out Tracker prediction artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider


@dataclass(frozen=True)
class TrackerOOFIndex:
    path: Path
    video_to_artifact: Mapping[str, Path]
    artifact_sha256: Mapping[Path, str]

    def provider_for(self, video_id: str) -> PrecomputedPredictedTrackProvider:
        try:
            artifact = self.video_to_artifact[video_id]
        except KeyError as exc:
            raise KeyError(f"OOF Tracker index has no held-out prediction for {video_id}") from exc
        if sha256_file(artifact) != self.artifact_sha256[artifact]:
            raise ValueError("OOF Tracker artifact digest changed after indexing")
        provider = PrecomputedPredictedTrackProvider.from_json(artifact)
        provider.reset(video_id)
        return provider


def write_tracker_oof_index(
    path: str | Path,
    *,
    video_to_artifact: Mapping[str, str | Path],
    fold_count: int,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if fold_count < 2:
        raise ValueError("Tracker OOF index requires at least two folds")
    if not video_to_artifact:
        raise ValueError("Tracker OOF index requires held-out video artifacts")
    resolved = {
        video_id: Path(artifact).expanduser().resolve()
        for video_id, artifact in video_to_artifact.items()
    }
    if any(not video_id or not artifact.is_file() for video_id, artifact in resolved.items()):
        raise ValueError("Tracker OOF index contains a missing video or artifact")
    unique_artifacts = tuple(sorted(set(resolved.values())))
    payload = {
        "schema_version": "tracker_oof_index_v1",
        "source_split": "Training",
        "fold_count": fold_count,
        "video_to_artifact": {
            video_id: artifact.relative_to(destination.parent).as_posix()
            for video_id, artifact in sorted(resolved.items())
        },
        "artifacts": {
            artifact.relative_to(destination.parent).as_posix(): sha256_file(artifact)
            for artifact in unique_artifacts
        },
    }
    atomic_write_json(destination, payload)
    load_tracker_oof_index(destination)
    return destination


def load_tracker_oof_index(path: str | Path) -> TrackerOOFIndex:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Tracker OOF index") from exc
    expected = {
        "schema_version",
        "source_split",
        "fold_count",
        "video_to_artifact",
        "artifacts",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Tracker OOF index fields do not match the schema")
    if raw["schema_version"] != "tracker_oof_index_v1" or raw["source_split"] != "Training":
        raise ValueError("unsupported or non-Training Tracker OOF index")
    fold_count = raw["fold_count"]
    video_map = raw["video_to_artifact"]
    hashes = raw["artifacts"]
    if (
        not isinstance(fold_count, int)
        or isinstance(fold_count, bool)
        or fold_count < 2
        or not isinstance(video_map, Mapping)
        or not isinstance(hashes, Mapping)
    ):
        raise ValueError("Tracker OOF index values are invalid")
    artifacts: dict[Path, str] = {}
    for relative, digest in hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise TypeError("Tracker OOF artifact index values must be text")
        artifact = (source.parent / relative).resolve()
        if not artifact.is_relative_to(source.parent):
            raise ValueError("Tracker OOF artifact escapes its index directory")
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError("Tracker OOF artifact is missing or digest-invalid")
        artifacts[artifact] = digest
    videos: dict[str, Path] = {}
    for video_id, relative in video_map.items():
        if not isinstance(video_id, str) or not video_id or not isinstance(relative, str):
            raise TypeError("Tracker OOF video mapping must contain text")
        artifact = (source.parent / relative).resolve()
        if artifact not in artifacts:
            raise ValueError("Tracker OOF video points to an unindexed artifact")
        encoded = json.loads(artifact.read_text(encoding="utf-8"))
        if video_id not in encoded.get("videos", {}):
            raise ValueError("Tracker OOF artifact does not contain its mapped video")
        videos[video_id] = artifact
    return TrackerOOFIndex(
        path=source,
        video_to_artifact=MappingProxyType(videos),
        artifact_sha256=MappingProxyType(artifacts),
    )


__all__ = ["TrackerOOFIndex", "load_tracker_oof_index", "write_tracker_oof_index"]
