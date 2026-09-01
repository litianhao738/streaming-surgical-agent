"""Atomic writer for auditable predicted-track runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.contracts import (
    PREDICTED_TRACK_INFERENCE_MODE,
    PREDICTED_TRACK_PRODUCER_VERSION,
    TRACK_CONTEXT_SCHEMA_VERSION,
    PredictedTrackFrame,
)
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_predicted_track_artifact(
    path: str | Path,
    *,
    provider: str,
    source_model_identifier: str,
    checkpoint_path: str | Path,
    inference_config_path: str | Path,
    repair_manifest_path: str | Path,
    videos: Mapping[str, tuple[DatasetSplit, tuple[PredictedTrackFrame, ...]]],
) -> Path:
    output = Path(path).expanduser().resolve()
    if not videos:
        raise ValueError("predicted-track artifact requires at least one video")
    encoded_videos: dict[str, object] = {}
    for video_id, (split, frames) in sorted(videos.items()):
        if not isinstance(split, DatasetSplit) or not frames:
            raise ValueError("each predicted-track video requires a split and frames")
        if tuple(frame.frame_id for frame in frames) != tuple(
            sorted({frame.frame_id for frame in frames})
        ):
            raise ValueError("predicted-track frame IDs must be unique and increasing")
        encoded_videos[video_id] = {
            "source_split": split.value,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "tracks": [track.as_mapping() for track in frame.tracks],
                }
                for frame in frames
            ],
        }
    payload = {
        "schema_version": TRACK_CONTEXT_SCHEMA_VERSION,
        "provider": provider,
        "source_model_identifier": source_model_identifier,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "causal": True,
        "inference_mode": PREDICTED_TRACK_INFERENCE_MODE,
        "producer_version": PREDICTED_TRACK_PRODUCER_VERSION,
        "dataset_repair_manifest_sha256": sha256_file(repair_manifest_path),
        "inference_config_sha256": sha256_file(inference_config_path),
        "videos": encoded_videos,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    PrecomputedPredictedTrackProvider.from_json(output)
    return output


__all__ = ["sha256_file", "write_predicted_track_artifact"]
