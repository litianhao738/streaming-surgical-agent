"""Gold-free contracts for predicted track context."""

from __future__ import annotations

import math
from dataclasses import dataclass

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.schemas import DatasetSplit

TRACK_CONTEXT_SCHEMA_VERSION = "predicted_track_context_v1"
PREDICTED_TRACK_INFERENCE_MODE = "online_forward_only"
PREDICTED_TRACK_PRODUCER_VERSION = "predicted_track_producer_v1"


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a stripped non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PredictedTrack:
    """One detector/associator prediction; annotations are not representable."""

    track_id: str
    instrument_id: int
    bbox_tlwh: tuple[float, float, float, float]
    score: float
    age: int

    def __post_init__(self) -> None:
        _nonempty_text(self.track_id, name="track_id")
        lower, upper = TASK_ID_BOUNDS["instrument"]
        if (
            not isinstance(self.instrument_id, int)
            or isinstance(self.instrument_id, bool)
            or not lower <= self.instrument_id <= upper
        ):
            raise ValueError("instrument_id is outside the ontology")
        bbox = tuple(self.bbox_tlwh)
        if len(bbox) != 4 or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in bbox
        ):
            raise ValueError("bbox_tlwh must contain four finite numbers")
        x, y, width, height = (float(value) for value in bbox)
        if (
            x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.0
            or y + height > 1.0
        ):
            raise ValueError("bbox_tlwh must be a positive normalized box")
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
            or not 0.0 <= float(self.score) <= 1.0
        ):
            raise ValueError("track score must be finite in [0, 1]")
        if not isinstance(self.age, int) or isinstance(self.age, bool) or self.age <= 0:
            raise ValueError("track age must be a positive integer")
        object.__setattr__(self, "bbox_tlwh", (x, y, width, height))
        object.__setattr__(self, "score", float(self.score))

    def as_mapping(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "instrument_id": self.instrument_id,
            "bbox_tlwh": self.bbox_tlwh,
            "score": self.score,
            "age": self.age,
        }


@dataclass(frozen=True)
class PredictedTrackFrame:
    frame_id: int
    tracks: tuple[PredictedTrack, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_id, int)
            or isinstance(self.frame_id, bool)
            or self.frame_id < 0
        ):
            raise ValueError("track frame_id must be non-negative")
        tracks = tuple(self.tracks)
        if any(not isinstance(track, PredictedTrack) for track in tracks):
            raise TypeError("tracks must contain PredictedTrack values")
        track_ids = tuple(track.track_id for track in tracks)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique within a frame")
        object.__setattr__(self, "tracks", tracks)


@dataclass(frozen=True)
class PredictedTrackVideo:
    video_id: str
    source_split: DatasetSplit
    frames: tuple[PredictedTrackFrame, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.video_id, name="video_id")
        if not isinstance(self.source_split, DatasetSplit):
            raise TypeError("source_split must be DatasetSplit")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("predicted-track videos require at least one frame")
        if any(not isinstance(frame, PredictedTrackFrame) for frame in frames):
            raise TypeError("frames must contain PredictedTrackFrame values")
        frame_ids = tuple(frame.frame_id for frame in frames)
        if tuple(sorted(set(frame_ids))) != frame_ids:
            raise ValueError("predicted-track frame IDs must be unique and increasing")
        object.__setattr__(self, "frames", frames)


__all__ = [
    "PREDICTED_TRACK_INFERENCE_MODE",
    "PREDICTED_TRACK_PRODUCER_VERSION",
    "TRACK_CONTEXT_SCHEMA_VERSION",
    "PredictedTrack",
    "PredictedTrackFrame",
    "PredictedTrackVideo",
    "_nonempty_text",
    "_sha256",
]
