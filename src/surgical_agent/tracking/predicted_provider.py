"""Read-only predicted-track artifacts for causal runtime context."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import freeze_snapshot
from surgical_agent.tracking.contracts import (
    PREDICTED_TRACK_INFERENCE_MODE,
    PREDICTED_TRACK_PRODUCER_VERSION,
    TRACK_CONTEXT_SCHEMA_VERSION,
    PredictedTrack,
    PredictedTrackFrame,
    PredictedTrackVideo,
    _nonempty_text,
    _sha256,
)

_TOP_FIELDS = {
    "schema_version",
    "provider",
    "source_model_identifier",
    "checkpoint_sha256",
    "causal",
    "inference_mode",
    "producer_version",
    "dataset_repair_manifest_sha256",
    "inference_config_sha256",
    "videos",
}
_VIDEO_FIELDS = {"source_split", "frames"}
_FRAME_FIELDS = {"frame_id", "tracks"}
_TRACK_FIELDS = {"track_id", "instrument_id", "bbox_tlwh", "score", "age"}


class PredictedTrackArtifactError(ValueError):
    """Raised when predicted tracking could leak labels or violate causality."""


def _exact_mapping(value: object, fields: set[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PredictedTrackArtifactError(f"{name} must be a mapping")
    if set(value) != fields:
        raise PredictedTrackArtifactError(f"{name} has unexpected fields")
    return value


class UnavailablePredictedTrackProvider:
    """Explicit no-track provider used by frames/workflow-only ablations."""

    artifact_sha256: str | None = None
    provider_name = "unavailable"

    def __init__(self) -> None:
        self._video_id: str | None = None

    def reset(self, video_id: str) -> None:
        self._video_id = _nonempty_text(video_id, name="video_id")

    def snapshot(self, sample: InferenceSample) -> Mapping[str, object]:
        if self._video_id != sample.video_id:
            raise PredictedTrackArtifactError("track provider crossed a video boundary")
        return freeze_snapshot(
            {
                "component": "predicted_track_context",
                "status": "UNAVAILABLE",
                "video_id": sample.video_id,
                "source_max_frame_id": None,
                "frames": (),
            },
            name="track",
        )


class PrecomputedPredictedTrackProvider:
    """Serve one auditable causal prediction artifact for the selected window.

    The required metadata makes the artifact inspectable, but does not prove that
    its producer was GT-free. Formal artifacts must come from the controlled
    AutoDL causal generator, not a hand-authored JSON file.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        source_model_identifier: str,
        checkpoint_sha256: str,
        inference_mode: str,
        producer_version: str,
        dataset_repair_manifest_sha256: str,
        inference_config_sha256: str,
        artifact_sha256: str,
        videos: Mapping[str, PredictedTrackVideo],
        max_tracks_per_frame: int = 32,
    ) -> None:
        self.provider_name = _nonempty_text(provider_name, name="provider")
        self.source_model_identifier = _nonempty_text(
            source_model_identifier,
            name="source_model_identifier",
        )
        self.checkpoint_sha256 = _sha256(
            checkpoint_sha256,
            name="checkpoint_sha256",
        )
        if inference_mode != PREDICTED_TRACK_INFERENCE_MODE:
            raise PredictedTrackArtifactError(
                "track artifact must declare inference_mode=online_forward_only"
            )
        self.inference_mode = inference_mode
        if producer_version != PREDICTED_TRACK_PRODUCER_VERSION:
            raise PredictedTrackArtifactError(
                "track artifact has an unsupported producer_version"
            )
        self.producer_version = producer_version
        self.dataset_repair_manifest_sha256 = _sha256(
            dataset_repair_manifest_sha256,
            name="dataset_repair_manifest_sha256",
        )
        self.inference_config_sha256 = _sha256(
            inference_config_sha256,
            name="inference_config_sha256",
        )
        self.artifact_sha256 = _sha256(artifact_sha256, name="artifact_sha256")
        if (
            not isinstance(max_tracks_per_frame, int)
            or isinstance(max_tracks_per_frame, bool)
            or max_tracks_per_frame <= 0
        ):
            raise ValueError("max_tracks_per_frame must be positive")
        self.max_tracks_per_frame = max_tracks_per_frame
        normalized = dict(videos)
        if not normalized or any(
            not isinstance(video, PredictedTrackVideo)
            or video_id != video.video_id
            for video_id, video in normalized.items()
        ):
            raise PredictedTrackArtifactError("videos must contain matching predictions")
        if any(
            len(frame.tracks) > max_tracks_per_frame
            for video in normalized.values()
            for frame in video.frames
        ):
            raise PredictedTrackArtifactError("track count exceeds the bounded context")
        self._videos = normalized
        self._video_id: str | None = None

    @property
    def available_video_ids(self) -> tuple[str, ...]:
        """Return the immutable, sorted video scope carried by this artifact."""

        return tuple(sorted(self._videos))

    def video(self, video_id: str) -> PredictedTrackVideo:
        """Expose one validated video for offline audit and evaluation only."""

        normalized = _nonempty_text(video_id, name="video_id")
        try:
            return self._videos[normalized]
        except KeyError as exc:
            raise PredictedTrackArtifactError(
                "track artifact has no requested video"
            ) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> PrecomputedPredictedTrackProvider:
        artifact_path = Path(path).expanduser().resolve()
        try:
            encoded = artifact_path.read_bytes()
            raw = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PredictedTrackArtifactError("invalid predicted-track artifact") from exc
        artifact = _exact_mapping(raw, _TOP_FIELDS, name="track artifact")
        if artifact["schema_version"] != TRACK_CONTEXT_SCHEMA_VERSION:
            raise PredictedTrackArtifactError("unsupported track artifact schema")
        if artifact["causal"] is not True:
            raise PredictedTrackArtifactError("track artifact must declare causal=true")
        videos_raw = artifact["videos"]
        if not isinstance(videos_raw, Mapping):
            raise PredictedTrackArtifactError("track artifact videos must be a mapping")
        videos: dict[str, PredictedTrackVideo] = {}
        for video_id, video_raw in videos_raw.items():
            if not isinstance(video_id, str):
                raise PredictedTrackArtifactError("track video IDs must be text")
            video = _exact_mapping(video_raw, _VIDEO_FIELDS, name="track video")
            try:
                source_split = DatasetSplit(video["source_split"])
            except (TypeError, ValueError) as exc:
                raise PredictedTrackArtifactError("invalid track video split") from exc
            frames_raw = video["frames"]
            if not isinstance(frames_raw, list):
                raise PredictedTrackArtifactError("track frames must be a list")
            frames: list[PredictedTrackFrame] = []
            for frame_raw in frames_raw:
                frame = _exact_mapping(frame_raw, _FRAME_FIELDS, name="track frame")
                tracks_raw = frame["tracks"]
                if not isinstance(tracks_raw, list):
                    raise PredictedTrackArtifactError("frame tracks must be a list")
                tracks: list[PredictedTrack] = []
                for track_raw in tracks_raw:
                    track = _exact_mapping(track_raw, _TRACK_FIELDS, name="track")
                    try:
                        tracks.append(
                            PredictedTrack(
                                track_id=track["track_id"],
                                instrument_id=track["instrument_id"],
                                bbox_tlwh=tuple(track["bbox_tlwh"]),
                                score=track["score"],
                                age=track["age"],
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise PredictedTrackArtifactError(str(exc)) from exc
                try:
                    frames.append(
                        PredictedTrackFrame(frame_id=frame["frame_id"], tracks=tuple(tracks))
                    )
                except (TypeError, ValueError) as exc:
                    raise PredictedTrackArtifactError(str(exc)) from exc
            try:
                videos[video_id] = PredictedTrackVideo(
                    video_id=video_id,
                    source_split=source_split,
                    frames=tuple(frames),
                )
            except (TypeError, ValueError) as exc:
                raise PredictedTrackArtifactError(str(exc)) from exc
        try:
            return cls(
                provider_name=artifact["provider"],
                source_model_identifier=artifact["source_model_identifier"],
                checkpoint_sha256=artifact["checkpoint_sha256"],
                inference_mode=artifact["inference_mode"],
                producer_version=artifact["producer_version"],
                dataset_repair_manifest_sha256=artifact[
                    "dataset_repair_manifest_sha256"
                ],
                inference_config_sha256=artifact["inference_config_sha256"],
                artifact_sha256=hashlib.sha256(encoded).hexdigest(),
                videos=videos,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PredictedTrackArtifactError):
                raise
            raise PredictedTrackArtifactError(str(exc)) from exc

    def reset(self, video_id: str) -> None:
        self._video_id = _nonempty_text(video_id, name="video_id")

    def snapshot(self, sample: InferenceSample) -> Mapping[str, object]:
        if not isinstance(sample, InferenceSample):
            raise TypeError("track snapshot requires an InferenceSample")
        if self._video_id is None:
            raise PredictedTrackArtifactError("track provider must reset before snapshot")
        if self._video_id != sample.video_id:
            raise PredictedTrackArtifactError("track provider crossed a video boundary")
        try:
            video = self._videos[sample.video_id]
        except KeyError as exc:
            raise PredictedTrackArtifactError("track artifact has no selected video") from exc
        if video.source_split is not sample.source_split:
            raise PredictedTrackArtifactError("track artifact split does not match sample")
        by_frame = {frame.frame_id: frame for frame in video.frames}
        if sample.target_frame_id not in by_frame:
            raise PredictedTrackArtifactError("track artifact is missing the target frame")
        if set(sample.causal_frame_ids) - set(by_frame):
            raise PredictedTrackArtifactError("track artifact is missing a causal frame")
        selected = tuple(by_frame[frame_id] for frame_id in sample.causal_frame_ids)
        if any(frame.frame_id > sample.target_frame_id for frame in selected):
            raise PredictedTrackArtifactError("track context references a future frame")
        return freeze_snapshot(
            {
                "component": "predicted_track_context",
                "status": "AVAILABLE",
                "video_id": sample.video_id,
                "provider": self.provider_name,
                "source_model_identifier": self.source_model_identifier,
                "checkpoint_sha256": self.checkpoint_sha256,
                "inference_mode": self.inference_mode,
                "producer_version": self.producer_version,
                "dataset_repair_manifest_sha256": self.dataset_repair_manifest_sha256,
                "inference_config_sha256": self.inference_config_sha256,
                "artifact_sha256": self.artifact_sha256,
                "source_max_frame_id": selected[-1].frame_id,
                "frames": tuple(
                    {
                        "frame_id": frame.frame_id,
                        "tracks": tuple(track.as_mapping() for track in frame.tracks),
                    }
                    for frame in selected
                ),
            },
            name="track",
        )


__all__ = [
    "PrecomputedPredictedTrackProvider",
    "PredictedTrackArtifactError",
    "UnavailablePredictedTrackProvider",
]
