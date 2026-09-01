"""Deterministic online association for predicted instrument detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.tracking.contracts import PredictedTrack


@dataclass(frozen=True)
class InstrumentDetection:
    instrument_id: int
    bbox_tlwh: tuple[float, float, float, float]
    score: float

    def __post_init__(self) -> None:
        lower, upper = TASK_ID_BOUNDS["instrument"]
        if (
            not isinstance(self.instrument_id, int)
            or isinstance(self.instrument_id, bool)
            or not lower <= self.instrument_id <= upper
        ):
            raise ValueError("instrument_id is outside the ontology")
        values = tuple(float(value) for value in self.bbox_tlwh)
        if len(values) != 4 or any(not math.isfinite(value) for value in values):
            raise ValueError("bbox_tlwh must contain four finite numbers")
        x, y, width, height = values
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("bbox_tlwh must be a positive normalized box")
        if not math.isfinite(float(self.score)) or not 0 <= float(self.score) <= 1:
            raise ValueError("score must be finite in [0, 1]")
        object.__setattr__(self, "bbox_tlwh", values)
        object.__setattr__(self, "score", float(self.score))


@dataclass
class _TrackState:
    track_id: str
    instrument_id: int
    bbox_tlwh: tuple[float, float, float, float]
    score: float
    age: int
    missed: int = 0


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


class CausalHungarianAssociator:
    """Associate current detections with state committed at earlier frames."""

    def __init__(self, *, iou_threshold: float, max_age: int) -> None:
        if not isinstance(iou_threshold, (int, float)) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age < 0:
            raise ValueError("max_age must be a non-negative integer")
        self.iou_threshold = float(iou_threshold)
        self.max_age = max_age
        self._video_id: str | None = None
        self._last_frame_id: int | None = None
        self._next_id = 1
        self._tracks: dict[str, _TrackState] = {}

    def reset(self, video_id: str) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be non-empty text")
        self._video_id = video_id.strip().upper()
        self._last_frame_id = None
        self._next_id = 1
        self._tracks.clear()

    def _new_state(self, detection: InstrumentDetection) -> _TrackState:
        assert self._video_id is not None
        state = _TrackState(
            track_id=f"{self._video_id}:pred:{self._next_id:06d}",
            instrument_id=detection.instrument_id,
            bbox_tlwh=detection.bbox_tlwh,
            score=detection.score,
            age=1,
        )
        self._next_id += 1
        return state

    def update(
        self,
        frame_id: int,
        detections: tuple[InstrumentDetection, ...],
    ) -> tuple[PredictedTrack, ...]:
        if self._video_id is None:
            raise RuntimeError("associator must reset before update")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frame IDs must be strictly increasing")
        if any(not isinstance(value, InstrumentDetection) for value in detections):
            raise TypeError("detections must contain InstrumentDetection values")

        active = list(self._tracks.values())
        matched_tracks: set[int] = set()
        matched_detections: dict[int, _TrackState] = {}
        if active and detections:
            costs = np.full((len(active), len(detections)), 1_000_000.0)
            overlaps = np.zeros_like(costs)
            for track_index, track in enumerate(active):
                for detection_index, detection in enumerate(detections):
                    if track.instrument_id != detection.instrument_id:
                        continue
                    overlap = _iou(track.bbox_tlwh, detection.bbox_tlwh)
                    overlaps[track_index, detection_index] = overlap
                    costs[track_index, detection_index] = 1.0 - overlap
            rows, columns = linear_sum_assignment(costs)
            for track_index, detection_index in zip(rows.tolist(), columns.tolist()):
                if (
                    costs[track_index, detection_index] >= 1_000_000.0
                    or overlaps[track_index, detection_index] < self.iou_threshold
                ):
                    continue
                detection = detections[detection_index]
                track = active[track_index]
                track.bbox_tlwh = detection.bbox_tlwh
                track.score = detection.score
                track.age += 1
                track.missed = 0
                matched_tracks.add(track_index)
                matched_detections[detection_index] = track

        for track_index, track in enumerate(active):
            if track_index not in matched_tracks:
                track.missed += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed <= self.max_age
        }

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = self._new_state(detection)
            self._tracks[track.track_id] = track
            matched_detections[detection_index] = track

        self._last_frame_id = frame_id
        return tuple(
            PredictedTrack(
                track_id=matched_detections[index].track_id,
                instrument_id=detection.instrument_id,
                bbox_tlwh=detection.bbox_tlwh,
                score=detection.score,
                age=matched_detections[index].age,
            )
            for index, detection in enumerate(detections)
        )


__all__ = ["CausalHungarianAssociator", "InstrumentDetection"]
