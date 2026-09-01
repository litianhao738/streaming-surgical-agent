"""Deterministic selection and evidence for a bounded causal image window."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor
from torch.nn import functional


@dataclass(frozen=True)
class AdaptiveCausalWindow:
    """Selected image indices plus auditable evidence from the full window."""

    selected_indices: tuple[int, ...]
    evidence: Mapping[str, object]


def _track_frames(snapshot: Mapping[str, object]) -> dict[int, tuple[Mapping[str, object], ...]]:
    if snapshot.get("status") != "AVAILABLE":
        return {}
    raw_frames = snapshot.get("frames", ())
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        return {}
    result: dict[int, tuple[Mapping[str, object], ...]] = {}
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, Mapping):
            continue
        frame_id = raw_frame.get("frame_id")
        raw_tracks = raw_frame.get("tracks", ())
        if (
            not isinstance(frame_id, int)
            or isinstance(frame_id, bool)
            or not isinstance(raw_tracks, Sequence)
            or isinstance(raw_tracks, (str, bytes))
        ):
            continue
        tracks = tuple(track for track in raw_tracks if isinstance(track, Mapping))
        result[frame_id] = tracks
    return result


def _track_id(track: Mapping[str, object]) -> str | None:
    value = track.get("track_id")
    return value if isinstance(value, str) and value else None


def _track_center(track: Mapping[str, object]) -> tuple[float, float] | None:
    bbox = track.get("bbox_tlwh")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    return x + width / 2.0, y + height / 2.0


def _instrument_id(track: Mapping[str, object]) -> int | None:
    value = track.get("instrument_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _track_change(
    first: tuple[Mapping[str, object], ...],
    second: tuple[Mapping[str, object], ...],
) -> tuple[float, float]:
    first_by_id = {
        track_id: track
        for track in first
        if (track_id := _track_id(track)) is not None
    }
    second_by_id = {
        track_id: track
        for track in second
        if (track_id := _track_id(track)) is not None
    }
    first_ids = set(first_by_id)
    second_ids = set(second_by_id)
    first_instruments = {
        value for track in first if (value := _instrument_id(track)) is not None
    }
    second_instruments = {
        value for track in second if (value := _instrument_id(track)) is not None
    }
    instrument_union = first_instruments | second_instruments
    set_change = (
        0.0
        if not instrument_union
        else 1.0
        - len(first_instruments & second_instruments) / len(instrument_union)
    )

    displacements: list[float] = []
    for track_id in first_ids & second_ids:
        first_center = _track_center(first_by_id[track_id])
        second_center = _track_center(second_by_id[track_id])
        if first_center is None or second_center is None:
            continue
        distance = math.dist(first_center, second_center) / math.sqrt(2.0)
        displacements.append(min(max(distance, 0.0), 1.0))
    if not displacements:
        for instrument_id in first_instruments & second_instruments:
            first_centers = [
                center
                for track in first
                if _instrument_id(track) == instrument_id
                and (center := _track_center(track)) is not None
            ]
            second_centers = [
                center
                for track in second
                if _instrument_id(track) == instrument_id
                and (center := _track_center(track)) is not None
            ]
            if first_centers and second_centers:
                distance = min(
                    math.dist(first_center, second_center)
                    for first_center in first_centers
                    for second_center in second_centers
                ) / math.sqrt(2.0)
                displacements.append(min(max(distance, 0.0), 1.0))
    displacement = sum(displacements) / len(displacements) if displacements else 0.0
    return displacement, set_change


def _round(value: float) -> float:
    return round(float(value), 6)


def select_adaptive_causal_window(
    frames: Tensor,
    *,
    frame_ids: tuple[int, ...],
    track_snapshot: Mapping[str, object],
    max_images: int,
) -> AdaptiveCausalWindow:
    """Use every frame for evidence while uploading only informative images.

    The target frame is always selected. Historical frames are ranked using
    visual change, predicted-track displacement, track-set change, and recency.
    No annotation or future-frame information is admitted.
    """

    if frames.ndim != 4 or frames.shape[0] != len(frame_ids):
        raise ValueError("adaptive window requires aligned [T,C,H,W] frames")
    if not frame_ids or tuple(sorted(set(frame_ids))) != frame_ids:
        raise ValueError("adaptive window frame IDs must be unique and increasing")
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images <= 0:
        raise ValueError("adaptive window max_images must be positive")

    frame_count = len(frame_ids)
    selected_count = min(max_images, frame_count)
    pooled = functional.adaptive_avg_pool2d(frames.to(dtype=torch.float32), (24, 24))
    visual_steps = [0.0]
    for index in range(1, frame_count):
        visual_steps.append(float((pooled[index] - pooled[index - 1]).abs().mean()))

    tracks_by_frame = _track_frames(track_snapshot)
    tracker_available = bool(tracks_by_frame)
    scores: list[dict[str, object]] = []
    for index, frame_id in enumerate(frame_ids):
        if index + 1 < frame_count:
            next_frame_id = frame_ids[index + 1]
            displacement, set_change = _track_change(
                tracks_by_frame.get(frame_id, ()),
                tracks_by_frame.get(next_frame_id, ()),
            )
            visual_change = max(visual_steps[index], visual_steps[index + 1])
        else:
            displacement = 0.0
            set_change = 0.0
            visual_change = visual_steps[index]
        recency = 1.0 if frame_count == 1 else index / (frame_count - 1)
        total = (
            0.40 * min(visual_change / 0.20, 1.0)
            + 0.30 * displacement
            + 0.15 * set_change
            + 0.15 * recency
        )
        scores.append(
            {
                "frame_id": frame_id,
                "visual_change": _round(visual_change),
                "track_displacement": _round(displacement),
                "track_set_change": _round(set_change),
                "recency": _round(recency),
                "selection_score": _round(total),
            }
        )

    historical_count = max(0, selected_count - 1)
    ranked_history = sorted(
        range(max(0, frame_count - 1)),
        key=lambda index: (float(scores[index]["selection_score"]), index),
        reverse=True,
    )
    selected_indices = tuple(sorted((*ranked_history[:historical_count], frame_count - 1)))
    selected_frame_ids = tuple(frame_ids[index] for index in selected_indices)
    omitted_frame_ids = tuple(
        frame_id for index, frame_id in enumerate(frame_ids) if index not in selected_indices
    )

    target_track_ids = {
        track_id
        for track in tracks_by_frame.get(frame_ids[-1], ())
        if (track_id := _track_id(track)) is not None
    }
    historical_sets = [
        {
            track_id
            for track in tracks_by_frame.get(frame_id, ())
            if (track_id := _track_id(track)) is not None
        }
        for frame_id in frame_ids[:-1]
    ]
    persistence = (
        0.0
        if not target_track_ids or not historical_sets
        else sum(
            sum(track_id in values for values in historical_sets) / len(historical_sets)
            for track_id in target_track_ids
        )
        / len(target_track_ids)
    )
    mean_visual_change = sum(visual_steps[1:]) / max(1, frame_count - 1)
    mean_track_displacement = sum(
        float(item["track_displacement"]) for item in scores[:-1]
    ) / max(1, frame_count - 1)
    motion_state = (
        "changing"
        if mean_visual_change >= 0.05 or mean_track_displacement >= 0.03
        else "stable"
    )
    evidence = MappingProxyType(
        {
            "schema_version": "adaptive_causal_window_v1",
            "candidate_frame_ids": frame_ids,
            "selected_image_frame_ids": selected_frame_ids,
            "omitted_image_frame_ids": omitted_frame_ids,
            "frame_scores": tuple(MappingProxyType(item) for item in scores),
            "summary": MappingProxyType(
                {
                    "window_frame_count": frame_count,
                    "uploaded_image_count": len(selected_indices),
                    "tracker_available": tracker_available,
                    "target_track_count": len(target_track_ids),
                    "target_track_persistence": _round(persistence),
                    "mean_visual_change": _round(mean_visual_change),
                    "mean_track_displacement": _round(mean_track_displacement),
                    "motion_state": motion_state,
                }
            ),
        }
    )
    return AdaptiveCausalWindow(
        selected_indices=selected_indices,
        evidence=evidence,
    )


__all__ = ["AdaptiveCausalWindow", "select_adaptive_causal_window"]
