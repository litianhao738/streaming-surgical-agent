"""Deterministic strict-causal visual window policy."""

from __future__ import annotations

from collections.abc import Iterable


def build_causal_frame_ids(
    available_frame_ids: Iterable[int],
    *,
    target_frame_id: int,
    max_frames: int,
) -> tuple[int, ...]:
    """Return the latest available frames ending at ``target_frame_id``."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    ordered = tuple(sorted(set(available_frame_ids)))
    if target_frame_id not in ordered:
        raise ValueError(f"target_frame_id {target_frame_id} is not available")
    causal = tuple(frame_id for frame_id in ordered if frame_id <= target_frame_id)
    result = causal[-max_frames:]
    if not result or result[-1] != target_frame_id:
        raise AssertionError("Causal window construction lost the target frame")
    if max(result) > target_frame_id:
        raise AssertionError("Future visual evidence entered the causal window")
    return result
