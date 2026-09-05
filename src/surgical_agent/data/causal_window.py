"""Deterministic strict-causal visual window policy."""

from __future__ import annotations

from collections.abc import Iterable


def build_causal_frame_ids(
    available_frame_ids: Iterable[int],
    *,
    target_frame_id: int,
    max_frames: int,
    expected_frame_id_step: int | None = None,
) -> tuple[int, ...]:
    """Return the latest contiguous frames ending at ``target_frame_id``.

    When ``expected_frame_id_step`` is supplied, history before a missing
    observation is excluded rather than being presented as adjacent motion.
    """

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if expected_frame_id_step is not None and (
        not isinstance(expected_frame_id_step, int)
        or isinstance(expected_frame_id_step, bool)
        or expected_frame_id_step <= 0
    ):
        raise ValueError("expected_frame_id_step must be a positive integer")
    ordered = tuple(sorted(set(available_frame_ids)))
    if target_frame_id not in ordered:
        raise ValueError(f"target_frame_id {target_frame_id} is not available")
    causal = tuple(frame_id for frame_id in ordered if frame_id <= target_frame_id)
    result = causal[-max_frames:]
    if expected_frame_id_step is not None:
        first = len(result) - 1
        while (
            first > 0
            and result[first] - result[first - 1] == expected_frame_id_step
        ):
            first -= 1
        result = result[first:]
    if not result or result[-1] != target_frame_id:
        raise AssertionError("Causal window construction lost the target frame")
    if max(result) > target_frame_id:
        raise AssertionError("Future visual evidence entered the causal window")
    return result
