"""Gold-free, deterministic selection for API rollout samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from surgical_agent.data.schemas import DatasetSplit, InferenceSample


@dataclass(frozen=True)
class RolloutSelection:
    """Immutable ordered rollout plan containing runtime samples only."""

    mode: str
    split: DatasetSplit | None
    video_ids: tuple[str, ...]
    samples: tuple[InferenceSample, ...]
    frame_counts: Mapping[str, int]

    @property
    def expected_provider_calls(self) -> int:
        return len(self.samples)


class _RolloutEntry(Protocol):
    split: DatasetSplit


class _InferenceSource(Protocol):
    entries: Mapping[str, _RolloutEntry]

    def iter_inference_video(
        self,
        video_id: str,
        *,
        max_samples: int | None = None,
    ) -> Iterator[InferenceSample]: ...


def _require_positive_int(value: int | None, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_samples(
    samples: Iterable[InferenceSample],
    *,
    video_ids: tuple[str, ...],
) -> tuple[InferenceSample, ...]:
    selected = tuple(samples)
    grouped: dict[str, list[InferenceSample]] = {video_id: [] for video_id in video_ids}
    previous_video_index = -1
    previous_frame_id = -1
    for sample in selected:
        try:
            video_index = video_ids.index(sample.video_id)
        except ValueError as exc:
            raise ValueError(
                f"sample video {sample.video_id!r} does not match selected videos"
            ) from exc
        if video_index < previous_video_index:
            raise ValueError("samples must be video-major in selected video order")
        if video_index != previous_video_index:
            previous_frame_id = -1
        if sample.target_frame_id <= previous_frame_id:
            raise ValueError("sample frame IDs must be strictly increasing within a video")
        grouped[sample.video_id].append(sample)
        previous_video_index = video_index
        previous_frame_id = sample.target_frame_id
    if tuple(video_id for video_id in video_ids if grouped[video_id]) != video_ids:
        raise ValueError("samples must match selected video_ids exactly")
    return selected


def resolve_rollout_selection(
    adapter: _InferenceSource,
    *,
    mode: str,
    video_id: str | None,
    max_frames: int | None,
    split: str | None,
) -> RolloutSelection:
    """Resolve an explicit engineering run or a complete official paper split."""

    if not isinstance(mode, str):
        raise TypeError("mode must be 'engineering' or 'paper'")
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"engineering", "paper"}:
        raise ValueError("mode must be 'engineering' or 'paper'")

    entries = adapter.entries
    iter_inference_video = adapter.iter_inference_video
    if normalized_mode == "engineering":
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("engineering mode requires one video_id")
        if split is not None:
            raise ValueError("engineering mode does not accept split")
        limit = _require_positive_int(max_frames, name="max_frames")
        video_ids = (video_id.upper(),)
        selected_split: DatasetSplit | None = None
    else:
        if video_id is not None:
            raise ValueError("paper mode does not accept video_id")
        if max_frames is not None:
            raise ValueError("paper mode forbids truncation")
        if not isinstance(split, str):
            raise TypeError("paper mode requires split")
        selected_split = DatasetSplit.parse(split)
        if selected_split not in {DatasetSplit.VALIDATION, DatasetSplit.TESTING}:
            raise ValueError("paper mode requires validation or testing split")
        video_ids = tuple(
            sorted(
                entry_video_id
                for entry_video_id, entry in entries.items()
                if entry.split is selected_split
            )
        )
        if not video_ids:
            raise ValueError(f"paper mode found no videos for {selected_split.value}")
        limit = None

    samples = _validate_samples(
        (
            sample
            for selected_video_id in video_ids
            for sample in iter_inference_video(selected_video_id, max_samples=limit)
        ),
        video_ids=video_ids,
    )
    frame_counts = MappingProxyType(
        {
            selected_video_id: sum(
                sample.video_id == selected_video_id for sample in samples
            )
            for selected_video_id in video_ids
        }
    )
    return RolloutSelection(
        mode=normalized_mode,
        split=selected_split,
        video_ids=video_ids,
        samples=samples,
        frame_counts=frame_counts,
    )
