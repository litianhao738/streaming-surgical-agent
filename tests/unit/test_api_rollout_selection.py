"""Gold-free rollout sample selection contracts."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from surgical_agent.data.api_rollout_selection import resolve_rollout_selection
from surgical_agent.data.dataset import (
    PNG_ALIGNMENT_VERSION,
    CholecTrack20DatasetAdapter,
)
from surgical_agent.data.schemas import DatasetSplit, InferenceSample


class FakeInferenceSource:
    def __init__(
        self,
        splits: dict[str, DatasetSplit],
        samples: dict[str, tuple[InferenceSample, ...]],
    ) -> None:
        self.entries = {
            video_id: SimpleNamespace(split=split)
            for video_id, split in splits.items()
        }
        self._samples = samples

    def iter_inference_video(
        self,
        video_id: str,
        *,
        max_samples: int | None = None,
    ) -> Iterator[InferenceSample]:
        selected = self._samples[video_id]
        yield from selected if max_samples is None else selected[:max_samples]


def _sample(
    video_id: str,
    frame_id: int,
    split: DatasetSplit = DatasetSplit.VALIDATION,
) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_id,
        causal_frame_ids=(frame_id,),
        media_refs=(f"{frame_id}.png",),
        source_split=split,
        alignment_version=PNG_ALIGNMENT_VERSION,
    )


def test_engineering_selection_requires_one_video_and_positive_limit() -> None:
    """A missing video or unbounded limit would make engineering calls unsafe."""

    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {"VID30": (_sample("VID30", 1), _sample("VID30", 2))},
    )

    selected = resolve_rollout_selection(
        source,
        mode="engineering",
        video_id="vid30",
        max_frames=1,
        split=None,
    )

    assert selected.video_ids == ("VID30",)
    assert [sample.target_frame_id for sample in selected.samples] == [1]
    assert dict(selected.frame_counts) == {"VID30": 1}
    with pytest.raises(TypeError):
        selected.frame_counts["VID30"] = 2  # type: ignore[index]


def test_engineering_selection_can_start_at_an_exact_target_frame() -> None:
    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {
            "VID30": tuple(
                _sample("VID30", frame_id) for frame_id in (1, 2, 3, 4, 5, 6, 7)
            )
        },
    )

    selected = resolve_rollout_selection(
        source,
        mode="engineering",
        video_id="VID30",
        max_frames=1,
        split=None,
        target_frame_id=6,
    )

    assert tuple(sample.target_frame_id for sample in selected.samples) == (6,)


@pytest.mark.parametrize(
    ("video_id", "max_frames"),
    [(None, 1), ("VID30", None), ("VID30", 0), ("VID30", True)],
)
def test_engineering_selection_rejects_incomplete_bounds(
    video_id: str | None,
    max_frames: int | None,
) -> None:
    """Relaxing either engineering bound could issue an unplanned API run."""

    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {"VID30": (_sample("VID30", 1),)},
    )

    with pytest.raises((TypeError, ValueError)):
        resolve_rollout_selection(
            source,
            mode="engineering",
            video_id=video_id,
            max_frames=max_frames,
            split=None,
        )


def test_paper_selection_uses_every_sorted_video_and_rejects_truncation() -> None:
    """Skipping, reordering, or truncating official paper samples breaks coverage."""

    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION, "VID11": DatasetSplit.VALIDATION},
        {
            "VID11": (_sample("VID11", 1), _sample("VID11", 2)),
            "VID30": (_sample("VID30", 1), _sample("VID30", 2)),
        },
    )

    selected = resolve_rollout_selection(
        source,
        mode="paper",
        video_id=None,
        max_frames=None,
        split="validation",
    )

    assert selected.video_ids == ("VID11", "VID30")
    assert [sample.video_id for sample in selected.samples] == [
        "VID11",
        "VID11",
        "VID30",
        "VID30",
    ]
    assert selected.expected_provider_calls == 4
    with pytest.raises(ValueError, match="paper mode forbids truncation"):
        resolve_rollout_selection(
            source,
            mode="paper",
            video_id=None,
            max_frames=1,
            split="validation",
        )


@pytest.mark.parametrize("split", ["train", "training"])
def test_paper_selection_rejects_training_split(split: str) -> None:
    """Allowing a training rollout would violate the paper-run split contract."""

    source = FakeInferenceSource(
        {"VID02": DatasetSplit.TRAINING},
        {"VID02": (_sample("VID02", 1, DatasetSplit.TRAINING),)},
    )

    with pytest.raises(ValueError, match="validation or testing"):
        resolve_rollout_selection(
            source,
            mode="paper",
            video_id=None,
            max_frames=None,
            split=split,
        )


def test_selection_rejects_out_of_order_samples() -> None:
    """A source that emits non-video-major frames would make API calls noncanonical."""

    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {"VID30": (_sample("VID30", 2), _sample("VID30", 1))},
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_rollout_selection(
            source,
            mode="engineering",
            video_id="VID30",
            max_frames=2,
            split=None,
        )


def test_inference_iterator_preserves_configured_causal_buffer() -> None:
    """The adapter emits the configured contiguous causal buffer."""

    adapter = object.__new__(CholecTrack20DatasetAdapter)
    adapter.causal_window_size = 5
    adapter.expected_frame_id_step = 1
    entry = SimpleNamespace(video_id="VID30", split=DatasetSplit.VALIDATION)
    resolver = SimpleNamespace(
        available_frame_ids=(1, 2, 3, 4),
        resolve=lambda frame_id: SimpleNamespace(media_path=f"{frame_id}.png"),
    )
    adapter._entry = lambda video_id: entry
    adapter._derived = lambda video_id: None
    adapter._png_resolver = lambda entry, derived: resolver

    sample = tuple(adapter.iter_inference_video("VID30"))[-1]

    assert sample.causal_frame_ids == (1, 2, 3, 4)


def test_inference_iterator_does_not_bridge_a_frame_clock_gap() -> None:
    adapter = object.__new__(CholecTrack20DatasetAdapter)
    adapter.causal_window_size = 3
    adapter.expected_frame_id_step = 25
    entry = SimpleNamespace(video_id="VID30", split=DatasetSplit.VALIDATION)
    resolver = SimpleNamespace(
        available_frame_ids=(1, 26, 51, 101, 126, 151),
        resolve=lambda frame_id: SimpleNamespace(media_path=f"{frame_id}.png"),
    )
    adapter._entry = lambda video_id: entry
    adapter._derived = lambda video_id: None
    adapter._png_resolver = lambda entry, derived: resolver

    samples = {
        sample.target_frame_id: sample
        for sample in adapter.iter_inference_video("VID30")
    }

    assert samples[51].causal_frame_ids == (1, 26, 51)
    assert samples[101].causal_frame_ids == (101,)
    assert samples[126].causal_frame_ids == (101, 126)
    assert samples[151].causal_frame_ids == (101, 126, 151)
