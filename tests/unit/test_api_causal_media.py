"""Exact causal API media loading contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import (
    MP4_ALIGNMENT_VERSION,
    PNG_ALIGNMENT_VERSION,
    DatasetContractError,
)
from surgical_agent.data.schemas import DatasetSplit, InferenceSample


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (6, 4), color).save(path)


class RecordingVideoFrameReader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[int, ...]]] = []

    def read_many_rgb(
        self,
        path: Path,
        decoder_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]:
        self.calls.append((path, decoder_indices))
        return tuple(
            np.full((4, 6, 3), index % 255, dtype=np.uint8)
            for index in decoder_indices
        )


def _sample(
    *,
    video_id: str,
    frame_ids: tuple[int, ...],
    media_refs: tuple[str, ...],
    split: DatasetSplit,
    alignment: str,
) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_ids[-1],
        causal_frame_ids=frame_ids,
        media_refs=media_refs,
        source_split=split,
        alignment_version=alignment,
    )


def test_png_window_loads_native_rgb_and_replaces_paths_with_safe_ids(
    tmp_path: Path,
) -> None:
    _write_rgb(tmp_path / "1.png", (255, 0, 0))
    _write_rgb(tmp_path / "2.png", (0, 255, 0))
    sample = _sample(
        video_id="VID30",
        frame_ids=(1, 2),
        media_refs=(str(tmp_path / "1.png"), str(tmp_path / "2.png")),
        split=DatasetSplit.VALIDATION,
        alignment=PNG_ALIGNMENT_VERSION,
    )

    loaded = CausalApiMediaLoader().load(sample)

    assert loaded.frames.shape == (2, 3, 4, 6)
    assert loaded.runtime_sample.media_refs == (
        "cholectrack20:VID30:frame:1",
        "cholectrack20:VID30:frame:2",
    )
    assert str(tmp_path) not in repr(loaded.runtime_sample)


def test_test_mp4_uses_annotation_id_minus_one_indices(tmp_path: Path) -> None:
    video = tmp_path / "vid01.mp4"
    video.touch()
    reader = RecordingVideoFrameReader()
    sample = _sample(
        video_id="VID01",
        frame_ids=(1, 26, 51),
        media_refs=(str(video), str(video), str(video)),
        split=DatasetSplit.TESTING,
        alignment=MP4_ALIGNMENT_VERSION,
    )

    loaded = CausalApiMediaLoader(video_reader=reader).load(sample)

    assert reader.calls == [(video.resolve(), (0, 25, 50))]
    assert loaded.frames.shape[0] == 3


def test_png_window_rejects_mismatched_frame_geometry(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "1.png", (255, 0, 0))
    Image.new("RGB", (5, 4), (0, 255, 0)).save(tmp_path / "2.png")
    sample = _sample(
        video_id="VID30",
        frame_ids=(1, 2),
        media_refs=(str(tmp_path / "1.png"), str(tmp_path / "2.png")),
        split=DatasetSplit.VALIDATION,
        alignment=PNG_ALIGNMENT_VERSION,
    )

    with pytest.raises(DatasetContractError, match="geometry"):
        CausalApiMediaLoader().load(sample)


def test_testing_sample_rejects_non_mp4_media_path(tmp_path: Path) -> None:
    png = tmp_path / "1.png"
    _write_rgb(png, (255, 0, 0))
    sample = _sample(
        video_id="VID01",
        frame_ids=(1,),
        media_refs=(str(png),),
        split=DatasetSplit.TESTING,
        alignment=MP4_ALIGNMENT_VERSION,
    )

    with pytest.raises(DatasetContractError, match="MP4"):
        CausalApiMediaLoader().load(sample)


def test_sample_rejects_wrong_media_alignment(tmp_path: Path) -> None:
    png = tmp_path / "1.png"
    _write_rgb(png, (255, 0, 0))
    sample = _sample(
        video_id="VID30",
        frame_ids=(1,),
        media_refs=(str(png),),
        split=DatasetSplit.VALIDATION,
        alignment=MP4_ALIGNMENT_VERSION,
    )

    with pytest.raises(DatasetContractError, match="alignment"):
        CausalApiMediaLoader().load(sample)


@pytest.mark.parametrize("frame_ids", ((-1, 1), (0,)))
def test_testing_sample_rejects_non_positive_annotation_frame_ids(
    tmp_path: Path,
    frame_ids: tuple[int, ...],
) -> None:
    video = tmp_path / "vid01.mp4"
    video.touch()
    sample = _sample(
        video_id="VID01",
        frame_ids=frame_ids,
        media_refs=tuple(str(video) for _ in frame_ids),
        split=DatasetSplit.TESTING,
        alignment=MP4_ALIGNMENT_VERSION,
    )

    with pytest.raises(DatasetContractError, match="positive"):
        CausalApiMediaLoader().load(sample)
