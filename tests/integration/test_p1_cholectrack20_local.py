"""Read-only integration checks against the declared local CholecTrack20 release."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.data.splits import discover_official_split_manifest

DATASET_ROOT = Path(
    os.environ.get("CHOLECTRACK20_ROOT", "__external_dataset_not_configured__")
)
pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="Declared local CholecTrack20 dataset is unavailable",
)


def _png_ids(path: Path) -> set[int]:
    return {int(item.stem) for item in path.glob("*.png")}


def test_local_release_has_official_split_partition_and_parses_all_json() -> None:
    entries = discover_official_split_manifest(DATASET_ROOT)
    split_counts = Counter(entry.split for entry in entries)

    assert split_counts == {
        DatasetSplit.TRAINING: 10,
        DatasetSplit.VALIDATION: 2,
        DatasetSplit.TESTING: 8,
    }
    assert len({entry.video_id for entry in entries}) == 20

    videos = {
        entry.video_id: parse_annotation_file(
            entry.annotation_file,
            expected_split=entry.split,
        )
        for entry in entries
    }
    assert sum(len(video.frames) for video in videos.values()) == 35_009
    assert (
        sum(len(frame.instances) for video in videos.values() for frame in video.frames)
        == 65_247
    )


def test_local_alignment_is_explicitly_qualified_not_silently_repaired() -> None:
    entries = {
        entry.video_id: entry
        for entry in discover_official_split_manifest(DATASET_ROOT)
    }
    videos = {
        video_id: parse_annotation_file(entry.annotation_file)
        for video_id, entry in entries.items()
        if video_id in {"VID02", "VID30", "VID31"}
    }

    vid02_media = _png_ids(Path(entries["VID02"].media_source))
    vid30_media = _png_ids(Path(entries["VID30"].media_source))
    vid31_media = _png_ids(Path(entries["VID31"].media_source))

    assert vid02_media == set(videos["VID02"].frame_ids)
    assert vid30_media != set(videos["VID30"].frame_ids)
    assert vid31_media != set(videos["VID31"].frame_ids)
    assert vid30_media == set(videos["VID31"].frame_ids)
    assert videos["VID30"].split is DatasetSplit.VALIDATION
    assert videos["VID31"].split is DatasetSplit.TRAINING
