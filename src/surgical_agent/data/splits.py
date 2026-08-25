"""Official-directory split discovery and train-only safety guards."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.data.schemas import DatasetSplit

OFFICIAL_SPLIT_DIRECTORIES = {
    DatasetSplit.TRAINING: "Training",
    DatasetSplit.VALIDATION: "Validation",
    DatasetSplit.TESTING: "Testing",
}


class SplitSafetyError(ValueError):
    """Raised when a video crosses or contradicts an official split."""


@dataclass(frozen=True)
class SplitManifestEntry:
    video_id: str
    split: DatasetSplit
    source: str
    available_modalities: tuple[str, ...]
    annotation_file: str
    media_source: str


def discover_official_split_manifest(
    dataset_root: str | Path,
) -> tuple[SplitManifestEntry, ...]:
    """Discover the release split directly from local official directories."""

    root = Path(dataset_root).expanduser().resolve()
    entries: list[SplitManifestEntry] = []
    seen: dict[str, DatasetSplit] = {}
    for split, directory_name in OFFICIAL_SPLIT_DIRECTORIES.items():
        split_dir = root / directory_name
        if not split_dir.is_dir():
            raise SplitSafetyError(f"Missing official split directory: {split_dir}")
        for video_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            annotation_file = video_dir / f"{video_dir.name.lower()}.json"
            if not annotation_file.is_file():
                raise SplitSafetyError(
                    f"Missing canonical raw annotation JSON: {annotation_file}"
                )
            try:
                raw = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
                video = raw["video"]
                video_id = str(video["name"]).upper()
                declared_split = DatasetSplit.parse(str(video["split"]))
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise SplitSafetyError(
                    f"Invalid split metadata in {annotation_file}: {exc}"
                ) from exc
            if declared_split is not split:
                raise SplitSafetyError(
                    f"{video_id} declares {declared_split.value} inside {split.value}"
                )
            if video_id in seen:
                raise SplitSafetyError(
                    f"{video_id} belongs to both {seen[video_id].value} and {split.value}"
                )
            seen[video_id] = split

            frames_dir = video_dir / "Frames"
            mp4_files = sorted(video_dir.glob("*.mp4"))
            modalities = ["annotation_json"]
            media_source: Path
            if frames_dir.is_dir():
                modalities.append("png_frames")
                media_source = frames_dir
            elif len(mp4_files) == 1:
                modalities.append("mp4")
                media_source = mp4_files[0]
            else:
                raise SplitSafetyError(f"No unambiguous media source in {video_dir}")

            entries.append(
                SplitManifestEntry(
                    video_id=video_id,
                    split=split,
                    source="local official release directory + raw JSON video.split",
                    available_modalities=tuple(modalities),
                    annotation_file=str(annotation_file.resolve()),
                    media_source=str(media_source.resolve()),
                )
            )
    return tuple(entries)


def require_train_only(entries: Iterable[SplitManifestEntry]) -> None:
    """Reject validation/test records from a train-derived statistic builder."""

    invalid = sorted(
        entry.video_id for entry in entries if entry.split is not DatasetSplit.TRAINING
    )
    if invalid:
        raise SplitSafetyError(
            f"Train-only operation received held-out videos: {invalid}"
        )
