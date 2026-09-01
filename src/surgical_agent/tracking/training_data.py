"""Read-only detector training helpers for CholecTrack20."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch
from PIL import Image

from surgical_agent.cli.progress import progress_bar
from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.schemas import BoundingBox, DatasetSplit


@dataclass(frozen=True)
class DetectionTrainingTarget:
    instrument_id: int
    bbox: BoundingBox
    is_crowd: bool


@dataclass(frozen=True)
class DetectionTrainingRecord:
    video_id: str
    frame_id: int
    media_path: Path
    targets: tuple[DetectionTrainingTarget, ...]


class InstrumentDetectionDataset:
    """Torchvision-compatible current-frame detector dataset."""

    def __init__(self, records: tuple[DetectionTrainingRecord, ...]) -> None:
        if not records:
            raise ValueError("detector dataset requires at least one record")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        with Image.open(record.media_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        boxes = [
            normalized_tlwh_to_xyxy(target.bbox, width=width, height=height)
            for target in record.targets
        ]
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(
            [instrument_to_model_label(target.instrument_id) for target in record.targets],
            dtype=torch.int64,
        )
        area = (box_tensor[:, 2] - box_tensor[:, 0]) * (
            box_tensor[:, 3] - box_tensor[:, 1]
        )
        return tensor, {
            "boxes": box_tensor,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.tensor(
                [int(target.is_crowd) for target in record.targets],
                dtype=torch.int64,
            ),
        }


def instrument_to_model_label(instrument_id: int) -> int:
    lower, upper = TASK_ID_BOUNDS["instrument"]
    if (
        not isinstance(instrument_id, int)
        or isinstance(instrument_id, bool)
        or not lower <= instrument_id <= upper
    ):
        raise ValueError("instrument ID is outside the CholecTrack20 ontology")
    return instrument_id + 1


def model_to_instrument_label(model_label: int) -> int:
    if (
        not isinstance(model_label, int)
        or isinstance(model_label, bool)
        or not 1 <= model_label <= 7
    ):
        raise ValueError("model label must be in 1..7")
    return model_label - 1


def normalized_tlwh_to_xyxy(
    bbox: BoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not bbox.has_positive_extent or not bbox.is_inside_unit_frame:
        raise ValueError("bbox must be a valid normalized positive box")
    return (
        bbox.x * width,
        bbox.y * height,
        (bbox.x + bbox.width) * width,
        (bbox.y + bbox.height) * height,
    )


def supervision_qualified_training_video_ids(adapter: Any) -> tuple[str, ...]:
    """Select Training videos whose audited instance fields are usable."""

    selected: list[str] = []
    derived_videos = adapter.derived_manifest.videos
    for video_id, entry in sorted(adapter.entries.items()):
        if entry.split is not DatasetSplit.TRAINING:
            continue
        derived = derived_videos.get(video_id)
        if derived is not None and not all(
            derived.allows(field) for field in ("instrument", "bbox", "track_ids")
        ):
            continue
        selected.append(video_id)
    return tuple(selected)


def deterministic_video_folds(
    video_ids: tuple[str, ...],
    fold_count: int,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(fold_count, int) or isinstance(fold_count, bool) or fold_count < 2:
        raise ValueError("fold_count must be at least two")
    ordered = tuple(sorted(video_ids))
    if len(ordered) < fold_count:
        raise ValueError("fold_count cannot exceed the number of videos")
    return tuple(tuple(ordered[index::fold_count]) for index in range(fold_count))


def build_detection_training_records(
    adapter: Any,
    video_ids: tuple[str, ...],
    *,
    max_samples_per_video: int | None = None,
    progress_enabled: bool = False,
    progress_file: IO[str] | None = None,
) -> tuple[DetectionTrainingRecord, ...]:
    qualified = set(supervision_qualified_training_video_ids(adapter))
    if set(video_ids) - qualified:
        raise ValueError("detector training received a non-qualified video")
    records: list[DetectionTrainingRecord] = []
    with progress_bar(
        total=len(video_ids),
        description="Tracker data",
        unit="video",
        enabled=progress_enabled,
        file=progress_file,
    ) as progress:
        for video_id in video_ids:
            progress.set_postfix_str(video_id, refresh=False)
            for resolved in adapter.iter_video(
                video_id,
                max_samples=max_samples_per_video,
            ):
                evaluation = resolved.evaluation
                if evaluation is None or not evaluation.instance_supervision_available:
                    continue
                targets = tuple(
                    DetectionTrainingTarget(
                        instrument_id=instance.instrument_id,
                        bbox=instance.bbox,
                        is_crowd=False,
                    )
                    for instance in evaluation.instances
                    if instance.mask.instrument
                    and instance.bbox.has_positive_extent
                    and instance.bbox.is_inside_unit_frame
                )
                if not targets:
                    continue
                records.append(
                    DetectionTrainingRecord(
                        video_id=video_id,
                        frame_id=resolved.inference.target_frame_id,
                        media_path=Path(resolved.inference.media_refs[-1]).resolve(),
                        targets=targets,
                    )
                )
            progress.update()
    if not records:
        raise ValueError("no qualified detector training samples were found")
    return tuple(records)


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


__all__ = [
    "DetectionTrainingRecord",
    "DetectionTrainingTarget",
    "InstrumentDetectionDataset",
    "build_detection_training_records",
    "detection_collate",
    "deterministic_video_folds",
    "instrument_to_model_label",
    "model_to_instrument_label",
    "normalized_tlwh_to_xyxy",
    "supervision_qualified_training_video_ids",
]
