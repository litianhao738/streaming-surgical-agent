from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from surgical_agent.data.schemas import BoundingBox, DatasetSplit
from surgical_agent.tracking.config import load_tracker_training_config
from surgical_agent.tracking.training_data import (
    DetectionTrainingRecord,
    DetectionTrainingTarget,
    InstrumentDetectionDataset,
    instrument_to_model_label,
    model_to_instrument_label,
    normalized_tlwh_to_xyxy,
    supervision_qualified_training_video_ids,
)


def test_instrument_detector_mapping_reserves_zero_for_background() -> None:
    assert tuple(instrument_to_model_label(value) for value in range(7)) == tuple(
        range(1, 8)
    )
    assert tuple(model_to_instrument_label(value) for value in range(1, 8)) == tuple(
        range(7)
    )
    with pytest.raises(ValueError, match="instrument"):
        instrument_to_model_label(7)
    with pytest.raises(ValueError, match="model label"):
        model_to_instrument_label(0)


def test_normalized_tlwh_is_converted_to_pixel_xyxy() -> None:
    assert normalized_tlwh_to_xyxy(
        BoundingBox(0.25, 0.5, 0.5, 0.25), width=200, height=100
    ) == (50.0, 50.0, 150.0, 75.0)
    with pytest.raises(ValueError, match="valid normalized"):
        normalized_tlwh_to_xyxy(
            BoundingBox(-1.0, -1.0, -1.0, -1.0), width=200, height=100
        )


def test_training_video_selection_excludes_disabled_vid31_instance_fields() -> None:
    entries = {
        "VID02": SimpleNamespace(split=DatasetSplit.TRAINING),
        "VID31": SimpleNamespace(split=DatasetSplit.TRAINING),
        "VID01": SimpleNamespace(split=DatasetSplit.VALIDATION),
    }
    qualified = SimpleNamespace(
        allows=lambda field: field not in {"instrument", "bbox", "track_ids"}
    )
    adapter = SimpleNamespace(
        entries=entries,
        derived_manifest=SimpleNamespace(videos={"VID31": qualified}),
    )

    assert supervision_qualified_training_video_ids(adapter) == ("VID02",)


def test_tracker_yaml_loads_frozen_architecture() -> None:
    path = Path("configs/tracker/fasterrcnn_mobilenet_v3.yaml")
    config = load_tracker_training_config(path)

    assert config.architecture == "fasterrcnn_mobilenet_v3_large_fpn"
    assert config.initial_weights == "COCO_V1"
    assert config.num_classes == 8
    assert config.oof_folds == 3
    assert 0.0 < config.score_threshold < 1.0


def test_detection_dataset_returns_torchvision_target(tmp_path: Path) -> None:
    image_path = tmp_path / "000001.png"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
    dataset = InstrumentDetectionDataset(
        (
            DetectionTrainingRecord(
                video_id="VID02",
                frame_id=1,
                media_path=image_path,
                targets=(
                    DetectionTrainingTarget(
                        instrument_id=0,
                        bbox=BoundingBox(0.1, 0.2, 0.4, 0.4),
                        is_crowd=False,
                    ),
                ),
            ),
        )
    )

    image, target = dataset[0]

    assert tuple(image.shape) == (3, 10, 20)
    assert target["labels"].tolist() == [1]
    assert target["boxes"][0].tolist() == pytest.approx([2.0, 2.0, 10.0, 6.0])
