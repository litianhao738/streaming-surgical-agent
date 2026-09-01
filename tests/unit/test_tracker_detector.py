from __future__ import annotations

from pathlib import Path

import pytest
import torch

from surgical_agent.tracking.config import TrackerTrainingConfig
from surgical_agent.tracking.detector import (
    build_instrument_detector,
    calculate_detection_metrics,
    decode_detections,
    load_tracker_checkpoint,
    save_tracker_checkpoint,
)
from surgical_agent.tracking.training_data import deterministic_video_folds


def _config() -> TrackerTrainingConfig:
    return TrackerTrainingConfig(
        architecture="fasterrcnn_mobilenet_v3_large_fpn",
        initial_weights="NONE",
        num_classes=8,
        seed=7,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        num_workers=0,
        min_size=64,
        max_size=64,
        score_threshold=0.35,
        nms_threshold=0.5,
        association_iou_threshold=0.3,
        max_age=1,
        oof_folds=3,
    )


def test_decode_detections_maps_labels_and_normalizes_boxes() -> None:
    detections = decode_detections(
        {
            "boxes": torch.tensor([[20.0, 10.0, 60.0, 30.0], [0.0, 0.0, 5.0, 5.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.1]),
        },
        width=100,
        height=50,
        score_threshold=0.35,
    )

    assert len(detections) == 1
    assert detections[0].instrument_id == 0
    assert detections[0].bbox_tlwh == pytest.approx((0.2, 0.2, 0.4, 0.4))


def test_detection_metrics_report_perfect_ap50() -> None:
    predictions = [[(0, (0.1, 0.1, 0.2, 0.2), 0.9)]]
    targets = [[(0, (0.1, 0.1, 0.2, 0.2))]]

    metrics = calculate_detection_metrics(predictions, targets, num_classes=7)

    assert metrics["ap50_by_instrument"]["0"] == pytest.approx(1.0)
    assert metrics["macro_ap50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_video_folds_are_deterministic_and_disjoint() -> None:
    folds = deterministic_video_folds(("VID09", "VID02", "VID04", "VID03"), 3)

    assert folds == (("VID02", "VID09"), ("VID03",), ("VID04",))
    assert sorted(video for fold in folds for video in fold) == [
        "VID02",
        "VID03",
        "VID04",
        "VID09",
    ]


def test_checkpoint_round_trip_preserves_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = save_tracker_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        metadata={"training_video_ids": ["VID02"], "architecture": "tiny-test"},
    )
    restored = torch.nn.Linear(2, 1)
    metadata = load_tracker_checkpoint(path, model=restored, map_location="cpu")

    assert metadata["training_video_ids"] == ["VID02"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_torchvision_factory_replaces_detector_head_without_download() -> None:
    model = build_instrument_detector(_config(), use_pretrained=False)

    assert model.roi_heads.box_predictor.cls_score.out_features == 8
