"""Torchvision instrument detector and deterministic validation metrics."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from surgical_agent.tracking.associator import InstrumentDetection
from surgical_agent.tracking.config import TrackerTrainingConfig
from surgical_agent.tracking.training_data import model_to_instrument_label

TRACKER_CHECKPOINT_SCHEMA_VERSION = "predicted_tracker_checkpoint_v1"


def build_instrument_detector(
    config: TrackerTrainingConfig,
    *,
    use_pretrained: bool | None = None,
) -> Any:
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Torchvision detection operators are unavailable") from exc
    pretrained = config.initial_weights == "COCO_V1" if use_pretrained is None else use_pretrained
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=config.min_size,
        max_size=config.max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)
    model.roi_heads.nms_thresh = config.nms_threshold
    return model


def decode_detections(
    output: Mapping[str, torch.Tensor],
    *,
    width: int,
    height: int,
    score_threshold: float,
) -> tuple[InstrumentDetection, ...]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("detector output arrays must have equal length")
    detections: list[InstrumentDetection] = []
    for box, label, score in zip(boxes, labels, scores):
        confidence = float(score.item())
        if confidence < score_threshold:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            InstrumentDetection(
                instrument_id=model_to_instrument_label(int(label.item())),
                bbox_tlwh=(
                    x1 / width,
                    y1 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ),
                score=confidence,
            )
        )
    return tuple(detections)


def _tlwh_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def calculate_detection_metrics(
    predictions: Sequence[Sequence[tuple[int, Sequence[float], float]]],
    targets: Sequence[Sequence[tuple[int, Sequence[float]]]],
    *,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target frame counts must match")
    per_class: dict[str, float | None] = {}
    total_tp = total_fp = total_gt = 0
    observed_ap: list[float] = []
    for class_id in range(num_classes):
        ground_truth = {
            frame_index: [tuple(box) for label, box in frame if label == class_id]
            for frame_index, frame in enumerate(targets)
        }
        gt_count = sum(len(boxes) for boxes in ground_truth.values())
        total_gt += gt_count
        ranked = sorted(
            (
                (float(score), frame_index, tuple(box))
                for frame_index, frame in enumerate(predictions)
                for label, box, score in frame
                if label == class_id
            ),
            reverse=True,
        )
        matched = {frame_index: set() for frame_index in ground_truth}
        true_positive: list[float] = []
        false_positive: list[float] = []
        for _, frame_index, predicted_box in ranked:
            candidates = ground_truth[frame_index]
            overlaps = [_tlwh_iou(predicted_box, target_box) for target_box in candidates]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if (
                best >= 0
                and overlaps[best] >= iou_threshold
                and best not in matched[frame_index]
            ):
                matched[frame_index].add(best)
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)
        total_tp += int(sum(true_positive))
        total_fp += int(sum(false_positive))
        if gt_count == 0:
            per_class[str(class_id)] = None
            continue
        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recalls = cumulative_tp / gt_count
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        recall_curve = np.concatenate(([0.0], recalls, [1.0]))
        precision_curve = np.concatenate(([0.0], precisions, [0.0]))
        for index in range(len(precision_curve) - 2, -1, -1):
            precision_curve[index] = max(precision_curve[index], precision_curve[index + 1])
        changes = np.where(recall_curve[1:] != recall_curve[:-1])[0]
        ap = float(
            np.sum(
                (recall_curve[changes + 1] - recall_curve[changes])
                * precision_curve[changes + 1]
            )
        )
        per_class[str(class_id)] = ap
        observed_ap.append(ap)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "schema_version": "tracker_detection_metrics_v1",
        "iou_threshold": float(iou_threshold),
        "ap50_by_instrument": per_class,
        "macro_ap50": float(np.mean(observed_ap)) if observed_ap else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "ground_truth": total_gt,
    }


def save_tracker_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    metadata: Mapping[str, object],
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRACKER_CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "metadata": dict(metadata),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def load_tracker_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device,
) -> dict[str, object]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or payload.get("schema_version") != TRACKER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported tracker checkpoint schema")
    state = payload.get("model_state")
    metadata = payload.get("metadata")
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("tracker checkpoint is incomplete")
    model.load_state_dict(state, strict=True)
    return dict(metadata)


__all__ = [
    "TRACKER_CHECKPOINT_SCHEMA_VERSION",
    "build_instrument_detector",
    "calculate_detection_metrics",
    "decode_detections",
    "load_tracker_checkpoint",
    "save_tracker_checkpoint",
]
