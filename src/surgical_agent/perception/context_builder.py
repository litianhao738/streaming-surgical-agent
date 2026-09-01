"""Construction of the gold-free causal input for joint perception."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from io import BytesIO
from types import MappingProxyType
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.data.schemas import (
    EvaluationTarget,
    FrameSupervisionTarget,
    InferenceSample,
    LabelMask,
)
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.adaptive_window import select_adaptive_causal_window

_FORBIDDEN_SNAPSHOT_KEYS = frozenset(
    {
        "ground_truth",
        "evaluation_target",
        "frame_supervision",
        "label_mask",
        "frame_task_mask",
        "labels",
        "future_state",
    }
)
_GOLD_BEARING_TYPES = (EvaluationTarget, FrameSupervisionTarget, LabelMask)


class PerceptionContextError(ValueError):
    """Raised when an input cannot safely enter causal joint perception."""


@dataclass(frozen=True)
class PerceptionContext:
    """The complete, gold-free causal context admitted to perception."""

    sample: InferenceSample
    frames: Tensor
    images: tuple[ApiImageInput, ...]
    workflow_snapshot: Mapping[str, Any]
    memory_snapshot: Mapping[str, Any]
    prior_finalized_prediction: PredictionRecord | None
    track_snapshot: Mapping[str, Any] = field(default_factory=dict)
    selected_image_frame_ids: tuple[int, ...] = ()
    image_details: tuple[str, ...] = ()
    temporal_evidence: Mapping[str, Any] = field(default_factory=dict)


def require_gold_free(value: object) -> None:
    """Reject precisely the GT-bearing snapshot content forbidden at runtime."""

    _require_gold_free(value, seen=set())


def _require_gold_free(value: object, *, seen: set[int]) -> None:
    if isinstance(value, _GOLD_BEARING_TYPES):
        raise PerceptionContextError("GT-bearing object is forbidden in a snapshot")

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for key, nested_value in value.items():
            if key in _FORBIDDEN_SNAPSHOT_KEYS:
                raise PerceptionContextError(
                    f"GT-bearing snapshot key is forbidden: {key}"
                )
            _require_gold_free(key, seen=seen)
            _require_gold_free(nested_value, seen=seen)
        return

    if isinstance(value, (tuple, list, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for nested_value in value:
            _require_gold_free(nested_value, seen=seen)
        return

    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for field in fields(value):
            _require_gold_free(getattr(value, field.name), seen=seen)


def freeze_snapshot(
    snapshot: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    """Validate and recursively freeze one accepted runtime snapshot."""

    if not isinstance(snapshot, Mapping):
        raise PerceptionContextError(f"{name} snapshot must be a mapping")
    frozen_snapshot = _freeze_snapshot_value(snapshot, active=set())
    require_gold_free(frozen_snapshot)
    return frozen_snapshot


def _freeze_snapshot_value(value: object, *, active: set[int]) -> object:
    if value is None or isinstance(value, (bool, int, float, str, bytes, Enum)):
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise PerceptionContextError("snapshot values must not contain cycles")
        active.add(identity)
        try:
            return MappingProxyType(
                {
                    key: _freeze_snapshot_value(nested_value, active=active)
                    for key, nested_value in value.items()
                }
            )
        finally:
            active.remove(identity)

    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise PerceptionContextError("snapshot values must not contain cycles")
        active.add(identity)
        try:
            return tuple(
                _freeze_snapshot_value(nested_value, active=active)
                for nested_value in value
            )
        finally:
            active.remove(identity)

    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise PerceptionContextError("snapshot values must not contain cycles")
        active.add(identity)
        try:
            return frozenset(
                _freeze_snapshot_value(nested_value, active=active)
                for nested_value in value
            )
        finally:
            active.remove(identity)

    if is_dataclass(value) and not isinstance(value, type):
        parameters = value.__dataclass_params__
        if not parameters.frozen:
            raise PerceptionContextError("snapshot dataclasses must be frozen")
        identity = id(value)
        if identity in active:
            raise PerceptionContextError("snapshot values must not contain cycles")
        active.add(identity)
        try:
            return replace(
                value,
                **{
                    field.name: _freeze_snapshot_value(
                        getattr(value, field.name),
                        active=active,
                    )
                    for field in fields(value)
                    if field.init
                },
            )
        finally:
            active.remove(identity)

    raise PerceptionContextError("snapshot contains a mutable or unsupported value")


def normalize_ordered_frames(frames: Tensor, *, expected_count: int) -> Tensor:
    """Validate and normalize one RGB causal window to ``[T, C, H, W]``."""

    if not isinstance(frames, Tensor):
        raise PerceptionContextError("context frames must be a torch Tensor")
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise PerceptionContextError("batched context frames must have batch size 1")
        frames = frames.squeeze(0)
    elif frames.ndim != 4:
        raise PerceptionContextError("context frames must be [T,C,H,W] or [1,T,C,H,W]")

    if frames.shape[0] != expected_count:
        raise PerceptionContextError("context frame count does not match the sample")
    if frames.shape[1] != 3:
        raise PerceptionContextError("context frames must be RGB tensors")
    if not torch.isfinite(frames).all().item():
        raise PerceptionContextError("context frame values must be finite")
    if not ((frames >= 0).all() and (frames <= 1).all()):
        raise PerceptionContextError("context frame values must lie in [0, 1]")
    return frames


def encode_rgb_png(frame: Tensor) -> bytes:
    """Encode one validated ``[3,H,W]`` RGB tensor deterministically as PNG."""

    pixels = (
        frame.detach()
        .to(device="cpu", dtype=torch.float32)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .contiguous()
        .numpy()
    )
    output = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


def validate_prior(
    sample: InferenceSample,
    prior_finalized_prediction: PredictionRecord | None,
) -> None:
    """Ensure prior state belongs to this video and predates the target frame."""

    if prior_finalized_prediction is None:
        return
    if not isinstance(prior_finalized_prediction, PredictionRecord):
        raise PerceptionContextError("prior finalized state must be a PredictionRecord")
    if prior_finalized_prediction.video_id != sample.video_id:
        raise PerceptionContextError("prior finalized state must be from the same video")
    if prior_finalized_prediction.frame_id >= sample.target_frame_id:
        raise PerceptionContextError(
            "prior finalized state must be strictly earlier than the target frame"
        )
    prior_causal_frame_ids = prior_finalized_prediction.causal_frame_ids
    if any(frame_id >= sample.target_frame_id for frame_id in prior_causal_frame_ids):
        raise PerceptionContextError(
            "prior causal frame IDs must be strictly earlier than the target frame"
        )
    if tuple(sorted(set(prior_causal_frame_ids))) != prior_causal_frame_ids:
        raise PerceptionContextError(
            "prior causal frame IDs must be unique and increasing"
        )


class CausalPerceptionContextBuilder:
    """Build the only causal, gold-free context allowed into joint perception."""

    def __init__(
        self,
        *,
        max_frames: int = 6,
        max_images: int = 3,
        selection_strategy: str = "adaptive",
        history_image_detail: str = "auto",
        target_image_detail: str = "auto",
    ) -> None:
        if not 1 <= max_frames <= 6:
            raise ValueError("max_frames must be in 1..6")
        if not 1 <= max_images <= max_frames:
            raise ValueError("max_images must be in 1..max_frames")
        if selection_strategy not in {"adaptive", "fixed_all"}:
            raise ValueError("selection_strategy must be adaptive or fixed_all")
        if selection_strategy == "fixed_all" and max_images != max_frames:
            raise ValueError("fixed_all requires max_images == max_frames")
        allowed_details = {"low", "high", "auto", "original"}
        if history_image_detail not in allowed_details:
            raise ValueError("history_image_detail is unsupported")
        if target_image_detail not in allowed_details:
            raise ValueError("target_image_detail is unsupported")
        self.max_frames = max_frames
        self.max_images = max_images
        self.selection_strategy = selection_strategy
        self.history_image_detail = history_image_detail
        self.target_image_detail = target_image_detail

    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
        track_snapshot: Mapping[str, Any] | None = None,
    ) -> PerceptionContext:
        frozen_workflow_snapshot = freeze_snapshot(
            workflow_snapshot,
            name="workflow",
        )
        frozen_memory_snapshot = freeze_snapshot(
            memory_snapshot,
            name="memory",
        )
        frozen_track_snapshot = freeze_snapshot(
            {} if track_snapshot is None else track_snapshot,
            name="track",
        )
        expected_count = len(sample.causal_frame_ids)
        if expected_count > self.max_frames:
            raise PerceptionContextError(
                f"context windows must contain at most {self.max_frames} frames"
            )
        ordered_frames = normalize_ordered_frames(
            frames,
            expected_count=expected_count,
        )
        validate_prior(sample, prior_finalized_prediction)
        ordered_frames = ordered_frames.detach().clone()
        if self.selection_strategy == "fixed_all":
            selected_indices = tuple(range(expected_count))
            selection_evidence: Mapping[str, Any] = {
                "schema_version": "fixed_causal_window_v1",
                "selection_strategy": "fixed_all",
                "candidate_frame_ids": sample.causal_frame_ids,
                "selected_image_frame_ids": sample.causal_frame_ids,
                "omitted_image_frame_ids": (),
                "window_frame_count": expected_count,
                "uploaded_image_count": expected_count,
            }
        else:
            adaptive_window = select_adaptive_causal_window(
                ordered_frames,
                frame_ids=sample.causal_frame_ids,
                track_snapshot=frozen_track_snapshot,
                max_images=self.max_images,
            )
            selected_indices = adaptive_window.selected_indices
            selection_evidence = adaptive_window.evidence
        selected_image_frame_ids = tuple(
            sample.causal_frame_ids[index]
            for index in selected_indices
        )
        images = tuple(
            ApiImageInput(
                sample.media_refs[index],
                "image/png",
                encode_rgb_png(ordered_frames[index]),
            )
            for index in selected_indices
        )
        image_details = tuple(
            self.target_image_detail
            if index == expected_count - 1
            else self.history_image_detail
            for index in selected_indices
        )
        return PerceptionContext(
            sample=sample,
            frames=ordered_frames,
            images=images,
            workflow_snapshot=frozen_workflow_snapshot,
            memory_snapshot=frozen_memory_snapshot,
            prior_finalized_prediction=prior_finalized_prediction,
            track_snapshot=frozen_track_snapshot,
            selected_image_frame_ids=selected_image_frame_ids,
            image_details=image_details,
            temporal_evidence=freeze_snapshot(
                selection_evidence,
                name="temporal_evidence",
            ),
        )
