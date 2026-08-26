"""Construction of the gold-free causal input for joint perception."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
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
    require_gold_free(snapshot)
    return _freeze_snapshot_value(snapshot, active=set())


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

    def __init__(self, *, max_frames: int = 3) -> None:
        if not 1 <= max_frames <= 3:
            raise ValueError("max_frames must be in 1..3")
        self.max_frames = max_frames

    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
    ) -> PerceptionContext:
        frozen_workflow_snapshot = freeze_snapshot(
            workflow_snapshot,
            name="workflow",
        )
        frozen_memory_snapshot = freeze_snapshot(
            memory_snapshot,
            name="memory",
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
        images = tuple(
            ApiImageInput(identifier, "image/png", encode_rgb_png(frame))
            for identifier, frame in zip(sample.media_refs, ordered_frames)
        )
        return PerceptionContext(
            sample=sample,
            frames=ordered_frames,
            images=images,
            workflow_snapshot=frozen_workflow_snapshot,
            memory_snapshot=frozen_memory_snapshot,
            prior_finalized_prediction=prior_finalized_prediction,
        )
