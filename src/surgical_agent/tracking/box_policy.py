"""Explicit, versioned handling of normalized detector annotation boxes."""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum

from surgical_agent.data.schemas import BoundingBox


class BBoxPolicy(str, Enum):
    LEGACY_STRICT_V1 = "legacy_strict_v1"
    CLIP_TO_FRAME_V2 = "clip_to_frame_v2"


@dataclass(frozen=True)
class BoxPolicyResult:
    policy: BBoxPolicy
    raw_bbox: tuple[float, float, float, float]
    bbox: BoundingBox | None
    status: str

    @property
    def clipped(self) -> bool:
        return self.status == "clipped"

    @property
    def dropped(self) -> bool:
        return self.bbox is None


def apply_bbox_policy(
    bbox: BoundingBox | Sequence[float],
    *,
    policy: BBoxPolicy | str = BBoxPolicy.LEGACY_STRICT_V1,
) -> BoxPolicyResult:
    """Keep strict boxes or intersect raw TLWH with [0,1]^2 without mutation.

    Invalid input shape/type is a contract error; non-finite, non-positive, or
    wholly off-image boxes are unusable annotations and receive drop reasons.
    """
    selected = BBoxPolicy(policy)
    raw = (bbox.x, bbox.y, bbox.width, bbox.height) if isinstance(bbox, BoundingBox) else tuple(bbox)
    if len(raw) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
        raise ValueError("bbox must contain four numeric normalized TLWH values")
    values = tuple(float(v) for v in raw)

    def result(status: str, kept: BoundingBox | None = None) -> BoxPolicyResult:
        return BoxPolicyResult(selected, values, kept, status)

    if not all(math.isfinite(v) for v in values):
        return result("dropped_non_finite")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        return result("dropped_degenerate")
    right, bottom = x + width, y + height
    if not math.isfinite(right) or not math.isfinite(bottom):
        return result("dropped_non_finite")
    canonical = bbox if isinstance(bbox, BoundingBox) else BoundingBox(*values)
    if canonical.is_inside_unit_frame:
        return result("kept_unchanged", canonical)
    if selected is BBoxPolicy.LEGACY_STRICT_V1:
        return result("dropped_out_of_bounds")
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(1.0, right), min(1.0, bottom)
    if right <= left or bottom <= top:
        return result("dropped_outside")
    return result("clipped", BoundingBox(left, top, right - left, bottom - top))


_COUNT_KEYS = (
    "input_boxes", "kept_unchanged", "clipped", "dropped_total",
    "dropped_non_finite", "dropped_degenerate", "dropped_outside", "dropped_out_of_bounds",
)


def new_box_audit(policy: BBoxPolicy | str) -> dict[str, int | str]:
    return {"bbox_policy": BBoxPolicy(policy).value, **dict.fromkeys(_COUNT_KEYS, 0)}


def record_box_policy_result(
    audit: MutableMapping[str, int | str], result: BoxPolicyResult
) -> None:
    """Count instrument-valid candidate boxes; preserve the chosen policy."""
    if not audit:
        audit.update(new_box_audit(result.policy))
    if audit.get("bbox_policy") != result.policy.value:
        raise ValueError("cannot combine different bbox policies in one audit")
    for key in ("input_boxes", result.status, *(["dropped_total"] if result.dropped else [])):
        audit[key] = int(audit.get(key, 0)) + 1


__all__ = ["BBoxPolicy", "BoxPolicyResult", "apply_bbox_policy", "new_box_audit", "record_box_policy_result"]
