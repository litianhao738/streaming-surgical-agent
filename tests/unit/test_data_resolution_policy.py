"""Task-label and VID30/VID31 resolution helper tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from surgical_agent.data.label_policy import (
    IvtConsistencyStatus,
    classify_ivt_consistency,
    is_valid_task_id,
)
from surgical_agent.data.masks import canonical_label_mask
from surgical_agent.data.qualification import (
    SupervisionQualificationError,
    load_field_level_supervision_manifest,
)
from tools.data_resolution.cholectrack20_vid30_vid31.resolve_identity import (
    _duplicate_diagnostic,
    _sample_quantiles,
    _similarity,
)


def test_task_ranges_reject_unknown_positive_and_negative_ids() -> None:
    assert is_valid_task_id("instrument", 6)
    assert not is_valid_task_id("instrument", 7)
    assert is_valid_task_id("triplet", 99)
    assert not is_valid_task_id("triplet", 100)
    assert not is_valid_task_id("verb", -1)

    mask = canonical_label_mask(
        instrument_id=7,
        verb_id=9,
        target_id=14,
        triplet_id=-2,
        phase_id=6,
    )
    assert not mask.instrument
    assert mask.verb and mask.target and mask.phase
    assert not mask.ivt


def test_ivt_consistency_keeps_scope_difference_separate_from_conflict() -> None:
    mapping = {12: (0, 0, 13), 43: (1, 1, 0)}

    exact = classify_ivt_consistency(
        instrument_id=1,
        verb_id=1,
        target_id=0,
        triplet_id=43,
        triplet_mapping=mapping,
    )
    assert exact.status is IvtConsistencyStatus.EXACT
    assert exact.use_for_component_consistency

    scope = classify_ivt_consistency(
        instrument_id=6,
        verb_id=0,
        target_id=13,
        triplet_id=12,
        triplet_mapping=mapping,
    )
    assert scope.status is IvtConsistencyStatus.SCOPE_DIFFERENCE
    assert not scope.consistency_applicable

    conflict = classify_ivt_consistency(
        instrument_id=0,
        verb_id=1,
        target_id=0,
        triplet_id=43,
        triplet_mapping=mapping,
    )
    assert conflict.status is IvtConsistencyStatus.UNRESOLVED_CONFLICT
    assert conflict.consistency_applicable
    assert not conflict.use_for_component_consistency

    partial = classify_ivt_consistency(
        instrument_id=0,
        verb_id=-1,
        target_id=-1,
        triplet_id=94,
        triplet_mapping=mapping,
    )
    assert partial.status is IvtConsistencyStatus.NOT_APPLICABLE_PARTIAL_LABEL


def test_resolution_helpers_are_deterministic() -> None:
    assert _sample_quantiles(list(range(100)), 5) == [0, 24, 49, 74, 99]

    left = {1: [{"instrument": 0, "phase": 1}]}
    right = {1: [{"instrument": 0, "phase": 2}]}
    diagnostic = _duplicate_diagnostic(left, right)
    assert diagnostic["equal_fields"] == ["instrument"]
    assert diagnostic["differing_fields"] == {"phase": 1}
    assert diagnostic["core_values_equal"]


def test_image_similarity_distinguishes_identical_content() -> None:
    pixels = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    image = Image.fromarray(pixels, mode="L")
    changed = Image.fromarray(np.fliplr(pixels), mode="L")

    identical = _similarity(image, image)
    different = _similarity(image, changed)
    assert identical["phash_hamming"] == 0
    assert identical["gray_correlation"] > 0.999
    assert identical["mean_absolute_error"] == 0.0
    assert different["phash_hamming"] > identical["phash_hamming"]


def test_field_level_manifest_blocks_only_listed_unresolved_video(tmp_path) -> None:
    fields = {
        "instrument": False,
        "verb": False,
        "target": False,
        "triplet": False,
        "phase": False,
        "bbox": False,
        "operator": False,
        "track_ids": False,
        "visual_conditions": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ct20_field_level_supervision_manifest_v1",
                "unlisted_video_policy": "USE_CANONICAL_TASK_MASKS",
                "videos": {
                    "VID30": {
                        "raw_assets_retained": True,
                        "field_supervision": fields,
                        "status": "PENDING_IDENTITY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_field_level_supervision_manifest(path)
    assert manifest.allows("VID02", "instrument")
    assert not manifest.allows("VID30", "instrument")
    with pytest.raises(SupervisionQualificationError, match="VID30"):
        manifest.require("VID30", "track_ids")
