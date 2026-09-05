from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from surgical_agent.data.schemas import BoundingBox, DatasetSplit
from surgical_agent.tracking.box_policy import (
    BBoxPolicy,
    apply_bbox_policy,
    new_box_audit,
    record_box_policy_result,
)
from surgical_agent.tracking.oof_evaluation import _metrics, _score_video
from surgical_agent.tracking.training_data import (
    InstrumentDetectionDataset,
    build_detection_training_records,
)


@pytest.mark.parametrize("raw,expected", [
    ((-0.1, 0.2, 0.4, 0.3), (0.0, 0.2, 0.3, 0.3)),
    ((0.8, 0.7, 0.4, 0.5), (0.8, 0.7, 0.2, 0.3)),
    ((-1.0, -2.0, 4.0, 5.0), (0.0, 0.0, 1.0, 1.0)),
])
def test_clip_is_intersection_not_independent_tlwh_clamp(raw, expected):
    source = BoundingBox(*raw)
    decision = apply_bbox_policy(source, policy="clip_to_frame_v2")
    box = decision.bbox
    assert decision.clipped and not decision.dropped and box is not None
    assert (box.x, box.y, box.width, box.height) == pytest.approx(expected)
    assert decision.raw_bbox == raw
    assert (source.x, source.y, source.width, source.height) == raw
    assert apply_bbox_policy(source).status == "dropped_out_of_bounds"


@pytest.mark.parametrize("raw,status", [
    ((1.0, 0.2, 0.2, 0.2), "dropped_outside"),
    ((-0.4, 0.2, 0.2, 0.2), "dropped_outside"),
    ((0.2, -0.3, 0.2, 0.3), "dropped_outside"),
    ((0.2, 0.2, 0.0, 0.2), "dropped_degenerate"),
    ((-1.0, -1.0, -1.0, -1.0), "dropped_degenerate"),
    ((0.2, float("nan"), 0.2, 0.2), "dropped_non_finite"),
    ((0.2, 0.2, float("inf"), 0.2), "dropped_non_finite"),
    ((1e308, 0.2, 1e308, 0.2), "dropped_non_finite"),
])
def test_unusable_boxes_are_dropped_with_reason(raw, status):
    result = apply_bbox_policy(raw, policy=BBoxPolicy.CLIP_TO_FRAME_V2)
    assert result.bbox is None
    assert result.dropped and not result.clipped
    assert result.status == status


def test_valid_box_is_identical_in_both_policies_and_invalid_policy_is_rejected():
    box = BoundingBox(0.0, 0.0, 1.0, 1.0)
    for policy in BBoxPolicy:
        result = apply_bbox_policy(box, policy=policy)
        assert result.bbox is box and result.status == "kept_unchanged"
    with pytest.raises(ValueError):
        apply_bbox_policy(box, policy="silently_guess")


def test_audit_counters_conserve_boxes_and_cannot_mix_policies():
    audit = new_box_audit("clip_to_frame_v2")
    for box in ((0.1, 0.1, 0.2, 0.2), (-0.1, 0.1, 0.2, 0.2), (2.0, 2.0, 0.2, 0.2)):
        record_box_policy_result(audit, apply_bbox_policy(box, policy="clip_to_frame_v2"))
    assert audit["input_boxes"] == audit["kept_unchanged"] + audit["clipped"] + audit["dropped_total"] == 3
    with pytest.raises(ValueError, match="different bbox policies"):
        record_box_policy_result(audit, apply_bbox_policy((0.1, 0.1, 0.2, 0.2)))


class _Adapter:
    def __init__(self, image: Path):
        self.image = image
        self.entries = {"VID02": SimpleNamespace(split=DatasetSplit.TRAINING)}
        self.derived_manifest = SimpleNamespace(videos={})
        self.boxes = (
            BoundingBox(0.1, 0.1, 0.2, 0.2),
            BoundingBox(-0.1, 0.2, 0.4, 0.3),
            BoundingBox(1.1, 0.2, 0.2, 0.3),
            BoundingBox(0.1, 0.1, 0.0, 0.3),
        )

    def iter_video(self, video_id, max_samples=None):
        for fid, box in enumerate(self.boxes, start=1):
            yield SimpleNamespace(
                inference=SimpleNamespace(target_frame_id=fid, media_refs=(str(self.image),)),
                evaluation=SimpleNamespace(
                    video_id=video_id, frame_id=fid, instance_supervision_available=True,
                    instances=(SimpleNamespace(instrument_id=6 if fid == 2 else 0,
                                               bbox=box, mask=SimpleNamespace(instrument=True)),),
                ),
            )


def test_training_defaults_preserve_legacy_and_clipping_restores_border_example(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (100, 50)).save(image)
    adapter = _Adapter(image)
    strict = build_detection_training_records(adapter, ("VID02",))
    audit = {}
    clipped = build_detection_training_records(adapter, ("VID02",), bbox_policy="clip_to_frame_v2", box_audit=audit)
    assert [r.frame_id for r in strict] == [1]
    assert [r.frame_id for r in clipped] == [1, 2]
    assert strict[0] == clipped[0]
    _, target = InstrumentDetectionDataset(clipped)[1]
    assert target["labels"].tolist() == [7]  # specimen_bag ID 6; model background is 0
    assert target["boxes"][0].tolist() == pytest.approx([0, 10, 30, 25])
    assert audit["input_boxes"] == 4
    assert audit["clipped"] == 1 and audit["dropped_total"] == 2


def test_scoring_policy_changes_gt_and_preserves_all_frame_predictions(tmp_path):
    adapter = _Adapter(tmp_path / "unused.png")
    predictions = {1: ((0, (0.1, 0.1, 0.2, 0.2), 0.9),),
                   2: ((6, (0.0, 0.2, 0.3, 0.3), 0.8),), 3: (), 4: ()}
    strict = _score_video(adapter, video_id="VID02", predictions_by_frame=predictions)
    clipped = _score_video(adapter, video_id="VID02", predictions_by_frame=predictions,
                           gt_bbox_policy="clip_to_frame_v2")
    assert strict.frame_count == clipped.frame_count == 4
    assert strict.predictions == clipped.predictions
    assert strict.ground_truth_count == 1 and clipped.ground_truth_count == 2
    assert strict.box_audit["clipped"] == 0 and clipped.box_audit["clipped"] == 1
    before = _metrics((strict,), num_instrument_classes=7, iou_threshold=0.5)
    after = _metrics((clipped,), num_instrument_classes=7, iou_threshold=0.5)
    assert before["true_positive"] == 1 and before["false_positive"] == 1
    assert after["true_positive"] == 2 and after["false_positive"] == 0


def test_oof_report_records_explicit_gt_policy_and_clip_counts(tmp_path):
    from surgical_agent.tracking.oof_evaluation import evaluate_tracker_oof
    from tests.unit.test_tracker_oof_evaluation import _build_fixture

    adapter, index_path, config_path = _build_fixture(tmp_path)
    original_iterator = adapter.iter_video

    def border_annotations(video_id):
        for sample in original_iterator(video_id):
            sample.evaluation.instances[0].bbox = BoundingBox(-0.1, 0.1, 0.4, 0.2)
            yield sample

    adapter.iter_video = border_annotations
    report = evaluate_tracker_oof(adapter=adapter, index_path=index_path,
                                  tracker_config_path=config_path,
                                  gt_bbox_policy="clip_to_frame_v2")
    assert report["metric_protocol"]["gt_bbox_policy"] == "clip_to_frame_v2"
    assert report["overall_pooled"]["gt_box_audit"]["clipped"] == 2
    assert report["overall_pooled"]["scored_ground_truth_count"] == 2
    assert report["per_video"]["VID01"]["gt_box_audit"]["clipped"] == 1
