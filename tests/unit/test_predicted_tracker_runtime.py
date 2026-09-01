from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.associator import (
    CausalHungarianAssociator,
    InstrumentDetection,
)
from surgical_agent.tracking.contracts import PredictedTrackFrame
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider


def _detection(instrument_id: int, x: float, score: float = 0.9) -> InstrumentDetection:
    return InstrumentDetection(
        instrument_id=instrument_id,
        bbox_tlwh=(x, 0.1, 0.2, 0.2),
        score=score,
    )


def test_associator_is_class_constrained_and_causal() -> None:
    associator = CausalHungarianAssociator(iou_threshold=0.3, max_age=1)
    associator.reset("VID02")

    first = associator.update(10, (_detection(0, 0.1),))
    second = associator.update(11, (_detection(0, 0.11), _detection(1, 0.11)))

    assert second[0].track_id == first[0].track_id
    assert second[0].age == 2
    assert second[1].track_id != first[0].track_id
    with pytest.raises(ValueError, match="increasing"):
        associator.update(11, (_detection(0, 0.12),))


def test_associator_expires_tracks_and_resets_video_identity() -> None:
    associator = CausalHungarianAssociator(iou_threshold=0.3, max_age=0)
    associator.reset("VID02")
    first = associator.update(1, (_detection(2, 0.2),))[0]
    assert associator.update(2, ()) == ()
    replacement = associator.update(3, (_detection(2, 0.2),))[0]
    assert replacement.track_id != first.track_id

    associator.reset("VID03")
    reset_track = associator.update(1, (_detection(2, 0.2),))[0]
    assert reset_track.track_id.startswith("VID03:")
    assert reset_track.track_id != replacement.track_id


def test_artifact_writer_emits_provider_compatible_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "tracker.yaml"
    repair = tmp_path / "repair_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("tracker: frozen\n", encoding="utf-8")
    repair.write_text("{}\n", encoding="utf-8")

    associator = CausalHungarianAssociator(iou_threshold=0.3, max_age=1)
    associator.reset("VID01")
    frames = (
        PredictedTrackFrame(frame_id=1, tracks=associator.update(1, (_detection(0, 0.1),))),
        PredictedTrackFrame(frame_id=2, tracks=associator.update(2, (_detection(0, 0.11),))),
    )
    output = write_predicted_track_artifact(
        tmp_path / "predicted_tracks.json",
        provider="torchvision_fasterrcnn_hungarian_v1",
        source_model_identifier="fasterrcnn_mobilenet_v3_large_fpn:COCO_V1:ct20",
        checkpoint_path=checkpoint,
        inference_config_path=config,
        repair_manifest_path=repair,
        videos={"VID01": (DatasetSplit.VALIDATION, frames)},
    )

    loaded = PrecomputedPredictedTrackProvider.from_json(output)
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert loaded.checkpoint_sha256 == raw["checkpoint_sha256"]
    assert raw["causal"] is True
    assert raw["videos"]["VID01"]["frames"][1]["tracks"][0]["age"] == 2
