from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.config import load_tracker_training_config
from surgical_agent.tracking.contracts import PredictedTrack, PredictedTrackFrame
from surgical_agent.tracking.oof_index import (
    load_tracker_oof_index,
    write_tracker_oof_index,
)
from surgical_agent.tracking.reassociation import (
    reassociate_frames,
    reassociate_tracker_artifact,
    reassociate_tracker_bundle,
)


def _frame(frame_id: int, *, present: bool = True, age: int = 9) -> PredictedTrackFrame:
    return PredictedTrackFrame(
        frame_id,
        (PredictedTrack("old:shared", 0, (0.1, 0.1, 0.2, 0.2), 0.9, age),)
        if present
        else (),
    )


def test_reassociation_clears_state_across_gap_and_keeps_contiguous_identity() -> None:
    frames = (_frame(1), _frame(26), _frame(101), _frame(126))
    result, stats = reassociate_frames("VID01", frames, iou_threshold=0.3, max_age=2)
    assert result[0].tracks[0].track_id == result[1].tracks[0].track_id
    assert result[2].tracks[0].track_id == result[3].tracks[0].track_id
    assert result[1].tracks[0].track_id != result[2].tracks[0].track_id
    assert [frame.tracks[0].age for frame in result] == [1, 2, 1, 2]
    assert stats["gap_count"] == 1
    assert stats["source_gap_boundaries_with_shared_adjacent_ids"] == 1
    assert stats["output_cross_segment_id_reuse_count"] == 0


def test_reassociation_resets_even_when_gap_first_frame_has_no_detections() -> None:
    result, _ = reassociate_frames(
        "VID01",
        (_frame(1), _frame(76, present=False), _frame(101)),
        iou_threshold=0.3,
        max_age=2,
    )
    assert not result[1].tracks
    assert result[2].tracks[0].age == 1
    assert result[0].tracks[0].track_id != result[2].tracks[0].track_id


def test_reassociation_preserves_duplicate_detections_and_empty_frames() -> None:
    duplicate = PredictedTrackFrame(
        1,
        (
            PredictedTrack("old:a", 0, (0.1, 0.1, 0.2, 0.2), 0.9, 1),
            PredictedTrack("old:b", 0, (0.1, 0.1, 0.2, 0.2), 0.9, 1),
        ),
    )
    frames = (duplicate, _frame(26, present=False), _frame(51))
    result, stats = reassociate_frames("VID01", frames, iou_threshold=0.3, max_age=2)
    for old, new in zip(frames, result):
        assert old.frame_id == new.frame_id
        assert Counter(
            (x.instrument_id, x.bbox_tlwh, x.score) for x in old.tracks
        ) == Counter((x.instrument_id, x.bbox_tlwh, x.score) for x in new.tracks)
    assert stats["detection_count"] == 3


@pytest.mark.parametrize("gap", [0, -1, True])
def test_reassociation_rejects_invalid_gap(gap: int) -> None:
    with pytest.raises(ValueError, match="max_frame_id_gap"):
        reassociate_frames(
            "VID01", (_frame(1),), iou_threshold=0.3, max_age=2, max_frame_id_gap=gap
        )


def test_reassociation_rejects_backwards_frames() -> None:
    with pytest.raises(ValueError, match="increasing"):
        reassociate_frames(
            "VID01", (_frame(26), _frame(1)), iou_threshold=0.3, max_age=2
        )


def _source(root: Path, video_id: str) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True)
    config_file = root / "config.yaml"
    project = Path(__file__).resolve().parents[2]
    config_file.write_bytes(
        (
            project / "configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml"
        ).read_bytes()
    )
    config = load_tracker_training_config(config_file)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"dummy checkpoint: never loaded")
    repair = root / "repair_manifest.json"
    repair.write_text("{}", encoding="utf-8")
    training = root / "training_manifest.json"
    training.write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(checkpoint),
                "dataset_repair_manifest_sha256": sha256_file(repair),
                "config": asdict(config),
            }
        ),
        encoding="utf-8",
    )
    artifact = write_predicted_track_artifact(
        root / "predicted_tracks.json",
        provider="test",
        source_model_identifier="test detector",
        checkpoint_path=checkpoint,
        inference_config_path=config_file,
        repair_manifest_path=repair,
        videos={video_id: (DatasetSplit.TRAINING, (_frame(1), _frame(76)))},
    )
    return artifact, config_file, training


def test_artifact_migration_preserves_source_and_binds_sidecar(tmp_path: Path) -> None:
    source, config, training = _source(tmp_path / "original", "VID01")
    original_digest = sha256_file(source)
    output = tmp_path / "new/predicted_tracks.json"
    result = reassociate_tracker_artifact(
        source_path=source,
        output_path=output,
        config_path=config,
        training_manifest_path=training,
    )
    assert sha256_file(source) == original_digest == result["source_artifact_sha256"]
    assert result["output_artifact_sha256"] == sha256_file(output)
    assert result["algorithm"]["max_frame_id_gap"] == 25
    assert result["training_performed"] is False
    assert result["detector_inference_performed"] is False
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        reassociate_tracker_artifact(
            source_path=source,
            output_path=output,
            config_path=config,
            training_manifest_path=training,
        )
    assert output.read_bytes() == before


def test_migration_rejects_mismatched_config_before_output(tmp_path: Path) -> None:
    source, config, training = _source(tmp_path / "original", "VID01")
    config.write_text(config.read_text() + "\n# changed", encoding="utf-8")
    output = tmp_path / "new/predicted_tracks.json"
    with pytest.raises(ValueError, match="inference config"):
        reassociate_tracker_artifact(
            source_path=source,
            output_path=output,
            config_path=config,
            training_manifest_path=training,
        )
    assert not output.exists()


def test_bundle_retains_index_layout_and_copies_checkpoints_independently(
    tmp_path: Path,
) -> None:
    first, config, _ = _source(tmp_path / "source/oof/fold_0", "VID01")
    second, _, _ = _source(tmp_path / "source/oof/fold_1", "VID02")
    full, full_config, training = _source(tmp_path / "source/full", "VID31")
    index = write_tracker_oof_index(
        tmp_path / "source/oof/index.json",
        video_to_artifact={"VID01": first, "VID02": second},
        fold_count=2,
    )
    output = tmp_path / "migrated"
    result = reassociate_tracker_bundle(
        oof_index_path=index,
        full_artifact_path=full,
        full_training_directory=training.parent,
        oof_config_path=config,
        full_config_path=full_config,
        output_root=output,
    )
    assert result["status"] == "PASS"
    migrated = load_tracker_oof_index(output / "oof/index.json")
    assert set(migrated.video_to_artifact) == {"VID01", "VID02"}
    copied = output / "full/checkpoint.pt"
    original_bytes = (training.parent / "checkpoint.pt").read_bytes()
    assert copied.read_bytes() == original_bytes
    copied.write_bytes(b"independence probe")
    assert (training.parent / "checkpoint.pt").read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        reassociate_tracker_bundle(
            oof_index_path=index,
            full_artifact_path=full,
            full_training_directory=training.parent,
            oof_config_path=config,
            full_config_path=full_config,
            output_root=output,
        )
