from __future__ import annotations

from pathlib import Path

import pytest

from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.contracts import PredictedTrackFrame
from surgical_agent.tracking.oof_index import (
    load_tracker_oof_index,
    write_tracker_oof_index,
)


def _write_artifact(root: Path, video_id: str) -> Path:
    checkpoint = root / "checkpoint.pt"
    config = root / "config.yaml"
    repair = root / "repair_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("config", encoding="utf-8")
    repair.write_text("{}", encoding="utf-8")
    return write_predicted_track_artifact(
        root / video_id / "predicted_tracks.json",
        provider="unit_test_tracker",
        source_model_identifier=f"held-out:{video_id}",
        checkpoint_path=checkpoint,
        inference_config_path=config,
        repair_manifest_path=repair,
        videos={
            video_id: (
                DatasetSplit.TRAINING,
                (PredictedTrackFrame(frame_id=1, tracks=()),),
            )
        },
    )


def test_tracker_oof_index_resolves_exact_held_out_video(tmp_path: Path) -> None:
    first = _write_artifact(tmp_path, "VID01")
    second = _write_artifact(tmp_path, "VID02")
    index_path = write_tracker_oof_index(
        tmp_path / "index.json",
        video_to_artifact={"VID01": first, "VID02": second},
        fold_count=2,
    )

    index = load_tracker_oof_index(index_path)

    provider = index.provider_for("VID02")
    assert provider.provider_name == "unit_test_tracker"
    with pytest.raises(KeyError, match="VID03"):
        index.provider_for("VID03")


def test_tracker_oof_index_detects_artifact_tampering(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, "VID01")
    index_path = write_tracker_oof_index(
        tmp_path / "index.json",
        video_to_artifact={"VID01": artifact},
        fold_count=2,
    )
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_tracker_oof_index(index_path)
