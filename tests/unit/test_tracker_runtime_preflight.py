from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.contracts import (
    PredictedTrack,
    PredictedTrackFrame,
    PredictedTrackVideo,
)
from surgical_agent.tracking.oof_index import write_tracker_oof_index
from surgical_agent.tracking.runtime_preflight import (
    build_validated_track_router,
    validate_gap_resets,
)


def _frame(frame_id: int, *, track_id: str = "one", instrument: int = 0, age: int = 1):
    return PredictedTrackFrame(
        frame_id=frame_id,
        tracks=(PredictedTrack(track_id, instrument, (0.1, 0.1, 0.2, 0.2), 0.9, age),),
    )


def _sample(video_id: str, frame_id: int = 1):
    return InferenceSample(
        video_id=video_id, target_frame_id=frame_id,
        causal_frame_ids=(frame_id,), media_refs=(f"{video_id}/{frame_id}.png",),
        source_split=DatasetSplit.TRAINING,
        alignment_version="test_png_exact",
    )


def _fixture(tmp_path: Path, *, leaked: bool = False, gap: bool = False):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    repair = dataset / "repair_manifest.json"
    repair.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("test", encoding="utf-8")
    adapter = SimpleNamespace(
        dataset_root=dataset,
        expected_frame_id_step=25,
        entries={video: SimpleNamespace(split=DatasetSplit.TRAINING) for video in ("VID01", "VID02")},
        derived_manifest=SimpleNamespace(videos={}),
    )
    artifact_map = {}
    for i, video in enumerate(("VID01", "VID02")):
        directory = tmp_path / "models/oof" / f"fold_{i}"
        directory.mkdir(parents=True)
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{i}".encode())
        training = "VID02" if i == 0 else "VID01"
        atomic_write_json(directory / "training_manifest.json", {
            "schema_version": "predicted_tracker_training_manifest_v1", "mode": "oof",
            "training_video_ids": [video if leaked and i == 0 else training],
            "excluded_video_ids": [video], "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_repair_manifest_sha256": sha256_file(repair),
            "tracker_config_sha256": sha256_file(config),
        })
        frames = (_frame(1, instrument=i),)
        if gap and i == 0:
            frames += (_frame(51, instrument=i, age=2),)
        artifact_map[video] = write_predicted_track_artifact(
            directory / "predicted_tracks.json", provider="test",
            source_model_identifier=f"fold-{i}", checkpoint_path=checkpoint,
            inference_config_path=config, repair_manifest_path=repair,
            videos={video: (DatasetSplit.TRAINING, frames)},
        )
    index = write_tracker_oof_index(
        tmp_path / "models/oof/index.json", video_to_artifact=artifact_map, fold_count=2,
    )
    return adapter, index, artifact_map


def test_router_changes_model_by_held_out_video(tmp_path: Path):
    adapter, index, _ = _fixture(tmp_path)
    samples = (_sample("VID01"), _sample("VID02"))
    router = build_validated_track_router(adapter=adapter, samples=samples, oof_index_path=index)
    for instrument, sample in enumerate(samples):
        router.reset(sample.video_id)
        snapshot = router.snapshot(sample)
        assert snapshot["status"] == "AVAILABLE"
        assert snapshot["frames"][0]["tracks"][0]["instrument_id"] == instrument
        assert snapshot["source_model_identifier"] == f"fold-{instrument}"
    assert router.audit["label_values_accessed"] is False
    assert all(row["coverage"] == 1.0 for row in router.audit["coverage"].values())


def test_router_rejects_model_trained_on_its_prediction_video(tmp_path: Path):
    adapter, index, _ = _fixture(tmp_path, leaked=True)
    with pytest.raises(ValueError, match="training partition"):
        build_validated_track_router(adapter=adapter, samples=(_sample("VID01"),), oof_index_path=index)


def test_router_rejects_missing_target_before_pipeline(tmp_path: Path):
    adapter, index, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        build_validated_track_router(adapter=adapter, samples=(_sample("VID01", 26),), oof_index_path=index)


def test_router_rejects_single_artifact_for_training(tmp_path: Path):
    adapter, _, artifacts = _fixture(tmp_path)
    with pytest.raises(ValueError, match="require --tracker-oof-index"):
        build_validated_track_router(
            adapter=adapter, samples=(_sample("VID01"),), artifact_path=artifacts["VID01"],
        )


def test_router_rejects_training_observation_relabelled_as_validation(tmp_path: Path):
    adapter, _, artifacts = _fixture(tmp_path)
    sample = replace(_sample("VID01"), source_split=DatasetSplit.VALIDATION)
    with pytest.raises(ValueError, match="canonical dataset entry"):
        build_validated_track_router(
            adapter=adapter, samples=(sample,), artifact_path=artifacts["VID01"],
        )


def test_router_rejects_old_gap_associations(tmp_path: Path):
    adapter, index, _ = _fixture(tmp_path, gap=True)
    with pytest.raises(ValueError, match="gap reset"):
        build_validated_track_router(adapter=adapter, samples=(_sample("VID01"),), oof_index_path=index)


def test_router_rejects_changed_checkpoint(tmp_path: Path):
    adapter, index, artifacts = _fixture(tmp_path)
    (artifacts["VID01"].parent / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint digest"):
        build_validated_track_router(adapter=adapter, samples=(_sample("VID01"),), oof_index_path=index)


@pytest.mark.parametrize("native_training_layout", [False, True])
def test_full_provider_validates_validation_and_test_without_labels(
    tmp_path: Path, native_training_layout: bool,
):
    adapter, _, _ = _fixture(tmp_path)
    directory = tmp_path / "models/full"
    directory.mkdir()
    checkpoint = directory / "checkpoint.pt"
    checkpoint.write_bytes(b"full-checkpoint")
    repair = adapter.dataset_root / "repair_manifest.json"
    config = tmp_path / "config.yaml"
    atomic_write_json(directory / "training_manifest.json", {
        "schema_version": "predicted_tracker_training_manifest_v1", "mode": "full",
        "training_video_ids": ["VID01", "VID02"], "excluded_video_ids": [],
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_repair_manifest_sha256": sha256_file(repair),
    })
    artifact = write_predicted_track_artifact(
        (directory.parent if native_training_layout else directory) / "predicted_tracks.json",
        provider="test",
        source_model_identifier="full", checkpoint_path=checkpoint,
        inference_config_path=config, repair_manifest_path=repair,
        videos={
            "VID110": (DatasetSplit.VALIDATION, (_frame(1),)),
            "VID06": (DatasetSplit.TESTING, (_frame(1, instrument=2),)),
        },
    )
    samples = (
        replace(_sample("VID110"), source_split=DatasetSplit.VALIDATION),
        replace(_sample("VID06"), source_split=DatasetSplit.TESTING),
    )
    adapter.entries.update({
        sample.video_id: SimpleNamespace(split=sample.source_split) for sample in samples
    })
    router = build_validated_track_router(adapter=adapter, samples=samples, artifact_path=artifact)
    for sample in samples:
        router.reset(sample.video_id)
        assert router.snapshot(sample)["status"] == "AVAILABLE"
    assert router.audit["source_kind"] == "full_artifact"


def test_gap_reset_rejects_id_revived_after_an_empty_frame():
    video = PredictedTrackVideo(
        "VID01", DatasetSplit.TRAINING,
        (_frame(1), PredictedTrackFrame(51, ()), _frame(76, age=2)),
    )
    with pytest.raises(ValueError, match="gap reset"):
        validate_gap_resets(video, max_frame_id_gap=25)


def test_gap_reset_allows_new_ids_and_normal_continuity():
    video = PredictedTrackVideo(
        "VID01", DatasetSplit.TRAINING,
        (_frame(1), _frame(26, age=2), _frame(76, track_id="new"), _frame(101, track_id="new", age=2)),
    )
    assert validate_gap_resets(video, max_frame_id_gap=25) == 1


def test_final_cli_tracker_sources_are_mutually_exclusive():
    from scripts.run_final_dataset_pipeline import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--mode", "engineering", "--max-provider-calls", "1", "--tracker-oof-index", "index.json",
    ])
    assert args.tracker_oof_index == Path("index.json")
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--mode", "engineering", "--max-provider-calls", "1",
            "--tracker-oof-index", "index.json", "--tracker-artifact", "tracks.json",
        ])
