from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.data.schemas import BoundingBox, DatasetSplit, InferenceSample
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.config import load_tracker_training_config
from surgical_agent.tracking.contracts import PredictedTrack, PredictedTrackFrame
from surgical_agent.tracking.oof_evaluation import (
    VID31_UNSCORED_REASON,
    evaluate_tracker_oof,
)
from surgical_agent.tracking.oof_index import write_tracker_oof_index


class _NoInstanceSupervision:
    def allows(self, field: str) -> bool:
        return False


class _FakeAdapter:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.entries = {
            video_id: SimpleNamespace(split=DatasetSplit.TRAINING)
            for video_id in ("VID01", "VID02", "VID31")
        }
        self.derived_manifest = SimpleNamespace(
            videos={"VID31": _NoInstanceSupervision()}
        )
        self.runtime_frames = {
            "VID01": (1,),
            "VID02": (1,),
            "VID31": (1,),
        }

    def _sample(self, video_id: str, frame_id: int) -> InferenceSample:
        return InferenceSample(
            video_id=video_id,
            target_frame_id=frame_id,
            causal_frame_ids=(frame_id,),
            media_refs=(str(self.dataset_root / f"{video_id}_{frame_id}.png"),),
            source_split=DatasetSplit.TRAINING,
            alignment_version="unit_test_alignment_v1",
        )

    def iter_inference_video(self, video_id: str):
        for frame_id in self.runtime_frames[video_id]:
            yield self._sample(video_id, frame_id)

    def iter_video(self, video_id: str):
        for frame_id in self.runtime_frames[video_id]:
            instance = SimpleNamespace(
                instrument_id=0,
                bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
                mask=SimpleNamespace(instrument=True),
            )
            yield SimpleNamespace(
                inference=self._sample(video_id, frame_id),
                evaluation=SimpleNamespace(
                    video_id=video_id,
                    frame_id=frame_id,
                    instances=(instance,),
                    instance_supervision_available=True,
                ),
            )


def _write_config(path: Path) -> Path:
    path.write_text(
        """tracker:
  schema_version: predicted_tracker_training_v1
  architecture: fasterrcnn_mobilenet_v3_large_fpn
  initial_weights: NONE
  num_classes: 8
  seed: 3407
  epochs: 1
  batch_size: 1
  learning_rate: 0.005
  weight_decay: 0.0005
  num_workers: 0
  min_size: 64
  max_size: 64
  score_threshold: 0.35
  nms_threshold: 0.5
  association_iou_threshold: 0.3
  max_age: 2
  oof_folds: 2
""",
        encoding="utf-8",
    )
    return path


def _prediction_frame(*, present: bool = True) -> PredictedTrackFrame:
    tracks = (
        PredictedTrack(
            track_id="track-1",
            instrument_id=0,
            bbox_tlwh=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            age=1,
        ),
    ) if present else ()
    return PredictedTrackFrame(frame_id=1, tracks=tracks)


def _write_training_manifest(
    path: Path,
    *,
    mode: str,
    training: tuple[str, ...],
    excluded: tuple[str, ...],
    checkpoint: Path,
    repair_manifest: Path,
    config_path: Path,
    include_config: bool,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "predicted_tracker_training_manifest_v1",
        "mode": mode,
        "training_video_ids": list(training),
        "excluded_video_ids": list(excluded),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_repair_manifest_sha256": sha256_file(repair_manifest),
    }
    if include_config:
        payload["tracker_config_sha256"] = sha256_file(config_path)
        payload["config"] = asdict(load_tracker_training_config(config_path))
    atomic_write_json(path, payload)


def _build_fixture(tmp_path: Path) -> tuple[_FakeAdapter, Path, Path]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    repair_manifest = dataset_root / "repair_manifest.json"
    repair_manifest.write_text("{}\n", encoding="utf-8")
    config_path = _write_config(tmp_path / "tracker_oof2.yaml")
    output_root = tmp_path / "tracker"
    oof_root = output_root / "oof"
    video_to_artifact: dict[str, Path] = {}

    for fold_index, (held_out, training) in enumerate(
        (("VID01", ("VID02",)), ("VID02", ("VID01",)))
    ):
        fold = oof_root / f"fold_{fold_index}"
        fold.mkdir(parents=True)
        checkpoint = fold / "checkpoint.pt"
        checkpoint.write_bytes(f"fold-{fold_index}".encode())
        artifact = write_predicted_track_artifact(
            fold / "predicted_tracks.json",
            provider="unit_test_tracker",
            source_model_identifier=f"held-out:{held_out}",
            checkpoint_path=checkpoint,
            inference_config_path=config_path,
            repair_manifest_path=repair_manifest,
            videos={
                held_out: (
                    DatasetSplit.TRAINING,
                    (_prediction_frame(),),
                )
            },
        )
        _write_training_manifest(
            fold / "training_manifest.json",
            mode="oof",
            training=training,
            excluded=(held_out,),
            checkpoint=checkpoint,
            repair_manifest=repair_manifest,
            config_path=config_path,
            include_config=True,
        )
        video_to_artifact[held_out] = artifact

    full = output_root / "full"
    full.mkdir(parents=True)
    full_checkpoint = full / "checkpoint.pt"
    full_checkpoint.write_bytes(b"full")
    _write_training_manifest(
        full / "training_manifest.json",
        mode="full",
        training=("VID01", "VID02"),
        excluded=(),
        checkpoint=full_checkpoint,
        repair_manifest=repair_manifest,
        config_path=config_path,
        include_config=False,
    )
    vid31 = oof_root / "vid31"
    vid31.mkdir(parents=True)
    video_to_artifact["VID31"] = write_predicted_track_artifact(
        vid31 / "predicted_tracks.json",
        provider="unit_test_tracker",
        source_model_identifier="full-without-vid31",
        checkpoint_path=full_checkpoint,
        inference_config_path=config_path,
        repair_manifest_path=repair_manifest,
        videos={
            "VID31": (
                DatasetSplit.TRAINING,
                (_prediction_frame(present=False),),
            )
        },
    )
    index_path = write_tracker_oof_index(
        oof_root / "index.json",
        video_to_artifact=video_to_artifact,
        fold_count=2,
    )
    return _FakeAdapter(dataset_root), index_path, config_path


def test_tracker_oof_evaluation_is_held_out_and_excludes_vid31_metrics(
    tmp_path: Path,
) -> None:
    adapter, index_path, config_path = _build_fixture(tmp_path)

    report = evaluate_tracker_oof(
        adapter=adapter,
        index_path=index_path,
        tracker_config_path=config_path,
    )

    assert report["status"] == "PASS"
    assert report["leakage_audit"]["status"] == "PASS"
    assert report["overall_pooled"]["metrics"]["macro_ap50"] == 1.0
    assert report["overall_pooled"]["metrics"]["true_positive"] == 2
    assert report["video_macro_mean"]["f1"] == 1.0
    assert report["per_video"]["VID31"] == {
        "metric_status": "NOT_SCORED",
        "reason": VID31_UNSCORED_REASON,
        "artifact": "vid31/predicted_tracks.json",
        "runtime_frame_count": 1,
        "runtime_prediction_count": 0,
        "predicted_frame_coverage": 1.0,
        "metrics": None,
    }


def test_tracker_oof_evaluation_rejects_training_leakage(tmp_path: Path) -> None:
    adapter, index_path, config_path = _build_fixture(tmp_path)
    manifest_path = index_path.parent / "fold_0/training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_video_ids"] = ["VID01", "VID02"]
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="training set violates the OOF partition"):
        evaluate_tracker_oof(
            adapter=adapter,
            index_path=index_path,
            tracker_config_path=config_path,
        )


def test_tracker_oof_evaluation_rejects_incomplete_frame_coverage(
    tmp_path: Path,
) -> None:
    adapter, index_path, config_path = _build_fixture(tmp_path)
    adapter.runtime_frames["VID01"] = (1, 2)

    with pytest.raises(ValueError, match="prediction coverage is not exact"):
        evaluate_tracker_oof(
            adapter=adapter,
            index_path=index_path,
            tracker_config_path=config_path,
        )
