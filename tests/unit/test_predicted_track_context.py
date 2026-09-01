"""Contracts for gold-free, precomputed predicted-track context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.tracking.predicted_provider import (
    PrecomputedPredictedTrackProvider,
    PredictedTrackArtifactError,
)


def _sample(*, target_frame_id: int = 12) -> InferenceSample:
    return InferenceSample(
        video_id="VID30",
        target_frame_id=target_frame_id,
        causal_frame_ids=(10, 11, target_frame_id),
        media_refs=("synthetic:10", "synthetic:11", f"synthetic:{target_frame_id}"),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="unit-test",
    )


def _write_artifact(
    path: Path,
    *,
    extra_track: dict[str, object] | None = None,
    provenance: bool = True,
) -> Path:
    track = {
        "track_id": "tool-1",
        "instrument_id": 0,
        "bbox_tlwh": [0.1, 0.2, 0.3, 0.4],
        "score": 0.9,
        "age": 3,
    }
    if extra_track:
        track.update(extra_track)
    artifact = {
        "schema_version": "predicted_track_context_v1",
        "provider": "predicted_tracker_v1",
        "source_model_identifier": "tool-detector-fold0",
        "checkpoint_sha256": "a" * 64,
        "causal": True,
        "videos": {
                    "VID30": {
                        "source_split": "validation",
                        "frames": [
                            {"frame_id": 10, "tracks": [track]},
                            {"frame_id": 11, "tracks": []},
                            {"frame_id": 12, "tracks": [track]},
                            {"frame_id": 13, "tracks": [track]},
                        ],
                    }
        },
    }
    if provenance:
        artifact.update(
            {
                "inference_mode": "online_forward_only",
                "producer_version": "predicted_track_producer_v1",
                "dataset_repair_manifest_sha256": "b" * 64,
                "inference_config_sha256": "c" * 64,
            }
        )
    path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    return path


def test_snapshot_exposes_only_requested_causal_frames(tmp_path: Path) -> None:
    provider = PrecomputedPredictedTrackProvider.from_json(
        _write_artifact(tmp_path / "tracks.json")
    )
    sample = _sample()

    provider.reset(sample.video_id)
    snapshot = provider.snapshot(sample)

    assert snapshot["status"] == "AVAILABLE"
    assert snapshot["source_max_frame_id"] == 12
    assert [frame["frame_id"] for frame in snapshot["frames"]] == [10, 11, 12]
    assert all(frame["frame_id"] <= sample.target_frame_id for frame in snapshot["frames"])
    assert snapshot["artifact_sha256"] == provider.artifact_sha256


def test_provider_exposes_required_online_provenance_in_snapshot(tmp_path: Path) -> None:
    """Catches a provider that accepts but hides its causal-generation evidence."""

    provider = PrecomputedPredictedTrackProvider.from_json(
        _write_artifact(tmp_path / "tracks.json", provenance=True)
    )
    provider.reset("VID30")

    snapshot = provider.snapshot(_sample())

    assert provider.inference_mode == "online_forward_only"
    assert provider.producer_version == "predicted_track_producer_v1"
    assert provider.dataset_repair_manifest_sha256 == "b" * 64
    assert provider.inference_config_sha256 == "c" * 64
    assert snapshot["inference_mode"] == "online_forward_only"
    assert snapshot["producer_version"] == "predicted_track_producer_v1"
    assert snapshot["dataset_repair_manifest_sha256"] == "b" * 64
    assert snapshot["inference_config_sha256"] == "c" * 64


def test_artifact_rejects_missing_online_provenance(tmp_path: Path) -> None:
    """Catches an artifact that omits the evidence needed to audit its origin."""

    with pytest.raises(PredictedTrackArtifactError, match="unexpected fields"):
        PrecomputedPredictedTrackProvider.from_json(
            _write_artifact(tmp_path / "tracks.json", provenance=False)
        )


def test_provider_requires_exact_target_coverage_and_matching_split(tmp_path: Path) -> None:
    provider = PrecomputedPredictedTrackProvider.from_json(
        _write_artifact(tmp_path / "tracks.json")
    )
    missing = _sample(target_frame_id=14)
    provider.reset(missing.video_id)

    with pytest.raises(PredictedTrackArtifactError, match="target frame"):
        provider.snapshot(missing)

    wrong_split = InferenceSample(
        video_id="VID30",
        target_frame_id=12,
        causal_frame_ids=(12,),
        media_refs=("synthetic:12",),
        source_split=DatasetSplit.TESTING,
        alignment_version="unit-test",
    )
    with pytest.raises(PredictedTrackArtifactError, match="split"):
        provider.snapshot(wrong_split)


def test_artifact_rejects_gt_or_annotation_fields(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path / "tracks.json",
        extra_track={"ground_truth": True},
    )

    with pytest.raises(PredictedTrackArtifactError, match="unexpected fields"):
        PrecomputedPredictedTrackProvider.from_json(path)


def test_video_reset_and_frame_order_are_fail_closed(tmp_path: Path) -> None:
    provider = PrecomputedPredictedTrackProvider.from_json(
        _write_artifact(tmp_path / "tracks.json")
    )

    with pytest.raises(PredictedTrackArtifactError, match="reset"):
        provider.snapshot(_sample())
    provider.reset("VID99")
    with pytest.raises(PredictedTrackArtifactError, match="video boundary"):
        provider.snapshot(_sample())
