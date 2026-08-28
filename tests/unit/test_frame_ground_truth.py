"""Split-local GT loading and conservative frame aggregation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_agent.data.masks import canonical_label_mask
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import (
    aggregate_frame_target,
    load_evaluation_data,
)
from surgical_agent.evaluation.offline_artifacts import OfflineEvaluationError


def _instance(instrument: int, verb: int) -> SimpleNamespace:
    return SimpleNamespace(
        instrument_id=instrument,
        verb_id=verb,
        target_id=1,
        triplet_id=2,
        phase_id=3,
        mask=canonical_label_mask(
            instrument_id=instrument,
            verb_id=verb,
            target_id=1,
            triplet_id=2,
            phase_id=3,
        ),
    )


def test_aggregate_frame_target_masks_partial_task_without_making_negative() -> None:
    frame = SimpleNamespace(
        video_id="VID110",
        frame_id=1,
        instances=(_instance(0, 2), _instance(1, -1)),
    )

    target = aggregate_frame_target(
        frame,
        allowed_tasks=frozenset({"instrument", "verb", "target", "ivt", "phase"}),
        source="Validation/VID110/vid110.json",
    )

    assert target.instrument_ids == (0, 1)
    assert target.mask.instrument is True
    assert target.verb_ids == ()
    assert target.mask.verb is False


def test_empty_frame_has_no_authoritative_negative_labels() -> None:
    frame = SimpleNamespace(video_id="VID110", frame_id=1, instances=())

    target = aggregate_frame_target(
        frame,
        allowed_tasks=frozenset({"instrument", "verb", "target", "ivt", "phase"}),
        source="Validation/VID110/vid110.json",
    )

    assert target.instrument_ids == ()
    assert not any(
        (
            target.mask.instrument,
            target.mask.verb,
            target.mask.target,
            target.mask.ivt,
            target.mask.phase,
        )
    )


def test_test_run_is_rejected_before_dataset_access_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = SimpleNamespace(effective_split=DatasetSplit.TESTING)

    def fail_if_called(*args: object, **kwargs: object) -> bool:
        raise AssertionError("dataset was accessed")

    monkeypatch.setattr(Path, "is_dir", fail_if_called)
    with pytest.raises(OfflineEvaluationError, match="authorization"):
        load_evaluation_data(
            run, tmp_path / "dataset", authorize_test_gt_evaluation=False
        )


def _annotation_payload() -> dict[str, object]:
    return {
        "info": {"dataset": "CholecTrack20", "version": 1.0},
        "video": {
            "name": "VID110",
            "num_frames": 1,
            "split": "validation",
            "height": 480,
            "width": 854,
        },
        "categories": {
            "tools": [{"id": 0, "name": "grasper"}],
            "operators": [{"id": 1, "name": "surgeon"}],
        },
        "annotations": {
            "1": [
                {
                    "instrument": 0,
                    "verb": 2,
                    "target": 1,
                    "phase": 3,
                    "triplet": 2,
                    "tool_bbox": [0.1, 0.1, 0.2, 0.2],
                    "operator": 1,
                    "iscrowd": 0,
                    "area": 0.04,
                    "score": 1.0,
                    "intraoperative_track": 1,
                    "intracorporeal_track": 1,
                    "visibility_track": 1,
                    "visibility": 1,
                    "crowded": 0,
                    "visible": 1,
                    "occluded": 0,
                    "bleeding": 0,
                    "smoke": 0,
                    "blurred": 0,
                    "undercoverage": 0,
                    "reflection": 0,
                    "stainedlens": 0,
                }
            ]
        },
    }


def test_validation_loader_reports_media_only_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    video_dir = dataset / "Validation" / "VID110"
    frames_dir = video_dir / "Frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "1.png").touch()
    (frames_dir / "2.png").touch()
    (video_dir / "vid110.json").write_text(
        json.dumps(_annotation_payload()), encoding="utf-8"
    )
    repair_path = dataset / "repair_manifest.json"
    repair_path.write_text("{}\n", encoding="utf-8")
    repair_sha = hashlib.sha256(repair_path.read_bytes()).hexdigest()
    predictions = (
        SimpleNamespace(video_id="VID110", frame_id=1),
        SimpleNamespace(video_id="VID110", frame_id=2),
    )
    run = SimpleNamespace(
        effective_split=DatasetSplit.VALIDATION,
        mode="engineering",
        declared_split=None,
        video_ids=("VID110",),
        predictions=predictions,
        prediction_identities=(("VID110", 1), ("VID110", 2)),
        repair_manifest_sha256=repair_sha,
    )

    data = load_evaluation_data(
        run, dataset, authorize_test_gt_evaluation=False
    )

    assert data.runtime_identities == (("VID110", 1), ("VID110", 2))
    assert tuple(data.targets) == (("VID110", 1),)
