from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts/train_tracker.py"
    spec = importlib.util.spec_from_file_location("train_tracker_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracker_cli_exposes_smoke_full_and_oof_modes() -> None:
    module = _module()
    parser = module.parser()

    assert parser.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert parser.parse_args(["--mode", "full"]).mode == "full"
    assert parser.parse_args(["--mode", "oof"]).mode == "oof"


def test_tracker_cli_enables_progress_by_default_and_can_disable_it() -> None:
    module = _module()
    parser = module.parser()

    assert parser.parse_args([]).no_progress is False
    assert parser.parse_args(["--no-progress"]).no_progress is True


def test_tracker_cli_defaults_to_new_bbox_policy_and_new_output_selection() -> None:
    args = _module().parser().parse_args([])
    assert args.bbox_policy == "clip_to_frame_v2"
    assert args.output_root is None
    assert args.allow_legacy_resume is False


def test_training_refuses_to_overwrite_existing_checkpoint_before_loading_data(tmp_path) -> None:
    import pytest
    import torch
    module = _module()
    (tmp_path / "checkpoint.pt").write_bytes(b"frozen")
    with pytest.raises(FileExistsError, match="already exists"):
        module._train(
            adapter=None, config=None, config_path=tmp_path / "unused.yaml",
            output_dir=tmp_path, training_video_ids=(), excluded_video_ids=(),
            device=torch.device("cpu"), epochs=1, max_train_batches=1,
            max_records=1, use_pretrained=False, resume=False, mode="smoke",
        )
    assert (tmp_path / "checkpoint.pt").read_bytes() == b"frozen"


def test_training_fingerprint_detects_annotation_and_resolved_target_changes(tmp_path) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from surgical_agent.data.schemas import BoundingBox
    from surgical_agent.tracking.training_data import (
        DetectionTrainingRecord,
        DetectionTrainingTarget,
    )

    module = _module()
    annotation = tmp_path / "vid02.json"
    annotation.write_text('{"label": 1}', encoding="utf-8")
    (tmp_path / "repair_manifest.json").write_text("{}", encoding="utf-8")
    adapter = SimpleNamespace(
        dataset_root=tmp_path,
        entries={"VID02": SimpleNamespace(annotation_file=annotation)},
    )
    record = DetectionTrainingRecord(
        "VID02", 1, tmp_path / "000001.png",
        (DetectionTrainingTarget(0, BoundingBox(0., 0., .2, .2), False),),
    )
    original = module._training_data_identity(adapter, (record,), ("VID02",))
    annotation.write_text('{"label": 2}', encoding="utf-8")
    changed_annotation = module._training_data_identity(adapter, (record,), ("VID02",))
    assert original != changed_annotation
    changed_target = replace(record, targets=(replace(record.targets[0], instrument_id=1),))
    assert changed_annotation != module._training_data_identity(adapter, (changed_target,), ("VID02",))


def test_truncated_smoke_metrics_use_only_predicted_frames(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "metrics.json"
    module._write_metrics(
        path=output,
        predictions={
            ("VID01", 1): [(0, (0.1, 0.1, 0.2, 0.2), 0.9)],
        },
        targets={
            ("VID01", 1): [(0, (0.1, 0.1, 0.2, 0.2))],
            ("VID01", 2): [(0, (0.1, 0.1, 0.2, 0.2))],
        },
    )

    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["ground_truth"] == 1
    assert metrics["recall"] == 1.0
