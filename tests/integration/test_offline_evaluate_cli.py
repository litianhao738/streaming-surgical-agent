"""Offline evaluation command-line contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import evaluate
from tests.unit.test_frame_ground_truth import _annotation_payload
from tests.unit.test_offline_artifacts import _write_completed_run


def test_cli_has_only_offline_evaluation_inputs() -> None:
    destinations = {action.dest for action in evaluate.build_parser()._actions}
    assert {
        "run_dir",
        "dataset_root",
        "output_dir",
        "authorize_test_gt_evaluation",
    } <= destinations
    assert not ({"api_key", "provider", "model", "endpoint"} & destinations)


def test_cli_prints_completed_report_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "evaluation_report.json"
    monkeypatch.setattr(
        evaluate,
        "evaluate_completed_run",
        lambda *args, **kwargs: SimpleNamespace(report_path=report_path),
    )

    result = evaluate.main(
        ["--run-dir", str(tmp_path / "run"), "--dataset-root", str(tmp_path / "data")]
    )

    assert result == 0
    assert str(report_path) in capsys.readouterr().out


def test_cli_writes_complete_synthetic_validation_report(
    tmp_path: Path, capsys
) -> None:
    run_dir = _write_completed_run(tmp_path / "runs")
    dataset = tmp_path / "dataset"
    video_dir = dataset / "Validation" / "VID110"
    frames_dir = video_dir / "Frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "1.png").touch()
    (video_dir / "vid110.json").write_text(
        json.dumps(_annotation_payload()), encoding="utf-8"
    )
    repair_path = dataset / "repair_manifest.json"
    repair_path.write_text("{}\n", encoding="utf-8")
    rollout_path = run_dir / "dataset_rollout_artifact.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout["repair_manifest_sha256"] = hashlib.sha256(
        repair_path.read_bytes()
    ).hexdigest()
    rollout_path.write_text(
        json.dumps(rollout, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    result = evaluate.main(
        ["--run-dir", str(run_dir), "--dataset-root", str(dataset)]
    )

    assert result == 0
    report_path = run_dir.parent / "eval-unit__evaluation/evaluation_report.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["scored_frame_counts"] == {
        "VID110": 1
    }
    assert str(report_path) in capsys.readouterr().out
