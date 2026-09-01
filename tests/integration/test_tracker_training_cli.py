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
