"""Subprocess-level P3 mock smoke and real-call fail-closed checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/smoke_api.py"


def test_p3_mock_smoke_uses_synthetic_image_cache_retry_and_usage(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(PROJECT_ROOT / "configs/api/mock.yaml"),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "integration",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P3 MOCK PASS / REAL API BLOCKED" in completed.stdout
    artifact = json.loads(
        (tmp_path / "integration/p3_mock_smoke.json").read_text(encoding="utf-8")
    )
    assert artifact["synthetic_non_sensitive_image"] is True
    assert artifact["track20_image_uploaded"] is False
    assert artifact["first_cache_hit"] is False
    assert artifact["second_cache_hit"] is True
    assert artifact["first_retry_count"] == 1
    assert artifact["usage"]["logical_calls"] == 2
    assert artifact["usage"]["provider_calls"] == 2
    assert artifact["real_api_smoke"].startswith("BLOCKED_")
    assert artifact["p4_prediction_granularity"] == "DEFERRED_NOT_FROZEN_BY_P3"


def test_real_smoke_is_blocked_before_any_transport_call(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(PROJECT_ROOT / "configs/api/default.yaml"),
            "--output-root",
            str(tmp_path),
            "--real",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "REAL_API_SMOKE_BLOCKED" in completed.stderr
    assert not list(tmp_path.rglob("api_cache"))
