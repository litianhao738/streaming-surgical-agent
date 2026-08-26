"""Subprocess-level P3 mock smoke and safe mode-consistency checks."""

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
    assert artifact["schema_version"] == "p3_api_smoke_artifact_v2"
    assert artifact["status"] == "PARTIAL"
    assert artifact["first"]["cache_hit"] is False
    assert artifact["second"]["cache_hit"] is True
    assert artifact["first"]["retry_count"] == 1
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert artifact["usage"]["logical_calls"] == 2
    assert artifact["usage"]["provider_calls"] == 2
    assert artifact["p4_prediction_granularity"] == "DEFERRED_NOT_FROZEN_BY_P3"


def test_real_flag_rejects_non_requesty_config_before_any_transport_call(
    tmp_path: Path,
) -> None:
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
    assert "P3 smoke failed: contract_error" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.rglob("api_cache"))
