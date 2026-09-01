from __future__ import annotations

import importlib.util
from pathlib import Path


def test_autodl_verifier_checks_matching_tracker_runtime() -> None:
    path = Path(__file__).parents[2] / "scripts/verify_autodl_bundle.py"
    spec = importlib.util.spec_from_file_location("verify_autodl_bundle_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    versions = module.tracker_runtime_versions()

    assert versions["torch"]
    assert versions["torchvision"]
    assert versions["torchvision_detection_ops"] == "AVAILABLE"
