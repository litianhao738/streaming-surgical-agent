"""Security and deterministic helper tests for the read-only Synapse audit."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("synapseclient")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "audit" / "audit_cholectrack20_synapse.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_cholectrack20_synapse", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synapse_audit_requires_runtime_environment_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN", raising=False)
    with pytest.raises(module.SynapseAuditError, match="runtime environment"):
        module._runtime_token()

    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", "runtime-only-test-token")
    assert module._runtime_token() == "runtime-only-test-token"


def test_synapse_annotation_and_png_helpers_are_deterministic() -> None:
    module = _load_module()
    annotations = module._plain_annotations(
        {
            "annotations": {
                "Videos": {"type": "LONG", "value": ["10"]},
            }
        }
    )
    frame_ids, nonnumeric = module._numeric_png_children(
        (
            {"id": "syn1", "name": "000026.png"},
            {"id": "syn2", "name": "000051.png"},
            {"id": "syn3", "name": "README.txt"},
        )
    )

    assert annotations == {"Videos": {"type": "LONG", "value": ["10"]}}
    assert frame_ids == {26: "syn1", 51: "syn2"}
    assert nonnumeric == ["README.txt"]


def test_semantic_json_hash_ignores_serialization_whitespace(tmp_path: Path) -> None:
    module = _load_module()
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    payload = {"video": {"name": "VID31"}, "annotations": {"26": []}}
    compact.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    pretty.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    assert module._md5(compact) != module._md5(pretty)
    assert module._semantic_json_sha256(compact) == module._semantic_json_sha256(pretty)


def test_generated_synapse_artifact_contains_no_jwt() -> None:
    artifact = PROJECT_ROOT / "artifacts" / "p1" / "synapse_metadata_audit.json"
    if not artifact.is_file():
        pytest.skip("Synapse audit artifact is generated only with authorized access")
    text = artifact.read_text(encoding="utf-8")

    assert not re.search(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", text)
    payload = json.loads(text)
    assert payload["read_only"] is True
    assert payload["authentication"]["token_serialized"] is False
    assert payload["authentication"]["write_operations_performed"] is False
