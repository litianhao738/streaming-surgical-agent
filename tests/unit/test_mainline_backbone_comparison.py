"""No-provider checks for paired base requests and default dispatch."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import compare_mainline_backbones as trial
from scripts import recover_qwen_backbone_alias as recovery
from scripts import run_pipeline


@pytest.mark.parametrize("name", tuple(trial.MODELS))
def test_base_substitution_preserves_semantic_payload_and_original(name):
    body = {"model": "google/gemini-3.8-flash", "provider": {"only": ["google-ai-studio"]},
            "temperature": 0, "max_tokens": 4096, "reasoning": {"effort": "low"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "frozen"}]}],
            "response_format": {"type": "json_schema", "json_schema": {"strict": True}}}
    before = deepcopy(body)
    changed = trial.base_body(body, name)
    assert body == before
    assert changed["model"] == trial.MODELS[name]["model"]
    assert changed["provider"]["allow_fallbacks"] is False
    assert ("temperature" not in changed) == (name == "sol")
    for field in ("messages", "response_format", "reasoning", "max_tokens"):
        assert changed[field] == before[field]


def test_selected_mainline_dispatches_preflight_without_legacy_diagnostics(tmp_path, monkeypatch):
    manifest = tmp_path / "default.json"
    manifest.write_text(json.dumps({"implementation_entrypoint": "scripts/run_prior_gated_joint_mainline.py",
                                   "accepts_dataset_root": False, "dataset_root": "D:/cholec_dataset"}))
    monkeypatch.setattr(run_pipeline, "DEFAULT_MANIFEST", manifest)
    monkeypatch.delenv("CHOLECTRACK20_ROOT", raising=False)
    commands = []
    def dispatch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_pipeline.subprocess, "run", dispatch)
    assert run_pipeline.main(["preflight", "--output", str(tmp_path)]) == 0
    assert commands[0][4] == "preflight"
    assert "run_prior_gated_joint_mainline.py" in commands[0][3]


def test_recovery_changes_only_exact_snapshot_identifier():
    body = {"model": "google/gemini-3.8-flash", "temperature": 0, "reasoning": {"effort": "low"},
            "messages": [{"role": "user", "content": "frozen"}], "response_format": {"type": "json_object"}}
    expected = trial.base_body(body, "qwen")
    expected["model"] = recovery.SNAPSHOT
    assert recovery.wire(body) == expected


@pytest.mark.parametrize("model,provider", [("qwen/unverified", "Alibaba"), (recovery.SNAPSHOT, "Other")])
def test_recovery_rejects_unverified_model_or_provider(tmp_path, monkeypatch, model, provider):
    monkeypatch.setattr(recovery.old, "gemini_h0_wire", lambda base: {"messages": [], "model": "google/gemini-3.8-flash"})
    folder = tmp_path / "qwen/calls/000_sample_h0_base"
    folder.mkdir(parents=True)
    request = trial.base_body({"messages": [], "model": "google/gemini-3.8-flash"}, "qwen")
    (folder / "request.json").write_text(json.dumps(request))
    (folder / "response.json").write_text(json.dumps({"http_status": 200, "body": {"model": model, "provider": provider}}))
    with pytest.raises(ValueError, match="verified Qwen snapshot"):
        recovery.verified_h0(tmp_path, {"index": 0, "target": "sample"}, None)
