"""Default dispatch, live wire binding, frozen plans, and legacy isolation."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import run_compact_parallel_repair as compact
from scripts import run_pipeline as entry
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


@pytest.fixture
def inputs(tmp_path):
    images = []
    for frame in (951, 976, 1001):
        path = tmp_path / f"{frame}.png"
        Image.new("RGB", (3, 2), (30, 40, 50)).save(path)
        images.append({"path": str(path), "sha256": compact.sha(path)})
    selected = {"key": "FIXTURE_1001", "frame_id": 1001, "causal_frame_ids": [951, 976, 1001], "images": images}
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=Path(i["path"]).read_bytes()) for i in images],
                           payload={"image_details": ["low", "low", "high"]})
    pool = make_pool({"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]})
    return selected, base, pool


@pytest.mark.parametrize("seat", SEATS)
def test_default_runtime_reaches_both_measured_prompt_builders(seat, inputs):
    selected, base, pool = inputs
    old_graph = compact.original.review_wire(seat, base, selected, pool)
    old_phase = compact.original.phase_wire(seat, selected)
    with compact.compact_protocol():
        graph = compact.original.review_wire(seat, base, selected, pool)
        phase = compact.original.phase_wire(seat, selected)
        assert compact.original_score.verify_plan is compact.verify_plan
    assert graph == compact.compact_body(old_graph, "graph")
    assert phase == compact.compact_body(old_phase, "phase")
    for before, after in ((old_graph, graph), (old_phase, phase)):
        assert after["messages"][0]["content"][0]["text"] != before["messages"][0]["content"][0]["text"]
        restored = deepcopy(after)
        restored["messages"][0]["content"][0]["text"] = before["messages"][0]["content"][0]["text"]
        assert restored == before
    assert compact.original.review_wire(seat, base, selected, pool) == old_graph
    assert compact.original.phase_wire(seat, selected) == old_phase


def test_context_restores_bindings_after_exception():
    original = {n: getattr(compact.original, n) for n in ("PROFILE", "review_wire", "phase_wire", "verify_plan")}
    scorer = compact.original_score.verify_plan
    with pytest.raises(RuntimeError), compact.compact_protocol():
        raise RuntimeError("fixture")
    assert {n: getattr(compact.original, n) for n in original} == original
    assert compact.original_score.verify_plan is scorer


def test_prepare_freezes_new_phase_fingerprints_and_leaves_proposer(inputs, tmp_path, monkeypatch):
    selected, _, _ = inputs
    key = selected["key"]
    output = tmp_path / "plan"
    plan = {"profile": compact.LEGACY_PROFILE, "source_sha256": {}, "phase_inputs": {key: selected},
            "wire_fingerprints": {key: {"proposal": "original-proposal", "phase": {}}},
            "policy": "original", "limitations": [], "selection": [selected]}
    compact.save(output / "plan.json", plan)
    monkeypatch.setattr(compact, "_VERIFY_PLAN", lambda p: None)
    result = compact.finalize_compact_plan(output)
    assert result["profile"] == compact.PROFILE
    assert result["verifier_prompt_profile"] == compact.PROMPT_PROFILE
    assert result["wire_fingerprints"][key]["proposal"] == "original-proposal"
    assert result["wire_fingerprints"][key]["phase"] == {
        s: compact.fingerprint(compact.phase_wire(s, selected)) for s in SEATS}
    assert all((output / "frozen_source" / p.relative_to(compact.ROOT)).exists() for p in compact.RUNTIME_FILES)
    with compact.compact_protocol():
        compact.verify_plan(result)
        result["source_sha256"].pop(compact.PROMPT_FILES[0].relative_to(compact.ROOT).as_posix())
        with pytest.raises(ValueError, match="missing or changed"):
            compact.verify_plan(result)


@pytest.mark.parametrize("command", ("execute", "score"))
@pytest.mark.parametrize("profile", (compact.PROFILE, compact.LEGACY_PROFILE))
def test_saved_profile_controls_execution_and_scoring(command, profile, tmp_path, monkeypatch):
    compact.save(tmp_path / "plan.json", {"profile": profile})
    def action(output, adapter):
        assert compact.original.PROFILE == profile
        assert (compact.original.review_wire is compact.review_wire) == (profile == compact.PROFILE)
        return "called"
    owner = compact.original if command == "execute" else compact.original_score
    monkeypatch.setattr(owner, command, action)
    assert compact.dispatch(command, tmp_path, None) == "called"
    assert compact.original.PROFILE == compact.LEGACY_PROFILE


@pytest.mark.parametrize("command", ("prepare", "execute", "score"))
def test_default_cli_routes_all_commands_to_selected_runner(command, tmp_path, monkeypatch):
    from scripts import run_glm_parallel_repair as selected
    archived = entry.DEFAULT_MANIFEST.parent / "configs/defaults/parallel-phase-repair-v1.3.0-glm-low_before_mainline_20260911.json"
    monkeypatch.setattr(entry, "DEFAULT_MANIFEST", archived)
    manifest = json.loads(entry.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["version"] == selected.VERSION
    assert manifest["profile"] == selected.PROFILE
    assert manifest["verifier_prompt_profile"] == compact.PROMPT_PROFILE
    def run(argv, **kwargs):
        assert argv[3].endswith("run_glm_parallel_repair.py")
        assert argv[4] == command
        assert "--dataset-root" in argv
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, "run", run)
    reports = []
    monkeypatch.setattr(entry, "write_review_diagnostics", reports.append)
    assert entry.main([command, "--output", str(tmp_path), "--dataset-root", "D:/fixture"]) == 0
    assert reports == ([tmp_path.resolve()] if command in ("execute", "score") else [])


def test_failed_inference_does_not_produce_a_success_diagnostic_report(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    def unexpected(output):
        pytest.fail("must not postprocess failed inference")
    monkeypatch.setattr(entry, "write_review_diagnostics", unexpected)
    assert entry.main(["execute", "--output", str(tmp_path)]) == 1
