"""Validation isolation and incomplete-confirmation regression checks."""
from types import SimpleNamespace

import pytest

from scripts import collect_validation_inputs as collector
from scripts import run_validation_confirmation as confirmation
from scripts.resume_validation_confirmation import ResumeCalls
from surgical_agent.data.schemas import DatasetSplit


@pytest.mark.parametrize("split", [DatasetSplit.TRAINING, DatasetSplit.TESTING])
def test_validation_builder_rejects_other_splits_before_loading_media(split):
    sample = SimpleNamespace(target_frame_id=25, source_split=split)
    adapter = SimpleNamespace(iter_inference_video=lambda _: iter([sample]))
    with pytest.raises(ValueError, match="Validation"):
        collector.build_validation_base(adapter, {"video_id": "example", "frame_id": 25})


def test_validation_builder_rejects_changed_causal_frames():
    sample = SimpleNamespace(target_frame_id=50, source_split=DatasetSplit.VALIDATION,
                             causal_frame_ids=(0, 25, 50))
    adapter = SimpleNamespace(iter_inference_video=lambda _: iter([sample]))
    with pytest.raises(ValueError, match="causal frame"):
        collector.build_validation_base(adapter, {"video_id": "example", "frame_id": 50,
                                                   "causal_frame_ids": [1, 25, 50]})


@pytest.mark.parametrize("fatal,calls,keys", [(None, 15, ["a"]),
                                            ("ValueError", 16, ["a"]),
                                            (None, 16, [])])
def test_incomplete_confirmation_cannot_be_scored(tmp_path, fatal, calls, keys):
    confirmation.save(tmp_path / "plan.json", {"max_calls": 16, "selection": [{"key": "a"}]})
    confirmation.save(tmp_path / "completion.json", {"fatal_error": fatal, "calls": calls})
    confirmation.save(tmp_path / "predictions.json", {"targets": [{"key": k} for k in keys]})
    with pytest.raises(ValueError, match="complete declared cohort"):
        confirmation.score(tmp_path, None)  # No truth adapter may be reached.


def test_resume_preserves_refusal_and_rejects_duplicate_dispatch(tmp_path):
    source = tmp_path / "source"
    limits = {"openrouter_usd": "2.5", "aliyun_cny": "2", "xai_usd": "0"}
    row = {"index": 0, "target": "a", "stage": "control_phase", "seat": "gpt", "status": "API_FAILED"}
    confirmation.save(source / "budget.json", {"stopped": True, "limits": limits,
                      "occupied": {k: "0" for k in limits}, "calls": [row]})
    body = {"model": "example", "messages": []}
    confirmation.save(source / "calls/000_a_control_phase_gpt/request.json", body)
    ledger = ResumeCalls(tmp_path / "out", source, {"limits": limits, "max_calls": 16}, offline=True)
    assert ledger.call("a", "control_phase", "gpt", body) is None
    with pytest.raises(ValueError, match="duplicate"):
        ledger.call("a", "control_phase", "gpt", body)
    with pytest.raises(ValueError, match="cannot dispatch"):
        ledger.call("b", "control_phase", "gpt", body)
    changed = ResumeCalls(tmp_path / "out", source, {"limits": limits, "max_calls": 16}, offline=True)
    with pytest.raises(ValueError, match="saved request differs"):
        changed.call("a", "control_phase", "gpt", {**body, "model": "different"})


def test_resume_retries_only_local_budget_write(monkeypatch):
    from scripts import resume_validation_confirmation as recovery

    attempts = []

    def persist(_):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("transient file lock")
        return "saved"

    monkeypatch.setattr(confirmation.joint.roster.GLMCalls, "persist", persist)
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    ledger = object.__new__(ResumeCalls)
    assert ledger.persist() == "saved"
    assert len(attempts) == 2
