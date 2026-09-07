import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.replay_grounded_contact_repair import _historical_cost, replay
from surgical_agent.research.verification.grounded_repair import proposed_labels
from tests.unit.test_final_only_grounded import _evidence


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run_fixture(path):
    h0 = {"instrument": [2, 0], "verb": [2, 1], "target": [0],
          "ivt": [60, 17], "phase": [3]}
    locator, proposal, review = _evidence(((0, [17]), (2, [60])))
    h1 = proposed_labels(h0, locator, proposal)
    _write(path / "predictions.json", [{
        "frame_id": 51, "status": "OK", "h0": h0, "h1": h1, "final": h1,
        "locator": locator, "proposal": proposal, "review": review,
        "proposal_slot": "FIRST", "admission": {"decision": "ACCEPT"},
    }])
    gt = deepcopy(h0)
    gt["target"] = None  # Deliberately unavailable, not an empty-set GT.
    _write(path / "summary.json", {
        "provider_calls": 4,
        "evaluation": {"h0": {"frames": [{
            "frame_id": 51, "h0": h0, "gt": gt,
            "mask": {task: task != "target" for task in h0},
        }]}},
    })
    _write(path / "plan.json", {
        "targets": [51], "input_metadata": {"51": {
            "prompt_version": "historical_fixture",
            "images": [{"identifier": "cholectrack20:VID02:frame:51"}],
        }},
    })
    for stage in ("h0", "locator", "proposal", "review"):
        _write(path / "calls" / f"51_{stage}" / "api_usage.jsonl", {
            "provider": "openrouter", "provider_call_count": 1,
            "provider_cost": .01,
        })


def test_replay_uses_saved_masks_and_separates_historical_cost(tmp_path, capsys):
    source, output = tmp_path / "source", tmp_path / "output"
    _run_fixture(source)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = replay(source, output)
    assert report["new_provider_calls"] == 0 and report["new_cost_usd"] == 0
    assert report["historical_cost"]["repair_added_provider_calls"] == 3
    assert report["historical_cost"]["repair_added_priced_cost_usd"] == .03
    assert report["decisions"][0]["legacy_decision"] == "ACCEPT"
    assert report["decisions"][0]["decision"] == "KEEP"
    assert report["decisions"][0]["review_required"] is False
    metrics = report["checked"]["arms"]["final"]["tasks"]
    assert metrics["target"]["valid_targets"] == 0
    assert metrics["target"]["micro_f1"] is None
    assert metrics["phase"]["exact_set_accuracy"] == 1
    assert report["source_unchanged"] and (output / "report.json").is_file()
    assert {p: p.read_bytes() for p in before} == before
    assert json.loads(capsys.readouterr().out)["new_provider_calls"] == 0


@pytest.mark.parametrize("invalid", ["existing_output", "nested_output", "duplicate", "coverage"])
def test_replay_refuses_overwrite_and_incomplete_target_alignment(tmp_path, invalid):
    source, output = tmp_path / "source", tmp_path / "output"
    _run_fixture(source)
    if invalid == "existing_output":
        output.mkdir()
    elif invalid == "nested_output":
        output = source / "new_output"
    else:
        plan = json.loads((source / "plan.json").read_text())
        plan["targets"] = [51, 51] if invalid == "duplicate" else [51, 76]
        _write(source / "plan.json", plan)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        replay(source, output)
    assert not (output / "report.json").exists()
    assert {p: p.read_bytes() for p in before} == before


def test_missing_provider_price_remains_explicitly_unpriced(tmp_path):
    path = Path(tmp_path) / "51_review" / "api_usage.jsonl"
    _write(path, {"provider": "openrouter", "provider_call_count": 1,
                  "provider_cost": None})
    result = _historical_cost([path])
    assert result["repair_added_provider_calls"] == 1
    assert result["repair_unpriced_calls"] == 1
    assert result["repair_added_priced_cost_usd"] == 0
