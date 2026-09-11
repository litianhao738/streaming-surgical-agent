from copy import deepcopy

import pytest

from scripts import score_prior_candidate_trial as scorer
from scripts.run_prior_panel_trial import save


def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "verify", lambda *args: None)
    row = {"key": "v_1", "video_id": "v", "frame_id": 1, "h0": None,
           "arms": {a: {"prediction": None, "status": "H0_FAILED"} for a in scorer.ARMS}}
    save(tmp_path / "plan.json", {"arms": list(scorer.ARMS), "selection": [{"key": "v_1", "video_id": "v", "frame_id": 1}]})
    save(tmp_path / "predictions.json", {"targets": [row]})
    save(tmp_path / "budget.json", {"stopped": True, "calls": []})
    done = {"closed_utc": "test", **{f"{n}_sha256": scorer.sha(tmp_path / f"{n}.json") for n in ("plan", "predictions", "budget")}}
    save(tmp_path / "completion.json", done)
    return done


def test_score_retains_failed_target_but_rejects_unclosed_or_changed_snapshot(tmp_path, monkeypatch):
    done = frozen(tmp_path, monkeypatch)
    assert scorer.validate(tmp_path)[1][0]["h0"] is None
    save(tmp_path / "budget.json", {"stopped": False, "calls": []})
    with pytest.raises(ValueError, match="closed"):
        scorer.validate(tmp_path)
    save(tmp_path / "budget.json", {"stopped": True, "calls": []})
    save(tmp_path / "predictions.json", {"targets": []})
    with pytest.raises(ValueError, match="snapshot"):
        scorer.validate(tmp_path)
    done["predictions_sha256"] = scorer.sha(tmp_path / "predictions.json")
    save(tmp_path / "completion.json", done)
    with pytest.raises(ValueError, match="planned"):
        scorer.validate(tmp_path)


def test_shared_panels_have_no_independent_latency():
    rows = [{"key": "v_1", "arms": {"control": {"status": "UNRESOLVED", "panel_seconds": 10},
             "prior_graph": {"status": "UNRESOLVED", "panel_seconds": None, "shared_from": "control"}}}]
    result = scorer.runtime(rows, {"calls": []})
    assert result["arms"]["prior_graph"]["measured_panels"] == 0
    assert result["paired_panel"] == []
    measured = deepcopy(rows)
    measured[0]["arms"]["prior_graph"] = {"status": "UNRESOLVED", "panel_seconds": 11}
    assert scorer.runtime(measured, {"calls": []})["median_paired_percent_added"] == pytest.approx(10)
