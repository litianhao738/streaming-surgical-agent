import json

import pytest

from scripts import score_graph_review_trial as scoring


def labels(**changes):
    return {**{task: [] for task in scoring.TASKS}, "phase": [0], **changes}


def make_archive(tmp_path):
    source = "scripts/score_graph_review_trial.py"
    target = {"key": "VID103_26", "video_id": "VID103", "frame_id": 26}
    plan = {"arms": list(scoring.ARMS), "selection": [target],
            "source_sha256": {source: scoring.sha(scoring.ROOT / source)}}
    predictions = {"targets": [{**target, "h0": labels(),
                                "arms": {a: {"prediction": labels(), "status": "KEEP"}
                                         for a in scoring.ARMS}}]}
    budget = {"stopped": True, "calls": []}
    for name, value in (("plan", plan), ("predictions", predictions), ("budget", budget)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    close_archive(tmp_path)
    return predictions


def close_archive(tmp_path):
    value = {f"{name}_sha256": scoring.sha(tmp_path / f"{name}.json")
             for name in ("plan", "predictions", "budget")}
    value["closed_utc"] = "2026-09-08T00:00:00Z"
    (tmp_path / "completion.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("failure", ["open", "modified", "missing_arm", "missing_target", "outstanding"])
def test_scoring_rejects_incomplete_or_modified_inference_before_gt(tmp_path, monkeypatch, failure):
    predictions = make_archive(tmp_path)
    if failure == "open":
        (tmp_path / "budget.json").write_text(json.dumps({"stopped": False, "calls": []}))
    elif failure == "outstanding":
        (tmp_path / "budget.json").write_text(json.dumps({"stopped": True, "calls": [{"status": "DISPATCHED"}]}))
    elif failure == "modified":
        (tmp_path / "predictions.json").write_text("{}")
    else:
        if failure == "missing_arm":
            predictions["targets"][0]["arms"].pop("flat")
        else:
            predictions["targets"] = []
        (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    if failure != "modified":
        close_archive(tmp_path)
    monkeypatch.setattr(scoring, "score_saved", lambda *_: pytest.fail("GT accessed before snapshot guards"))
    with pytest.raises(ValueError):
        scoring.score(tmp_path, object())


def test_edit_attribution_masks_gt_and_exposes_cross_head_harm():
    before = labels(verb=[0], target=[0], ivt=[0])
    after = labels(verb=[1], target=[1], ivt=[1])
    gt = {"instrument": [], "verb": [1], "target": [0], "ivt": None, "phase": [0]}
    mask = {task: task != "ivt" for task in scoring.TASKS}
    result = scoring.frame_delta(before, after, gt, mask)
    assert result["category"] == "mixed"
    assert "ivt" not in result["tasks"]
    assert result["fixed_label_errors"] == 2
    assert result["introduced_label_errors"] == 2
    assert result["net_errors_removed"] == 0
    assert result["both_beneficial_and_harmful_edits"] is True


def test_same_loss_replacement_and_missing_prediction_are_not_success():
    gt = labels(verb=[0])
    mask = dict.fromkeys(scoring.TASKS, True)
    result = scoring.frame_delta(labels(verb=[1]), labels(verb=[2]), gt, mask)
    assert result["category"] == "no_benefit_replacement"
    assert result["correct_deletions"] == 1
    assert result["wrong_additions"] == 1
    failed = scoring.frame_delta(None, gt, gt, mask)
    assert failed["category"] == "recovered_prediction"
    assert failed["wrong_to_exact"] is False


def test_successful_score_retains_failed_h0_and_masks(tmp_path, monkeypatch):
    predictions = make_archive(tmp_path)
    predictions["targets"][0]["h0"] = None
    for arm in scoring.ARMS:
        predictions["targets"][0]["arms"][arm]["prediction"] = None
        predictions["targets"][0]["arms"][arm]["status"] = "H0_FAILED"
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    close_archive(tmp_path)
    mask = dict.fromkeys(scoring.TASKS, True)
    mask["target"] = False
    truth = [{"video_id": "VID103", "frame_id": 26, "h0": None, "h1": None, "final": None,
              "gt": {**labels(), "target": None}, "mask": mask}]
    monkeypatch.setattr(scoring, "score_saved", lambda *_: (scoring.compute_repair_comparison(truth), truth))
    report = scoring.score(tmp_path, object())
    assert report["coverage"]["selected_targets"] == 1
    assert report["coverage"]["h0_available"] == 0
    assert report["metrics"]["h0"]["tasks"]["target"]["valid_targets"] == 0
    assert report["metrics"]["h0"]["tasks"]["instrument"]["exact_set_accuracy"] == 0
    assert report["comparisons"]["h0_to_graph"]["categories"] == {"unchanged_failure": 1}


def test_shared_panels_excluded_from_latency_and_charges_remain_separate():
    targets = [{"key": "example", "arms": {
        "none": {"panel_seconds": 10, "reference_chars": 0},
        "flat": {"panel_seconds": 12, "retrieval_seconds": .001, "reference_chars": 200},
        "graph": {"panel_seconds": 0, "shared_from": "flat", "reference_chars": 200},
    }}]
    budget = {"calls": [
        {"stage": "none_review", "account": "openrouter_usd", "charge": ".01",
         "charge_kind": "native", "status": "JSON_PARSED", "elapsed_seconds": 10,
         "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        {"stage": "flat_review", "account": "aliyun_cny", "charge": ".02",
         "charge_kind": "conservative_estimate", "status": "JSON_PARSED"},
    ]}
    result = scoring.runtime_summary(targets, budget)
    assert result["timing"]["graph"]["panel_seconds"]["n"] == 0
    assert result["timing"]["graph"]["paired_to_none"]["pairs"] == []
    assert result["timing"]["flat"]["paired_to_none"]["median_percent_added"] == pytest.approx(20)
    assert result["dispatched_calls"]["all"]["charges_by_account"] == {
        "openrouter_usd": "0.01", "aliyun_cny": "0.02"}
