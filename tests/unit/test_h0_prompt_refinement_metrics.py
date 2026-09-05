"""Synthetic-only checks: no project GT, credentials, or provider requests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_h0_prompt_refinement_smoke as runner
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION


def synthetic_evaluation():
    truth = {task: [0] for task in runner.study.TASKS}
    truth["instrument"] = [0, 1]
    partial = deepcopy(truth)
    partial["instrument"] = [0]
    frames = []
    for index, prediction in enumerate((truth, partial, None)):
        frames.append({
            "frame_id": index + 1, "mask": {task: True for task in truth},
            "gt": deepcopy(truth), "h0": deepcopy(prediction),
            "status": "OK" if prediction is not None else "API_FAILURE",
            "exact": {task: set(prediction[task]) == set(truth[task])
                      if prediction is not None else None for task in truth},
        })
    metrics = {task: {
        "valid_gt": 3, "scored": 2, "failed_with_gt": 1,
        "correct": sum(frame["exact"][task] is True for frame in frames),
    } for task in truth}
    return {"frames": frames, "tasks": metrics}


def test_failed_response_stays_in_exact_and_recall_denominators():
    result = runner.enrich_evaluation(synthetic_evaluation())
    instrument = result["tasks"]["instrument"]
    assert (instrument["tp"], instrument["fp"], instrument["fn"]) == (3, 0, 3)
    assert instrument["micro_precision"] == 1
    assert instrument["micro_recall"] == 0.5
    assert instrument["micro_f1"] == pytest.approx(2 / 3)
    assert instrument["exact_accuracy_all_valid_gt"] == pytest.approx(1 / 3)
    assert result["tasks"]["phase"]["exact_accuracy_all_valid_gt"] == pytest.approx(2 / 3)
    assert result["all_five_exact"] == {"correct": 1, "denominator": 3, "accuracy": 1 / 3}


def test_five_head_claim_rejects_missing_gt_mask():
    evaluation = synthetic_evaluation()
    evaluation["frames"][0]["mask"]["phase"] = False
    with pytest.raises(ValueError, match="five-head GT"):
        runner.enrich_evaluation(evaluation)


def test_no_shared_success_produces_unavailable_metrics_without_loading_gt():
    class ForbiddenAdapter:
        def iter_video(self, *args, **kwargs):
            raise AssertionError("Empty shared cohort must not read data")

    result, per_video = runner.evaluate_rows(ForbiddenAdapter(), [])
    assert per_video == {}
    assert result["all_five_exact"] == {"correct": 0, "denominator": 0, "accuracy": None}
    for task in result["tasks"].values():
        assert task["valid_gt"] == 0
        assert task["micro_f1"] is None
        assert task["exact_accuracy_all_valid_gt"] is None


def test_uncached_price_equivalent_retains_cached_tokens_and_billed_reasoning():
    native = [
        {"prompt_tokens": 2000, "completion_tokens": 100, "cost": 0.001,
         "completion_tokens_details": {"reasoning_tokens": 80},
         "prompt_tokens_details": {"cached_tokens": 500}},
        {"prompt_tokens": 3000, "completion_tokens": 300, "cost": 0.005,
         "completion_tokens_details": {"reasoning_tokens": 200},
         "prompt_tokens_details": {"cached_tokens": 1500}},
    ]
    result = runner.usage_means(native)
    assert result["provider_cost"] == pytest.approx(0.003)
    assert result["cached_tokens"] == 1000
    assert result["reasoning_tokens"] == 140
    assert result["visible_tokens"] == 60
    assert result["uncached_price_equivalent_not_actual_cost"] == pytest.approx(0.0062)


def test_missing_reasoning_tokens_do_not_become_zero():
    result = runner.usage_means([{"prompt_tokens": 100, "completion_tokens": 50}])
    assert result["reasoning_tokens"] is None
    assert result["provider_cost"] is None
    assert "visible_tokens" not in result
    assert runner.usage_means([]) == {}


def test_native_usage_reads_final_sse_usage_without_summing_repeated_snapshots(tmp_path):
    runner.study.atomic_write_json(tmp_path / "http_response.json", {"body": (
        'data: {"usage":{"prompt_tokens":10}}\n'
        'data: invalid\n'
        'data: {"usage":{"prompt_tokens":20,"completion_tokens":5}}\n'
        'data: [DONE]\n'
    )})
    assert runner.native_usage(tmp_path) == {"prompt_tokens": 20, "completion_tokens": 5}


def test_analyze_counts_complete_schema_and_rejects_empty_shared_cohort(tmp_path, monkeypatch):
    """Exercise real schema validation and adoption with wholly synthetic rows."""
    evaluation = runner.enrich_evaluation(synthetic_evaluation())
    empty = runner.enrich_evaluation({"frames": [], "tasks": {
        task: {"valid_gt": 0, "scored": 0, "failed_with_gt": 0, "correct": 0}
        for task in runner.study.TASKS
    }})
    monkeypatch.setattr(runner.study, "CholecTrack20DatasetAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "evaluate_rows", lambda adapter, rows: (deepcopy(evaluation if rows else empty), {}))
    monkeypatch.setattr(runner, "wire_audit", lambda *args: {"status": "PASS"})
    labels = {task: [0] for task in runner.study.TASKS}
    rows = [{"video_id": "synthetic", "frame_id": index, "arm": arm, "key": arm,
             "status": "OK", "selected_ids": labels}
            for index, arm in enumerate(runner.ARMS)]
    good = {"schema_version": FINAL_ONLY_SCHEMA_VERSION,
            **{task: {"selected_ids": [0]} for task in labels if task != "phase"},
            "phase": {"selected_id": 0}}
    for arm in runner.ARMS:
        payload = deepcopy(good)
        if arm == "schema_only":
            del payload["phase"]
        runner.study.atomic_write_json(tmp_path / "calls" / arm / "raw_responses" / "sample.json",
                                       {"parsed_payload": payload})
    runner.study.atomic_write_json(tmp_path / "predictions.json", rows)
    before = runner.study.sha256_file(tmp_path / "predictions.json")
    result = runner.analyze(SimpleNamespace(output=tmp_path, dataset=tmp_path),
                            {"call_order": rows, "plan_sha256": "synthetic"}, rows, {})
    assert result["groups"]["baseline"]["complete_schema"]["valid"] == 1
    assert result["groups"]["schema_only"]["complete_schema"]["valid"] == 0
    assert result["groups"]["tuned"]["complete_schema"]["valid"] == 1
    assert result["paired_same_success_count"] == 0
    decision = result["adoption_assessment"]["schema_only"]
    assert not decision["pilot_rule_pass"]
    assert "paired_same_success.instrument.exact_accuracy_all_valid_gt" in decision["failed_requirements"]
    assert "complete_schema_rate" in decision["failed_requirements"]
    assert "incomplete_experiment" in decision["failed_requirements"]
    assert result["predictions_sha256_after_scoring"] == before
