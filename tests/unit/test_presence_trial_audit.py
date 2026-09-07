import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from surgical_agent.evaluation.repair_comparison import TASKS

SCRIPT = Path(__file__).resolve().parents[2] / "tools/audit/audit_presence_trial.py"
SPEC = importlib.util.spec_from_file_location("presence_trial_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def labels(**updates):
    return {task: updates.get(task, [0]) for task in TASKS}


def row(frame_id=1, **updates):
    return {
        "video_id": "VID02", "frame_id": frame_id,
        "h0": labels(), "h1": labels(), "old_final": labels(),
        "final_a": labels(), "final_b": labels(), "gt": labels(),
        "mask": {task: True for task in TASKS}, **updates,
    }


def test_actual_edit_benefit_and_harm_are_scored_individually():
    rows = [row(
        h0=labels(ivt=[0, 1]), h1=labels(ivt=[1, 2, 3]), gt=labels(ivt=[0, 2]),
        old_final=labels(ivt=[1, 2, 3]), final_a=labels(ivt=[0, 1]),
        final_b=labels(ivt=[0, 1, 2]),
    )]
    original = deepcopy(rows)
    report = audit.compute_presence_audit(rows)
    old = report["edit_application"]["old_final"]["tasks"]["ivt"]
    assert (old["beneficial_proposed"], old["harmful_proposed"]) == (1, 2)
    assert old["beneficial_application_rate"] == old["harmful_application_rate"] == 1
    partial = report["edit_application"]["final_b"]["tasks"]["ivt"]
    assert partial["beneficial_application_rate"] == 1
    assert partial["harmful_application_rate"] == 0
    assert report["arms"]["final_b"]["tasks"]["ivt"]["micro_f1"] == .8
    assert report["independent_metric_counts_match"] is True
    assert rows == original


def test_masks_exclude_edit_gt_but_do_not_hide_unexpected_changes():
    report = audit.compute_presence_audit([row(
        gt={"verb": [0]}, mask={task: task == "verb" for task in TASKS},
        final_b=labels(target=[1]),
    )])
    result = report["edit_application"]["final_b"]
    assert result["applied_subset_of_proposed"] is False
    assert result["unexpected_edits"][0]["task_gt_valid"] is False
    assert result["tasks"]["target"]["beneficial_application_rate"] is None
    assert result["tasks"]["target"]["masked_targets"] == 1
    assert report["arms"]["final_b"]["tasks"]["target"]["micro_f1"] is None
    assert report["candidate_ivt_diagnostic"]["valid_targets_with_h0"] == 0


def test_failures_and_missing_candidates_remain_in_main_denominator():
    report = audit.compute_presence_audit([
        row(1, h0=None, h1=None, old_final=None, final_a=None, final_b=None),
        row(2, h1=None),
        row(3, h0=labels(ivt=[1]), h1=labels(), final_a=labels(ivt=[1])),
    ])
    assert report["coverage"]["targets"] == 3
    assert report["candidate_available"]["targets"] == 1
    task = report["arms"]["h1_policy"]["tasks"]["ivt"]
    assert task["valid_targets"] == 3
    assert task["failed_predictions"] == 1
    assert task["exact_set_accuracy"] == pytest.approx(2 / 3)
    assert report["edit_application"]["final_a"]["tasks"]["ivt"]["missing_h0_targets"] == 1


def test_candidate_oracle_reports_common_miss_and_optimistic_bound():
    report = audit.compute_presence_audit([row(
        h0=labels(ivt=[0, 4]), h1=labels(ivt=[1, 3]), gt=labels(ivt=[0, 1, 2]),
    )])
    diagnostic = report["candidate_ivt_diagnostic"]
    counts = diagnostic["counts"]
    assert counts["h0_missing_gt"] == 2
    assert counts["correct_ivt_additions_exposed"] == 1
    assert counts["missed_by_both"] == 1
    assert counts["h0_correct_removed_by_h1"] == 1
    assert counts["false_ivt_additions"] == 1
    assert counts["false_h0_ivt_removals_exposed"] == 1
    assert diagnostic["labelwise_oracle_ivt_metrics"]["micro_f1"] == .8
    assert diagnostic["targets_with_potential_labelwise_improvement"] == 1
    assert diagnostic["targets_with_whole_candidate_improvement"] == 0
    assert diagnostic["either_whole_answer_exact_targets"] == 0
    assert diagnostic["labelwise_oracle_exact_targets"] == 0


def test_no_edits_has_null_rates_and_phase_change_is_flagged():
    report = audit.compute_presence_audit([row(final_b=labels(phase=[1]))])
    result = report["edit_application"]["final_b"]
    assert result["pooled"]["beneficial_application_rate"] is None
    assert result["pooled"]["harmful_application_rate"] is None
    assert result["phase_changed_targets"] == [{"video_id": "VID02", "frame_id": 1}]
    with pytest.raises(ValueError, match="duplicate"):
        audit.compute_presence_audit([row(), row()])


def test_cli_records_cost_verbatim_and_refuses_to_overwrite(tmp_path):
    (tmp_path / "scored_predictions.json").write_text(json.dumps([row()]), encoding="utf-8")
    cost = {"added_cost_usd": None, "known_added_cost_usd": .02, "unpriced_calls": 1}
    (tmp_path / "calls_summary.json").write_text(json.dumps(cost), encoding="utf-8")
    audit.main(["--run-dir", str(tmp_path)])
    output = tmp_path / "presence_audit.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["calls_summary_as_recorded"] == cost
    assert len(report["sources"]["scored_predictions_sha256"]) == 64
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main(["--run-dir", str(tmp_path)])
    assert output.read_bytes() == before
