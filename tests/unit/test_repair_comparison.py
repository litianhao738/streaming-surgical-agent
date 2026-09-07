from copy import deepcopy

import pytest

from surgical_agent.evaluation.repair_comparison import TASKS, compute_repair_comparison


def labels(**updates):
    return {task: updates.get(task, [0]) for task in TASKS}


def row(frame_id=1, *, h0=None, h1=None, final=None, gt=None, mask=None):
    return {
        "video_id": "VID110",
        "frame_id": frame_id,
        "h0": h0,
        "h1": h1,
        "final": final,
        "gt": labels() if gt is None else gt,
        "mask": {task: True for task in TASKS} if mask is None else mask,
    }


def test_partial_gain_equal_loss_replacement_and_harm_are_distinct():
    rows = [
        row(1, h0=labels(verb=[1], ivt=[1]), h1=labels(ivt=[1]), final=labels(ivt=[1])),
        row(2, h0=labels(ivt=[1]), h1=labels(ivt=[2]), final=labels(ivt=[2])),
        row(3, h0=labels(), h1=labels(ivt=[1]), final=labels(ivt=[1])),
        row(4, h0=labels(ivt=[1]), h1=labels(), final=labels()),
    ]
    original = deepcopy(rows)
    report = compute_repair_comparison(rows)
    frames = report["paired"]["h0_to_final"]["frames"]
    assert frames == {
        "valid_targets": 4,
        "improved": 2,
        "worsened": 1,
        "equal_loss_changed": 1,
        "unchanged": 0,
        "mixed": 0,
        "unscored": 0,
        "wrong_to_exact": 1,
        "exact_to_wrong": 1,
    }
    assert report["details"][0]["pairs"]["h0_to_final"]["wrong_to_exact"] is False
    assert report["arms"]["final"]["tasks"]["ivt"]["micro_f1"] == 0.25
    assert report["arms"]["final"]["tasks"]["verb"]["exact_set_accuracy"] == 1
    assert rows == original


def test_candidate_availability_has_policy_fallback_and_matched_subset():
    report = compute_repair_comparison(
        [
            row(1, h0=labels(), final=labels()),
            row(2, h0=labels(ivt=[1]), h1=labels(), final=labels(ivt=[1])),
        ]
    )
    assert report["candidate_availability"] == {"available": 1, "total": 2, "rate": 0.5}
    assert report["arms"]["h1_policy"]["tasks"]["ivt"]["micro_f1"] == 1
    subset = report["candidate_available"]
    assert subset["targets"] == 1
    assert subset["arms"]["h0"]["tasks"]["ivt"]["micro_f1"] == 0
    assert subset["paired"]["h1_policy_to_final"]["frames"]["worsened"] == 1


def test_masks_exclude_missing_gt_without_excluding_other_heads():
    mask = {task: task == "verb" for task in TASKS}
    report = compute_repair_comparison(
        [
            row(
                h0=labels(),
                h1=labels(target=[1]),
                final=labels(target=[1]),
                gt={"verb": [0]},
                mask=mask,
            ),
        ]
    )
    final = report["arms"]["final"]
    assert final["tasks"]["target"]["valid_targets"] == 0
    assert final["tasks"]["target"]["micro_f1"] is None
    assert final["tasks"]["verb"]["micro_f1"] == 1
    assert final["all_valid_heads_exact"]["accuracy"] == 1
    assert report["paired"]["h0_to_final"]["frames"]["unchanged"] == 1


def test_failure_never_earns_exact_including_on_valid_empty_ground_truth():
    empty = labels(**{task: [] for task in TASKS})
    report = compute_repair_comparison(
        [
            row(1, gt=empty),
            row(2, h0=empty, final=empty, gt=empty),
            row(3),
        ]
    )
    task = report["arms"]["h0"]["tasks"]["ivt"]
    assert task["valid_targets"] == 3
    assert (task["tp"], task["fp"], task["fn"]) == (0, 0, 1)
    assert task["failed_predictions"] == 2
    assert task["exact_matches"] == 1
    assert task["exact_set_accuracy"] == pytest.approx(1 / 3)
    assert report["arms"]["h0"]["all_valid_heads_exact"]["exact_matches"] == 1


def test_opposite_head_changes_are_mixed_even_with_net_gain():
    report = compute_repair_comparison(
        [
            row(
                h0=labels(verb=[1, 2, 3]),
                h1=labels(target=[1]),
                final=labels(target=[1]),
            ),
        ]
    )
    detail = report["details"][0]["pairs"]["h0_to_final"]
    assert detail["category"] == "mixed"
    assert detail["loss_before"] > detail["loss_after"]
    assert detail["wrong_to_exact"] is False


def test_micro_metrics_pool_label_counts_not_frame_scores():
    report = compute_repair_comparison(
        [
            row(
                1,
                h0=labels(ivt=[0, 1, 2]),
                final=labels(ivt=[0, 1, 2]),
                gt=labels(ivt=[0, 1]),
            ),
            row(2, h0=labels(ivt=[]), final=labels(ivt=[]), gt=labels(ivt=[2])),
        ]
    )
    task = report["arms"]["h0"]["tasks"]["ivt"]
    assert (task["tp"], task["fp"], task["fn"]) == (2, 1, 1)
    assert task["micro_precision"] == pytest.approx(2 / 3)
    assert task["micro_recall"] == pytest.approx(2 / 3)
    assert task["micro_f1"] == pytest.approx(2 / 3)
    assert task["exact_set_accuracy"] == 0


def test_empty_cohort_and_all_masked_frame_are_not_exact_successes():
    empty = compute_repair_comparison([])
    assert empty["candidate_availability"]["rate"] is None
    assert empty["arms"]["final"]["all_valid_heads_exact"]["accuracy"] is None
    report = compute_repair_comparison(
        [
            row(
                h0=labels(), final=labels(), gt={}, mask={task: False for task in TASKS}
            ),
        ]
    )
    assert report["arms"]["final"]["all_valid_heads_exact"]["valid_targets"] == 0
    assert report["paired"]["h0_to_final"]["frames"]["unscored"] == 1


def test_bad_identity_mask_and_labels_fail_before_scoring():
    with pytest.raises(ValueError, match="duplicate"):
        compute_repair_comparison([row(), row()])
    malformed = row()
    malformed["mask"]["ivt"] = 1
    with pytest.raises(TypeError, match="mask"):
        compute_repair_comparison([malformed])
    with pytest.raises(TypeError, match="integers"):
        compute_repair_comparison([row(h0=labels(ivt=[True]))])
    with pytest.raises(ValueError, match="0..99"):
        compute_repair_comparison([row(h0=labels(ivt=[100]))])
    with pytest.raises(TypeError, match="gt.instrument"):
        compute_repair_comparison([row(gt={})])
