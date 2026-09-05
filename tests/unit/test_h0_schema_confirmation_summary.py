from copy import deepcopy

import pytest

from scripts.summarize_h0_schema_confirmation import TASKS, aggregate, compare


def row(frame, predicted, truth, status="OK"):
    return {"video_id": "VID1", "frame_id": frame, "status": status,
            "mask": dict.fromkeys(TASKS, True),
            "gt": {task: list(truth) for task in TASKS},
            "h0": {task: list(predicted) for task in TASKS} if status == "OK" else None}


def test_pooling_recalculates_counts_instead_of_averaging_f1():
    first = [row(1, [1], [1])]
    second = [row(2, [2, 3, 4], [5, 6, 7])]
    combined = aggregate(first + second)
    assert combined["tasks"]["ivt"]["micro_f1"] == 0.25
    assert combined["tasks"]["ivt"]["micro_f1"] != (
        aggregate(first)["tasks"]["ivt"]["micro_f1"] + aggregate(second)["tasks"]["ivt"]["micro_f1"]
    ) / 2
    assert combined["all_five_exact"] == 1


def test_failures_keep_ground_truth_and_exact_denominators():
    result = aggregate([row(1, [], [1], status="API_FAILURE"), row(2, [1], [1])])
    assert result["tasks"]["ivt"]["fn"] == 1
    assert result["tasks"]["ivt"]["micro_recall"] == 0.5
    assert result["tasks"]["ivt"]["exact_accuracy"] == 0.5


def test_duplicates_missing_masks_and_unpaired_truth_rejected():
    source = row(1, [1], [1])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([source, source])
    missing = deepcopy(source)
    missing["mask"]["phase"] = False
    with pytest.raises(ValueError, match="five"):
        aggregate([missing])
    with pytest.raises(ValueError, match="unpaired"):
        compare([source], [row(2, [1], [1])])
    with pytest.raises(ValueError, match="GT differs"):
        compare([source], [row(1, [1], [2])])
