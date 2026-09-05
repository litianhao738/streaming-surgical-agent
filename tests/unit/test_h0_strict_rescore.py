import json

import pytest

from scripts.rescore_h0_strict import TASKS, labels_from_visible_stream, score


def example(frame, predicted, truth, missing=(), valid=True):
    return {"video_id": "synthetic", "frame_id": frame,
            "mask": {t: t not in missing for t in TASKS},
            "gt": {t: (list(truth) if t == "ivt" else [0]) if t not in missing else None for t in TASKS},
            "prediction": {t: list(predicted) if t == "ivt" else [0] for t in TASKS} if valid else None,
            "valid_response": valid}


def test_partial_hit_counts_for_f1_but_not_exact():
    result = score([example(1, [17, 60], [17])])
    ivt = result["tasks"]["ivt"]
    assert (ivt["tp"], ivt["fp"], ivt["fn"]) == (1, 1, 0)
    assert ivt["micro_f1"] == pytest.approx(2 / 3)
    assert ivt["set_exact_accuracy"] == 0
    assert result["joint_all_five_exact"]["correct"] == 0


def test_missing_ivt_does_not_remove_phase_and_failures_stay():
    result = score([example(1, [0], [0], missing=("ivt",)), example(2, [], [0], valid=False)])
    assert result["tasks"]["ivt"]["valid_gt"] == 1
    assert result["tasks"]["phase"]["valid_gt"] == 2
    assert result["tasks"]["phase"]["set_exact_accuracy"] == 0.5
    assert result["joint_all_five_exact"]["valid_gt"] == 1


def test_valid_null_class_is_scored_and_wrong_ivt_has_no_component_credit():
    result = score([example(1, [17], [94]), example(2, [59], [60])])
    assert result["tasks"]["ivt"]["valid_gt"] == 2
    assert result["tasks"]["ivt"]["tp"] == 0
    assert result["tasks"]["ivt"]["fn"] == 2


def test_raw_reconstruction_ignores_reasoning_and_rejects_duplicate_json_keys():
    payload = {"schema_version": "joint_perception_final_only_v1", **{
        t: {"selected_id": 1} if t == "phase" else {"selected_ids": [0]} for t in TASKS}}
    text = json.dumps(payload)
    events = [{"model": "qwen/qwen3.8-max-0902", "choices": [{"index": 0,
               "delta": {"content": text, "reasoning": "Never read this as the answer"}, "finish_reason": "stop"}]}]
    http = {"status_code": 200, "body": "\n".join("data: " + json.dumps(e) for e in events)}
    assert labels_from_visible_stream(http)["phase"] == [1]
    events[0]["choices"][0]["delta"]["content"] = text[:-1] + ',"phase":{"selected_id":2}}'
    http["body"] = "data: " + json.dumps(events[0])
    with pytest.raises(ValueError, match="duplicate"):
        labels_from_visible_stream(http)
