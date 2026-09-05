"""Runtime Testing supervision hiding must never masquerade as absent raw GT."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.rescore_window_density_native_gt import score_native_rows
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask


def adapter(target):
    return SimpleNamespace(iter_video=lambda video, frame_ids: iter([
        SimpleNamespace(inference=SimpleNamespace(target_frame_id=30001),
                        frame_supervision=target, evaluation=None)]))


def prediction():
    return [{"frame_id": 30001, "status": "OK", "selected_ids": {
        "instrument": [0, 2], "verb": [1, 2], "target": [0, 2], "ivt": [17, 58], "phase": [1]}}]


def target():
    return FrameSupervisionTarget("VID06", 30001, (0, 2), (1, 2), (0, 2), (17, 58), 1,
        FrameTaskMask(True, True, True, True, True), "frame_multilabel", "native-test-fixture")


def test_inference_only_testing_adapter_raises_instead_of_reporting_missing_annotations():
    # This is the exact Testing runtime adapter contract: no evaluation object
    # and no frame_supervision, despite an available annotation frame key.
    with pytest.raises(ValueError, match="inference_adapter_hidden_gt"):
        score_native_rows(adapter(None), "VID06", prediction())


def test_explicit_native_targets_score_with_task_masks_and_partial_missing_is_not_fatal():
    scored = score_native_rows(adapter(target()), "VID06", prediction())
    assert all(metrics["valid_gt"] == 1 and metrics["micro_f1"] == 1
               for metrics in scored["tasks"].values())
    assert scored["tasks_without_valid_gt"] == []
    partially_masked = replace(target(), mask=FrameTaskMask(True, False, True, True, True), verb_ids=())
    scored = score_native_rows(adapter(partially_masked), "VID06", prediction())
    assert scored["tasks_without_valid_gt"] == ["verb"]
    assert scored["tasks"]["verb"]["micro_f1"] is None
    assert scored["tasks"]["ivt"]["micro_f1"] == 1
