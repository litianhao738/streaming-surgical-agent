from __future__ import annotations

from copy import deepcopy

import pytest

from surgical_agent.research.gate.final_only_training import (
    FEATURE_ORDER,
    FEATURE_VERSION,
    TASKS,
    TRACKER_FEATURES,
    canonical_labels,
    extract_features,
    label_outcome,
    mask_tracker,
    readiness,
)


def h0():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]}


def snapshot(frames=(1, 26, 51)):
    return {"status": "AVAILABLE", "source_max_frame_id": frames[-1],
            "frames": [{"frame_id": f, "tracks": [{"track_id": "a", "instrument_id": 0, "score": 0.9}]}
                       for f in frames]}


def examples():
    rows = []
    for video in ("VID02", "VID04", "VID11"):
        for frame in range(1, 5):
            features = extract_features(h0(), target_frame_id=frame, causal_frame_ids=[frame])
            rows.append({"sample_id": f"{video}:{frame}", "video_id": video, "frame_id": frame,
                         "source_split": "Training", "policy_id": "frozen-policy", "feature_version": FEATURE_VERSION,
                         "features_no_tracker": mask_tracker(features), "features_with_tracker": features,
                         "labels": {"benefit": frame % 2, "utility_delta": 0.2 if frame % 2 else -0.1,
                                    "any_head_harm": frame % 2 == 0}})
    return rows


def test_hard_labels_are_counts_not_fabricated_rank_confidence():
    values = extract_features(h0(), target_frame_id=51, causal_frame_ids=[1, 26, 51], tracker_snapshot=snapshot())
    assert tuple(values) == FEATURE_ORDER
    assert not any("margin" in key or "confidence" in key for key in values)
    assert values["tracker_detector_score_mean"] == pytest.approx(0.9)
    assert values["h0_instrument_count"] == 1
    assert all(mask_tracker(values)[key] == 0 for key in TRACKER_FEATURES)
    assert values["tracker_available"] == 1


def test_independent_head_extra_is_soft_feature_not_deleted():
    original = h0()
    original["target"].append(8)
    values = extract_features(original, target_frame_id=1, causal_frame_ids=[1])
    assert original["target"] == [0, 8]
    assert values["h0_target_independent_extra_count"] >= 1
    assert values["causal_image_count"] == 1


@pytest.mark.parametrize("frames", [[1, 26, 76], [1, 1, 51], [51, 26, 1], [1, 51], [True]])
def test_future_repeated_reversed_or_padded_windows_rejected(frames):
    with pytest.raises(ValueError, match="causal"):
        extract_features(h0(), target_frame_id=51, causal_frame_ids=frames)


def test_tracker_must_match_current_target_and_real_short_history():
    with pytest.raises(ValueError, match="exact causal"):
        extract_features(h0(), target_frame_id=51, causal_frame_ids=[1, 26, 51], tracker_snapshot=snapshot((26, 51, 76)))
    empty = snapshot((1,))
    empty["frames"][0]["tracks"] = []
    values = extract_features(h0(), target_frame_id=1, causal_frame_ids=[1], tracker_snapshot=empty)
    assert values["tracker_available"] == 1
    assert values["tracker_tool_count"] == 0
    assert values["tracker_h0_extra_class_count"] == 1


def test_duplicate_tracks_and_boolean_scores_rejected():
    bad = snapshot((1,))
    bad["frames"][0]["tracks"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        extract_features(h0(), target_frame_id=1, causal_frame_ids=[1], tracker_snapshot=bad)
    bad = snapshot((1,))
    bad["frames"][0]["tracks"][0]["score"] = True
    with pytest.raises(ValueError, match="score"):
        extract_features(h0(), target_frame_id=1, causal_frame_ids=[1], tracker_snapshot=bad)


def test_topk_payload_is_not_accepted_as_final_only():
    bad = h0()
    bad["confidence"] = 0.9
    with pytest.raises(ValueError, match="exactly five"):
        canonical_labels(bad)


def test_missing_gt_never_becomes_error_or_no_benefit_label():
    truth = h0()
    truth["target"] = None
    mask = dict.fromkeys(TASKS, True)
    mask["target"] = False
    values = label_outcome(h0(), h0(), gt=truth, mask=mask, repair_observed=True)
    assert values["h0_error_by_task"]["target"] is None
    assert values["h0_error"] is None
    assert values["benefit"] is None
    assert values["safe_benefit"] is None


def test_empty_pool_unverified_is_not_a_successful_negative_episode():
    values = label_outcome(h0(), h0(), gt=h0(), mask=dict.fromkeys(TASKS, True), repair_observed=False)
    assert values["benefit"] is None
    assert values["h0_error"] == 0
    assert values["utility_delta"] is None


def test_benefit_is_distinct_from_h0_error_and_detects_harm():
    wrong = h0()
    wrong["target"] = [8]
    values = label_outcome(wrong, h0(), gt=h0(), mask=dict.fromkeys(TASKS, True), repair_observed=True)
    assert (values["h0_error"], values["benefit"], values["safe_benefit"]) == (1, 1, 1)
    values = label_outcome(wrong, wrong, gt=h0(), mask=dict.fromkeys(TASKS, True), repair_observed=True)
    assert (values["h0_error"], values["benefit"]) == (1, 0)
    values = label_outcome(h0(), wrong, gt=h0(), mask=dict.fromkeys(TASKS, True), repair_observed=True)
    assert values["any_head_harm"] is True
    assert values["utility_delta"] < 0


def test_gate_cannot_silently_use_phase_repair_labels():
    changed = h0()
    changed["phase"] = [2]
    with pytest.raises(ValueError, match="Phase"):
        label_outcome(h0(), changed, gt=h0(), mask=dict.fromkeys(TASKS, True), repair_observed=True)


def test_both_classes_required_in_each_video_held_out_fit():
    rows = examples()
    assert readiness(rows)["can_fit_pilot"]
    for row in rows:
        row["labels"]["benefit"] = int(row["video_id"] == "VID02")
    result = readiness(rows)
    assert not result["can_fit_pilot"]
    assert "INSUFFICIENT_FIT_CLASSES_WHEN_HOLDING_OUT_VID02" in result["blocking_reasons"]


def test_tracker_and_no_tracker_rows_cannot_have_different_h0_features():
    rows = examples()
    rows[0]["features_no_tracker"]["h0_target_count"] = 99
    with pytest.raises(ValueError, match="paired views"):
        readiness(rows)


@pytest.mark.parametrize("change,match", [("duplicate", "duplicate"), ("split", "Training"),
                                         ("policy", "mix"), ("leak", "feature names")])
def test_training_rejects_duplicate_frames_test_split_policy_mix_and_extra_features(change, match):
    rows = examples()
    if change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "split":
        rows[0]["source_split"] = "Testing"
    elif change == "policy":
        rows[0]["policy_id"] = "another-repair-model"
    else:
        rows[0]["features_with_tracker"]["gt_ivt_error"] = 1
    with pytest.raises(ValueError, match=match):
        readiness(rows)
