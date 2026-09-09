"""Phase repair must add a decision without reopening interaction predictions."""

from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    choose_history,
    phase_apply,
    phase_choice_error,
    phase_pool,
)


def prediction(phase=2):
    # Independent heads and their order are deliberate: Phase must not close,
    # reorder or prune any interaction head as a side effect of a stage change.
    return {
        "instrument": [2, 0],
        "verb": [6, 0],
        "target": [4, 0],
        "ivt": [60, 0],
        "phase": [phase],
    }


def scores(value=2.0):
    return {f"phase_{i}": value for i in range(7)}


def test_pool_contains_exactly_seven_named_phase_alternatives_and_no_interaction():
    pool = phase_pool()
    props = pool["propositions"]
    assert len(props) == 7
    assert {p["id"] for p in props} == set(scores())
    for i, prop in enumerate(props):
        assert prop == {
            "id": f"phase_{i}", "task": "phase", "label_id": i,
            "name": _TASK_NAMES["phase"][i], "components": None,
        }
    props[0]["name"] = "caller mutation"
    assert phase_pool()["propositions"][0]["name"] == _TASK_NAMES["phase"][0]


@pytest.mark.parametrize("old_phase", range(7))
def test_phase_can_switch_from_every_stage_without_changing_any_other_head(old_phase):
    current = prediction(old_phase)
    means = scores()
    winner = (old_phase + 1) % 7
    means[f"phase_{winner}"] = 4.0
    snapshot = deepcopy((current, means))

    result, decision = phase_apply(current, means)

    assert result == {**current, "phase": [winner]}
    assert decision["before"] == old_phase
    assert decision["after"] == winner
    assert decision["reason"] == "UNIQUE_BETTER_SUPPORTED_PHASE"
    assert (current, means) == snapshot
    for task in current:
        assert result[task] is not current[task]


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"phase_1": 3.99}, "NO_SUPPORTED_ALTERNATIVE"),
        ({"phase_1": 4, "phase_3": 4}, "TIED_PHASE_SUPPORT"),
        ({"phase_1": 4, "phase_2": 4}, "TIED_PHASE_SUPPORT"),
        ({"phase_1": 4, "phase_2": 4.2}, "CURRENT_PHASE_BEST"),
        ({"phase_2": 5}, "CURRENT_PHASE_BEST"),
        ({"phase_1": 5, "phase_2": None}, "INVALID_CURRENT_PHASE_EVIDENCE"),
    ],
)
def test_uncertain_tied_or_weaker_alternative_preserves_the_complete_prediction(updates, reason):
    current = prediction()
    means = {**scores(), **updates}
    result, decision = phase_apply(current, means)
    assert result == current
    assert result is not current
    assert decision == {"before": 2, "after": 2, "reason": reason}


def test_all_invalid_means_cannot_become_zero_scores_or_force_a_phase_change():
    current = prediction()
    result, decision = phase_apply(current, scores(None))
    assert result == current
    assert decision["reason"] == "INVALID_CURRENT_PHASE_EVIDENCE"


def test_missing_unrelated_phase_evidence_does_not_block_a_supported_switch():
    means = scores(None)
    means.update(phase_2=2.0, phase_4=4.6)
    result, decision = phase_apply(prediction(), means)
    assert result["phase"] == [4]
    assert decision["reason"] == "UNIQUE_BETTER_SUPPORTED_PHASE"


@pytest.mark.parametrize("invalid", [True, False, "4", float("nan"), float("inf"), -float("inf"), 0, 5.01, [], {}])
def test_malformed_mean_cannot_enter_the_phase_vote(invalid):
    means = scores()
    means["phase_1"] = invalid
    with pytest.raises(ValueError):
        phase_apply(prediction(), means)


@pytest.mark.parametrize("bad_id", ["phase_7", "phase_01", "phase_-1", "Phase_1", 1, "instrument_1"])
def test_mean_ids_must_match_the_complete_phase_ontology(bad_id):
    means = scores()
    means[bad_id] = means.pop("phase_1")
    with pytest.raises(ValueError):
        phase_apply(prediction(), means)


def test_extra_or_missing_phase_mean_is_rejected():
    means = scores()
    del means["phase_6"]
    with pytest.raises(ValueError):
        phase_apply(prediction(), means)
    means = scores()
    means["ivt_6"] = 4
    with pytest.raises(ValueError):
        phase_apply(prediction(), means)


@pytest.mark.parametrize("threshold", [3.5, 4.1, True, "4", None, float("nan")])
def test_evaluation_threshold_cannot_drift_during_phase_repair(threshold):
    with pytest.raises(ValueError):
        phase_apply(prediction(), scores(), threshold=threshold)


@pytest.mark.parametrize("phases", [[], [1, 2], [7], [-1], [True], ["2"]])
def test_invalid_current_phase_is_not_silently_repaired(phases):
    current = prediction()
    current["phase"] = phases
    with pytest.raises(ApiSchemaError):
        phase_apply(current, scores())


def test_history_uses_original_25fps_frame_offsets_and_ignores_future_frames():
    frames = list(range(1, 1502, 25))
    snapshot = frames.copy()
    assert choose_history(frames, 1001) == [251, 751, 1001]
    assert frames == snapshot


@pytest.mark.parametrize("target,expected", [(1, [1]), (226, [226]), (251, [1, 251]), (751, [1, 501, 751])])
def test_history_at_video_start_uses_only_existing_anchors_without_padding(target, expected):
    assert choose_history(list(range(1, 1002, 25)), target) == expected


def test_history_never_crosses_a_missing_sample_to_fill_an_older_anchor():
    frames = [f for f in range(1, 1502, 25) if f != 801]
    # After the gap, only 7 seconds of real contiguous history exist.
    assert choose_history(frames, 1001) == [1001]
    # At 18 seconds after the gap, the -10 second anchor becomes valid.
    assert choose_history(frames, 1276) == [1026, 1276]


def test_history_never_crosses_an_off_grid_frame_gap():
    frames = list(range(1, 501, 25)) + list(range(502, 1003, 25))
    assert choose_history(frames, 1002) == [752, 1002]


def test_history_supports_a_short_declared_offset_set_and_a_target_only_window():
    frames = list(range(1, 1002, 25))
    assert choose_history(frames, 501, offsets_seconds=(2, 1, 0)) == [451, 476, 501]
    assert choose_history(frames, 501, offsets_seconds=(0,)) == [501]


@pytest.mark.parametrize(
    "frames,target",
    [([], 1), ([1, 26], 51), ([26, 1], 26), ([1, 26, 26], 26),
     ([-24, 1], 1), ([1, True], 1), ([1, 26.0], 1), ([1, 26], True)],
)
def test_history_requires_ordered_unique_real_integer_frame_ids_and_an_available_target(frames, target):
    with pytest.raises(ValueError):
        choose_history(frames, target)


@pytest.mark.parametrize("offsets", [(), (30, 10), (0, 10), (10, 10, 0), (10, -1, 0), (10.0, 0), (True, 0)])
def test_history_rejects_ambiguous_or_noncausal_offset_contracts(offsets):
    with pytest.raises(ValueError):
        choose_history([1, 26, 51], 51, offsets_seconds=offsets)


def choice(phase=2, image_count=3):
    return {
        "phase_id": phase,
        "image_indices": [] if phase is None else [image_count - 1],
        "observation": "The current frame supplies evidence for this stage.",
    }


def choice_panel(phases, image_count=3):
    assert len(phases) == len(SEATS)
    return {seat: choice(phase, image_count) for seat, phase in zip(SEATS, phases, strict=True)}


@pytest.mark.parametrize("phase", [*range(7), None])
@pytest.mark.parametrize("image_count", [1, 2, 3])
def test_one_phase_choice_or_explicit_abstention_is_valid(phase, image_count):
    assert phase_choice_error(choice(phase, image_count), image_count) is None


@pytest.mark.parametrize("phase", [True, False, -1, 7, "2", 2.0, [2], {"phase": 2}])
def test_choice_cannot_assign_multiple_labels_or_coerce_invalid_phase_ids(phase):
    assert phase_choice_error(choice(phase), 3) is not None


@pytest.mark.parametrize(
    "indices",
    [[], [0], [1], [-1, 2], [2, 3], [2, 2], [True, 2], [2.0], "2", None],
)
def test_conclusive_choice_needs_unique_real_indices_including_the_current_image(indices):
    raw = {**choice(4), "image_indices": indices}
    assert phase_choice_error(raw, 3) is not None


def test_choice_can_cite_both_past_and_current_but_past_alone_is_not_current_evidence():
    raw = {**choice(4), "image_indices": [0, 1, 2]}
    assert phase_choice_error(raw, 3) is None
    raw["image_indices"] = [0, 1]
    assert phase_choice_error(raw, 3) is not None


@pytest.mark.parametrize("observation", ["", " " * 2, "x" * 1001, None, 1, []])
def test_choice_requires_a_bounded_nonblank_observation(observation):
    assert phase_choice_error({**choice(), "observation": observation}, 3) is not None


def test_choice_observation_accepts_the_declared_1000_character_limit():
    assert phase_choice_error({**choice(), "observation": "x" * 1000}, 3) is None


@pytest.mark.parametrize("field", ["phase_id", "image_indices", "observation"])
def test_missing_choice_field_is_not_filled_by_the_coordinator(field):
    raw = choice()
    del raw[field]
    assert phase_choice_error(raw, 3) is not None


@pytest.mark.parametrize("extra", [{"rating": 5}, {"phase_ids": [2, 3]}, {"confidence": 1}, {"judgments": {}}])
def test_choice_rejects_extra_scores_or_conflicting_alternative_output_formats(extra):
    assert phase_choice_error({**choice(), **extra}, 3) is not None


@pytest.mark.parametrize("raw", [None, [], "phase_2", 2])
def test_missing_or_nondictionary_choice_is_invalid(raw):
    assert phase_choice_error(raw, 3) is not None


@pytest.mark.parametrize("image_count", [0, 4, True, 3.0, None])
def test_choice_rejects_an_invalid_image_count(image_count):
    assert phase_choice_error(choice(), image_count) is not None


@pytest.mark.parametrize("old", range(7))
def test_a_three_of_five_majority_switches_only_phase_and_does_not_mutate_inputs(old):
    current = prediction(old)
    winner = (old + 1) % 7
    panel = choice_panel([winner, winner, winner, old, None])
    snapshot = deepcopy((current, panel))

    result, decision = apply_phase_choices(current, panel, 3)

    assert result == {**current, "phase": [winner]}
    assert decision["before"] == old
    assert decision["after"] == winner
    assert decision["reason"] == "MAJORITY_PHASE_SWITCH"
    assert (current, panel) == snapshot
    for task in current:
        assert result[task] is not current[task]


@pytest.mark.parametrize("phases", [[2, 2, 2, 4, None], [2, 2, 2, 2, 2]])
def test_current_stage_majority_preserves_every_prediction(phases):
    current = prediction()
    result, decision = apply_phase_choices(current, choice_panel(phases), 3)
    assert result == current
    assert decision["reason"] == "CURRENT_PHASE_MAJORITY"


@pytest.mark.parametrize("phases", [[4, 4, 2, 2, None], [4, 4, None, None, None], [0, 1, 2, 3, 4], [None] * 5])
def test_two_votes_or_abstentions_never_create_a_three_vote_majority(phases):
    current = prediction()
    result, decision = apply_phase_choices(current, choice_panel(phases), 3)
    assert result == current
    assert decision["reason"] == "NO_MAJORITY"


def test_three_valid_votes_can_win_when_the_other_two_explicitly_abstain():
    result, decision = apply_phase_choices(prediction(), choice_panel([4, None, 4, None, 4]), 3)
    assert result["phase"] == [4]
    assert decision["reason"] == "MAJORITY_PHASE_SWITCH"


@pytest.mark.parametrize("bad_seat", SEATS)
def test_a_malformed_response_invalidates_the_panel_even_with_four_matching_votes(bad_seat):
    current = prediction()
    panel = choice_panel([4] * 5)
    panel[bad_seat] = {**choice(4), "image_indices": [0]}
    result, decision = apply_phase_choices(current, panel, 3)
    assert result == current
    assert decision["reason"] == "INVALID_PANEL"
    assert all(result[task] is not current[task] for task in current)


@pytest.mark.parametrize("bad_panel", [None, {}, [], {"other": choice(4)}])
def test_invalid_panel_envelope_cannot_switch_phase(bad_panel):
    current = prediction()
    result, decision = apply_phase_choices(current, bad_panel, 3)
    assert result == current
    assert decision["reason"] == "INVALID_PANEL"


def test_missing_or_extra_reviewer_cannot_be_treated_as_five_distinct_valid_seats():
    current = prediction()
    panel = choice_panel([4] * 5)
    del panel["deepseek"]
    result, decision = apply_phase_choices(current, panel, 3)
    assert result == current
    assert decision["reason"] == "INVALID_PANEL"
    panel = choice_panel([4] * 5)
    panel["extra"] = choice(4)
    result, decision = apply_phase_choices(current, panel, 3)
    assert result == current
    assert decision["reason"] == "INVALID_PANEL"


def test_abstention_with_a_fake_image_reference_is_still_invalid_not_a_valid_empty_vote():
    panel = choice_panel([4, 4, 4, None, None])
    panel["deepseek"]["image_indices"] = [3]
    result, decision = apply_phase_choices(prediction(), panel, 3)
    assert result == prediction()
    assert decision["reason"] == "INVALID_PANEL"
