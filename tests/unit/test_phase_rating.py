from copy import deepcopy

import pytest

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.phase_rating import (
    apply_ratings,
    rating_error,
)


def current():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]}


def reviews(old_scores, alternative_scores):
    result = {}
    for s, old, alternative in zip(SEATS, old_scores, alternative_scores, strict=True):
        result[s] = {"ratings": {str(p): old if p == 1 else alternative if p == 3 else 1 for p in range(7)},
            "image_indices": [2], "observation": "Visible evidence fixture."}
    return result


@pytest.mark.parametrize("old,alternative,expected", [
    ([3]*5, [4]*5, 3),  # Exact mean-4 boundary passes.
    ([3]*5, [4, 4, 4, 4, 3], 1),
    ([4]*5, [4]*5, 1),  # Tie with H0 preserves it.
    ([5]*5, [4]*5, 1),
    ([3]*5, [5, 5, 5, 1, 1], 1),  # Three strong supporters do not override two strong opponents.
])
def test_threshold_and_h0_comparison(old, alternative, expected):
    before = current()
    saved = deepcopy(before)
    result, _ = apply_ratings(before, reviews(old, alternative))
    assert result["phase"] == [expected]
    assert before == saved
    assert all(result[t] == before[t] for t in before if t != "phase")


def test_two_alternatives_tied_preserve_h0():
    raw = reviews([2]*5, [4]*5)
    for r in raw.values():
        r["ratings"]["5"] = 4
    result, decision = apply_ratings(current(), raw)
    assert result == current()
    assert decision["reason"] == "TIED_PHASE_SUPPORT"


@pytest.mark.parametrize("bad", [True, 0, 6, 4.0, "4", None])
def test_non_integer_or_out_of_range_rating_invalidates_panel(bad):
    raw = reviews([2]*5, [5]*5)
    raw[SEATS[0]]["ratings"]["0"] = bad
    result, decision = apply_ratings(current(), raw)
    assert result == current()
    assert not decision["valid_panel"]
    assert all(v is None for v in decision["means"].values())


def test_missing_seat_and_extra_schema_fields_are_not_votes():
    raw = reviews([2]*5, [5]*5)
    del raw[SEATS[0]]
    assert apply_ratings(current(), raw)[0] == current()
    raw = reviews([2]*5, [5]*5)
    raw[SEATS[0]]["type"] = "object"
    assert rating_error(raw[SEATS[0]], 3) == "INVALID_FIELDS"
    assert apply_ratings(current(), raw)[0] == current()


def test_all_seven_ratings_and_current_image_required():
    raw = reviews([2]*5, [5]*5)
    del raw[SEATS[0]]["ratings"]["6"]
    assert apply_ratings(current(), raw)[0] == current()
    raw = reviews([2]*5, [5]*5)
    raw[SEATS[0]]["image_indices"] = [0]
    assert rating_error(raw[SEATS[0]], 3) == "CURRENT_IMAGE_REQUIRED"
