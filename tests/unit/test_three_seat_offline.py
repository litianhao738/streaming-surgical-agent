"""The offline subset keeps evidence validity and changes only declared votes."""
from copy import deepcopy

import pytest

from scripts.run_three_seat_offline_trial import (
    SELECTED_SEATS,
    three_interaction,
    three_phase,
)
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_normalization import normalize_review


def prediction():
    return {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [1]}


def review_item(rating):
    return {"rating": rating, "finding": "MATCH" if rating >= 4 else "REFUTED",
            "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "Current-frame evidence."}


def graph_record(h0, votes):
    pool = make_pool(h0, {"instrument": [1], "verb": [], "target": [], "ivt": []})
    raw = {seat: {"judgments": {"instrument_0": review_item(5), "instrument_1": review_item(votes[seat])}}
           for seat in SEATS}
    reviews = {s: normalize_review(raw[s], pool, seat=s, image_count=3)[0] for s in SEATS}
    return {"pool": pool, "raw_reviews": raw, "reviews": reviews}


def test_only_three_requested_seats_enter_the_mean_without_changing_phase():
    h0 = prediction()
    record = graph_record(h0, {"gpt": 5, "gemini": 5, "deepseek": 2, "grok": 1, "qwen": 1})
    snapshot = deepcopy((h0, record))
    result = three_interaction(h0, record, 3)
    assert result["means"]["instrument_1"] == 4
    assert result["prediction"]["instrument"] == [0, 1]
    assert result["prediction"]["phase"] == [1]
    assert (h0, record) == snapshot


@pytest.mark.parametrize("bad_seat", SELECTED_SEATS)
def test_one_invalid_retained_vote_blocks_that_candidate_instead_of_becoming_neutral(bad_seat):
    h0 = prediction()
    record = graph_record(h0, dict.fromkeys(SEATS, 5))
    record["raw_reviews"][bad_seat]["judgments"]["instrument_1"]["image_indices"] = [0]
    record["reviews"][bad_seat] = normalize_review(record["raw_reviews"][bad_seat], record["pool"],
                                                   seat=bad_seat, image_count=3)[0]
    result = three_interaction(h0, record, 3)
    assert result["means"]["instrument_1"] is None
    assert result["means"]["instrument_0"] == 5
    assert result["prediction"] == h0


@pytest.mark.parametrize("phases,expected,reason", [
    ([2, 2, None], 2, "MAJORITY_PHASE_SWITCH"),
    ([2, None, None], 1, "NO_MAJORITY"),
    ([2, 3, 4], 1, "NO_MAJORITY"),
    ([1, 1, 2], 1, "CURRENT_PHASE_MAJORITY"),
])
def test_three_phase_vote_uses_two_matching_choices_and_retains_other_heads(phases, expected, reason):
    raw = {s: {"phase_id": p, "image_indices": [2], "observation": "Current phase evidence."}
           for s, p in zip(SELECTED_SEATS, phases, strict=True)}
    current = prediction()
    out, decision = three_phase(current, raw, 3)
    assert out == {**current, "phase": [expected]}
    assert decision["reason"] == reason


def test_three_phase_does_not_drop_an_invalid_seat_to_accept_the_other_two():
    raw = {s: {"phase_id": 2, "image_indices": [2], "observation": "Current phase evidence."}
           for s in SELECTED_SEATS}
    raw["deepseek"]["extra"] = "Not allowed"
    out, decision = three_phase(prediction(), raw, 3)
    assert out == prediction()
    assert decision["reason"] == "INVALID_PANEL"
