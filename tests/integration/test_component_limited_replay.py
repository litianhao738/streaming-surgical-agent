"""Counterfactual replay must stop where cached model inputs cease to match."""

from copy import deepcopy

import pytest

from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from tools.audit import replay_component_limited_repair as replay


def initial():
    return {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}


def test_saved_pending_counts_only_normalize_json_object_keys():
    actual = {"proposer_count": {7: 2, 60: 1, 17: 3}, "queued_ivt": [17, 7, 60]}
    saved = {"proposer_count": {"7": 2, "60": 1, "17": 3}, "queued_ivt": [17, 7, 60]}
    assert replay.saved_json_equal(actual, saved)
    assert not replay.saved_json_equal(actual, {**saved, "queued_ivt": [7, 17, 60]})
    assert not replay.saved_json_equal(actual, {**saved, "proposer_count": {"7": 3, "60": 1, "17": 3}})
    assert not replay.saved_json_equal({"valid": 1}, {"valid": True})


def answers(pool, *, proposal=None, ivt_rating=3):
    raw = {}
    for seat in SEATS:
        items = {}
        for p in pool["propositions"]:
            rating = ivt_rating if p["task"] == "ivt" else 5
            items[p["id"]] = {"rating": rating, "finding": "MATCH" if rating >= 4 else "REFUTED" if rating <= 2 else "UNCLEAR",
                               "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "Synthetic fixture."}
        raw[seat] = ({"rows": [{"candidate_id": key, **v} for key, v in items.items()]}
                     if seat == "gemini" else {"judgments": items})
        raw[seat]["proposals"] = {"ivt": proposal or []}
    return {"raw_reviews": raw, "image_count": 3}


def test_only_relation_supported_new_components_are_added_and_input_is_unchanged():
    h0 = initial()
    h0["target"] = [2]  # Existing independent labels are not rebuilt from IVT.
    before = deepcopy(h0)
    pool = make_pool(h0, {"instrument": [], "verb": [], "target": [1], "ivt": [7]})
    means = {p["id"]: 5 for p in pool["propositions"]}
    after = replay.component_limited_select(h0, pool, means)
    assert h0 == before
    assert after["ivt"] == [7]
    assert after["verb"] == [0]
    assert after["target"] == [0, 2]  # Standalone target_1 high score is insufficient.
    assert after["phase"] == [2]


def test_original_component_deletion_is_preserved():
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [2]}
    pool = make_pool(h0)
    means = {p["id"]: 1 for p in pool["propositions"]}
    assert replay.component_limited_select(h0, pool, means) == {
        "instrument": [], "verb": [], "target": [], "ivt": [], "phase": [2]}


def test_fixed_matching_requests_support_real_stateful_three_round_replay():
    seen = []

    def cached(number, pool):
        seen.append((number, deepcopy(pool)))
        return answers(pool, proposal=[7] if number == 1 else [])

    original = replay.replay_policy(initial(), cached, limited=False)
    original_requests = deepcopy(seen)
    seen.clear()
    limited = replay.replay_policy(initial(), cached, limited=True)
    assert original_requests == seen
    assert original["complete"] and limited["complete"]
    assert original["final"]["verb"] == [0] and limited["final"]["verb"] == []
    assert limited["history"][2]["before"] == limited["history"][1]["after"]
    assert original["history"][2]["before"] != limited["history"][2]["before"]
    assert original["review_calls_reused"] == limited["review_calls_reused"] == 15
    assert limited["status"] == "UNRESOLVED"


def test_changed_stop_branch_cannot_fabricate_an_uncached_third_round():
    def cached(number, pool):
        if number == 3:
            raise replay.ReplayUnavailable("ROUND_NOT_AVAILABLE:fixture:3")
        return answers(pool, proposal=[7] if number == 1 else [], ivt_rating=1)

    original = replay.replay_policy(initial(), cached, limited=False)
    limited = replay.replay_policy(initial(), cached, limited=True)
    assert original["complete"] and original["status"] == "MODEL_PASS"
    assert original["review_calls_reused"] == 10
    assert not limited["complete"] and limited["final"] is None
    assert limited["status"] == "CACHE_INCOMPLETE" and "ROUND_NOT_AVAILABLE" in limited["reason"]
    assert limited["review_calls_reused"] == 10


def test_request_mismatch_is_not_treated_as_a_bad_model_prediction():
    def changed(number, pool):
        raise replay.ReplayUnavailable("EXACT_REQUEST_MISMATCH:fixture")

    result = replay.replay_policy(initial(), changed, limited=True)
    assert result["final"] is None and not result["complete"]
    assert result["last_valid_prediction"] == initial()
    assert result["review_calls_reused"] == 0


@pytest.mark.parametrize("running,complete", [(True, True), (False, False)])
def test_running_or_incomplete_replay_cannot_select_a_favorable_scoring_subset(tmp_path, running, complete):
    replay.save(tmp_path / "predictions.json", [])
    replay.save(tmp_path / "manifest.json", {"source_was_running": running, "all_complete": complete,
                                             "prediction_sha256": replay.sha(tmp_path / "predictions.json")})
    with pytest.raises(ValueError, match="partial|incomplete"):
        replay.score(tmp_path, None)
