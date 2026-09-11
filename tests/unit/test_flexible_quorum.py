"""Evidence validity and bounded policy tests; no GT and no model calls."""

from copy import deepcopy

import pytest

from surgical_agent.research.verification import flexible_quorum as quorum
from surgical_agent.research.verification import recent_mean_panel as original
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def item(rating, *, scope="WHOLE_FRAME"):
    return {"rating": rating, "finding": "MATCH" if rating >= 4 else "REFUTED" if rating <= 2 else "UNCLEAR",
            "scope": scope, "image_indices": [2], "observation": "Synthetic observation for a policy test."}


def example():
    current = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    pool = make_pool(current, {"instrument": [], "verb": [], "target": [8], "ivt": [19]})
    reviews = {seat: {"judgments": {p["id"]: item(4) for p in pool["propositions"]}} for seat in SEATS}
    return current, pool, reviews


def score_candidate(reviews, pid, values):
    for seat, value in zip(SEATS, values, strict=True):
        if value is None:
            reviews[seat]["judgments"].pop(pid, None)
        else:
            reviews[seat]["judgments"][pid] = item(value)


def test_four_actual_votes_are_averaged_without_inventing_a_neutral():
    _, pool, reviews = example()
    score_candidate(reviews, "target_8", [4, 4, 5, 5, None])
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert means["target_8"] == 4.5
    assert diagnostics["target_8"]["scores"] == [4, 4, 5, 5, None]
    assert diagnostics["target_8"]["valid_count"] == 4
    assert diagnostics["target_8"]["valid_seats"] == list(SEATS[:-1])
    assert diagnostics["target_8"]["invalid"] == {"deepseek": "SCHEMA_INVALID"}
    assert quorum.aggregate(reviews, pool, minimum_valid=5)[0]["target_8"] is None


@pytest.mark.parametrize("floor", [4, 5])
def test_three_votes_cannot_satisfy_either_floor(floor):
    _, pool, reviews = example()
    score_candidate(reviews, "target_8", [5, 5, 5, None, None])
    means, diagnostics = quorum.aggregate(reviews, pool, minimum_valid=floor)
    assert means["target_8"] is None and diagnostics["target_8"]["valid_count"] == 3


def test_actual_uncertain_three_is_valid_and_counts_in_the_average():
    _, pool, reviews = example()
    score_candidate(reviews, "target_8", [3, 3, 4, 4, None])
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert means["target_8"] == 3.5 and diagnostics["target_8"]["valid_count"] == 4
    assert diagnostics["target_8"]["scores"][:2] == [3, 3]


def test_local_negative_remains_invalid_and_does_not_create_a_conflict():
    _, pool, reviews = example()
    reviews["grok"]["judgments"]["target_8"] = item(1, scope="LOCAL_REGION")
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert means["target_8"] == 4.0
    assert diagnostics["target_8"]["invalid"]["grok"] == "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"
    assert diagnostics["target_8"]["scores"][0] is None
    assert diagnostics["target_8"]["explicit_conflict"] is False
    assert reviews["grok"]["judgments"]["target_8"]["scope"] == "LOCAL_REGION"


def test_whole_failed_seat_abstains_in_every_proposition():
    _, pool, reviews = example()
    reviews["deepseek"] = None
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert all(value == 4 for value in means.values())
    assert all(d["valid_count"] == 4 and d["scores"][-1] is None for d in diagnostics.values())


@pytest.mark.parametrize("floor", [0, 3, 6, True, 4.0, "4", None])
def test_invalid_or_unregistered_floors_are_rejected(floor):
    _, pool, reviews = example()
    with pytest.raises(ValueError, match="minimum_valid"):
        quorum.aggregate(reviews, pool, minimum_valid=floor)


@pytest.mark.parametrize("pool", [{}, {"propositions": []}, {"propositions": None}, {"propositions": [None]}])
def test_malformed_and_empty_pools_do_not_pass(pool):
    with pytest.raises((ValueError, TypeError)):
        quorum.aggregate({seat: {"judgments": {}} for seat in SEATS}, pool)


def test_pool_identity_and_all_five_named_seats_are_required():
    _, pool, reviews = example()
    bad_pool = deepcopy(pool)
    bad_pool["propositions"][0]["id"] = "instrument_999"
    with pytest.raises(ValueError):
        quorum.aggregate(reviews, bad_pool)
    del reviews["qwen"]
    with pytest.raises(ValueError, match="five named reviewer seats"):
        quorum.aggregate(reviews, pool)


def test_floor_five_reproduces_original_means_and_diagnostics_fields():
    _, pool, reviews = example()
    score_candidate(reviews, "target_8", [5, 5, 5, 5, 1])
    score_candidate(reviews, "ivt_19", [4, None, 3, 1, 4])
    old_means, old_diagnostics = original.aggregate(reviews, pool)
    means, diagnostics = quorum.aggregate(reviews, pool, minimum_valid=5)
    assert means == old_means
    assert {pid: {key: value[key] for key in old_diagnostics[pid]} for pid, value in diagnostics.items()} == old_diagnostics


def test_selected_uncertain_missing_and_unselected_conflicts_enter_sorted_queue():
    current, pool, reviews = example()
    score_candidate(reviews, "ivt_7", [3, 3, 3, 3, 3])
    score_candidate(reviews, "target_0", [4, 4, 4, None, None])
    score_candidate(reviews, "target_8", [1, 4, 3, 3, 3])
    score_candidate(reviews, "ivt_19", [3, 3, 3, 3, 3])
    means, diagnostics = quorum.aggregate(reviews, pool)
    queue = quorum.recheck_queue(current, pool, means, diagnostics)
    by_id = {row["candidate_id"]: row for row in queue}
    assert list(by_id) == sorted(by_id)
    assert by_id["ivt_7"]["reasons"] == ["SELECTED_BELOW_SUPPORT_THRESHOLD"]
    assert by_id["target_0"]["reasons"] == ["SELECTED_MISSING_OR_INVALID"]
    assert by_id["target_8"]["reasons"] == ["VALID_REVIEWER_CONFLICT"]
    assert by_id["target_8"]["currently_selected"] is False
    assert "ivt_19" not in by_id


def test_conflict_remains_a_recheck_reason_even_at_high_average():
    current, pool, reviews = example()
    score_candidate(reviews, "instrument_0", [5, 5, 5, 5, 1])
    means, diagnostics = quorum.aggregate(reviews, pool)
    queue = quorum.recheck_queue(current, pool, means, diagnostics)
    match = next(row for row in queue if row["candidate_id"] == "instrument_0")
    assert match["mean"] == 4.2 and match["reasons"] == ["VALID_REVIEWER_CONFLICT"]


def test_queue_uses_each_policys_mean_and_current_not_a_shared_strict_decision():
    current, pool, reviews = example()
    score_candidate(reviews, "target_8", [4, 4, 4, 4, None])
    means4, diagnostics4 = quorum.aggregate(reviews, pool, minimum_valid=4)
    means5, diagnostics5 = quorum.aggregate(reviews, pool, minimum_valid=5)
    policy4 = quorum.select(current, pool, means4)
    policy5 = quorum.select(current, pool, means5)
    assert 8 in policy4["target"] and 8 not in policy5["target"]
    assert not any(row["candidate_id"] == "target_8" for row in quorum.recheck_queue(policy4, pool, means4, diagnostics4))
    assert not any(row["candidate_id"] == "target_8" for row in quorum.recheck_queue(policy5, pool, means5, diagnostics5))
    score_candidate(reviews, "target_0", [4, 4, 4, 4, None])
    means4, diagnostics4 = quorum.aggregate(reviews, pool, minimum_valid=4)
    means5, diagnostics5 = quorum.aggregate(reviews, pool, minimum_valid=5)
    assert not any(row["candidate_id"] == "target_0" for row in quorum.recheck_queue(current, pool, means4, diagnostics4))
    assert any(row["candidate_id"] == "target_0" for row in quorum.recheck_queue(current, pool, means5, diagnostics5))


def test_empty_queue_does_not_add_unverified_candidates_or_return_model_pass():
    current, pool, reviews = example()
    for p in pool["propositions"]:
        if p["label_id"] not in current[p["task"]]:
            score_candidate(reviews, p["id"], [3, 3, 3, 3, 3])
    before = deepcopy((current, pool, reviews))
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert quorum.recheck_queue(current, pool, means, diagnostics) == []
    assert quorum.select(current, pool, means) == current
    assert (current, pool, reviews) == before


def test_queue_output_does_not_mutate_scores_or_invalid_reasons():
    current, pool, reviews = example()
    score_candidate(reviews, "target_0", [4, 4, 4, None, None])
    means, diagnostics = quorum.aggregate(reviews, pool)
    before = deepcopy(diagnostics)
    queue = quorum.recheck_queue(current, pool, means, diagnostics)
    queue[0]["scores"][0] = 1
    queue[0]["invalid"].clear()
    assert diagnostics == before


@pytest.mark.parametrize("damage", ["fabricated_mean", "neutral_missing", "wrong_seat_count"])
def test_queue_rejects_fabricated_or_inconsistent_diagnostics(damage):
    current, pool, reviews = example()
    score_candidate(reviews, "target_0", [4, 4, 4, 4, None])
    means, diagnostics = quorum.aggregate(reviews, pool)
    if damage == "fabricated_mean":
        means["target_0"] = 3.8
    elif damage == "neutral_missing":
        diagnostics["target_0"]["scores"][-1] = 3
    else:
        diagnostics["target_0"]["valid_count"] = 5
    with pytest.raises(ValueError):
        quorum.recheck_queue(current, pool, means, diagnostics)


def test_selection_is_original_policy_and_still_requires_ivt_components():
    current, pool, reviews = example()
    assert quorum.select is original.select
    score_candidate(reviews, "ivt_19", [5, 5, 5, 5, None])
    score_candidate(reviews, "target_8", [3, 3, 3, 3, None])
    means, _ = quorum.aggregate(reviews, pool)
    assert quorum.select(current, pool, means)["ivt"] == [7]
    score_candidate(reviews, "target_8", [4, 4, 4, 4, None])
    means, _ = quorum.aggregate(reviews, pool)
    result = quorum.select(current, pool, means)
    assert result["ivt"] == [7, 19] and result["phase"] == current["phase"]


def test_local_absence_cannot_delete_even_when_four_valid_negative_votes_would():
    current, pool, reviews = example()
    for seat in SEATS[:4]:
        reviews[seat]["judgments"]["ivt_7"] = item(1, scope="LOCAL_REGION")
    means, diagnostics = quorum.aggregate(reviews, pool)
    assert means["ivt_7"] is None and diagnostics["ivt_7"]["valid_count"] == 1
    assert quorum.select(current, pool, means)["ivt"] == [7, 19]
    for seat in SEATS[:4]:
        reviews[seat]["judgments"]["ivt_7"] = item(1)
    reviews["deepseek"]["judgments"].pop("ivt_7")
    means, _ = quorum.aggregate(reviews, pool)
    assert means["ivt_7"] == 1 and quorum.select(current, pool, means)["ivt"] == [19]


def test_refuted_component_needed_by_retained_ivt_is_protected():
    current, pool, reviews = example()
    score_candidate(reviews, "target_0", [1, 1, 1, 1, None])
    means, _ = quorum.aggregate(reviews, pool)
    result = quorum.select(current, pool, means)
    assert 7 in result["ivt"] and 0 in result["target"]
    assert result["phase"] == current["phase"]
