"""Feedback forwards verified bindings, never invented reviewer evidence."""

from copy import deepcopy

import pytest

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_feedback import build_review_feedback


def fixture():
    current = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}
    pool = make_pool(current, {"instrument": [], "verb": [], "target": [], "ivt": [7]})
    reviews = {seat: {"judgments": {p["id"]: {
        "rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [2],
        "observation": "Contact is visible but the action is uncertain."}
        for p in pool["propositions"]}} for seat in SEATS}
    return current, pool, reviews


def issues_for(current, pool, reviews, image_count=3):
    means, diagnostics = panel.aggregate(reviews, pool, image_count=image_count)
    return panel.unresolved(current, pool, means, diagnostics)


def test_feedback_is_deterministic_preserves_all_five_and_does_not_mutate():
    current, pool, reviews = fixture()
    issues = issues_for(current, pool, reviews)
    inputs = deepcopy((pool, reviews, issues))
    result = build_review_feedback(pool, reviews, issues)
    assert (pool, reviews, issues) == inputs
    assert result["schema_version"] == "candidate_review_feedback_v1"
    assert "fallible, not ground truth" in result["instruction"]
    assert "Verify every claim" in result["instruction"]
    assert [c["candidate_id"] for c in result["candidates"]] == sorted(i["candidate_id"] for i in issues)
    for candidate in result["candidates"]:
        assert [item["reviewer_seat"] for item in candidate["valid_observations"]] == list(SEATS)
        assert candidate["invalid_reviewers"] == []
        assert candidate["proposition"]["id"] == candidate["candidate_id"]
    reversed_pool = {"propositions": list(reversed(pool["propositions"]))}
    reversed_reviews = dict(reversed(list(reviews.items())))
    assert build_review_feedback(reversed_pool, reversed_reviews, list(reversed(issues))) == result
    result["candidates"][0]["valid_observations"][0]["image_indices"].append(0)
    assert (pool, reviews, issues) == inputs


def test_invalid_local_negative_is_only_an_invalid_reason_not_visual_feedback():
    current, pool, reviews = fixture()
    reviews["gpt"]["judgments"]["ivt_7"].update(
        rating=1, finding="REFUTED", scope="LOCAL_REGION", observation="Invalid local absence must not be forwarded.")
    issues = issues_for(current, pool, reviews)
    result = build_review_feedback(pool, reviews, issues)
    candidate = next(c for c in result["candidates"] if c["candidate_id"] == "ivt_7")
    assert len(candidate["valid_observations"]) == 4
    assert candidate["invalid_reviewers"] == [{
        "reviewer_seat": "gpt", "reason": "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"}]
    assert "Invalid local absence" not in str(result)


def test_genuine_missing_normalized_item_is_recorded_without_an_observation():
    current, pool, reviews = fixture()
    reviews["gemini"]["judgments"].pop("ivt_7")
    issues = issues_for(current, pool, reviews)
    result = build_review_feedback(pool, reviews, issues)
    candidate = next(c for c in result["candidates"] if c["candidate_id"] == "ivt_7")
    assert candidate["invalid_reviewers"] == [{"reviewer_seat": "gemini", "reason": "SCHEMA_INVALID"}]
    assert len(candidate["valid_observations"]) == 4


@pytest.mark.parametrize("mutation", ["removed_evidence", "fake_missing", "wrong_score", "wrong_mean", "wrong_reason"])
def test_issue_cannot_fabricate_valid_or_missing_review_evidence(mutation):
    current, pool, reviews = fixture()
    issues = issues_for(current, pool, reviews)
    issue = next(i for i in issues if i["candidate_id"] == "ivt_7")
    if mutation == "removed_evidence":
        reviews["gpt"]["judgments"].pop("ivt_7")
    elif mutation == "fake_missing":
        issue["invalid"] = {"gpt": "SCHEMA_INVALID"}
        issue["scores"][SEATS.index("gpt")] = None
        issue["mean"] = None
    elif mutation == "wrong_score":
        issue["scores"][0] = 5
    elif mutation == "wrong_mean":
        issue["mean"] = 4
    else:
        issue["invalid"] = {"gpt": "NOT_REAL"}
    with pytest.raises(ValueError, match="disagrees"):
        build_review_feedback(pool, reviews, issues)


@pytest.mark.parametrize("mutation", ["missing_seat", "extra_seat", "wrong_seat_case", "unknown_review_id",
                                       "unknown_issue_id", "duplicate_issue", "duplicate_pool", "false_binding",
                                       "wrong_components", "wrong_name"])
def test_ambiguous_seat_and_candidate_bindings_are_rejected(mutation):
    current, pool, reviews = fixture()
    issues = issues_for(current, pool, reviews)
    if mutation == "missing_seat":
        reviews.pop("gpt")
    elif mutation == "extra_seat":
        reviews["extra"] = reviews["gpt"]
    elif mutation == "wrong_seat_case":
        reviews["GPT"] = reviews.pop("gpt")
    elif mutation == "unknown_review_id":
        reviews["gpt"]["judgments"]["ivt_100"] = reviews["gpt"]["judgments"]["ivt_7"]
    elif mutation == "unknown_issue_id":
        issues[0]["candidate_id"] = "ivt_100"
    elif mutation == "duplicate_issue":
        issues.append(deepcopy(issues[0]))
    elif mutation == "duplicate_pool":
        pool["propositions"].append(deepcopy(pool["propositions"][0]))
    elif mutation == "false_binding":
        pool["propositions"][0]["id"] = "instrument_1"
    elif mutation == "wrong_components":
        next(p for p in pool["propositions"] if p["task"] == "ivt")["components"]["target"] = 1
    else:
        pool["propositions"][0]["name"] = "scissors"
    with pytest.raises(ValueError):
        build_review_feedback(pool, reviews, issues)


@pytest.mark.parametrize("mutation", ["past_only", "rating_boolean", "rating_finding", "too_long"])
def test_invalid_core_evidence_is_never_forwarded_as_a_reason(mutation):
    current, pool, reviews = fixture()
    item = reviews["qwen"]["judgments"]["ivt_7"]
    item.update(rating=4, finding="MATCH", scope="LOCAL_REGION")
    if mutation == "past_only":
        item["image_indices"] = [0]
    elif mutation == "rating_boolean":
        item["rating"] = True
    elif mutation == "rating_finding":
        item["finding"] = "REFUTED"
    else:
        item["observation"] = "x" * 1001
    issues = issues_for(current, pool, reviews)
    result = build_review_feedback(pool, reviews, issues)
    candidate = next(c for c in result["candidates"] if c["candidate_id"] == "ivt_7")
    assert len(candidate["valid_observations"]) == 4
    assert candidate["invalid_reviewers"][0]["reviewer_seat"] == "qwen"


@pytest.mark.parametrize("image_count", [0, 4, True, 1.0])
def test_invalid_image_count_rejected(image_count):
    current, pool, reviews = fixture()
    with pytest.raises(ValueError, match="causal images"):
        build_review_feedback(pool, reviews, issues_for(current, pool, reviews), image_count=image_count)


def test_real_short_history_uses_actual_last_image_and_no_future_padding():
    current, pool, reviews = fixture()
    for raw in reviews.values():
        for item in raw["judgments"].values():
            item["image_indices"] = [0]
    result = build_review_feedback(pool, reviews, issues_for(current, pool, reviews, 1), image_count=1)
    assert all(item["image_indices"] == [0] for c in result["candidates"] for item in c["valid_observations"])


def test_empty_issue_list_does_not_invent_feedback_but_still_requires_five_seats():
    _, pool, reviews = fixture()
    assert build_review_feedback(pool, reviews, [])["candidates"] == []
    reviews.pop("gpt")
    with pytest.raises(ValueError, match="five"):
        build_review_feedback(pool, reviews, [])
