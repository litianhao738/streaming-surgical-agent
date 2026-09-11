from copy import deepcopy

from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification.repair_v2 import (
    aggregate_quorum,
    complete_target_hypotheses,
)


def fixture():
    h0 = {"instrument": [0, 2], "verb": [1, 2], "target": [1], "ivt": [59], "phase": [3]}
    pool = old.make_pool(h0)
    value = {"rating": 4, "finding": "MATCH", "scope": "LOCAL_REGION", "image_indices": [2], "observation": "Tool acts on tissue."}
    reviews = {s: {"judgments": {p["id"]: deepcopy(value) for p in pool["propositions"]}} for s in old.SEATS}
    return h0, pool, reviews


def test_completion_adds_legal_unseen_targets_without_accepting_or_using_gt():
    h0, pool, _ = fixture()
    before = deepcopy((h0, pool))
    expanded, info = complete_target_hypotheses(h0, pool)
    ids = {p["id"] for p in expanded["propositions"]}
    assert {"ivt_19", "ivt_60", "target_8", "target_0"} <= ids
    assert info["status"] == "EXPANDED" and (h0, pool) == before
    for p in expanded["propositions"]:
        if p["task"] == "ivt":
            assert p["components"]["instrument"] in h0["instrument"]
            assert p["components"]["verb"] in h0["verb"]
    assert complete_target_hypotheses(h0, expanded)[0] == expanded


def test_completion_overflow_preserves_original_pool(monkeypatch):
    h0, pool, _ = fixture()
    monkeypatch.setattr(old, "MAX_POOL", len(pool["propositions"]))
    assert complete_target_hypotheses(h0, pool) == (pool, {"status": "CAP_KEEP_ORIGINAL", "legal_alternatives": 20})


def test_one_invalid_seat_abstains_but_two_cannot_be_silently_dropped():
    _, pool, reviews = fixture()
    reviews["gemini"] = None
    means, clean = aggregate_quorum(reviews, pool)
    assert set(means.values()) == {4}
    assert clean["gpt"]["quorum"]["target_1"]["abstained_seats"] == ["gemini"]
    reviews["qwen"] = None
    means, _ = aggregate_quorum(reviews, pool)
    assert set(means.values()) == {3}


def test_conclusive_negative_still_blocks_positive_majority():
    _, pool, reviews = fixture()
    for item in reviews["qwen"]["judgments"].values():
        item.update(rating=1, finding="REFUTED", scope="WHOLE_FRAME")
    means, _ = aggregate_quorum(reviews, pool)
    assert set(means.values()) == {3}


def test_local_negative_is_never_used_as_frame_deletion_vote():
    _, pool, reviews = fixture()
    for seat in old.SEATS:
        for item in reviews[seat]["judgments"].values():
            item.update(rating=1, finding="REFUTED", scope="LOCAL_REGION")
    means, clean = aggregate_quorum(reviews, pool)
    assert set(means.values()) == {3}
    assert all(len(r["errors"]) == len(pool["propositions"]) for r in clean.values())
