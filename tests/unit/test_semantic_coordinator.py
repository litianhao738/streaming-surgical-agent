from copy import deepcopy

import pytest

from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification import semantic_coordinator as semantic


def judgment(rating=3, finding="UNCLEAR", scope="UNCERTAIN", images=None):
    return {"rating": rating, "finding": finding, "scope": scope,
            "image_indices": [2] if images is None else images, "observation": "Observed tool and tissue relation."}


def fixture():
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    pool = old.make_pool(h0, {"instrument": [], "verb": [], "target": [8], "ivt": []})
    raw = {s: {"judgments": {p["id"]: judgment() for p in pool["propositions"]}} for s in old.SEATS}
    return h0, pool, raw


@pytest.mark.parametrize("value,error", [
    (judgment(5, "VISIBLE_ONLY", "WHOLE_FRAME"), "RATING_FINDING_CONFLICT"),
    (judgment(1, "REFUTED", "LOCAL_REGION"), "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"),
    (judgment(5, "MATCH", "LOCAL_REGION", [0, 1]), "NO_CURRENT_FRAME_EVIDENCE"),
    (judgment(3.0), "NON_INTEGER"),
    (judgment(images=[2, 2]), "DUPLICATE_IMAGE"),
    (judgment(images=[3]), "SCHEMA_INVALID"),
])
def test_invalid_evidence(value, error):
    assert semantic.item_error(value, "target", 3) == error


def test_conflict_blocks_high_mean_without_breaking_other_candidates():
    h0, pool, raw = fixture()
    for seat in old.SEATS:
        raw[seat]["judgments"]["target_8"] = judgment(5, "MATCH", "LOCAL_REGION")
        raw[seat]["judgments"]["ivt_7"] = judgment(1, "REFUTED", "WHOLE_FRAME")
    raw["qwen"]["judgments"]["target_8"] = judgment(1, "INDIRECT_EFFECT", "WHOLE_FRAME")
    means, clean = semantic.aggregate(raw, pool)
    assert means["target_8"] == 3
    assert clean["gpt"]["blocked"]["target_8"]["valid_items_mean"] == 4.2
    final = old.select(h0, pool, means)
    assert final["target"] == [0] and final["ivt"] == []
    assert final["verb"] == [0] and final["instrument"] == [0]


def test_missing_item_is_not_silently_averaged_and_extra_score_rejected():
    h0, pool, raw = fixture()
    for seat in old.SEATS:
        raw[seat]["judgments"]["target_8"] = judgment(5, "MATCH", "LOCAL_REGION")
    del raw["qwen"]["judgments"]["target_8"]
    means, _ = semantic.aggregate(raw, pool)
    assert old.select(h0, pool, means) == h0
    raw["gpt"]["scores"] = {"target_8": 5}
    _, clean = semantic.aggregate(raw, pool)
    assert set(clean["gpt"]["errors"]) == {p["id"] for p in pool["propositions"]}


def test_no_projection_deletion_and_no_h0_mutation():
    h0, pool, raw = fixture()
    frozen = deepcopy(h0)
    for seat in old.SEATS:
        raw[seat]["judgments"]["target_0"] = judgment(1, "REFUTED", "WHOLE_FRAME")
        raw[seat]["judgments"]["target_8"] = judgment(4, "MATCH", "LOCAL_REGION")
    means, _ = semantic.aggregate(raw, pool)
    out = old.select(h0, pool, means)
    assert out["target"] == [0, 8]  # IVT 7 still requires 0; unrelated new target allowed.
    assert out["ivt"] == h0["ivt"] and h0 == frozen


def test_short_history_and_no_new_candidate_stop():
    assert semantic.item_error(judgment(4, "MATCH", "LOCAL_REGION", [0]), "target", 1) is None
    h0, _, _ = fixture()
    proposals, rounds = [], []
    def propose(n, state, pool, issues):
        proposals.append(n)
        return {q: [] for q in old.TASKS}
    def review(n, state, pool):
        rounds.append(n)
        return {s: {"judgments": {p["id"]: judgment() for p in pool["propositions"]}} for s in old.SEATS}
    result = old.run(h0, propose, review, lambda *_: None, aggregate_review=semantic.aggregate)
    assert result["final"] == h0 and result["stop_reason"] == "NO_NEW_CANDIDATES"
    assert proposals == [0, 1] and rounds == [1]


def test_one_round_control_does_not_refill_or_fake_more_rounds():
    h0, _, _ = fixture()
    proposals = []
    def propose(n, state, pool, issues):
        proposals.append(n)
        return {q: [] for q in old.TASKS}
    def review(n, state, pool):
        return {s: {"judgments": {p["id"]: judgment() for p in pool["propositions"]}} for s in old.SEATS}
    result = old.run(h0, propose, review, lambda *_: None,
                     aggregate_review=semantic.aggregate, max_rounds=1)
    assert proposals == [0] and len(result["history"]) == 1
    assert result["stop_reason"] == "MAX_ROUNDS"
    assert len(result["snapshots"]) == 3 and all(s == h0 for s in result["snapshots"])
