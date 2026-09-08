from copy import deepcopy

import pytest

from surgical_agent.research.verification import recent_mean_panel as mean_panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def setup_case():
    current = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    pool = make_pool(current, {"instrument": [], "verb": [], "target": [8], "ivt": [19]})
    reviews = {s: {"judgments": {p["id"]: item(3) for p in pool["propositions"]}} for s in SEATS}
    return current, pool, reviews


def item(rating):
    return {"rating": rating, "finding": "MATCH" if rating >= 4 else "REFUTED" if rating <= 2 else "UNCLEAR",
            "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "Short visual observation."}


def test_real_mean_allows_explicit_disagreement_but_preserves_record():
    current, pool, reviews = setup_case()
    for seat, rating in zip(SEATS, (5, 5, 5, 5, 1), strict=True):
        reviews[seat]["judgments"]["target_8"] = item(rating)
    means, diag = mean_panel.aggregate(reviews, pool)
    assert means["target_8"] == 4.2 and diag["target_8"]["explicit_conflict"]
    assert mean_panel.select(current, pool, means)["target"] == [0, 8]


def test_missing_vote_is_none_not_neutral_or_four_seat_mean():
    current, pool, reviews = setup_case()
    for seat in SEATS:
        reviews[seat]["judgments"]["target_8"] = item(5)
    del reviews["qwen"]["judgments"]["target_8"]
    means, diag = mean_panel.aggregate(reviews, pool)
    assert means["target_8"] is None and diag["target_8"]["scores"][1] is None
    assert mean_panel.select(current, pool, means) == current


def test_predeclared_thresholds_and_no_mutation():
    current, pool, _ = setup_case()
    before = deepcopy(current)
    means = {p["id"]: 3 for p in pool["propositions"]}
    means["target_8"] = 3.6
    assert mean_panel.select(current, pool, means, threshold=3.5)["target"] == [0, 8]
    assert mean_panel.select(current, pool, means, threshold=4) == current == before
    with pytest.raises(ValueError):
        mean_panel.select(current, pool, means, threshold=3)


def test_new_ivt_needs_supported_relation_and_components():
    current, pool, _ = setup_case()
    means = {p["id"]: 3 for p in pool["propositions"]}
    means["ivt_19"] = 5
    assert mean_panel.select(current, pool, means)["ivt"] == [7]
    for p in pool["propositions"]:
        if p["id"] == "ivt_19":
            for task, label in p["components"].items():
                means[f"{task}_{label}"] = 4
    out = mean_panel.select(current, pool, means)
    assert out["ivt"] == [7, 19] and out["phase"] == current["phase"]


def test_refutation_does_not_delete_shared_component_or_project_whole_heads():
    current, pool, _ = setup_case()
    means = {p["id"]: 3 for p in pool["propositions"]}
    means["target_0"] = 1
    assert mean_panel.select(current, pool, means)["target"] == [0]
    means["ivt_7"] = 1
    out = mean_panel.select(current, pool, means)
    assert out["target"] == [] and out["ivt"] == []
    assert out["verb"] == current["verb"] and out["instrument"] == current["instrument"]


def test_uncertainty_and_unselected_positive_do_not_pass():
    current, pool, _ = setup_case()
    means = {p["id"]: (5 if p["label_id"] in current[p["task"]] else 1) for p in pool["propositions"]}
    diag = {p["id"]: {"scores": [means[p["id"]]] * 5, "invalid": {}} for p in pool["propositions"]}
    assert not mean_panel.unresolved(current, pool, means, diag)
    means["target_8"] = 3
    assert mean_panel.unresolved(current, pool, means, diag)[0]["candidate_id"] == "target_8"


def test_empty_candidate_pool_is_not_a_five_model_pass():
    with pytest.raises(ValueError, match="empty candidate pool"):
        mean_panel.aggregate({s: {"judgments": {}} for s in SEATS}, {"propositions": []})
