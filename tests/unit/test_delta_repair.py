from copy import deepcopy

import pytest

from surgical_agent.research.verification.delta_repair import (
    apply_reviewed_delta,
    compile_delta,
    delta_schema,
)


def initial():
    return {"instrument": [2], "verb": [2], "target": [1], "ivt": [59], "phase": [1]}


def edit(pid, op="ADD"):
    return {"candidate_id": pid, "operation": op, "rating": 5 if op == "ADD" else 1,
            "finding": "MATCH" if op == "ADD" else "REFUTED", "scope": "WHOLE_FRAME",
            "image_indices": [2], "observation": "Observed action-object relation at the tool tip."}


def test_empty_delta_preserves_all_heads_without_returning_prediction():
    h0 = initial()
    assert set(delta_schema()["properties"]) == {"changes"}
    compiled = compile_delta(h0, {"changes": []})
    assert compiled["tentative"] == h0
    assert compiled["review_pool"] == {"propositions": []}
    assert apply_reviewed_delta(compiled, {}) == h0


def test_full_ontology_addition_and_explicit_components_are_bound():
    h0 = initial()
    saved = deepcopy(h0)
    delta = {"changes": [edit("ivt_60"), edit("target_0")]}
    compiled = compile_delta(h0, delta)
    assert compiled["tentative"]["ivt"] == [59, 60]
    assert compiled["tentative"]["target"] == [0, 1]
    ids = {p["id"] for p in compiled["review_pool"]["propositions"]}
    assert ids == {"ivt_60", "target_0", "instrument_2", "verb_2"}
    assert h0 == saved and compiled["tentative"]["phase"] == h0["phase"]


@pytest.mark.parametrize("changes", [
    [edit("ivt_60")],  # Required new target omitted.
    [edit("target_1", "REMOVE")],  # Retained IVT still requires it.
    [edit("instrument_2")],  # No-op ADD.
    [edit("instrument_0", "REMOVE")],  # No-op REMOVE.
    [edit("target_0"), edit("target_0")],
    [edit("phase_2")],
    [edit("target_99")],
    [{**edit("target_0"), "image_indices": [0]}],
    [{**edit("ivt_59", "REMOVE"), "scope": "LOCAL_REGION"}],
])
def test_invalid_edits_never_become_tentative_answers(changes):
    with pytest.raises((ValueError, TypeError)):
        compile_delta(initial(), {"changes": changes})


def test_deleting_ivt_does_not_implicitly_delete_any_component():
    compiled = compile_delta(initial(), {"changes": [edit("ivt_59", "REMOVE")]})
    final = apply_reviewed_delta(compiled, {"ivt_59": 1})
    assert final == {**initial(), "ivt": []}


def test_panel_cannot_change_unrequested_dependency_and_can_block_new_ivt():
    compiled = compile_delta(initial(), {"changes": [edit("ivt_60"), edit("target_0")]})
    means = {p["id"]: 5 for p in compiled["review_pool"]["propositions"]}
    means["instrument_2"] = 1
    final = apply_reviewed_delta(compiled, means)
    assert final["instrument"] == [2]  # Dependency vote is not an authorized deletion.
    assert final["ivt"] == [59]  # Added relation lacks component support.
    assert final["target"] == [0, 1]  # Independently reviewed target ADD can pass.


def test_uncertain_patch_does_not_change_current_state():
    compiled = compile_delta(initial(), {"changes": [edit("ivt_60"), edit("target_0")]})
    means = {p["id"]: 3 for p in compiled["review_pool"]["propositions"]}
    assert apply_reviewed_delta(compiled, means) == initial()
    with pytest.raises(ValueError):
        apply_reviewed_delta(compiled, {})
