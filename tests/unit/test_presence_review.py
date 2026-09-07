"""Presence polarity, neutral request binding and shared A/B safety checks."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.diff_review import (
    DIFF_REVIEW_SCHEMA,
    DIFF_REVIEW_VERSION,
    build_change_claims,
    evaluate_diff_review,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_SCHEMA,
    PRESENCE_REVIEW_VERSION,
    build_presence_review_input,
    evaluate_presence_review,
    validate_presence_review,
    validate_presence_review_shape,
)

REFS = ["frame:25", "frame:50", "frame:75", "crop:1", "crop:2"]
FULL = "frame:75"


def _h0():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def _h1():
    return {"instrument": [0], "verb": [1], "target": [0], "ivt": [17], "phase": [1]}


def _body(h0, h1):
    return build_presence_review_input(
        h0, h1, allowed_evidence_refs=REFS, full_frame_ref=FULL)


def _review(h0, h1, presence="UNCLEAR"):
    return {"schema_version": PRESENCE_REVIEW_VERSION, "assessments": [
        {"proposition_id": proposition["proposition_id"], "presence": presence,
         "observation": "The visible interaction provides the stated evidence.",
         "evidence_refs": [FULL], "scope": "FRAME", "full_frame_reviewed": True}
        for proposition in _body(h0, h1)["propositions"]]}


def _all_changes_review(h0, h1):
    review = _review(h0, h1)
    claims = {(claim["task"], claim["label_id"]): claim
              for claim in build_change_claims(h0, h1)}
    for proposition, assessment in zip(_body(h0, h1)["propositions"],
                                       review["assessments"], strict=True):
        claim = claims[(proposition["task"], proposition["label_id"])]
        assessment["presence"] = "PRESENT" if claim["operation"] == "ADD" else "ABSENT"
    return review


def _evaluate(h0, h1, review, **kwargs):
    return evaluate_presence_review(
        h0, h1, review, allowed_evidence_refs=kwargs.get("refs", REFS),
        full_frame_ref=kwargs.get("full", FULL))


def test_standalone_import_and_schema_copy_preserve_historical_contract():
    result = subprocess.run(
        [sys.executable, "-c", "import surgical_agent.research.verification.presence_review"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    old = DIFF_REVIEW_SCHEMA["properties"]["assessments"]["items"]["properties"]
    assert "change_id" in old and "verdict" in old and "presence" not in old
    assert DIFF_REVIEW_SCHEMA["properties"]["schema_version"]["const"] == DIFF_REVIEW_VERSION


def test_neutral_body_identical_when_edit_direction_is_reversed():
    # Sorting by the label, rather than REMOVE first, is necessary to avoid
    # encoding edit direction in opaque IDs and proposition ordering.
    h0, h1 = _h1(), _h0()
    forward, reversed_body = _body(h0, h1), _body(h1, h0)
    assert forward == reversed_body
    assert set(forward) == {"propositions", "allowed_evidence_refs", "full_frame_ref"}
    assert [(item["proposition_id"], item["task"], item["label_id"])
            for item in forward["propositions"]] == [
                ("p001", "verb", 0), ("p002", "verb", 1),
                ("p003", "ivt", 7), ("p004", "ivt", 17)]
    for item in forward["propositions"]:
        assert set(item) == {"proposition_id", "task", "label_id", "statement"}
    encoded = json.dumps(forward)
    assert all(word not in encoded for word in ("h0", "h1", "ADD", "REMOVE", "operation"))


@pytest.mark.parametrize("operation,presence,accepted", [
    ("REMOVE", "PRESENT", False), ("REMOVE", "ABSENT", True),
    ("REMOVE", "UNCLEAR", False), ("ADD", "PRESENT", True),
    ("ADD", "ABSENT", False), ("ADD", "UNCLEAR", False),
])
def test_presence_polarity_controls_edits_without_model_operation_judgment(
        operation, presence, accepted):
    h0, h1 = _h0(), _h0()
    if operation == "REMOVE":
        h1["ivt"] = []
    else:
        h1["instrument"].append(6)
    review = _review(h0, h1, presence)
    # Regression for the observed bug: presence is explicit, rather than
    # interpreting "contradicted" as either opposing deletion or label absence.
    if operation == "REMOVE" and presence == "PRESENT":
        review["assessments"][0]["observation"] = "The original label is present in the frame."
    result = _evaluate(h0, h1, review)
    assert result["schema_version"] == PRESENCE_REVIEW_VERSION
    assert result["decision_a"] == result["decision_b"] == ("ACCEPT" if accepted else "KEEP")
    assert result["final_a"] == result["final_b"] == (h1 if accepted else h0)


@pytest.mark.parametrize("modification", [
    {"scope": "INSTANCE"}, {"full_frame_reviewed": False},
    {"evidence_refs": ["crop:1"]}, {"evidence_refs": []},
])
def test_absence_does_not_authorize_removal_without_full_frame_evidence(modification):
    h0, h1 = _h0(), _h0()
    h1["ivt"] = []
    review = _review(h0, h1, "ABSENT")
    review["assessments"][0].update(modification)
    result = _evaluate(h0, h1, review)
    assert result["final_a"] == result["final_b"] == h0


@pytest.mark.parametrize("refs", [["frame:25"], ["frame:25", "frame:50"], ["crop:1"]])
def test_presence_in_history_or_crop_alone_cannot_authorize_target_addition(refs):
    h0, h1 = _h0(), _h0()
    h1["instrument"].append(6)
    review = _review(h0, h1, "PRESENT")
    review["assessments"][0]["evidence_refs"] = refs
    assert _evaluate(h0, h1, review)["final_b"] == h0


def test_b_reuses_existing_shared_component_protection_without_reprojecting_heads():
    h0, h1 = _h0(), _h1()
    review = _all_changes_review(h0, h1)
    # Preserve old IVT 7 despite the proposed removal of its shared verb 0.
    old_ivt_id = next(item["proposition_id"] for item in _body(h0, h1)["propositions"]
                      if item["task"] == "ivt" and item["label_id"] == 7)
    next(item for item in review["assessments"]
         if item["proposition_id"] == old_ivt_id)["presence"] = "UNCLEAR"
    result = _evaluate(h0, h1, review)
    assert result["decision_a"] == "KEEP"
    assert result["final_b"]["ivt"] == [7, 17]
    assert result["final_b"]["verb"] == [0, 1]
    deletion = next(item for item in result["claim_decisions"]
                    if item["change_id"] == "verb:REMOVE:0")
    assert not deletion["applied_b"]
    assert deletion["reason_b"] == "REFERENCED_BY_REMAINING_IVT"


def test_equivalent_evidence_preserves_existing_a_b_policies_and_inputs():
    h0, h1 = _h0(), _h1()
    review = _all_changes_review(h0, h1)
    saved = deepcopy((h0, h1, review))
    old = {"schema_version": DIFF_REVIEW_VERSION, "assessments": [
        {"change_id": claim["change_id"],
         "verdict": "SUPPORTED" if claim["operation"] == "ADD" else "CONTRADICTED",
         "observation": "The visible interaction provides the stated evidence.",
         "evidence_refs": [FULL], "scope": "FRAME", "full_frame_reviewed": True}
        for claim in build_change_claims(h0, h1)]}
    old_result = evaluate_diff_review(h0, h1, old, allowed_evidence_refs=REFS, full_frame_ref=FULL)
    result = _evaluate(h0, h1, review)
    assert result == {**old_result, "schema_version": PRESENCE_REVIEW_VERSION}
    assert result["final_a"] == result["final_b"] == h1
    assert (h0, h1, review) == saved


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "unknown_id", "wrong_version", "unknown_ref", "extra_field",
    "old_verdict", "encoded_id", "wrong_boolean", "blank_observation", "duplicate_ref",
])
def test_invalid_reviews_fail_closed_and_validation_rejects(mutation):
    review = _review(_h0(), _h1())
    first = review["assessments"][0]
    if mutation == "missing":
        review["assessments"].pop()
    elif mutation == "duplicate":
        review["assessments"].append(deepcopy(first))
    elif mutation == "unknown_id":
        first["proposition_id"] = "p040"
    elif mutation == "wrong_version":
        review["schema_version"] = DIFF_REVIEW_VERSION
    elif mutation == "unknown_ref":
        first["evidence_refs"] = ["frame:future"]
    elif mutation == "extra_field":
        first["operation"] = "REMOVE"
    elif mutation == "old_verdict":
        first["presence"] = "CONTRADICTED"
    elif mutation == "encoded_id":
        first["proposition_id"] = "verb:REMOVE:0"
    elif mutation == "wrong_boolean":
        first["full_frame_reviewed"] = 1
    elif mutation == "blank_observation":
        first["observation"] = "   "
    else:
        first["evidence_refs"] = [FULL, FULL]
    with pytest.raises(ApiSchemaError):
        validate_presence_review(review, h0=_h0(), h1=_h1(),
                                 allowed_evidence_refs=REFS, full_frame_ref=FULL)
    result = _evaluate(_h0(), _h1(), review)
    assert result["final_a"] == result["final_b"] == _h0()
    assert result["reason_a"] == result["reason_b"] == "INVALID_REVIEW"


@pytest.mark.parametrize("identity", ["p000", "p041", "p1", "p001:ADD", "phase:1"])
def test_shape_rejects_out_of_bound_or_nonopaque_ids(identity):
    review = _review(_h0(), _h1())
    review["assessments"][0]["proposition_id"] = identity
    with pytest.raises(ApiSchemaError):
        validate_presence_review_shape(review)


def test_invalid_baseline_raises_but_optional_candidate_and_absent_review_keep():
    with pytest.raises(ApiSchemaError):
        _evaluate({**_h0(), "instrument": [True]}, _h1(), None)
    for candidate in (None, {**_h1(), "phase": [2]}, {**_h1(), "ivt": [100]}):
        result = _evaluate(_h0(), candidate, None)
        assert result["reason_a"] == "INVALID_CANDIDATE"
        assert result["final_a"] == result["final_b"] == _h0()
    assert _evaluate(_h0(), _h1(), None)["reason_a"] == "INVALID_REVIEW"
    assert _evaluate(_h0(), _h0(), None)["reason_a"] == "NO_LABEL_CHANGE"


@pytest.mark.parametrize("refs", [None, FULL, [FULL, FULL], ["crop:1"], [FULL, 7]])
def test_invalid_evidence_manifest_fails_closed(refs):
    assert _evaluate(_h0(), _h1(), _review(_h0(), _h1()), refs=refs)["final_b"] == _h0()


def test_schema_capacity_matches_legacy_difference_limit():
    assert PRESENCE_REVIEW_SCHEMA["properties"]["assessments"]["maxItems"] == 40


def test_wire_labels_produce_identical_neutral_input():
    wire = {"schema_version": "joint_perception_final_only_v1"}
    wire.update({task: {"selected_ids": labels} for task, labels in _h0().items()
                 if task != "phase"})
    wire["phase"] = {"selected_id": 1}
    assert _body(wire, _h1()) == _body(_h0(), _h1())
