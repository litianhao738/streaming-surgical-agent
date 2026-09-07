import subprocess
import sys
from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.final_only import final_only_schema
from surgical_agent.research.verification.diff_review import (
    DIFF_REVIEW_SCHEMA,
    DIFF_REVIEW_VERSION,
    MAX_CHANGE_CLAIMS,
    build_change_claims,
    build_diff_review_input,
    evaluate_diff_review,
    validate_diff_review_shape,
)

REFS = ["frame:25", "frame:50", "frame:75", "crop:1", "crop:2"]
FULL = "frame:75"


def _h0():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def _h1():
    return {"instrument": [0], "verb": [1], "target": [0], "ivt": [17], "phase": [1]}


def _review(h0, h1):
    return {"schema_version": DIFF_REVIEW_VERSION, "assessments": [
        {"change_id": claim["change_id"],
         "verdict": "SUPPORTED" if claim["operation"] == "ADD" else "CONTRADICTED",
         "observation": "Observable interaction.", "evidence_refs": [FULL],
         "scope": "FRAME", "full_frame_reviewed": True}
        for claim in build_change_claims(h0, h1)]}


def _item(review, identity):
    return next(item for item in review["assessments"] if item["change_id"] == identity)


def _evaluate(h0=None, h1=None, review=None, **kwargs):
    h0, h1 = _h0() if h0 is None else h0, _h1() if h1 is None else h1
    return evaluate_diff_review(
        h0, h1, _review(h0, h1) if review is None else review,
        allowed_evidence_refs=kwargs.get("allowed_evidence_refs", REFS),
        full_frame_ref=kwargs.get("full_frame_ref", FULL))


def test_standalone_import_does_not_require_api_import_order():
    script = "import surgical_agent.research.verification.diff_review"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_claims_are_deterministic_symmetric_differences_and_ignore_order():
    h0 = _h0()
    h0["target"] = [8, 0]
    h1 = deepcopy(h0)
    h1["target"] = [0, 8]
    assert build_change_claims(h0, h1) == []
    assert build_change_claims(_h0(), _h1()) == [
        {"change_id": "verb:REMOVE:0", "task": "verb", "label_id": 0, "operation": "REMOVE"},
        {"change_id": "verb:ADD:1", "task": "verb", "label_id": 1, "operation": "ADD"},
        {"change_id": "ivt:REMOVE:7", "task": "ivt", "label_id": 7, "operation": "REMOVE"},
        {"change_id": "ivt:ADD:17", "task": "ivt", "label_id": 17, "operation": "ADD"},
    ]


def test_schema_supports_every_final_only_difference():
    schema = final_only_schema()["properties"]
    bound = sum(2 * schema[task]["properties"]["selected_ids"]["maxItems"]
                for task in ("instrument", "verb", "target", "ivt"))
    assert MAX_CHANGE_CLAIMS >= bound
    assert DIFF_REVIEW_SCHEMA["properties"]["assessments"]["maxItems"] >= bound


def test_all_claims_approved_applies_both_and_does_not_mutate_inputs():
    h0, h1 = _h0(), _h1()
    review = _review(h0, h1)
    saved = deepcopy((h0, h1, review))
    result = _evaluate(h0, h1, review)
    assert result["decision_a"] == result["decision_b"] == "ACCEPT"
    assert result["final_a"] == result["final_b"] == h1
    assert (h0, h1, review) == saved


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invented", "extra", "wrong_id"])
def test_malformed_or_unbound_review_keeps_both(mutation):
    review = _review(_h0(), _h1())
    if mutation == "missing":
        review["assessments"].pop()
    elif mutation == "duplicate":
        review["assessments"].append(deepcopy(review["assessments"][0]))
    elif mutation == "invented":
        review["assessments"][0]["evidence_refs"] = ["frame:future"]
    elif mutation == "extra":
        review["gt"] = [7]
    else:
        review["assessments"][0]["change_id"] = "verb:REMOVE:9"
    result = _evaluate(review=review)
    assert result["reason_a"] == result["reason_b"] == "INVALID_REVIEW"
    assert result["final_a"] == result["final_b"] == _h0()


@pytest.mark.parametrize("change", [
    {"scope": "INSTANCE"}, {"full_frame_reviewed": False},
    {"evidence_refs": ["crop:1"]}, {"verdict": "INSUFFICIENT"},
    {"verdict": "SUPPORTED"}, {"evidence_refs": []},
])
def test_removal_cannot_use_one_instance_or_missing_evidence(change):
    h0, h1 = _h0(), _h0()
    h1["ivt"] = []
    review = _review(h0, h1)
    review["assessments"][0].update(change)
    result = _evaluate(h0, h1, review)
    assert result["final_a"] == result["final_b"] == h0
    assert result["decision_a"] == result["decision_b"] == "KEEP"


def test_positive_instance_evidence_can_establish_frame_presence():
    h0, h1 = _h0(), _h0()
    h1["instrument"].append(6)
    review = _review(h0, h1)
    review["assessments"][0].update(scope="INSTANCE", full_frame_reviewed=False,
                                    evidence_refs=[FULL, "crop:1"])
    assert _evaluate(h0, h1, review)["final_b"] == h1


@pytest.mark.parametrize("refs", [["frame:25"], ["frame:25", "frame:50"], ["crop:1"]])
def test_addition_requires_target_frame_reference(refs):
    h0, h1 = _h0(), _h0()
    h1["instrument"].append(6)
    review = _review(h0, h1)
    review["assessments"][0]["evidence_refs"] = refs
    result = _evaluate(h0, h1, review)
    assert result["final_a"] == result["final_b"] == h0
    assert result["claim_decisions"][0]["reason"] == "ADDITION_REQUIRES_TARGET_FRAME_EVIDENCE"


def test_b_keeps_unsupported_deletion_but_applies_supported_ivt_group():
    h0, h1 = _h0(), _h1()
    review = _review(h0, h1)
    _item(review, "ivt:REMOVE:7")["verdict"] = "INSUFFICIENT"
    result = _evaluate(h0, h1, review)
    assert result["decision_a"] == "KEEP"
    assert result["final_b"]["ivt"] == [7, 17]
    # The retained grasp IVT still requires verb 0, despite its deletion verdict.
    assert result["final_b"]["verb"] == [0, 1]
    decision = next(item for item in result["claim_decisions"]
                    if item["change_id"] == "verb:REMOVE:0")
    assert not decision["applied_b"]
    assert decision["reason_b"] == "REFERENCED_BY_REMAINING_IVT"


def test_b_cannot_add_ivt_without_approved_missing_component():
    h0, h1 = _h0(), _h1()
    review = _review(h0, h1)
    _item(review, "verb:ADD:1")["verdict"] = "INSUFFICIENT"
    _item(review, "ivt:REMOVE:7")["verdict"] = "INSUFFICIENT"
    _item(review, "verb:REMOVE:0")["verdict"] = "INSUFFICIENT"
    result = _evaluate(h0, h1, review)
    assert result["final_b"] == h0
    decision = next(item for item in result["claim_decisions"]
                    if item["change_id"] == "ivt:ADD:17")
    assert decision["reason_b"] == "IVT_COMPONENT_NOT_SUPPORTED"


def test_b_does_not_invent_component_missing_from_both_hypotheses():
    h0, h1 = _h0(), _h0()
    h1["ivt"].append(17)  # Verb 1 has no addition claim and cannot be invented.
    result = _evaluate(h0, h1)
    assert result["final_b"] == h0


def test_ivt_removal_does_not_automatically_remove_independent_heads():
    h0, h1 = _h0(), _h0()
    h0["instrument"].append(6)
    h1["instrument"].append(6)
    h1["ivt"] = []
    result = _evaluate(h0, h1)
    assert result["final_b"] == h1
    assert result["final_b"]["instrument"] == [0, 6]
    assert result["final_b"]["verb"] == [0]
    assert result["final_b"]["target"] == [0]


def test_supported_component_deletion_allowed_without_remaining_reference():
    h0, h1 = _h0(), _h0()
    h0["target"].append(8)
    result = _evaluate(h0, h1)
    assert result["final_b"] == h1


def test_over_capacity_local_union_rolls_back_without_arbitrary_subset():
    h0, h1 = _h0(), _h0()
    h0["instrument"] = [0, 1, 6]
    h1["instrument"] = [0, 2, 6]
    review = _review(h0, h1)
    _item(review, "instrument:REMOVE:1")["verdict"] = "INSUFFICIENT"
    result = _evaluate(h0, h1, review)
    assert result["final_b"] == h0
    assert result["reason_b"] == "LOCAL_REPAIR_OUTSIDE_FINAL_ONLY_CONTRACT"
    assert not any(item["applied_b"] for item in result["claim_decisions"])


@pytest.mark.parametrize("field,value", [
    ("verdict", "YES"), ("full_frame_reviewed", 1),
    ("observation", "  "), ("evidence_refs", [FULL, FULL]),
    ("change_id", "instrument:ADD:7"), ("change_id", "ivt:ADD:100"),
    ("change_id", "ivt:ADD:01"), ("change_id", "phase:ADD:1"),
])
def test_strict_shape_validation(field, value):
    review = _review(_h0(), _h1())
    review["assessments"][0][field] = value
    with pytest.raises(ApiSchemaError):
        validate_diff_review_shape(review)


def test_invalid_h0_raises_but_invalid_optional_h1_or_phase_keeps():
    h0 = _h0()
    h0["instrument"] = [True]
    with pytest.raises(ApiSchemaError):
        _evaluate(h0, _h1(), {})
    for h1 in (None, {**_h1(), "phase": [2]}, {**_h1(), "ivt": [100]}):
        result = evaluate_diff_review(_h0(), h1, {},
                                      allowed_evidence_refs=REFS, full_frame_ref=FULL)
        assert result["final_a"] == result["final_b"] == _h0()
        assert result["reason_a"] == "INVALID_CANDIDATE"


def test_no_change_needs_no_review_and_preserves_order():
    h0 = _h0()
    h0["instrument"] = [6, 0]
    h1 = deepcopy(h0)
    h1["instrument"] = [0, 6]
    result = _evaluate(h0, h1, {})
    assert result["reason_a"] == result["reason_b"] == "NO_LABEL_CHANGE"
    assert result["final_a"] == result["final_b"] == h0


def test_wire_labels_work_and_prompt_context_contains_no_gt():
    wire = {"schema_version": "joint_perception_final_only_v1"}
    wire.update({task: {"selected_ids": labels} for task, labels in _h0().items()
                 if task != "phase"})
    wire["phase"] = {"selected_id": 1}
    context = build_diff_review_input(wire, _h1(), allowed_evidence_refs=REFS,
                                      full_frame_ref=FULL)
    assert context["h0"] == _h0()
    assert set(context) == {"h0", "h1", "claims", "allowed_evidence_refs", "full_frame_ref"}
    assert _evaluate(wire, _h1())["final_a"] == _h1()


def test_missing_full_frame_or_invalid_reference_collection_fails_closed():
    for refs, full in ((["crop:1"], FULL), (FULL, FULL), ([FULL, FULL], FULL),
                       ([FULL, 7], FULL), (None, FULL)):
        result = _evaluate(allowed_evidence_refs=refs, full_frame_ref=full)
        assert result["final_a"] == result["final_b"] == _h0()
