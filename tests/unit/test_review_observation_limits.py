"""Versioned review-only text limits; evidence/admission semantics stay fixed."""

from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.research.verification.diff_review import (
    DIFF_REVIEW_1000_VERSION,
    DIFF_REVIEW_VERSION,
    evaluate_diff_review,
)
from surgical_agent.research.verification.final_only_grounded import (
    finalize_grounded_repair,
)
from surgical_agent.research.verification.grounded_repair import (
    PROPOSAL_VERSION,
    REVIEW_1000_VERSION,
    REVIEW_VERSION,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    PRESENCE_REVIEW_VERSION,
    evaluate_presence_review,
)
from tests.unit.test_diff_review import FULL, REFS, _h0, _h1, _review
from tests.unit.test_grounded_pipeline import _responses
from tests.unit.test_presence_review import _all_changes_review


def diff_payload(version, length):
    payload = _review(_h0(), _h1())
    payload["schema_version"] = version
    for item in payload["assessments"]:
        item["observation"] = "e" * length
    return payload


def presence_payload(version, length):
    payload = _all_changes_review(_h0(), _h1())
    payload["schema_version"] = version
    for item in payload["assessments"]:
        item["observation"] = "e" * length
    return payload


def contrast_payload(version, length):
    payload = _responses()["review"]
    payload["schema_version"] = version
    for item in payload["instances"]:
        item["distinguishing_observation"] = "e" * length
    return payload


@pytest.mark.parametrize("old,new,make", [
    (DIFF_REVIEW_VERSION, DIFF_REVIEW_1000_VERSION, diff_payload),
    (PRESENCE_REVIEW_VERSION, PRESENCE_REVIEW_1000_VERSION, presence_payload),
    (REVIEW_VERSION, REVIEW_1000_VERSION, contrast_payload),
])
def test_new_1000_boundary_and_legacy_300_contract(old, new, make):
    validator_for(new)(make(new, 1000))
    with pytest.raises(ApiSchemaError):
        validator_for(new)(make(new, 1001))
    validator_for(old)(make(old, 300))
    with pytest.raises(ApiSchemaError):
        validator_for(old)(make(old, 301))
    with pytest.raises(ApiSchemaError):
        validator_for(new)(make(old, 1000))
    with pytest.raises(ApiSchemaError):
        validator_for(old)(make(new, 300))


@pytest.mark.parametrize("old,new,collection,field", [
    (DIFF_REVIEW_VERSION, DIFF_REVIEW_1000_VERSION, "assessments", "observation"),
    (PRESENCE_REVIEW_VERSION, PRESENCE_REVIEW_1000_VERSION, "assessments", "observation"),
    (REVIEW_VERSION, REVIEW_1000_VERSION, "instances", "distinguishing_observation"),
])
def test_schema_diff_contains_only_version_and_review_text_limit(old, new, collection, field):
    expected = schema_for(old)
    expected["properties"]["schema_version"]["const"] = new
    expected["properties"][collection]["items"]["properties"][field]["maxLength"] = 1000
    assert expected == schema_for(new)


def test_proposal_observation_still_300_and_not_aliased_to_new_review_schema():
    schema = schema_for(PROPOSAL_VERSION)
    assert schema["properties"]["instances"]["items"]["properties"]["contact_observation"]["maxLength"] == 300
    payload = _responses()["proposal"]
    payload["instances"][0]["contact_observation"] = "e" * 300
    validator_for(PROPOSAL_VERSION)(payload)
    payload["instances"][0]["contact_observation"] += "e"
    with pytest.raises(ApiSchemaError):
        validator_for(PROPOSAL_VERSION)(payload)


@pytest.mark.parametrize("evaluate,version,make", [
    (evaluate_diff_review, DIFF_REVIEW_1000_VERSION, diff_payload),
    (evaluate_presence_review, PRESENCE_REVIEW_1000_VERSION, presence_payload),
])
def test_full_admission_accepts_1000_and_preserves_raw_text(evaluate, version, make):
    payload = make(version, 1000)
    untouched = deepcopy(payload)
    result = evaluate(_h0(), _h1(), payload, allowed_evidence_refs=REFS,
                      full_frame_ref=FULL, schema_version=version)
    assert result["decision_a"] == "ACCEPT" and result["final_a"] == _h1()
    assert result["decision_b"] == "ACCEPT" and result["final_b"] == _h1()
    assert payload == untouched
    invalid = make(version, 1001)
    rejected = evaluate(_h0(), _h1(), invalid, allowed_evidence_refs=REFS,
                        full_frame_ref=FULL, schema_version=version)
    assert rejected["reason_a"] == "INVALID_REVIEW" and rejected["final_a"] == _h0()


def test_contrast_admit_1000_and_original_12_nonspace_threshold():
    responses = _responses()
    def run(review):
        return finalize_grounded_repair(responses["h0"], responses["locator"], responses["proposal"],
                                         review, proposal_slot="FIRST",
                                         review_schema_version=REVIEW_1000_VERSION)
    review = contrast_payload(REVIEW_1000_VERSION, 1000)
    untouched = deepcopy(review)
    assert run(review)["decision"] == "ACCEPT" and review == untouched
    assert run(contrast_payload(REVIEW_1000_VERSION, 1001))["reason"] == "INVALID_REVIEW"
    assert run(contrast_payload(REVIEW_1000_VERSION, 11))["reason"] == "MISSING_VISUAL_OBSERVATION"
    assert run(contrast_payload(REVIEW_1000_VERSION, 12))["decision"] == "ACCEPT"
    review["instances"][0]["distinguishing_observation"] = " " * 989 + "x" * 11
    assert run(review)["reason"] == "MISSING_VISUAL_OBSERVATION"


@pytest.mark.parametrize("corruption", ["bad_enum", "unknown_ref", "missing_assessment", "missing_field", "blank_text"])
def test_long_presence_text_does_not_relax_evidence_or_structure(corruption):
    review = presence_payload(PRESENCE_REVIEW_1000_VERSION, 1000)
    item = review["assessments"][0]
    if corruption == "bad_enum":
        item["presence"] = "MAYBE"
    elif corruption == "unknown_ref":
        item["evidence_refs"] = ["frame:future"]
    elif corruption == "missing_assessment":
        review["assessments"].pop()
    elif corruption == "missing_field":
        del item["scope"]
    else:
        item["observation"] = " " * 1000
    result = evaluate_presence_review(_h0(), _h1(), review, allowed_evidence_refs=REFS,
        full_frame_ref=FULL, schema_version=PRESENCE_REVIEW_1000_VERSION)
    assert result["reason_a"] == "INVALID_REVIEW" and result["final_a"] == _h0()


def test_long_contrast_text_does_not_relax_boolean_or_instance_checks():
    responses = _responses()
    review = contrast_payload(REVIEW_1000_VERSION, 1000)
    review["instances"][0]["crop_relevant"] = False
    result = finalize_grounded_repair(responses["h0"], responses["locator"], responses["proposal"],
                                     review, proposal_slot="FIRST", review_schema_version=REVIEW_1000_VERSION)
    assert result["reason"] == "INSUFFICIENT_DISCRIMINATING_EVIDENCE"
