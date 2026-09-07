"""Offline contract tests for the optional factored presence adapter."""

import json
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from scripts import run_verifier_variant_trial as runner
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.research.verification.factored_presence import (
    FACTORED_PRESENCE_PROMPT_VERSION,
    merge_factored_presence_reviews,
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
)
from tests.integration.test_presence_review_trial import base, stage_response
from tests.unit.test_grounded_pipeline import _responses


@pytest.fixture
def source():
    responses = _responses()

    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)

    original = base()
    row = runner.collect_target(original, Calls(), "offline-target")
    request, _, _ = runner.build_review_request(original, row)
    return runner.variant_request(request, "names1000"), row


def review_parts(request, *, presence="PRESENT"):
    body = json.loads(request.payload["input_text"])
    return [{"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": [{
        "proposition_id": proposition["proposition_id"], "presence": presence,
        "observation": "Synthetic offline evidence; this is not a model observation.",
        "evidence_refs": [body["full_frame_ref"]], "scope": "FRAME", "full_frame_reviewed": True,
    }]} for proposition in body["propositions"]]


def evaluate(source, review):
    request, row = source
    body = json.loads(request.payload["input_text"])
    return evaluate_presence_review(row["h0"], row["h1"], review,
        allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"],
        schema_version=PRESENCE_REVIEW_1000_VERSION)


def test_split_preserves_source_images_schema_parameters_and_original_opaque_ids(source):
    request, _ = source
    original_payload = thaw_json(request.payload)
    original_hash = canonical_request_metadata(request).request_hash
    body = json.loads(original_payload["input_text"])
    split = split_factored_presence_request(request)
    assert len(split) == len(body["propositions"]) > 1
    for proposition, part in zip(body["propositions"], split, strict=True):
        assert part.prompt_version == FACTORED_PRESENCE_PROMPT_VERSION
        for field in fields(request):
            if field.name not in {"payload", "prompt_version"}:
                assert getattr(part, field.name) == getattr(request, field.name)
        payload = thaw_json(part.payload)
        split_body = json.loads(payload.pop("input_text"))
        assert split_body == {**body, "propositions": [proposition]}
        assert payload == {k: v for k, v in original_payload.items() if k != "input_text"}
        assert [image.sha256 for image in part.images] == [image.sha256 for image in request.images]
    assert thaw_json(request.payload) == original_payload
    assert canonical_request_metadata(request).request_hash == original_hash
    hashes = [canonical_request_metadata(part).request_hash for part in split]
    assert len(set(hashes)) == len(hashes) and original_hash not in hashes


def test_complete_merge_has_exact_original_order_and_existing_a_b_semantics(source):
    request, _ = source
    parts = review_parts(request)
    parts[0]["assessments"][0]["observation"] = "x" * 1000
    expected = {"schema_version": PRESENCE_REVIEW_1000_VERSION,
                "assessments": [deepcopy(part["assessments"][0]) for part in parts]}
    merged = merge_factored_presence_reviews(request, parts)
    assert merged == expected
    assert evaluate(source, merged) == evaluate(source, expected)
    parts[0]["assessments"][0]["evidence_refs"].append("not-actually-supplied")
    assert merged == expected  # No mutable provider-response aliases are retained.


@pytest.mark.parametrize("problem", [
    "missing_response", "extra_response", "none", "wrong_version", "empty", "two_assessments",
    "wrong_id", "bad_ref", "duplicate_ref", "bad_presence", "overlong_observation", "bad_scope",
    "bad_bool", "missing_field", "extra_field", "wrong_order",
])
def test_any_incomplete_or_invalid_response_causes_whole_review_fallback(source, problem):
    request, row = source
    parts = review_parts(request)
    item = parts[0]["assessments"][0]
    if problem == "missing_response":
        parts.pop()
    elif problem == "extra_response":
        parts.append(deepcopy(parts[0]))
    elif problem == "none":
        parts[0] = None
    elif problem == "wrong_version":
        parts[0]["schema_version"] = "frame_label_presence_review_v2"
    elif problem == "empty":
        parts[0]["assessments"] = []
    elif problem == "two_assessments":
        parts[0]["assessments"].append(deepcopy(parts[1]["assessments"][0]))
    elif problem == "wrong_id":
        item["proposition_id"] = parts[1]["assessments"][0]["proposition_id"]
    elif problem == "bad_ref":
        item["evidence_refs"] = ["frame:future"]
    elif problem == "duplicate_ref":
        item["evidence_refs"] *= 2
    elif problem == "bad_presence":
        item["presence"] = "SUPPORTED"
    elif problem == "overlong_observation":
        item["observation"] = "x" * 1001
    elif problem == "bad_scope":
        item["scope"] = "CROP"
    elif problem == "bad_bool":
        item["full_frame_reviewed"] = "true"
    elif problem == "missing_field":
        del item["observation"]
    elif problem == "extra_field":
        item["confidence"] = .9
    elif problem == "wrong_order":
        parts.reverse()
    merged = merge_factored_presence_reviews(request, parts)
    assert merged is None
    result = evaluate(source, merged)
    assert result["final_a"] == result["final_b"] == row["h0"]


@pytest.mark.parametrize("problem", [
    "numeric", "old_schema", "already_factored", "no_propositions", "missing_task", "bad_task",
    "bool_label", "missing_name", "wrong_name", "duplicate_id", "duplicate_label", "bad_id",
    "non_object", "non_string_statement", "operation_leak", "missing_full_frame", "duplicate_ref",
])
def test_wrong_or_malformed_source_is_rejected_before_splitting_or_merging(source, problem):
    request, _ = source
    payload = thaw_json(request.payload)
    body = json.loads(payload["input_text"])
    item = body["propositions"][0]
    if problem == "numeric":
        request = replace(request, prompt_version=PRESENCE_REVIEW_1000_VERSION + "_causal_v1")
    elif problem == "old_schema":
        request = replace(request, response_schema_version="frame_label_presence_review_v2")
    elif problem == "already_factored":
        request = replace(request, prompt_version=FACTORED_PRESENCE_PROMPT_VERSION)
    elif problem == "no_propositions":
        body["propositions"] = []
    elif problem == "missing_task":
        del item["task"]
    elif problem == "bad_task":
        item["task"] = ["ivt"]
    elif problem == "bool_label":
        item["label_id"] = True
    elif problem == "missing_name":
        del item["decoded_label_name"]
    elif problem == "wrong_name":
        item["decoded_label_name"] = "not_the_label"
    elif problem == "duplicate_id":
        body["propositions"][1]["proposition_id"] = item["proposition_id"]
    elif problem == "duplicate_label":
        duplicate = deepcopy(item)
        duplicate["proposition_id"] = body["propositions"][1]["proposition_id"]
        body["propositions"][1] = duplicate
    elif problem == "bad_id":
        item["proposition_id"] = "ivt:REMOVE:59"
    elif problem == "non_object":
        body["propositions"][0] = None
    elif problem == "non_string_statement":
        item["statement"] = 17
    elif problem == "operation_leak":
        item["operation"] = "ADD"
    elif problem == "missing_full_frame":
        body["full_frame_ref"] = "frame:not_supplied"
    elif problem == "duplicate_ref":
        body["allowed_evidence_refs"] *= 2
    payload["input_text"] = json.dumps(body)
    request = replace(request, payload=payload)
    with pytest.raises((TypeError, ValueError)):
        split_factored_presence_request(request)
    with pytest.raises((TypeError, ValueError)):
        merge_factored_presence_reviews(request, [])


def test_unclear_and_local_evidence_are_left_to_existing_admission_rules(source):
    request, _ = source
    parts = review_parts(request, presence="UNCLEAR")
    for part in parts:
        item = part["assessments"][0]
        item.update(evidence_refs=["crop:1"], scope="INSTANCE", full_frame_reviewed=False)
    merged = merge_factored_presence_reviews(request, parts)
    assert merged is not None and all(item["presence"] == "UNCLEAR" for item in merged["assessments"])
    assert evaluate(source, merged)["decision_b"] == "KEEP"
