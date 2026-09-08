import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from scripts.check_candidate_panel_providers import ACADEMIC
from scripts.panel_proposal_wire import (
    review_and_propose_wire,
    split_review_and_proposals,
)
from scripts.run_complete_gt_semantic_trial import normalize_review_wire
from scripts.run_recent_mean_panel_trial import MODELS, review_wire
from surgical_agent.perception.ontology_prompt import load_prompt_ontology_text
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import TASKS
from surgical_agent.research.verification.review_normalization import normalize_review


def inputs(*, empty=False, image_count=3):
    frames = [1, 26, 51][-image_count:]
    base = SimpleNamespace(
        images=[SimpleNamespace(mime_type="image/png", content=f"image-{i}".encode())
                for i in range(image_count)],
        payload={"image_details": ["low"] * (image_count - 1) + ["high"]})
    selected = {"frame_id": 51, "causal_frame_ids": frames, "h0": "MUST_NOT_LEAK",
                "gt": "MUST_NOT_LEAK", "other_reviews": "MUST_NOT_LEAK"}
    pool = make_pool({"instrument": [], "verb": [], "target": [],
                      "ivt": [] if empty else [7], "phase": [0]})
    return base, selected, pool


def review_response(seat, pool, *, proposals=None):
    item = {"rating": 4, "finding": "MATCH", "scope": "LOCAL_REGION",
            "image_indices": [2], "observation": "The target image supports the candidate."}
    if seat == "gemini":
        raw = {"rows": [{"candidate_id": p["id"], **item} for p in pool["propositions"]]}
    else:
        raw = {"judgments": {p["id"]: dict(item) for p in pool["propositions"]}}
    raw["proposals"] = {"ivt": [0, 1]} if proposals is None else proposals
    return raw


@pytest.mark.parametrize("seat", SEATS)
@pytest.mark.parametrize("empty", [False, True])
def test_wire_preserves_provider_images_and_updates_both_schema_copies(seat, empty):
    base, selected, pool = inputs(empty=empty)
    old_body = review_wire(seat, base, selected, pool)
    old_packet = json.loads(old_body["messages"][0]["content"][0]["text"])
    before = deepcopy(pool)
    body = review_and_propose_wire(seat, base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context" and packet["academic_context"] == ACADEMIC
    assert body["model"] == MODELS[seat]
    assert body["messages"][0]["content"][1:] == old_body["messages"][0]["content"][1:]
    assert len(body["messages"][0]["content"]) == 4
    for key in ("provider", "reasoning", "reasoning_effort", "enable_thinking", "temperature", "max_tokens"):
        assert body.get(key) == old_body.get(key)
    assert packet["images"] == old_packet["images"]
    assert packet["proposition_semantics"] == old_packet["proposition_semantics"]
    assert packet["output_contract_clarification"] == old_packet["output_contract_clarification"]
    assert "MUST_NOT_LEAK" not in json.dumps(body)
    assert not {"h0", "gt", "ground_truth", "current_prediction", "other_reviews"} & set(packet)
    assert packet["ontology"] == load_prompt_ontology_text()
    assert "99=(" in packet["ontology"]
    assert "Use English" in packet["proposal_instructions"]
    assert "later round" in packet["proposal_instructions"]
    assert "own score cannot accept them in this round" in packet["proposal_instructions"]
    assert "Return fewer or []" in packet["proposal_instructions"]
    assert "Return judgments only" not in packet["instructions"]
    assert "Return rows only" not in packet["instructions"]
    assert pool == before
    schema = packet["response_schema"]
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["rows" if seat == "gemini" else "judgments", "proposals"]
    if body["response_format"]["type"] == "json_schema":
        assert body["response_format"]["json_schema"]["schema"] == schema
        assert body["response_format"]["json_schema"]["strict"] is True
    else:
        assert body["response_format"] == {"type": "json_object"}
    assert Draft202012Validator(schema).is_valid(review_response(seat, pool))
    invalid = review_response(seat, pool, proposals={"ivt": [0, 1, 2]})
    assert not Draft202012Validator(schema).is_valid(invalid)


@pytest.mark.parametrize("seat", SEATS)
@pytest.mark.parametrize("image_count", [1, 2])
def test_short_real_history_is_preserved(seat, image_count):
    base, selected, pool = inputs(image_count=image_count)
    body = review_and_propose_wire(seat, base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert len(body["messages"][0]["content"]) == image_count + 1
    assert packet["output_contract_clarification"]["current_image_index"] == image_count - 1


@pytest.mark.parametrize("frames", [[1, 26, 76], [1, 26, 26], [51, 26, 1], [1, 25, 51], [True, 26, 51]])
def test_future_duplicate_or_noncanonical_frames_are_rejected(frames):
    base, selected, pool = inputs()
    selected["causal_frame_ids"] = frames
    with pytest.raises(ValueError, match="causal"):
        review_and_propose_wire("gpt", base, selected, pool)


def test_pool_is_rebuilt_without_origin_or_gt_metadata():
    base, selected, pool = inputs()
    pool["h0"] = "MUST_NOT_LEAK"
    for proposition in pool["propositions"]:
        proposition.update(origin="MUST_NOT_LEAK", gt="MUST_NOT_LEAK", name="MUST_NOT_LEAK")
    body = review_and_propose_wire("gpt", base, selected, pool)
    assert "MUST_NOT_LEAK" not in json.dumps(body)
    assert pool["propositions"][0]["origin"] == "MUST_NOT_LEAK"


@pytest.mark.parametrize("seat", SEATS)
def test_split_preserves_channel_and_candidates_without_accepting_proposals(seat):
    _, _, pool = inputs()
    raw = review_response(seat, pool)
    before = deepcopy(raw)
    review_raw, proposals, diagnostics = split_review_and_proposals(raw, seat, pool)
    assert set(review_raw) == {"rows" if seat == "gemini" else "judgments"}
    normalized = normalize_review_wire(seat, review_raw, pool)
    assert set(normalized["judgments"]) == {p["id"] for p in pool["propositions"]}
    assert "ivt_0" not in normalized["judgments"]
    assert proposals == {"instrument": [], "verb": [], "target": [], "ivt": [0, 1]}
    assert diagnostics["proposal_status"] == "VALID"
    assert diagnostics["review_status"] == "EXTRACTED"
    assert diagnostics["proposals_require_next_round_review"] is True
    assert diagnostics["review_requires_separate_validation"] is True
    assert raw == before
    proposals["ivt"].append(2)
    assert raw == before


@pytest.mark.parametrize("seat", ["gpt", "gemini"])
@pytest.mark.parametrize("proposal,error", [
    ({"ivt": [0, 0]}, "DUPLICATE_PROPOSAL_ID"),
    ({"ivt": [7]}, "PROPOSAL_ALREADY_IN_POOL"),
    ({"ivt": [0, 1, 2]}, "TOO_MANY_PROPOSALS"),
    ({"ivt": [100]}, "INVALID_PROPOSAL_ID"),
    ({"ivt": [-1]}, "INVALID_PROPOSAL_ID"),
    ({"ivt": [True]}, "INVALID_PROPOSAL_ID"),
    ({"ivt": [1.0]}, "INVALID_PROPOSAL_ID"),
    ({"ivt": ["1"]}, "INVALID_PROPOSAL_ID"),
    ({"ivt": "0"}, "PROPOSAL_IDS_NOT_LIST"),
    ({"ivt": [0], "scores": [5]}, "INVALID_PROPOSAL_FIELDS"),
    ({"target": [0]}, "INVALID_PROPOSAL_FIELDS"),
    ([], "INVALID_PROPOSAL_FIELDS"),
])
def test_bad_proposals_are_rejected_without_erasing_valid_review(seat, proposal, error):
    _, _, pool = inputs()
    raw = review_response(seat, pool, proposals=proposal)
    review_raw, proposed, diagnostics = split_review_and_proposals(raw, seat, pool)
    review_key = "rows" if seat == "gemini" else "judgments"
    assert review_raw == {review_key: raw[review_key]}
    assert proposed == {task: [] for task in TASKS}
    assert diagnostics["proposal_status"] == "INVALID"
    assert diagnostics["proposal_errors"] == [error]
    assert diagnostics["review_status"] == "EXTRACTED"


@pytest.mark.parametrize("seat", SEATS)
def test_missing_proposal_is_distinct_from_explicit_empty_proposal(seat):
    _, _, pool = inputs(empty=True)
    raw = review_response(seat, pool, proposals={"ivt": []})
    valid_review, explicit_empty, valid_diag = split_review_and_proposals(raw, seat, pool)
    del raw["proposals"]
    missing_review, missing_empty, missing_diag = split_review_and_proposals(raw, seat, pool)
    assert valid_review == missing_review
    assert explicit_empty == missing_empty == {task: [] for task in TASKS}
    assert valid_diag["proposal_status"] == "VALID"
    assert missing_diag["proposal_status"] == "MISSING"
    assert missing_diag["proposal_errors"] == ["MISSING_PROPOSALS"]


@pytest.mark.parametrize("raw", [None, [], "not a JSON object", 1])
def test_malformed_response_cannot_claim_success(raw):
    _, _, pool = inputs()
    review, proposals, diagnostics = split_review_and_proposals(raw, "gpt", pool)
    assert review is None
    assert proposals == {task: [] for task in TASKS}
    assert diagnostics["review_status"] == diagnostics["proposal_status"] == "MALFORMED_RESPONSE"


def test_proposal_self_scores_are_not_silently_made_part_of_the_review_pool():
    _, _, pool = inputs()
    raw = review_response("gemini", pool)
    raw["rows"].append({**raw["rows"][0], "candidate_id": "ivt_0"})
    review, proposals, diagnostics = split_review_and_proposals(raw, "gemini", pool)
    assert proposals["ivt"] == [0, 1]
    assert diagnostics["review_requires_separate_validation"] is True
    assert normalize_review_wire("gemini", review, pool) == {"wire_error": "UNKNOWN_OR_DUPLICATE_ROW_ID"}


def test_proposal_extraction_does_not_certify_or_repair_a_malformed_review():
    _, _, pool = inputs()
    raw = {"judgments": [], "proposals": {"ivt": [0]}, "explanation": "extra"}
    review, proposals, diagnostics = split_review_and_proposals(raw, "gpt", pool)
    assert review == {"judgments": []}
    assert diagnostics["review_status"] == "MALFORMED_REVIEW_CONTAINER"
    assert diagnostics["ignored_top_level_fields"] == ["explanation"]
    assert proposals["ivt"] == [0]


@pytest.mark.parametrize("seat", ["unknown", "base", None])
def test_unknown_seat_is_rejected(seat):
    base, selected, pool = inputs()
    with pytest.raises(ValueError, match="seat"):
        review_and_propose_wire(seat, base, selected, pool)
    with pytest.raises(ValueError, match="seat"):
        split_review_and_proposals({}, seat, pool)


def test_candidate_id_cannot_be_misbound_or_duplicated():
    base, selected, pool = inputs()
    pool["propositions"][0]["id"] = "ivt_99"
    with pytest.raises(ValueError, match="bound"):
        review_and_propose_wire("gpt", base, selected, pool)
    base, selected, pool = inputs()
    pool["propositions"].append(deepcopy(pool["propositions"][0]))
    with pytest.raises(ValueError, match="unique"):
        split_review_and_proposals({}, "gpt", pool)


def test_gpt_omits_unsupported_uniqueitems_from_both_schemas_but_rejects_duplicates_locally():
    base, selected, pool = inputs()
    body = review_and_propose_wire("gpt", base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    transport_schema = body["response_format"]["json_schema"]["schema"]
    assert packet["response_schema"] == transport_schema
    assert "uniqueItems" not in json.dumps(transport_schema)
    assert "distinct IVT IDs" in packet["proposal_instructions"]
    raw = review_response("gpt", pool, proposals={"ivt": [0, 0]})
    # The provider cannot enforce uniqueness; local code still enforces it.
    assert Draft202012Validator(transport_schema).is_valid(raw)
    review_raw, proposals, diagnostics = split_review_and_proposals(raw, "gpt", pool)
    assert review_raw == {"judgments": raw["judgments"]}
    assert proposals == {task: [] for task in TASKS}
    assert diagnostics["proposal_status"] == "INVALID"
    assert diagnostics["proposal_errors"] == ["DUPLICATE_PROPOSAL_ID"]


@pytest.mark.parametrize("seat,container", [("qwen", "rows"), ("gemini", "judgments")])
def test_single_valid_envelope_is_recoverable_across_provider_container_variants(seat, container):
    _, _, pool = inputs()
    raw = review_response("gemini" if container == "rows" else "gpt", pool)
    original = deepcopy(raw)
    review_raw, proposals, diagnostics = split_review_and_proposals(raw, seat, pool)
    assert review_raw == {container: raw[container]}
    assert diagnostics["review_status"] == "EXTRACTED"
    assert diagnostics["proposal_status"] == "VALID"
    assert proposals["ivt"] == [0, 1]
    normalized, validation = normalize_review(review_raw, pool, seat=seat)
    assert set(normalized["judgments"]) == {p["id"] for p in pool["propositions"]}
    assert validation["errors"] == {} and validation["envelope_errors"] == []
    assert raw == original


@pytest.mark.parametrize("seat", SEATS)
def test_conflicting_review_envelopes_are_preserved_and_rejected_by_normalizer(seat):
    _, _, pool = inputs()
    raw = review_response("gpt", pool)
    raw["rows"] = [{**row, "rating": 1, "finding": "REFUTED", "scope": "WHOLE_FRAME"}
                   for row in review_response("gemini", pool)["rows"]]
    original = deepcopy(raw)
    review_raw, proposals, diagnostics = split_review_and_proposals(raw, seat, pool)
    assert set(review_raw) == {"judgments", "rows"}
    assert diagnostics["review_status"] == "AMBIGUOUS_REVIEW_CONTAINERS"
    assert diagnostics["ignored_top_level_fields"] == []
    assert proposals["ivt"] == [0, 1] and diagnostics["proposal_status"] == "VALID"
    normalized, validation = normalize_review(review_raw, pool, seat=seat)
    assert normalized == {"judgments": {}}
    assert validation["envelope_errors"] == ["MISSING_OR_AMBIGUOUS_REVIEW_ENVELOPE"]
    assert set(validation["errors"]) == {p["id"] for p in pool["propositions"]}
    assert raw == original
