import json
from copy import deepcopy

import pytest

from surgical_agent.research.verification import recent_mean_panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_normalization import (
    normalize_review,
    parse_review_json,
)
from surgical_agent.research.verification.semantic_coordinator import item_error


def case():
    state = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    pool = make_pool(state, {"instrument": [], "verb": [], "target": [], "ivt": []})
    return pool, {"judgments": {p["id"]: {
        "rating": 4, "finding": "MATCH", "scope": "WHOLE_FRAME",
        "image_indices": [2], "observation": "Visible contact in the target frame."
    } for p in pool["propositions"]}}


def test_filters_observed_metadata_without_mutation_or_semantic_change():
    pool, raw = case()
    expected = deepcopy(raw)
    raw["metadata"] = {"latency": 1}
    raw["judgments"]["ivt_7"].update({
        "image_indices_string": "[2]", "scope_notes": "WHOLE_FRAME",
        "scope_correction": None, "scope_status": "WHOLE_FRAME",
        "scope_validated": True, "observation_token_count": 8, "ivt_59": None,
    })
    original, original_pool = deepcopy(raw), deepcopy(pool)
    out, diag = normalize_review(raw, pool, seat="gemini")
    assert out == expected
    assert raw == original and pool == original_pool
    assert len(diag["ignored_fields"]) == 8 and not diag["errors"]
    out["judgments"]["ivt_7"]["image_indices"].append(0)
    assert raw == original


@pytest.mark.parametrize("field,value", [
    ("score", 1), ("ratingScore", 1), ("Rating", 1), ("Scope", "LOCAL_REGION"),
    ("Finding", "REFUTED"), ("imageIndices", [0]), ("scope_correction", "LOCAL_REGION"),
    ("scope_notes", "LOCAL_REGION"), ("scope_status", "UNCERTAIN"),
    ("verdict", "REFUTED"), ("image_indices_string", "[0]"),
    ("image_indices_string", "[2.0]"), ("image_indices_string", "not JSON"),
    ("image_indices_correction", [1]), ("candidate_id", "ivt_9"),
    ("id", "ivt_9"), ("label_id", 9), ("task", "target"), ("ivt_59", {"rating": 5}),
])
def test_conflicting_alias_or_binding_rejects_only_related_item(field, value):
    pool, raw = case()
    raw["judgments"]["ivt_7"][field] = value
    out, diag = normalize_review(raw, pool, seat="gemini")
    assert "ivt_7" not in out["judgments"] and "ivt_7" in diag["errors"]
    assert len(out["judgments"]) == len(pool["propositions"]) - 1


def test_matching_aliases_do_not_replace_core_and_missing_core_is_not_filled():
    pool, raw = case()
    raw["judgments"]["ivt_7"]["score"] = 4
    out, diag = normalize_review(raw, pool, seat="gpt")
    assert out["judgments"]["ivt_7"]["rating"] == 4 and not diag["errors"]
    del raw["judgments"]["ivt_7"]["rating"]
    out, diag = normalize_review(raw, pool, seat="gpt")
    assert "ivt_7" not in out["judgments"]
    assert "MISSING_CORE_FIELDS:rating" in diag["errors"]["ivt_7"]


@pytest.mark.parametrize("field,value,error", [
    ("rating", "4", "SCHEMA_INVALID"), ("rating", True, "SCHEMA_INVALID"),
    ("image_indices", [2.0], "NON_INTEGER"), ("image_indices", [True], "SCHEMA_INVALID"),
    ("image_indices", [2, 2], "DUPLICATE_IMAGE"),
    ("image_indices", [0], "NO_CURRENT_FRAME_EVIDENCE"),
    ("scope", "UNCERTAIN", "UNCERTAIN_SUPPORT"),
    ("observation", "", "SCHEMA_INVALID"),
    ("observation", "x" * 1001, "SCHEMA_INVALID"),
    ("finding", "REFUTED", "RATING_FINDING_CONFLICT"),
])
def test_semantic_validation_remains_strict(field, value, error):
    pool, raw = case()
    raw["judgments"]["ivt_7"][field] = value
    out, diag = normalize_review(raw, pool, seat="qwen")
    assert "ivt_7" not in out["judgments"]
    assert error in diag["errors"]["ivt_7"]


def test_local_absence_cannot_be_silently_changed_to_whole_frame():
    pool, raw = case()
    raw["judgments"]["ivt_7"].update({"rating": 1, "finding": "REFUTED",
                                       "scope": "LOCAL_REGION", "scope_validated": True})
    out, diag = normalize_review(raw, pool, seat="qwen")
    assert "ivt_7" not in out["judgments"]
    assert "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL" in diag["errors"]["ivt_7"]


def test_gemini_rows_bind_by_id_not_order_and_keep_other_valid_rows():
    pool, raw = case()
    rows = [{"candidate_id": pid, **item} for pid, item in reversed(list(raw["judgments"].items()))]
    out, diag = normalize_review({"rows": rows}, pool, seat="gemini")
    assert out == raw and not diag["errors"]
    rows.append(deepcopy(rows[0]))
    duplicated = rows[0]["candidate_id"]
    out, diag = normalize_review({"rows": rows}, pool, seat="gemini")
    assert duplicated not in out["judgments"]
    assert diag["errors"][duplicated] == ["DUPLICATE_CANDIDATE_ID"]
    assert len(out["judgments"]) == len(raw["judgments"]) - 1


def test_unknown_missing_and_unbound_rows_are_recorded_never_remapped():
    pool, raw = case()
    rows = [{"candidate_id": pid, **item} for pid, item in raw["judgments"].items()]
    missing = rows[0].pop("candidate_id")
    rows.append({"candidate_id": "ivt_999", **raw["judgments"][missing]})
    out, diag = normalize_review({"rows": rows}, pool, seat="gemini")
    assert missing not in out["judgments"] and "ivt_999" not in out["judgments"]
    assert diag["errors"][missing] == ["MISSING_CANDIDATE_ID"]
    assert diag["envelope_errors"] == ["INVALID_ROW_ID:0", "UNKNOWN_CANDIDATE_ID:ivt_999"]


def test_duplicate_json_keys_and_ambiguous_envelopes_fail_closed():
    pool, raw = case()
    encoded = json.dumps(raw)
    out, diag = normalize_review(encoded, pool, seat="gpt")
    assert out == raw and not diag["errors"]
    bad = encoded.replace('"rating": 4', '"rating": 1, "rating": 4', 1)
    out, diag = normalize_review(bad, pool, seat="gpt")
    assert out == {"judgments": {}}
    assert diag["envelope_errors"] == ["DUPLICATE_JSON_KEY"]
    out, diag = normalize_review({**raw, "rows": []}, pool, seat="gpt")
    assert out == {"judgments": {}}
    assert diag["envelope_errors"] == ["MISSING_OR_AMBIGUOUS_REVIEW_ENVELOPE"]


def test_output_still_uses_existing_five_seat_evidence_rules():
    pool, raw = case()
    normalized = {}
    for seat in SEATS:
        review = deepcopy(raw)
        if seat == "gemini":
            review["judgments"]["ivt_7"]["scope_notes"] = "WHOLE_FRAME"
        normalized[seat], _ = normalize_review(review, pool, seat=seat)
    means, _ = recent_mean_panel.aggregate(normalized, pool)
    assert means["ivt_7"] == 4
    for proposition in pool["propositions"]:
        assert item_error(normalized["gemini"]["judgments"][proposition["id"]],
                          proposition["task"], 3) is None
    normalized["gpt"]["judgments"].pop("ivt_7")
    means, diag = recent_mean_panel.aggregate(normalized, pool)
    assert means["ivt_7"] is None and "gpt" in diag["ivt_7"]["invalid"]


@pytest.mark.parametrize("raw", [None, [], {}, {"rows": {}}, {"judgments": []}])
def test_invalid_envelope_cannot_supply_votes(raw):
    pool, _ = case()
    out, diag = normalize_review(raw, pool, seat="grok")
    assert out == {"judgments": {}} and diag["envelope_errors"]


def test_configuration_validation():
    pool, raw = case()
    with pytest.raises(ValueError, match="reviewer"):
        normalize_review(raw, pool, seat="other")
    with pytest.raises(ValueError, match="histories"):
        normalize_review(raw, pool, seat="gpt", image_count=True)
    pool["propositions"].append(deepcopy(pool["propositions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_review(raw, pool, seat="gpt")


@pytest.mark.parametrize("prefix,suffix,removed", [
    ("", "", []), ("", "```", ["trailing"]),
    ("", "\n```", ["trailing"]),
    ("```json\n", "\n```", ["leading", "trailing"]),
    ("```\n", "```", ["leading", "trailing"]),
])
def test_parse_review_json_removes_only_boundary_fences(prefix, suffix, removed):
    raw = {"judgments": {"ivt_7": {"observation": "A literal ``` inside JSON is unchanged."}}}
    text = " \n" + prefix + json.dumps(raw) + suffix + "\n "
    original = text
    parsed, diag = parse_review_json(text)
    assert parsed == raw and text == original
    assert diag == {"removed_fences": removed, "error": None}


@pytest.mark.parametrize("text", [
    '{"rows": [', '{"rows": [```', '{"rows": []} {"rows": []}',
    'Here is the result: {"rows": []}', '{"rows": []} completed',
    '```json\n{"rows": []}', '```python\n{"rows": []}\n```',
    '```json\n{"rows": []}\n```\nExtra explanation.',
    '```json\n{"rows": []}\n```\n```json\n{"rows": []}\n```',
    '{"rows": [], "rows": []}```', '[]', 'null', None,
])
def test_parse_review_json_rejects_incomplete_prose_multiple_objects_and_duplicates(text):
    parsed, diag = parse_review_json(text)
    assert parsed is None and diag["error"]
