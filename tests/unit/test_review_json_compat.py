import json

import pytest

from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)
from surgical_agent.research.verification.review_normalization import (
    normalize_review,
    parse_review_json,
)


def item_text(**updates):
    item = {"candidate_id": "ivt_7", "rating": 4, "finding": "MATCH",
            "scope": "LOCAL_REGION", "image_indices": [2], "observation": "Visible tool contact."}
    item.update(updates)
    return json.dumps(item)


def duplicate_field(text, key, repeated):
    return text[:-1] + ", " + json.dumps(key) + ": " + json.dumps(repeated) + "}"


@pytest.mark.parametrize("field,value", [
    ("scope", "LOCAL_REGION"), ("rating", 4), ("finding", "MATCH"),
    ("image_indices", [2]), ("observation", "Visible tool contact."),
])
@pytest.mark.parametrize("envelope", ["rows", "judgments"])
def test_only_identical_core_item_duplicates_are_recovered_with_path(field, value, envelope):
    repeated = duplicate_field(item_text(), field, value)
    text = '{"rows": [' + repeated + "]}" if envelope == "rows" else '{"judgments": {"ivt_7": ' + repeated + "}}"
    assert parse_review_json(text)[1]["error"] == "DUPLICATE_JSON_KEY"
    parsed, diag = parse_review_json_compatible(text)
    expected = json.loads(item_text())
    assert parsed == ({"rows": [expected]} if envelope == "rows" else {"judgments": {"ivt_7": expected}})
    path = "$.rows[0]." if envelope == "rows" else "$.judgments.ivt_7."
    assert diag["identical_duplicates"] == [{"path": path + field, "field": field, "occurrences": 2}]
    assert diag["error"] is None and not diag["duplicate_errors"]


@pytest.mark.parametrize("field,value", [
    ("scope", "WHOLE_FRAME"), ("rating", True), ("rating", 4.0), ("rating", "4"),
    ("finding", "REFUTED"), ("image_indices", [2.0]), ("image_indices", [0]),
    ("observation", "Different observation."),
])
def test_conflicting_core_duplicates_never_cast_a_vote(field, value):
    text = '{"rows": [' + duplicate_field(item_text(), field, value) + "]}"
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is None and diag["error"] == "DUPLICATE_JSON_KEY"
    assert diag["duplicate_errors"][0]["reason"] == "CONFLICTING_TYPE_OR_VALUE"


@pytest.mark.parametrize("field,value", [
    ("candidate_id", "ivt_7"), ("label_id", 7), ("task", "ivt"),
    ("score", 4), ("scope_notes", "LOCAL_REGION"), ("metadata", None),
])
def test_binding_and_unknown_duplicates_are_forbidden_even_when_identical(field, value):
    original = item_text(**{field: value})
    text = '{"rows": [' + duplicate_field(original, field, value) + "]}"
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is None and diag["error"] == "DUPLICATE_JSON_KEY"
    assert diag["duplicate_errors"][0]["reason"] == "FORBIDDEN_FIELD_OR_PATH"


@pytest.mark.parametrize("text", [
    '{"rows": [], "rows": []}', '{"judgments": {}, "judgments": {}}',
    '{"scope": "LOCAL_REGION", "scope": "LOCAL_REGION"}',
    '{"judgments": {"ivt_7": {}, "ivt_7": {}}}',
    '{"rows": [{"metadata": {"scope": "LOCAL_REGION", "scope": "LOCAL_REGION"}}]}',
    '{"metadata": {"rows": [{"scope": "LOCAL_REGION", "scope": "LOCAL_REGION"}]}}',
])
def test_duplicate_containers_candidate_keys_and_nested_objects_fail_closed(text):
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is None and diag["error"] == "DUPLICATE_JSON_KEY" and diag["duplicate_errors"]


@pytest.mark.parametrize("prefix,suffix,fences", [
    ("", "", []), ("", "```", ["trailing"]),
    ("```json\n", "\n```", ["leading", "trailing"]),
    ("```\n", "```", ["leading", "trailing"]),
])
def test_original_boundary_fence_contract_is_preserved(prefix, suffix, fences):
    text = prefix + '{"rows": [' + duplicate_field(item_text(), "scope", "LOCAL_REGION") + "]}" + suffix
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is not None and diag["removed_fences"] == fences and diag["error"] is None


@pytest.mark.parametrize("suffix", [" second answer", ' {"rows": []}', ",", "\n```\nexplanation"])
def test_reparsing_must_consume_whole_document_after_duplicate_was_detected(suffix):
    text = '{"rows": [' + duplicate_field(item_text(), "scope", "LOCAL_REGION") + "]}" + suffix
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is None and diag["error"] == "INVALID_JSON"


@pytest.mark.parametrize("text", [
    '{"rows": [', 'Before JSON {"rows": []}', '```json\n{"rows": []}',
    '```python\n{"rows": []}\n```', '[]', 'null', None,
])
def test_other_original_parser_errors_are_unchanged(text):
    original, old_diag = parse_review_json(text)
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is original is None
    assert diag["error"] == old_diag["error"] and diag["removed_fences"] == old_diag["removed_fences"]


def test_plain_json_and_literal_in_json_fences_are_not_rewritten():
    text = '{"rows": [' + item_text(observation="Literal ``` inside observation.") + "]}"
    parsed, diag = parse_review_json_compatible(text)
    assert parsed == json.loads(text)
    assert diag == {"error": None, "removed_fences": [], "identical_duplicates": [], "duplicate_errors": []}


def test_identical_duplicates_do_not_bypass_existing_semantic_rejection():
    state = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    pool = make_pool(state, {"instrument": [], "verb": [], "target": [], "ivt": []})
    item = item_text(rating=1, finding="REFUTED", scope="LOCAL_REGION")
    text = '{"rows": [' + duplicate_field(item, "scope", "LOCAL_REGION") + "]}"
    parsed, diag = parse_review_json_compatible(text)
    assert parsed is not None and diag["error"] is None
    normalized, semantic_diag = normalize_review(parsed, pool, seat="gemini")
    assert "ivt_7" not in normalized["judgments"]
    assert "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL" in semantic_diag["errors"]["ivt_7"]


def test_three_duplicates_all_have_to_agree():
    two = duplicate_field(item_text(), "scope", "LOCAL_REGION")
    three = duplicate_field(two, "scope", "LOCAL_REGION")
    parsed, diag = parse_review_json_compatible('{"rows": [' + three + "]}")
    assert parsed and diag["identical_duplicates"][0]["occurrences"] == 3
    conflict = duplicate_field(two, "scope", "WHOLE_FRAME")
    parsed, diag = parse_review_json_compatible('{"rows": [' + conflict + "]}")
    assert parsed is None and diag["duplicate_errors"]
