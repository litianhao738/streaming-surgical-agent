from copy import deepcopy

from surgical_agent.research.verification.review_diagnostic_details import (
    explain_invalid_items,
)


def test_semantic_reason_survives_normalization_without_becoming_a_vote():
    original = {"verb_0": {"scores": [None, 3, 3, 3, 3], "invalid": {"grok": "SCHEMA_INVALID"}}}
    formatting = {"grok": {"errors": {"verb_0": ["LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"]}}}
    snapshots = deepcopy((original, formatting))
    report = explain_invalid_items(original, formatting)
    assert (original, formatting) == snapshots
    assert report["original_diagnostics"] == original
    detail = report["invalid_item_details"]["verb_0"]["grok"]
    assert detail["recorded_reasons"] == ["LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"]
    assert detail["semantic_rejection"] is True
    assert detail["valid_vote"] is False


def test_missing_reason_does_not_invent_a_semantic_explanation():
    report = explain_invalid_items({"ivt_0": {"invalid": {"qwen": "SCHEMA_INVALID"}}}, {})
    detail = report["invalid_item_details"]["ivt_0"]["qwen"]
    assert detail["recorded_reasons"] == ["SCHEMA_INVALID"]
    assert detail["semantic_rejection"] is False
