from types import SimpleNamespace

import pytest

from scripts.run_target_interaction_diagnosis import diagnostic_body, validate_evidence
from surgical_agent.research.verification.candidate_coordinator import make_pool


def pool():
    return make_pool({"instrument": [0], "verb": [], "target": [8], "ivt": [], "phase": [0]})


def answer():
    return {"scores": {"instrument_0": 5, "target_8": 1}, "target_evidence": {"target_8": {
        "visible_score": 5, "interaction_score": 1, "instrument": "NONE", "action": "NONE",
        "contact_bbox_xyxy": [], "observation": "Visible in the background.", "uncertainty": "None claimed."}}}


def test_visible_is_not_interaction_and_box_optional_for_absence():
    assert validate_evidence(answer(), pool()) == answer()


def test_inconsistent_score_and_invalid_box_rejected():
    a = answer()
    a["scores"]["target_8"] = 5
    with pytest.raises(ValueError):
        validate_evidence(a, pool())
    a = answer()
    a["target_evidence"]["target_8"]["contact_bbox_xyxy"] = [0.8, 0.8, 0.1, 0.1]
    with pytest.raises(ValueError):
        validate_evidence(a, pool())


def test_same_images_pool_and_disabled_thinking():
    import json
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    selection = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    body = diagnostic_body("qwen", base, selection, pool())
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert packet["propositions"] == pool()["propositions"]
    assert "gt" not in packet and "previous_scores" not in packet
    assert body["enable_thinking"] is False
    assert len(body["messages"][0]["content"]) == 4


def test_single_field_removes_duplicate_and_separates_invalid_evidence():
    from scripts.run_target_interaction_single_field import analyze, single_body
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    body, schema = single_body("gpt", base, {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}, pool())
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert schema["required"] == ["target_evidence"]
    assert "scores" not in schema["properties"]
    raw = answer()
    raw.pop("scores")
    assert analyze(raw, pool(), schema)["evidence_status"] == "VALID"
    raw["target_evidence"]["target_8"]["contact_bbox_xyxy"] = [100, 100, 300, 300]
    result = analyze(raw, pool(), schema)
    assert result["numeric_status"] == "VALID"
    assert result["evidence_status"] == "ValidationError"
