import json
from types import SimpleNamespace

from scripts.check_candidate_panel_providers import ACADEMIC, body_for, redact_images
from surgical_agent.research.verification.candidate_coordinator import make_pool


def fixture():
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test-image")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    selection = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    pool = make_pool({"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]})
    return base, selection, pool


def test_academic_context_first_causal_images_and_no_h0_or_gt_labels():
    for seat in ("grok", "qwen", "gpt", "gemini", "deepseek"):
        body = body_for(seat, *fixture())
        assert body["max_tokens"] == 512
        content = body["messages"][0]["content"]
        packet = json.loads(content[0]["text"])
        assert next(iter(packet)) == "academic_context" and packet["academic_context"] == ACADEMIC
        assert "h0" not in packet and "current_prediction" not in packet and "gt" not in packet
        assert [x["seconds_relative_to_target"] for x in packet["images"]] == [-2, -1, 0]
        assert [x["image_url"]["detail"] for x in content[1:]] == ["low", "low", "high"]
        safe = redact_images(body)
        assert "data:image" not in json.dumps(safe)
        assert content[1]["image_url"]["url"].startswith("data:image")


def test_non_thinking_modes_and_only_gemini_wire_format_differs():
    assert body_for("qwen", *fixture())["enable_thinking"] is False
    assert "reasoning" not in body_for("gpt", *fixture())
    a = body_for("gemini", *fixture())
    b = body_for("gemini", *fixture(), gemini_json_object=True)
    assert a["reasoning"] == {"enabled": False}
    assert a["provider"]["allow_fallbacks"] is False
    assert a.pop("response_format")["type"] == "json_schema"
    assert b.pop("response_format") == {"type": "json_object"}
    assert a == b
    ds = body_for("deepseek", *fixture())
    assert ds["model"] == "deepseek/deepseek-v4-flash-vision-exp"
    assert ds["provider"]["only"] == ["fireworks"]
    assert ds["reasoning"] == {"enabled": False}
    assert "enable_thinking" not in ds


def test_failed_json_keeps_real_bill_and_stops_after_inference(monkeypatch, tmp_path):
    from scripts import run_candidate_panel_trial as trial
    from surgical_agent.api.credentials import SecretValue
    monkeypatch.setattr(trial, "key_for", lambda s: SecretValue("test-secret"))
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {
            "model": "openai/gpt-4.1-mini", "provider": "OpenAI", "usage": {"cost": 0.001},
            "choices": [{"finish_reason": "stop", "message": {"content": "bad json"}}]})
    monkeypatch.setattr(trial.requests, "post", post)
    calls = trial.Calls(tmp_path)
    body = body_for("gpt", *fixture())
    assert calls.call("test", "review_1", "gpt", body) is None
    assert calls.rows[0]["charge"] == "0.001" and calls.rows[0]["charge_kind"] == "native"
    assert calls.rows[0]["status"] == "FAILED"
    calls.stopped = True
    assert calls.call("test", "review_2", "gpt", body) is None
    assert len(sent) == 1


def test_transport_failure_reserves_cost_no_retry(monkeypatch, tmp_path):
    from scripts import run_candidate_panel_trial as trial
    from surgical_agent.api.credentials import SecretValue
    monkeypatch.setattr(trial, "key_for", lambda s: SecretValue("test-secret"))
    sent = []
    def post(*args, **kwargs):
        sent.append(1)
        raise trial.requests.Timeout()
    monkeypatch.setattr(trial.requests, "post", post)
    calls = trial.Calls(tmp_path)
    assert calls.call("test", "review_1", "gpt", body_for("gpt", *fixture())) is None
    assert calls.rows[0]["charge"] == calls.rows[0]["reserve"]
    assert calls.rows[0]["charge_kind"] == "unknown_reserved"
    assert len(sent) == 1


def test_proposer_has_full_ontology_and_cannot_emit_patch():
    from scripts.run_candidate_panel_trial import BASE_MODEL, proposal_body
    from surgical_agent.research.verification.candidate_coordinator import (
        proposal_schema,
    )
    base, selected, pool = fixture()
    h0 = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]}
    body = proposal_body(base, selected, h0, pool, [])
    assert body["model"] == BASE_MODEL and body["reasoning"] == {"effort": "low"}
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert packet["full_ontology"]
    expected = proposal_schema()
    for field in expected["properties"].values():
        field.pop("uniqueItems")
    assert body["response_format"]["json_schema"]["schema"] == expected
    assert all(f["uniqueItems"] for f in proposal_schema()["properties"].values())
    assert "edits" not in proposal_schema()["properties"]


def test_semantic_contract_and_gemini_proposer_are_separate_from_h0():
    from scripts.run_semantic_candidate_trial import gemini_proposal, semantic_body
    base, selected, pool = fixture()
    for seat in ("grok", "qwen", "gpt", "gemini", "deepseek"):
        body = semantic_body(seat, base, selected, pool)
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        assert "current_prediction" not in packet and "gt" not in packet
        assert "judgments" in packet["response_schema"]["properties"]
        assert "scores" not in packet["response_schema"]["properties"]
    proposal = gemini_proposal(base, selected, {}, pool, [])
    assert proposal["model"] == "google/gemini-3.8-flash"
    assert proposal["reasoning"] == {"effort": "low"}
    assert proposal["provider"]["only"] == ["google-ai-studio"]


def test_redaction_preserves_system_message_and_covers_all_images():
    body = body_for("gpt", *fixture())
    body["messages"].insert(0, {"role": "system", "content": "Frozen H0 prompt"})
    safe = redact_images(body)
    assert safe["messages"][0] == body["messages"][0]
    assert "data:image" not in json.dumps(safe)
    assert "data:image" in json.dumps(body)


def test_wire_clarification_preserves_pool_images_schema_and_parameters():
    from scripts.replay_semantic_review import review_body_v2
    from scripts.run_semantic_candidate_trial import semantic_body
    for seat in ("grok", "qwen", "gpt", "gemini", "deepseek"):
        before, after = semantic_body(seat, *fixture()), review_body_v2(seat, *fixture())
        a = json.loads(before["messages"][0]["content"][0]["text"])
        b = json.loads(after["messages"][0]["content"][0]["text"])
        assert b.pop("output_contract_clarification")["current_image_index"] == 2
        assert a == b
        before["messages"][0]["content"][0]["text"] = "packet"
        after["messages"][0]["content"][0]["text"] = "packet"
        assert before == after


def test_gemini_uniform_rows_preserve_evidence_contract_and_images():
    from scripts.replay_semantic_review import review_body_v2
    from scripts.run_complete_gt_semantic_trial import review_body_v3
    old = review_body_v2("gemini", *fixture())
    new = review_body_v3("gemini", *fixture())
    a = json.loads(old["messages"][0]["content"][0]["text"])
    b = json.loads(new["messages"][0]["content"][0]["text"])
    expected = a["response_schema"]["properties"]["judgments"]["properties"]["instrument_0"]
    item = b["response_schema"]["properties"]["rows"]["items"]
    assert item["properties"].pop("candidate_id")["enum"] == ["instrument_0"]
    item["required"].remove("candidate_id")
    assert item == expected and new["response_format"] == {"type": "json_object"}
    assert old["messages"][0]["content"][1:] == new["messages"][0]["content"][1:]


def test_gemini_row_mapping_rejects_missing_unknown_and_duplicate_ids():
    from scripts.run_complete_gt_semantic_trial import normalize_review_wire
    _, _, pool = fixture()
    row = {"candidate_id": "instrument_0", "rating": 4, "finding": "MATCH",
           "scope": "LOCAL_REGION", "image_indices": [2], "observation": "Grasper at left."}
    good = normalize_review_wire("gemini", {"rows": [row]}, pool)
    assert good["judgments"]["instrument_0"]["rating"] == 4
    assert "candidate_id" not in good["judgments"]["instrument_0"]
    for bad in ({"rows": []}, {"rows": [row, row]}, {"rows": [{**row, "candidate_id": "target_8"}]}):
        assert "wire_error" in normalize_review_wire("gemini", bad, pool)


def test_blind_proposer_gets_no_h0_pool_or_previous_judge_opinions():
    from scripts.run_blind_candidate_replay import blind_proposal_body
    base, selected, _ = fixture()
    body = blind_proposal_body(base, selected)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert not {"current_prediction", "candidate_pool", "issues", "gt"} & set(packet)
    assert packet["full_ontology"] and next(iter(packet)) == "academic_context"
    assert body["model"] == "google/gemini-3.8-flash" and body["reasoning"] == {"effort": "low"}
    assert [p["image_url"]["detail"] for p in body["messages"][0]["content"][1:]] == ["low", "low", "high"]
