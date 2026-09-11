import json
from types import SimpleNamespace

from scripts.run_second_round_delta_trial import delta_body
from surgical_agent.research.verification.candidate_coordinator import make_pool


def test_delta_request_keeps_images_ontology_but_only_outputs_changes():
    h0 = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]}
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    pool = make_pool(h0)
    history = {"pool": pool, "means": {"instrument_0": 5}, "normalized": {
        seat: {"scores": {"instrument_0": 5}, "judgments": {}, "errors": {}}
        for seat in ("grok", "qwen", "gpt", "gemini", "deepseek")}}
    body = delta_body(base, selected, h0, history)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert packet["current_prediction"] == h0 and packet["full_ontology"]
    assert set(packet["response_schema"]["properties"]) == {"changes"}
    assert packet["issues"] == []
    assert not {"gt", "ground_truth", "task_masks"} & set(packet)
    assert packet["current_image_index"] == 2
    assert [x["image_url"]["detail"] for x in body["messages"][0]["content"][1:]] == ["low", "low", "high"]
    assert body["model"] == "google/gemini-3.8-flash"
    assert body["provider"]["only"] == ["google-ai-studio"]
    assert body["reasoning"] == {"effort": "medium"}
