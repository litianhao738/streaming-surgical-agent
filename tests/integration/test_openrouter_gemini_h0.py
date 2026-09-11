import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.run_openrouter_gemini_h0_trial import (
    PROPOSER,
    gemini_h0_wire,
    gemini_request,
)
from scripts.run_presence_review_trial import wire_body
from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.main_h0 import (
    MAIN_H0_PROMPT_VERSION,
    load_main_h0_prompt,
)


def template_request():
    return ApiRequest(
        provider="openrouter", model_identifier="qwen/qwen3.8-max-0902",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        prompt_version=MAIN_H0_PROMPT_VERSION, response_schema_version=FINAL_ONLY_SCHEMA_VERSION,
        generation_parameters={"temperature": 0, "max_output_tokens": 4096, "reasoning": {"effort": "low"}},
        payload={"openrouter_routing_profile": "strict_alibaba", "system_text": load_main_h0_prompt(),
                 "input_text": '{"causal_frame_ids":[1,26,51]}', "image_details": ["low", "low", "high"]},
        images=tuple(ApiImageInput(str(i), "image/png", b"test-image") for i in range(3)),
    )


def test_gemini_change_is_only_model_and_route_with_distinct_cache_identity():
    template = template_request()
    request = gemini_request(template)
    old, new = wire_body(template), gemini_h0_wire(request)
    assert new.pop("model") == PROPOSER
    assert old.pop("model") == "qwen/qwen3.8-max-0902"
    assert new["provider"].pop("only") == ["google-ai-studio"]
    assert old["provider"].pop("only") == ["alibaba"]
    assert new == old
    assert canonical_request_metadata(request) != canonical_request_metadata(template)
    assert request.images == template.images
    assert template.payload["openrouter_routing_profile"] == "strict_alibaba"


def test_model_route_and_generation_drift_are_rejected():
    request = gemini_request(template_request())
    with pytest.raises(ValueError):
        gemini_h0_wire(template_request())
    with pytest.raises(ValueError):
        gemini_h0_wire(replace(request, model_identifier="other/model"))
    with pytest.raises(ValueError):
        gemini_h0_wire(replace(request, endpoint_identifier="https://example.invalid/chat/completions"))
    with pytest.raises(ValueError):
        gemini_h0_wire(replace(request, generation_parameters={"max_output_tokens": 8192}))


def test_gemini_h0_dispatch_uses_openrouter_key_and_google_response(monkeypatch, tmp_path):
    from scripts import run_candidate_panel_trial as trial

    seats, sent = [], []
    def secret(seat):
        seats.append(seat)
        return SecretValue("test-secret")

    def post(url, **kwargs):
        sent.append((url, kwargs["json"]))
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {
            "model": PROPOSER, "provider": "Google AI Studio", "usage": {"cost": 0.001},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"ok": True})}}]})

    monkeypatch.setattr(trial, "key_for", secret)
    monkeypatch.setattr(trial.requests, "post", post)
    calls = trial.Calls(tmp_path, providers={PROPOSER: "Google AI Studio"}, max_calls=1)
    assert calls.call("test", "h0", "base", gemini_h0_wire(gemini_request(template_request()))) == {"ok": True}
    assert seats == ["gpt"]  # The existing OpenRouter credential slot, not Aliyun's Qwen slot.
    assert sent[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent[0][1]["provider"] == {"only": ["google-ai-studio"], "allow_fallbacks": False,
                                     "require_parameters": True}
    assert calls.rows[0]["account"] == "openrouter_usd"
