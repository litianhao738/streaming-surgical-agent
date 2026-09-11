"""Both branches, credential routing/accounting, and historical isolation."""
import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts import run_lightweight_parallel_repair as light
from surgical_agent.api.credentials import SecretValue
from tests.unit.test_default_compact_repair import inputs  # noqa: F401


@pytest.mark.parametrize("seat", light.MODELS)
def test_roster_reaches_both_branches_without_changing_prompt_or_selector(seat, inputs):  # noqa: F811
    selected, base, pool = inputs
    old_graph = light.compact.review_wire(seat, base, selected, pool)
    old_phase = light.compact.phase_wire(seat, selected)
    with light.lightweight_protocol():
        assert light.compact.original.MODELS == light.MODELS
        assert light.compact.original_score.verify_plan is light.verify_plan
        graph = light.compact.original.review_wire(seat, base, selected, pool)
        phase = light.compact.original.phase_wire(seat, selected)
    for old, new in ((old_graph, graph), (old_phase, phase)):
        assert new["model"] == light.MODELS[seat]
        assert new["messages"] == old["messages"]
        assert new["response_format"] == old["response_format"]
        if seat == "grok":
            assert "reasoning_effort" not in new
            assert new["provider"]["only"] == ["mistral"]
            assert new["provider"]["allow_fallbacks"] is False
        elif seat == "qwen":
            assert new["enable_thinking"] is False
        else:
            assert new == old
    assert light.compact.review_wire(seat, base, selected, pool) == old_graph
    assert light.compact.phase_wire(seat, selected) == old_phase


def test_context_restores_transport_and_runner_on_failure():
    owners = ((light.transport, ("MODELS", "key_for", "bucket")),
        (light.compact.original, ("PROFILE", "MODELS", "PROVIDERS", "RATES_V2", "TimedCalls", "phase_wire", "review_wire", "verify_plan")))
    old = [(owner, {name: getattr(owner, name) for name in names}) for owner, names in owners]
    with pytest.raises(RuntimeError), light.lightweight_protocol():
        assert light.transport.bucket("grok") == "openrouter_usd"
        raise RuntimeError("fixture")
    for owner, values in old:
        assert {name: getattr(owner, name) for name in values} == values


@pytest.mark.parametrize("reasoning_tokens", (0, 1))
def test_mistral_uses_openrouter_key_endpoint_and_native_billing(reasoning_tokens, inputs, tmp_path, monkeypatch):  # noqa: F811
    selected, _, _ = inputs
    accesses, requests = [], []

    def key_for(seat):
        accesses.append(seat)
        return SecretValue("test-credential-never-real")

    def post(url, *, json, headers, timeout):
        requests.append((url, deepcopy(json)))
        assert headers["Authorization"] == "Bearer test-credential-never-real"
        response = {"model": light.MODELS["grok"], "provider": "Mistral",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"phase_id":1,"image_indices":[2],"observation":"fixture"}'}}],
            "usage": {"cost": 0.002, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}}
        return SimpleNamespace(status_code=200, ok=True, json=lambda: response)

    monkeypatch.setattr(light, "_KEY_FOR", key_for)
    monkeypatch.setattr(light.transport.requests, "post", post)
    with light.lightweight_protocol():
        calls = light.LightweightCalls(tmp_path, limits={"openrouter_usd": Decimal(1), "xai_usd": Decimal(0), "aliyun_cny": Decimal(0)},
            rates=light.RATES, providers=light.PROVIDERS, max_calls=1, reasoning_seats=("grok", "gemini"))
        result = calls.call(selected["key"], "phase_review", "grok", light.phase_wire("grok", selected))
    assert accesses == ["gpt"]
    assert requests[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert (result is None) == bool(reasoning_tokens)
    assert calls.rows[0]["account"] == "openrouter_usd"
    assert calls.rows[0]["charge_kind"] == "native"
    assert calls.occupied["openrouter_usd"] == Decimal("0.002")
    assert calls.occupied["xai_usd"] == 0
    assert all("test-credential-never-real" not in p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))


def test_new_plan_freezes_models_transports_and_preserves_proposal(inputs, tmp_path, monkeypatch):  # noqa: F811
    selected, _, _ = inputs
    plan = {"profile": light.compact.PROFILE, "verifier_prompt_profile": light.compact.PROMPT_PROFILE,
        "source_sha256": {}, "phase_inputs": {selected["key"]: selected},
        "wire_fingerprints": {selected["key"]: {"proposal": "same-proposal", "phase": {}}}, "limitations": []}
    light.save(tmp_path / "plan.json", plan)
    monkeypatch.setattr(light.compact, "_VERIFY_PLAN", lambda p: None)
    updated = light.finalize_plan(tmp_path)
    assert updated["models"] == light.MODELS
    assert updated["reviewer_transports"] == light.TRANSPORTS
    assert updated["wire_fingerprints"][selected["key"]]["proposal"] == "same-proposal"
    with light.lightweight_protocol():
        light.verify_plan(updated)
        updated["reviewer_transports"] = {}
        with pytest.raises(ValueError, match="transport"):
            light.verify_plan(updated)


@pytest.mark.parametrize("command", ("execute", "score"))
@pytest.mark.parametrize("profile", (light.compact.PROFILE, light.compact.LEGACY_PROFILE))
def test_old_prepared_directories_keep_original_dispatch(command, profile, tmp_path, monkeypatch):
    light.save(tmp_path / "plan.json", {"profile": profile})
    observed = []
    monkeypatch.setattr(light.compact, "dispatch", lambda *args, **kwargs: observed.append(args))
    light.dispatch(command, tmp_path, None)
    assert observed[0][0] == command
    assert light.compact.original.PROFILE == light.compact.LEGACY_PROFILE


def test_selected_roster_matches_default_manifest():
    manifest = json.loads((light.ROOT / "configs/defaults/parallel-phase-repair-v1.2.0-lightweight-reviewers.json").read_text(encoding="utf-8"))
    assert manifest["version"] == light.VERSION
    assert manifest["reviewer_models"] == light.MODELS
    assert manifest["reviewer_families"] == light.FAMILIES
    assert manifest["policy"]["phase_matching_votes_required"] == 3
