import json
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run_prior_panel_trial import (
    DirectBudget,
    merge_dispatch_history,
    provider_schema,
    request_body,
    was_dispatched,
)
from surgical_agent.api.credentials import SecretValue


def test_reserve_whole_panel_and_persist_unknown_cost(tmp_path):
    plan = {"budget_usd": "15.00", "max_calls": 188, "models": [
        {"model": "m", "reserve_usd": "6.00"}]}
    ledger = DirectBudget(tmp_path, plan, SecretValue("synthetic-nonsecret"))
    assert not ledger.reserve([("a", "m"), ("b", "m"), ("c", "m")])
    assert not ledger.state["calls"]


def test_reopened_reservation_and_unsent_release(tmp_path):
    plan = {"budget_usd": "15.00", "max_calls": 188, "models": [
        {"model": "m", "reserve_usd": "2.00"}]}
    ledger = DirectBudget(tmp_path, plan, SecretValue("synthetic-nonsecret"))
    assert ledger.reserve([("patch", "m"), ("j1", "m"), ("j2", "m"), ("j3", "m")])
    restored = DirectBudget(tmp_path, plan, SecretValue("synthetic-nonsecret"))
    assert sum(Decimal(r["reserve_usd"]) for r in restored.state["calls"].values()) == 8
    restored.state["calls"]["patch"]["state"] = "DISPATCHED"
    restored.release_unsent()
    assert restored.state["calls"]["patch"]["cost_usd"] is None
    assert restored.state["calls"]["j1"]["cost_usd"] == "0"


def test_google_wire_simplification_keeps_full_local_schema():
    original = {"type": "object", "required": ["x"], "additionalProperties": False,
                "properties": {"x": {"type": "string", "maxLength": 1000, "enum": ["a", "b"]}}}
    wire = provider_schema(original, "google-vertex/eu")
    assert "maxLength" not in wire["properties"]["x"]
    assert original["properties"]["x"]["maxLength"] == 1000
    assert wire["properties"]["x"]["enum"] == ["a", "b"]
    assert wire["required"] == ["x"] and wire["additionalProperties"] is False
    assert provider_schema(original, "openai") == original


def test_actual_request_requires_whole_universe_without_mutating_schema():
    schema = {"type": "object", "properties": {"schema_version": {"const": "test"},
              "assessments": {"type": "array", "minItems": 1, "maxItems": 48}}}
    base = SimpleNamespace(images=[], payload={"image_details": []})
    body = request_body(base, {"tag": "openai", "model": "m", "max_output": 16}, schema,
                        "Test", {"propositions": [{}, {}, {}]})
    spec = body["response_format"]["json_schema"]["schema"]["properties"]["assessments"]
    assert spec["minItems"] == spec["maxItems"] == 3
    assert schema["properties"]["assessments"]["minItems"] == 1


def test_parallel_settlements_preserve_all_costs(tmp_path):
    barrier = Barrier(3)
    native = {"model": "m", "provider": "p", "id": "synthetic",
              "usage": {"cost": 0.1}, "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}

    def sender(*args, **kwargs):
        barrier.wait(timeout=5)
        return SimpleNamespace(status_code=200, text=json.dumps(native), json=lambda: native)

    model = {"model": "m", "tag": "p", "provider_name": "p", "max_output": 16, "reserve_usd": "0.20"}
    plan = {"budget_usd": "1.00", "max_calls": 3, "models": [model]}
    ledger = DirectBudget(tmp_path, plan, SecretValue("synthetic-nonsecret"), sender=sender)
    assert ledger.reserve([(f"j{i}", "m") for i in range(3)])
    body = {"model": "m", "max_tokens": 16, "provider": {"only": ["p"], "allow_fallbacks": False,
             "require_parameters": True}, "messages": [{"content": "test"}, {"content": [{"type": "text", "text": "test"}]}]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: ledger.call(f"j{i}", body, {"type": "object"}), range(3)))
    assert results == [{"ok": True}] * 3
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert all(r["state"] == "OK" for r in saved["calls"].values())
    assert sum(Decimal(r["cost_usd"]) for r in saved["calls"].values()) == Decimal("0.3")


def test_continuation_never_retries_attempted_failed_panel():
    calls = {"x/prior/r1_judge_0": {"state": "FAILED"},
             "x/no_prior/r1_judge_0": {"state": "NOT_SENT"},
             "y/h0": {"state": "RESERVED"}, "x/h0": {"state": "OK"}}
    assert was_dispatched(calls, "x/prior")
    assert was_dispatched(calls, "x/h0")
    assert not was_dispatched(calls, "x/no_prior")
    assert not was_dispatched(calls, "y/h0")


def test_transitive_continuation_retains_cached_h0_and_both_failed_arms():
    first = {"x/h0": {"state": "OK"}, "x/prior/r1_judge_0": {"state": "FAILED"}}
    second = {"x/no_prior/r1_judge_0": {"state": "FAILED"}}
    combined = merge_dispatch_history([first, second])
    assert was_dispatched(combined, "x/h0")
    assert was_dispatched(combined, "x/prior")
    assert was_dispatched(combined, "x/no_prior")
    assert not was_dispatched(combined, "y/h0")
