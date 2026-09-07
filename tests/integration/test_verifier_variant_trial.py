"""Offline tests for the shared goal budget and frozen verifier ablation."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest
import requests

from scripts import run_verifier_variant_trial as runner
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.request_hash import canonical_request_metadata
from tests.integration.test_presence_review_trial import base, response, stage_response
from tests.unit.test_grounded_pipeline import _responses


def ledger(tmp_path):
    return runner.GoalBudgetLedger(tmp_path / "goal.jsonl", "offline-goal")


def reserve(book, key, phase="development"):
    book.reserve(key, run_id="offline-run", key=key, variant="test", phase=phase)


def evidence(tmp_path, cost):
    path = tmp_path / "native_result.json"
    path.write_text(json.dumps({"cost_usd": cost}), encoding="utf-8")
    return path


def pricing():
    return {"response": {"data": {"id": runner.MODEL, "endpoints": [{
        "tag": "alibaba", "provider_name": "Alibaba", "status": 0,
        "context_length": 1000000, "max_prompt_tokens": 983616, "max_completion_tokens": 131072,
        "pricing": {"prompt": ".000002", "completion": ".000006",
                    "input_cache_read": ".00000025", "input_cache_write": ".0000025", "discount": 0},
    }]}}}


def test_pricing_envelope_uses_full_context_and_highest_cache_input_rate():
    _, bound = runner.checked_pricing(pricing())
    assert bound["maximum_input_rate"] == "0.0000025"
    assert Decimal(bound["calculated_upper_usd"]) == Decimal("2.524576")
    assert Decimal(bound["reserved_usd"]) == Decimal("2.60")


@pytest.mark.parametrize("change", ["price_increase", "extra_fee", "oversized_context", "missing_cache_rate"])
def test_uncovered_pricing_or_context_fails_closed(change):
    data = pricing()
    endpoint = data["response"]["data"]["endpoints"][0]
    if change == "price_increase":
        endpoint["pricing"]["input_cache_write"] = ".000003"
    elif change == "extra_fee":
        endpoint["pricing"]["image"] = ".01"
    elif change == "oversized_context":
        endpoint["context_length"] = 2000000
    else:
        del endpoint["pricing"]["input_cache_write"]
    with pytest.raises(ValueError):
        runner.checked_pricing(data)


def test_development_and_global_caps_are_separate_and_known_cost_releases_reserve(tmp_path):
    book = ledger(tmp_path)
    for key in ("a", "b", "c"):
        reserve(book, key)
    with pytest.raises(runner.GoalBudgetStop):
        reserve(book, "too_much_development")
    reserve(book, "validation_a", "validation")
    with pytest.raises(runner.GoalBudgetStop):
        reserve(book, "too_much_global", "validation")
    book.settle("a", .03, evidence_path=evidence(tmp_path, .03))
    reserve(book, "validation_b", "validation")
    state = book.snapshot()
    assert Decimal(state["occupied_usd"]) == Decimal("10.43")
    assert Decimal(state["occupied_by_phase"]["development"]) == Decimal("5.23")
    assert Decimal(state["occupied_by_phase"]["validation"]) == Decimal("5.20")


def test_unknown_and_crashed_calls_stay_reserved_across_process_lifetimes(tmp_path):
    book = ledger(tmp_path)
    reserve(book, "unknown")
    reserve(book, "crashed")
    before = book.path.read_bytes()
    book.unknown("unknown", evidence_path=evidence(tmp_path, None), reason="timeout")
    restored = runner.GoalBudgetLedger(book.path, "offline-goal")
    state = restored.snapshot()
    assert Decimal(state["held_reserve_usd"]) == Decimal("5.20")
    assert state["unknown_open"] == ["unknown"]
    assert state["open_reservations"] == ["crashed", "unknown"]
    assert book.path.read_bytes().startswith(before)
    restored.settle("unknown", .02, evidence_path=evidence(tmp_path, .02))
    assert Decimal(restored.snapshot()["occupied_usd"]) == Decimal("2.62")


def test_duplicate_dispatch_settlement_and_goal_identity_are_rejected(tmp_path):
    book = ledger(tmp_path)
    reserve(book, "once")
    with pytest.raises(runner.GoalBudgetStop):
        reserve(book, "once")
    book.settle("once", .01, evidence_path=evidence(tmp_path, .01))
    with pytest.raises(runner.GoalBudgetStop):
        book.settle("once", .01, evidence_path=evidence(tmp_path, .01))
    with pytest.raises(runner.GoalBudgetStop):
        runner.GoalBudgetLedger(book.path, "another-goal")


def test_native_cost_over_reservation_is_preserved_and_halts_every_phase(tmp_path):
    book = ledger(tmp_path)
    reserve(book, "expensive")
    book.settle("expensive", 2.7, evidence_path=evidence(tmp_path, 2.7))
    assert Decimal(book.snapshot()["known_cost_usd"]) == Decimal("2.7")
    with pytest.raises(runner.GoalBudgetStop):
        reserve(book, "next", "validation")


def test_atomic_reservations_cannot_race_past_development_cap(tmp_path):
    book = ledger(tmp_path)

    def attempt(index):
        try:
            reserve(book, str(index))
            return True
        except runner.GoalBudgetStop:
            return False

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt, range(10)))
    assert sum(results) == 3
    assert Decimal(book.snapshot()["occupied_usd"]) == Decimal("7.80")


def test_budget_ledger_tampering_is_detected(tmp_path):
    book = ledger(tmp_path)
    reserve(book, "first")
    content = book.path.read_text().replace('"reserve_usd":"2.60"', '"reserve_usd":"0.01"')
    book.path.write_text(content)
    with pytest.raises(runner.GoalBudgetStop):
        book.snapshot()


def test_paid_success_timeout_and_schema_failure_all_share_native_accounting(tmp_path):
    book = ledger(tmp_path)
    responses = [response(cost=.03), requests.Timeout("offline timeout"), response(cost=.02, payload={"bad": True})]

    def sender(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    calls = runner.BudgetedCalls(tmp_path / "run", 3, SecretValue("offline-test-key"), book, sender=sender)
    assert calls.call("a", "h0", base()) is not None
    assert calls.call("b", "h0", base()) is None
    assert calls.call("c", "h0", base()) is None
    assert calls.used == 3 and calls.stopped is None
    assert Decimal(book.snapshot()["known_cost_usd"]) == Decimal(".05")
    assert Decimal(book.snapshot()["held_reserve_usd"]) == Decimal("2.60")
    assert len(book.snapshot()["unknown_open"]) == 1


@pytest.mark.parametrize("response_args", [{"status": 401}, {"status": 402}, {"status": 403},
                                           {"model": "wrong"}, {"provider": "wrong"}])
def test_fatal_api_or_identity_error_halts_future_runners(tmp_path, response_args):
    book = ledger(tmp_path)
    calls = runner.BudgetedCalls(tmp_path / "one", 2, SecretValue("offline-test-key"), book,
                                  sender=lambda *a, **k: response(**response_args))
    assert calls.call("a", "h0", base()) is None
    another = runner.BudgetedCalls(tmp_path / "two", 2, SecretValue("offline-test-key"), book,
                                    phase="validation", sender=lambda *a, **k: pytest.fail("must not dispatch"))
    assert another.call("b", "h0", base()) is None
    assert another.used == 0 and book.snapshot()["halt_reasons"]


def test_collection_never_dispatches_old_or_new_review():
    seen, responses = [], _responses()

    class Calls:
        def call(self, key, stage, request):
            seen.append(stage)
            return stage_response(responses, stage, request)

    row = runner.collect_target(base(), Calls(), "target")
    assert seen == ["h0", "locator", "proposal"]
    assert row["h1"] is not None and row["final"] == row["h0"]
    assert row["review"] is None and row["review_execution"].startswith("SKIPPED")


def test_named_variant_only_adds_program_decoded_names_to_same_request():
    responses = _responses()
    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)
    original = base()
    row = runner.collect_target(original, Calls(), "target")
    request, _, _ = runner.build_review_request(original, row)
    numeric, names = runner.variant_request(request, "numeric1000"), runner.variant_request(request, "names1000")
    assert numeric is request
    assert numeric.images == names.images and numeric.generation_parameters == names.generation_parameters
    assert numeric.payload["system_text"] == names.payload["system_text"]
    old, new = json.loads(numeric.payload["input_text"]), json.loads(names.payload["input_text"])
    names_only = deepcopy(new)
    for proposition in names_only["propositions"]:
        proposition.pop("decoded_components", None)
        proposition.pop("decoded_label_name", None)
    assert names_only == old
    assert any("decoded_components" in p for p in new["propositions"])
    assert canonical_request_metadata(numeric).request_hash != canonical_request_metadata(names).request_hash


@pytest.mark.parametrize("split", ["Testing", "testing", "Unknown"])
def test_testing_is_always_forbidden(split):
    with pytest.raises(ValueError):
        runner.require_split(split, confirmation=True, variants=["names1000"])


def test_validation_requires_single_preselected_variant_and_explicit_confirmation():
    runner.require_split("Validation", confirmation=True, variants=["names1000"])
    with pytest.raises(ValueError):
        runner.require_split("Validation", confirmation=False, variants=["names1000"])
    with pytest.raises(ValueError):
        runner.require_split("Validation", confirmation=True, variants=list(runner.VARIANTS))
    with pytest.raises(ValueError):
        runner.require_split("Training", confirmation=True, variants=["names1000"])


def test_collection_saves_predictions_without_touching_gt_scoring(tmp_path, monkeypatch):
    output = tmp_path / "collect"
    output.mkdir()
    plan = {"source_split": "Training", "selection": [{"key": "VID103_101", "video_id": "VID103", "frame_id": 101}],
            "plan_sha256": "offline", "max_provider_calls": 3}
    responses = _responses()
    class Calls:
        stopped = None
        records = ()
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)
    monkeypatch.setattr(runner, "prepare_collect", lambda args: (object(), plan, {"VID103_101": base()}))
    monkeypatch.setattr(runner, "start_execution", lambda *args: (
        SecretValue("offline-test-key"), SimpleNamespace(snapshot=dict), Calls()))
    monkeypatch.setattr(runner, "assert_frozen", lambda plan: None)
    from scripts import run_grounded_api_pipeline
    monkeypatch.setattr(run_grounded_api_pipeline, "score_saved", lambda *a, **k: pytest.fail("GT must wait for verifier outputs"))
    summary = runner.run_collect(SimpleNamespace(max_calls=3, timeout_seconds=180, execute=True,
                                                api_key_file=tmp_path / "unused", output=output))
    assert summary["gt_scoring_performed"] is False
    assert (output / "predictions.json").exists() and not (output / "scored_predictions.json").exists()
