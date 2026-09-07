"""No-network checks for optional temporal reviews, fallback, and shared budget."""

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from scripts import run_temporal_memory_trial as runner
from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import ApiImageInput, thaw_json
from surgical_agent.api.credentials import SecretValue
from tests.integration.test_presence_review_trial import (
    base,
    neutral_response,
    response,
    stage_response,
)
from tests.unit.test_grounded_pipeline import _responses


def eight_images(count=8):
    original = base()
    payload = thaw_json(original.payload)
    payload["image_details"] = ["low"] * (count - 1) + ["high"]
    images = tuple(ApiImageInput(f"offline:{i}", "image/png", original.images[0].content) for i in range(count))
    return replace(original, images=images, payload=payload)


def names_case():
    original, responses = base(), _responses()

    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)

    row = shared.collect_target(original, Calls(), "target")
    raw, _, _ = shared.build_review_request(original, row)
    names = shared.variant_request(raw, "names1000")
    review = neutral_response(names, presence="PRESENT")
    first = {**runner.evaluate(names, row, review), "review": review, "status": "OK"}
    row["variants"] = {"initial_names1000": first}
    assert first["final_b"] != row["h0"]
    return row, names, first


def test_eight_image_transport_reserves_full_amount_before_sending_and_settles_native(tmp_path):
    book = shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "offline")
    original = eight_images()
    with pytest.raises(ValueError):
        shared.check_request_envelope(original)

    def sender(*args, **kwargs):
        assert Decimal(book.snapshot()["held_reserve_usd"]) == Decimal("2.60")
        wire = json.loads(kwargs["data"])
        assert len(wire["messages"][1]["content"]) == 9
        return response(cost=.02)

    calls = runner.TemporalBudgetedCalls(tmp_path / "run", 1, SecretValue("offline-secret"), book, sender=sender)
    assert calls.call("one", "repeat1000", original) is not None
    assert Decimal(book.snapshot()["known_cost_usd"]) == Decimal(".02")
    assert Decimal(book.snapshot()["held_reserve_usd"]) == 0
    assert calls.call("two", "memory1000", original) is None
    assert calls.used == 1


@pytest.mark.parametrize("bad", ["nine_images", "text", "pixel", "bytes"])
def test_other_envelope_limits_still_reject_before_any_reservation(tmp_path, bad):
    import io

    from PIL import Image

    req = eight_images(9 if bad == "nine_images" else 8)
    if bad == "text":
        payload = thaw_json(req.payload)
        payload["system_text"] = "x" * 40001
        req = replace(req, payload=payload)
    elif bad == "pixel":
        output = io.BytesIO()
        Image.new("RGB", (2049, 1)).save(output, format="PNG")
        req = replace(req, images=(ApiImageInput("large", "image/png", output.getvalue()),))
    elif bad == "bytes":
        req = replace(req, images=(ApiImageInput("large", "image/png", b"x" * (12 * 1024 * 1024 + 1)),))
    book = shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "offline")
    calls = runner.TemporalBudgetedCalls(tmp_path / "run", 1, SecretValue("offline-secret"), book,
                                         sender=lambda *a, **k: pytest.fail("must not send"))
    with pytest.raises(ValueError):
        calls.call("bad", "memory1000", req)
    assert calls.used == 0 and Decimal(book.snapshot()["occupied_usd"]) == 0


@pytest.mark.parametrize("prefill_validation", [False, True])
def test_unknown_calls_cannot_bypass_development_or_global_caps(tmp_path, prefill_validation):
    book = shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "offline")
    if prefill_validation:
        for i in range(3):
            book.reserve(f"held-{i}", run_id="another", key=str(i), variant="fixed", phase="validation")

    def sender(*args, **kwargs):
        raise requests.Timeout("synthetic timeout")

    calls = runner.TemporalBudgetedCalls(tmp_path / "run", 18, SecretValue("offline-secret"), book, sender=sender)
    for i in range(4):
        assert calls.call(str(i), "memory1000", eight_images()) is None
    assert calls.used == (1 if prefill_validation else 3)
    assert Decimal(book.snapshot()["occupied_usd"]) == (Decimal("10.40") if prefill_validation else Decimal("7.80"))
    assert calls.stopped == "SHARED_GOAL_BUDGET_STOP"
    assert len(book.snapshot()["unknown_open"]) == calls.used


@pytest.mark.parametrize("kwargs", [{"status": 401}, {"status": 402}, {"status": 403},
                                   {"model": "wrong"}, {"provider": "wrong"}])
def test_identity_and_auth_errors_halt_the_shared_goal(tmp_path, kwargs):
    book = shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "offline")
    calls = runner.TemporalBudgetedCalls(tmp_path / "one", 18, SecretValue("offline-secret"), book,
                                         sender=lambda *a, **k: response(**kwargs))
    assert calls.call("first", "memory1000", eight_images()) is None
    assert book.snapshot()["halt_reasons"]
    later = runner.TemporalBudgetedCalls(tmp_path / "two", 18, SecretValue("offline-secret"), book,
                                         sender=lambda *a, **k: pytest.fail("halt must persist"))
    assert later.call("later", "repeat1000", eight_images()) is None
    assert later.used == 0


@pytest.mark.parametrize("failure", ["none", "missing_id", "wrong_ref", "1001_characters", "bad_enum", "not_dispatched"])
def test_failed_supplement_preserves_non_h0_first_names_result(failure):
    row, req, first = names_case()
    review = neutral_response(req)
    if failure in ("none", "not_dispatched"):
        review = None
    elif failure == "missing_id":
        review["assessments"].pop()
    elif failure == "wrong_ref":
        review["assessments"][0]["evidence_refs"] = ["frame:999999"]
    elif failure == "1001_characters":
        review["assessments"][0]["observation"] = "x" * 1001
    elif failure == "bad_enum":
        review["assessments"][0]["presence"] = "SUPPORTED"
    final = runner.apply_supplement(row, req, review, dispatched=failure != "not_dispatched")
    assert final["review"] == first["review"]
    assert final["final_a"] == first["final_a"] and final["final_b"] == first["final_b"]
    assert final["final_b"] != row["h0"]
    assert final["review_source"] == "initial_names1000"
    assert final["supplemental_review"] == review
    assert final["supplemental_status"].endswith("FALLBACK")


def test_valid_second_response_replaces_whole_review_without_cherry_picking():
    row, req, first = names_case()
    second = neutral_response(req)
    second["assessments"][0]["observation"] = "x" * 1000
    final = runner.apply_supplement(row, req, second, dispatched=True)
    assert final["review"] == second and final["review"] != first["review"]
    assert final["final_b"] == row["h0"] != first["final_b"]
    assert final["supplemental_status"] == "VALID_REPLACED"


def test_history_stops_at_first_gap_even_if_older_png_exists(tmp_path):
    folder = tmp_path / "Frames"
    folder.mkdir()
    (folder / "000001.png").write_bytes(b"older exists")
    selected = {"key": "VID1_101", "video_id": "VID1", "frame_id": 101,
                "causal_frame_ids": [51, 76, 101], "images": [{"path": str(folder / "000101.png")}]}
    feasible = {"key": "VID1_101", "video_id": "VID1", "target_frame_id": 101,
                "source_split": "Training", "extra_frames": []}
    found, absent = runner.check_history(selected, feasible, {})
    assert found == [] and Path(absent).name == "000026.png"


def test_all_sixteen_outputs_persist_before_gt_and_unattempted_arm_keeps_first(tmp_path, monkeypatch):
    row, req, first = names_case()
    rows = []
    for index in range(16):
        item = {"video_id": "VID1", "frame_id": index + 1, "h0": row["h0"], "h1": row["h1"],
                "variants": {}}
        for variant in runner.VARIANTS:
            item["variants"][variant] = {**deepcopy(first), "review_source": "initial_names1000",
                "supplemental_status": "NOT_DISPATCHED_FALLBACK" if index == 0 else "NOT_TRIGGERED"}
        rows.append(item)
    plan = {"plan_sha256": "offline", "variants": list(runner.VARIANTS),
            "selection": [{"key": "VID1_1", "variant": v} for v in runner.SUPPLEMENTAL],
            "trigger_targets": 1, "dispatch_targets": 1, "fallback_policy": "first",
            "interpretation": "offline synthetic"}
    book = shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "offline")
    args = SimpleNamespace(execute=True, api_key_file=tmp_path / "unused.txt", output=tmp_path / "run",
                           max_calls=18, timeout_seconds=10)
    args.output.mkdir()
    monkeypatch.setattr(runner, "prepare", lambda _: (plan, {("VID1_1", v): req for v in runner.SUPPLEMENTAL}, rows))
    monkeypatch.setattr(runner, "assert_frozen", lambda _: None)
    monkeypatch.setattr(shared, "start_execution", lambda *_: (SecretValue("offline-secret"), book, None))
    original = runner.TemporalBudgetedCalls
    monkeypatch.setattr(runner, "TemporalBudgetedCalls", lambda *a, **k: original(
        *a, **k, sender=lambda *a, **k: response(status=401)))
    seen = []

    def score(_args, saved, variants):
        persisted = json.loads((Path(_args.output) / "predictions.json").read_text())
        assert len(persisted) == 16 and persisted == saved
        assert persisted[0]["variants"]["memory1000"]["supplemental_status"] == "NOT_DISPATCHED_FALLBACK"
        assert persisted[0]["variants"]["memory1000"]["final_b"] == first["final_b"] != row["h0"]
        seen.append(True)
        return [{**r, "gt": r["h0"], "mask": {t: True for t in r["h0"]}} for r in saved], {}

    monkeypatch.setattr(shared, "score_saved_rows", score)
    summary = runner.run(args)
    assert seen == [True] and summary["targets"] == 16 and summary["provider_calls"] == 1
    assert summary["supplemental_status_counts"]["memory1000"]["NOT_DISPATCHED_FALLBACK"] == 1
