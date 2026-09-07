import hashlib
import json
from copy import deepcopy
from decimal import Decimal

import pytest

from surgical_agent.api.contracts import canonical_json_bytes
from surgical_agent.evaluation.repair_comparison import TASKS, compute_repair_comparison
from tools.audit import audit_verifier_variants as audit

VARIANTS = ("numeric1000", "names1000")


def labels(**updates):
    return {task: updates.get(task, [0]) for task in TASKS}


def row(frame=1, **updates):
    return {"video_id": "VID103", "frame_id": frame, "h0": labels(), "h1": labels(),
            "gt": labels(), "mask": {task: True for task in TASKS},
            "variants": {name: {"status": "OK", "final_a": labels(), "final_b": labels()}
                         for name in VARIANTS}, **updates}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def events_file(tmp_path, body_events):
    previous = None
    lines = []
    for index, fields in enumerate(body_events):
        event = {"sequence": index, "previous_sha256": previous, **fields}
        event["sha256"] = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        previous = event["sha256"]
        lines.append(json.dumps(event))
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reservation(run, identity, phase="development"):
    return {"event": "RESERVE", "reservation_id": identity, "run_id": str(run.resolve()),
            "key": identity, "variant": "h0", "phase": phase, "reserve_usd": "2.60"}


def evidence(run, identity, cost, *, bad_native=False, non_json=False):
    directory = run / "calls" / identity / "h0"
    result = {"key": identity, "stage": "h0", "provider_calls": 1, "request_hash": "hash",
              "cost_usd": cost, "status": "OK" if cost is not None else "FAILURE",
              "provider_request_id": identity}
    write(directory / "result.json", result)
    write(directory / "request.json", {
        "metadata": {"request_hash": "hash", "requested_model_identifier": audit.MODEL},
        "wire": {"provider": {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}},
    })
    native = {"id": identity, "provider": "Alibaba", "model": audit.MODEL,
              "usage": {"cost": .99 if bad_native else cost}}
    write(directory / "http_response.json", {"status_code": 502 if non_json else 200,
                                              "body": "upstream error" if non_json else json.dumps(native)})
    return {"event": "SETTLE" if cost is not None else "UNKNOWN", "reservation_id": identity,
            "cost_usd": str(cost), "evidence_path": str(directory / "result.json"),
            "evidence_sha256": audit.sha(directory / "result.json"), "provider_request_id": identity}


def init():
    return {"event": "INIT", "goal_id": "test", "budget_usd": "12.00", "development_budget_usd": "8.00"}


def test_partial_masks_are_not_reported_as_five_head_exactness():
    rows = [row(), row(2, gt={"instrument": [0], "phase": [0]},
                        mask={task: task in ("instrument", "phase") for task in TASKS}),
            row(3, h0=None, h1=None, variants={name: {"status": "H0_FAILURE", "final_a": None, "final_b": None}
                                             for name in VARIANTS})]
    original = deepcopy(rows)
    report = audit.compute_variant_audit(rows, VARIANTS)
    assert report["coverage"]["complete_five_head_gt_targets"] == 2
    assert report["coverage"]["valid_by_task"]["ivt"] == 2
    overall = report["all_targets"]["arms"]["names1000_final_b"]
    complete = report["complete_five_head_gt"]["arms"]["names1000_final_b"]
    assert overall["all_valid_heads_exact"]["accuracy"] == pytest.approx(2 / 3)
    assert complete["all_valid_heads_exact"]["accuracy"] == .5
    assert overall["tasks"]["ivt"]["failed_predictions"] == 1
    assert rows == original


def test_variant_edit_rates_and_candidate_common_misses():
    example = row(h0=labels(ivt=[0, 1]), h1=labels(ivt=[1, 2, 3]), gt=labels(ivt=[0, 2, 4]))
    example["variants"]["numeric1000"].update(final_a=example["h1"], final_b=example["h1"])
    example["variants"]["names1000"].update(final_a=example["h0"], final_b=labels(ivt=[0, 1, 2]))
    report = audit.compute_variant_audit([example], VARIANTS)
    numerical = report["edit_application"]["numeric1000_final_b"]["tasks"]["ivt"]
    named = report["edit_application"]["names1000_final_b"]["tasks"]["ivt"]
    assert numerical["beneficial_application_rate"] == numerical["harmful_application_rate"] == 1
    assert named["beneficial_application_rate"] == 1 and named["harmful_application_rate"] == 0
    assert report["candidate_ivt_diagnostic"]["counts"]["missed_by_both"] == 1
    with pytest.raises(ValueError, match="retain every"):
        audit.compute_variant_audit([row(variants={})], VARIANTS)


def test_native_cost_is_independently_verified_unknown_and_crash_stay_held(tmp_path):
    run = tmp_path / "run"
    events = [init(), reservation(run, "paid"), evidence(run, "paid", .04),
              reservation(run, "unknown"), evidence(run, "unknown", None, non_json=True),
              reservation(run, "crash")]
    report = audit.audit_ledger(events_file(tmp_path, events), run_dir=run)
    assert report["known_cost_usd"] == "0.04"
    assert report["total_cost_usd"] is None
    assert report["held_reserve_usd"] == "5.20"
    assert report["occupied_usd"] == "5.24"
    assert len(report["open_reservations"]) == 2
    assert sum(item["explicit_unknown"] for item in report["open_reservations"]) == 1
    assert report["current_run"]["reservations"] == 3


def test_native_cost_disagreement_rejected_even_with_valid_ledger_hash(tmp_path):
    run = tmp_path / "run"
    events = [init(), reservation(run, "paid"), evidence(run, "paid", .04, bad_native=True)]
    with pytest.raises(ValueError, match="native HTTP"):
        audit.audit_ledger(events_file(tmp_path, events))


def test_development_limit_and_hash_chain_checked_independently(tmp_path):
    events = [init(), *(reservation(tmp_path, str(index)) for index in range(4))]
    with pytest.raises(ValueError, match="reservation violates"):
        audit.audit_ledger(events_file(tmp_path, events))
    path = events_file(tmp_path, [init()])
    path.write_text(path.read_text().replace('"12.00"', '"13.00"'), encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        audit.audit_ledger(path)


def test_incomplete_run_rejected_before_any_gt_read(tmp_path, monkeypatch):
    def forbidden(_path):
        pytest.fail("no JSON should be read without completed summary")
    monkeypatch.setattr(audit, "read", forbidden)
    with pytest.raises(ValueError, match="completed summary"):
        audit.audit_run(tmp_path)


def test_completed_run_matches_saved_metrics_and_native_budget(tmp_path):
    run = tmp_path / "run"
    scored = [row()]
    predictions = [{key: value for key, value in item.items() if key not in ("gt", "mask")} for item in scored]
    write(run / "predictions.json", predictions)
    write(run / "scored_predictions.json", scored)
    book = events_file(tmp_path, [init(), reservation(run, "paid"), evidence(run, "paid", .04)])
    write(run / "calls_summary.json", [audit.read(run / "calls" / "paid" / "h0" / "result.json")])
    plan = {"source_split": "Training", "variants": list(VARIANTS), "budget_ledger": str(book)}
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    write(run / "plan.json", plan)
    comparisons = {variant: {policy: compute_repair_comparison([
        {**item, "final": item["variants"][variant][policy]} for item in scored])
        for policy in ("final_a", "final_b")} for variant in VARIANTS}
    write(run / "summary.json", {"plan_sha256": plan["plan_sha256"],
        "predictions_sha256": audit.sha(run / "predictions.json"), "targets": 1,
        "variants": list(VARIANTS), "comparisons": comparisons, "known_cost_usd": .04, "cost_usd": .04,
        "provider_calls": 1, "unpriced_calls": 0})
    result = audit.audit_run(run)
    assert result["recorded_metrics_match"] is True
    assert result["recorded_costs_match"] is True
    assert result["provider_calls_by_audit"] == 0


def summary_cost_case(amounts, stages=None):
    records = [{"key": str(i), "stage": (stages or ["one"] * len(amounts))[i], "cost_usd": amount}
               for i, amount in enumerate(amounts)]
    groups = {}
    for record in records:
        groups.setdefault(record["stage"], 0.0)
        groups[record["stage"]] += record["cost_usd"] or 0.0
    known = sum(groups.values())
    unknown = sum(amount is None for amount in amounts)
    summary = {"known_cost_usd": known, "cost_usd": None if unknown else known,
               "provider_calls": len(amounts), "unpriced_calls": unknown}
    current = {"known_cost_usd": str(sum((Decimal(str(a)) for a in amounts if a is not None), Decimal(0))),
               "reservations": len(amounts), "open_reservations": unknown}
    return summary, records, current


def test_binary_float_summation_representation_is_accepted_only_with_exact_decimal_ledger():
    summary, records, current = summary_cost_case([.1, .2])
    assert summary["known_cost_usd"] == .30000000000000004
    report = audit.audit_summary_cost_representation(summary, records, current)
    assert report["exact_known_cost_usd"] == "0.3"
    assert report["reported_minus_exact_usd"] == "4E-17"
    assert report["money_tolerance_usd"] == "0"


@pytest.mark.parametrize("field", ["known_cost_usd", "cost_usd"])
@pytest.mark.parametrize("difference", [.01, .000001, .000000000001])
def test_real_money_difference_is_never_absorbed_as_float_noise(field, difference):
    summary, records, current = summary_cost_case([.1, .2])
    summary[field] += difference
    with pytest.raises(ValueError, match="exact replay"):
        audit.audit_summary_cost_representation(summary, records, current)


def test_ledger_or_per_call_amount_change_is_rejected_even_when_summary_float_replays():
    summary, records, current = summary_cost_case([.1, .2])
    current["known_cost_usd"] = "0.300001"
    with pytest.raises(ValueError, match="exact native ledger"):
        audit.audit_summary_cost_representation(summary, records, current)


def test_frozen_stage_grouping_order_is_used_instead_of_global_call_sum():
    summary, records, current = summary_cost_case([.1] * 6, ["one", "two"] * 3)
    report = audit.audit_summary_cost_representation(summary, records, current)
    assert report["stage_order"] == ["one", "two"]
    assert report["replayed_stage_grouped_float"] == (.1 + .1 + .1) + (.1 + .1 + .1)
    assert report["call_order_float_sum"] == .6 != report["replayed_stage_grouped_float"]
    assert report["exact_known_cost_usd"] == "0.6"
    # Different real orders are replayed according to their own saved sequence,
    # never sorted by key, stage name or numerical amount.
    reversed_summary, reversed_records, reversed_current = summary_cost_case([.3, .2, .1], ["two", "one", "two"])
    assert audit.audit_summary_cost_representation(reversed_summary, reversed_records, reversed_current)["stage_order"] == ["two", "one"]


def test_unknown_cost_remains_null_and_still_occupies_exact_reservation_count():
    summary, records, current = summary_cost_case([.1, None, .2])
    result = audit.audit_summary_cost_representation(summary, records, current)
    assert result["exact_total_cost_usd"] is None and result["exact_known_cost_usd"] == "0.3"
    summary["cost_usd"] = summary["known_cost_usd"]
    with pytest.raises(ValueError, match="remain unknown"):
        audit.audit_summary_cost_representation(summary, records, current)
