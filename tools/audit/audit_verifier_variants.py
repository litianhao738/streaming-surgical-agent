"""Independent post-run masked metrics and native-cost audit for verifier variants.

Never dispatch requests. A completed summary and its durable prediction hash
are required before any scored/GT rows are read. Overall exactness concerns all
scorable heads; only complete_five_head_gt represents five-head exactness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.api.contracts import canonical_json_bytes
from surgical_agent.evaluation.repair_comparison import TASKS, compute_repair_comparison
from tools.audit.audit_presence_trial import (
    _assert_counts,
    _candidate_diagnostic,
    _edit_admission,
)

MODEL = "qwen/qwen3.8-max-0902"
RESERVE = Decimal("2.60")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cohort(rows, arms):
    comparisons = {
        arm: compute_repair_comparison([{**row, "final": row[arm]} for row in rows])
        for arm in arms
    }
    reference = next(iter(comparisons.values()))
    metrics = {"h0": reference["arms"]["h0"], "h1_policy": reference["arms"]["h1_policy"]}
    metrics.update({arm: report["arms"]["final"] for arm, report in comparisons.items()})
    _assert_counts(rows, metrics)
    return {
        "targets": len(rows), "arms": metrics,
        "h1_policy_change_vs_h0": reference["paired"]["h0_to_h1_policy"],
        "changes_vs_h0": {arm: report["paired"]["h0_to_final"] for arm, report in comparisons.items()},
        "change_details": {
            arm: [{"video_id": detail["video_id"], "frame_id": detail["frame_id"],
                   **detail["pairs"]["h0_to_final"]} for detail in report["details"]]
            for arm, report in comparisons.items()
        },
    }


def compute_variant_audit(rows, variants):
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("distinct variants required")
    flattened = []
    arms = [f"{variant}_{policy}" for variant in variants for policy in ("final_a", "final_b")]
    for row in rows:
        if set(row["variants"]) != set(variants):
            raise ValueError("each target must retain every declared variant")
        flattened.append({
            **row, **{f"{variant}_{policy}": row["variants"][variant][policy]
                      for variant in variants for policy in ("final_a", "final_b")},
        })
    full = _cohort(flattened, arms)
    available = [row for row in flattened if row["h1"] is not None]
    complete = [row for row in flattened if all(row["mask"][task] for task in TASKS)]
    return {
        "schema_version": "verifier_variants_independent_metrics_v1",
        "variants": list(variants), "independent_metric_counts_match": True,
        "interpretation": {
            "all_valid_heads_exact": "all heads with mask=true match; this is NOT necessarily five-head exactness",
            "complete_five_head_gt": "every one of the five task masks is true; five-head exactness may be reported here",
            "changes": "symmetric-difference label loss; mixed retains both gains and harms, even if net loss falls",
            "application_rates": "actual output edits, not self-reported verifier approval",
        },
        "coverage": {
            "targets": len(rows), "candidate_available": len(available),
            "complete_five_head_gt_targets": len(complete),
            "valid_by_task": {task: sum(row["mask"][task] for row in rows) for task in TASKS},
            "videos": dict(Counter(row["video_id"] for row in rows)),
            "variant_status": {
                variant: dict(Counter(row["variants"][variant].get("status", "UNRECORDED") for row in rows))
                for variant in variants
            },
        },
        "all_targets": full,
        "candidate_available": _cohort(available, arms),
        "complete_five_head_gt": _cohort(complete, arms),
        "edit_application": {arm: _edit_admission(flattened, arm) for arm in arms},
        "candidate_ivt_diagnostic": _candidate_diagnostic(flattened),
    }


def _checked_event_evidence(event, reservation):
    path = Path(event["evidence_path"])
    if sha(path) != event["evidence_sha256"]:
        raise ValueError("ledger evidence digest mismatch")
    result = read(path)
    if (result["key"] != reservation["key"] or result["stage"] != reservation["variant"]
            or result["provider_calls"] != 1):
        raise ValueError("ledger evidence does not name its reserved request")
    request = read(path.parent / "request.json")
    if result["request_hash"] != request["metadata"]["request_hash"]:
        raise ValueError("request/result hash binding mismatch")
    route = request["wire"]["provider"]
    if (route != {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
            or request["metadata"]["requested_model_identifier"] != MODEL):
        raise ValueError("recorded request left the frozen model/provider route")
    http_path = path.parent / "http_response.json"
    http = read(http_path) if http_path.exists() else None
    try:
        native = json.loads(http["body"]) if http is not None else None
    except json.JSONDecodeError:
        native = None  # Non-JSON HTTP failure remains an unknown-cost reservation.
    return path, result, http, native


def audit_ledger(ledger_path, *, run_dir=None):
    """Recompute every reservation/settlement without using the runner's totals."""
    ledger_path = Path(ledger_path)
    raw = ledger_path.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not events:
        raise ValueError("budget ledger is empty")
    reserves, settlements = {}, {}
    unknown, halts, details = set(), [], []
    occupied = {"development": Decimal(0), "validation": Decimal(0)}
    known = Decimal(0)
    previous = None
    for index, event in enumerate(events):
        body = {key: value for key, value in event.items() if key != "sha256"}
        if (event["sequence"] != index or event["previous_sha256"] != previous
                or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != event["sha256"]):
            raise ValueError("budget event hash chain mismatch")
        previous = event["sha256"]
        kind, identity = event["event"], event.get("reservation_id")
        if kind == "INIT":
            if index != 0 or Decimal(event["budget_usd"]) != 12 or Decimal(event["development_budget_usd"]) != 8:
                raise ValueError("unexpected goal budget initialization")
        elif kind == "RESERVE":
            phase = event["phase"]
            if (halts or identity in reserves or Decimal(event["reserve_usd"]) != RESERVE
                    or phase not in occupied or sum(occupied.values()) + RESERVE > 12
                    or phase == "development" and occupied[phase] + RESERVE > 8):
                raise ValueError("reservation violates the goal/development limit or lifecycle")
            reserves[identity] = event
            occupied[phase] += RESERVE
        elif kind in ("SETTLE", "UNKNOWN"):
            if identity not in reserves or identity in settlements:
                raise ValueError("settlement/unknown does not name an open reservation")
            path, result, http, native = _checked_event_evidence(event, reserves[identity])
            if kind == "UNKNOWN":
                if result.get("cost_usd") is not None:
                    raise ValueError("unknown cost event contradicts saved known cost")
                unknown.add(identity)
                continue
            amount = Decimal(event["cost_usd"])
            if not amount.is_finite() or amount < 0 or Decimal(str(result["cost_usd"])) != amount:
                raise ValueError("settled cost is invalid or differs from result")
            if native is None or Decimal(str(native["usage"]["cost"])) != amount:
                raise ValueError("settled cost differs from native HTTP usage.cost")
            if event.get("provider_request_id") != result.get("provider_request_id"):
                raise ValueError("settlement provider id differs from saved response")
            identity_match = http["status_code"] != 200 or (
                native.get("model") == MODEL and native.get("provider") == "Alibaba"
                and native.get("id") == result.get("provider_request_id") and bool(native.get("id")))
            settlements[identity] = event
            occupied[reserves[identity]["phase"]] += amount - RESERVE
            known += amount
            details.append({
                "reservation_id": identity, "run_id": reserves[identity]["run_id"],
                "key": reserves[identity]["key"], "stage": reserves[identity]["variant"],
                "cost_usd": str(amount), "http_status": http["status_code"],
                "result_status": result["status"], "native_identity_match": identity_match,
                "exceeded_reservation": amount > RESERVE,
                "native_cost_verified": True, "evidence_path": str(path),
                "evidence_sha256": sha(path), "http_response_sha256": sha(path.parent / "http_response.json"),
            })
        elif kind == "HALT":
            halts.append(event["reason"])
        else:
            raise ValueError("unknown budget event")
    if any(row["exceeded_reservation"] for row in details) and "NATIVE_COST_EXCEEDED_RESERVATION" not in halts:
        raise ValueError("over-reservation native cost was not halted")
    if any(not row["native_identity_match"] for row in details) and "MODEL_OR_PROVIDER_IDENTITY_MISMATCH" not in halts:
        raise ValueError("native model/provider mismatch was not halted")
    open_ids = sorted(set(reserves) - set(settlements))
    if ledger_path.read_bytes() != raw:
        raise ValueError("budget changed during this audit; take a fresh snapshot")
    current_run = str(Path(run_dir).resolve()) if run_dir is not None else None
    current_reserves = {key: value for key, value in reserves.items() if value["run_id"] == current_run}
    current_settlements = [settlements[key] for key in current_reserves if key in settlements]
    current_known = sum((Decimal(e["cost_usd"]) for e in current_settlements), Decimal(0))
    current_open = sum(key not in settlements for key in current_reserves)
    return {
        "schema_version": "independent_goal_budget_native_cost_audit_v1",
        "snapshot_only": True, "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "goal_id": events[0]["goal_id"], "events": len(events),
        "reservations": len(reserves), "settled_calls": len(settlements),
        "known_cost_usd": str(known), "total_cost_usd": None if open_ids else str(known),
        "held_reserve_usd": str(RESERVE * len(open_ids)),
        "occupied_usd": str(sum(occupied.values())),
        "occupied_by_phase": {key: str(value) for key, value in occupied.items()},
        "open_reservations": [{"reservation_id": key, "run_id": reserves[key]["run_id"],
                               "key": reserves[key]["key"], "stage": reserves[key]["variant"],
                               "reserve_usd": str(RESERVE), "explicit_unknown": key in unknown}
                              for key in open_ids],
        "halt_reasons": halts, "settlement_details": details,
        "current_run": {"run_id": current_run, "reservations": len(current_reserves),
                        "settled_calls": len(current_settlements),
                        "known_cost_usd": str(current_known),
                        "total_cost_usd": None if current_open else str(current_known),
                        "open_reservations": current_open},
        "reservation_limits_checked_independently": True,
        "native_costs_checked_independently": True,
    }


def audit_summary_cost_representation(summary, records, current):
    """Reproduce the frozen float operation order; never tolerate money drift.

    Native/result/ledger amounts are checked as exact Decimals by audit_ledger.
    The runner accumulates Python floats within each stage, then sums stage
    totals in first-seen order. Only that exact representation is accepted.
    """
    exact = Decimal(0)
    sequential_float = 0.0
    stages = {}
    identities = set()
    unknown = 0
    for record in records:
        identity = record["key"], record["stage"]
        if identity in identities:
            raise ValueError("duplicate actual cost record")
        identities.add(identity)
        amount = record["cost_usd"]
        if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0):
            raise ValueError("invalid actual cost representation")
        if amount is None:
            unknown += 1
        else:
            exact += Decimal(str(amount))
        value = amount or 0.0
        sequential_float += value
        stage = stages.setdefault(record["stage"], {"float_sum": 0.0, "exact_sum": Decimal(0)})
        stage["float_sum"] += value
        stage["exact_sum"] += Decimal(str(amount)) if amount is not None else Decimal(0)
    grouped_float = sum(item["float_sum"] for item in stages.values())
    if (exact != Decimal(current["known_cost_usd"])
            or len(records) != current["reservations"] or unknown != current["open_reservations"]
            or summary["provider_calls"] != len(records) or summary["unpriced_calls"] != unknown):
        raise ValueError("actual call costs/counts differ from exact native ledger")
    fields = ("known_cost_usd", "cost_usd") if not unknown else ("known_cost_usd",)
    if unknown and summary["cost_usd"] is not None:
        raise ValueError("total cost must remain unknown while reservations are open")
    for field in fields:
        observed = summary[field]
        if type(observed) not in (int, float) or not math.isfinite(observed) or observed != grouped_float:
            raise ValueError("summary cost differs from exact replay of frozen stage-grouped float summation")
    return {"policy": "exact per-call Decimal ledger equality plus exact frozen stage-grouped Python-float replay; no tolerance",
            "exact_known_cost_usd": str(exact), "exact_total_cost_usd": None if unknown else str(exact),
            "reported_known_cost_usd": summary["known_cost_usd"], "reported_total_cost_usd": summary["cost_usd"],
            "replayed_stage_grouped_float": grouped_float, "call_order_float_sum": sequential_float,
            "reported_minus_exact_usd": str(Decimal(str(summary["known_cost_usd"])) - exact),
            "stage_order": list(stages), "stages": {name: {
                "exact_known_cost_usd": str(item["exact_sum"]), "replayed_float_sum": item["float_sum"]}
                for name, item in stages.items()}, "money_tolerance_usd": "0"}


def audit_run(run_dir, *, ledger_path=None):
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").is_file():
        raise ValueError("completed summary required before any GT read")
    summary, plan = read(run_dir / "summary.json"), read(run_dir / "plan.json")
    if (sha(run_dir / "predictions.json") != summary["predictions_sha256"]
            or plan["plan_sha256"] != summary["plan_sha256"]):
        raise ValueError("completed run prediction/plan binding mismatch")
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned_plan)).hexdigest() != plan["plan_sha256"]:
        raise ValueError("plan self-hash mismatch")
    if plan["source_split"] not in ("Training", "Validation"):
        raise ValueError("Testing is excluded from this experiment")
    rows = read(run_dir / "predictions.json")
    if any("gt" in row or "mask" in row for row in rows):
        raise ValueError("live prediction artifacts must not contain GT")
    scored = read(run_dir / "scored_predictions.json")
    if len(scored) != len(rows) or summary["targets"] != len(rows):
        raise ValueError("scored/main target coverage mismatch")
    for original, target in zip(rows, scored, strict=True):
        if {key: value for key, value in target.items() if key not in ("gt", "mask")} != original:
            raise ValueError("prediction values changed while attaching GT")
    variants = summary["variants"]
    if variants != plan["variants"]:
        raise ValueError("variant set changed after freezing")
    metrics = compute_variant_audit(scored, variants)
    for variant in variants:
        for policy in ("final_a", "final_b"):
            arm = f"{variant}_{policy}"
            recorded = summary["comparisons"][variant][policy]
            if metrics["all_targets"]["arms"][arm] != recorded["arms"]["final"]:
                raise ValueError("recorded final-arm metrics differ from independent audit")
            if metrics["all_targets"]["changes_vs_h0"][arm] != recorded["paired"]["h0_to_final"]:
                raise ValueError("recorded change counts differ from independent audit")
            for baseline in ("h0", "h1_policy"):
                if metrics["all_targets"]["arms"][baseline] != recorded["arms"][baseline]:
                    raise ValueError("recorded baseline metrics differ from independent audit")
    budget = audit_ledger(ledger_path or plan["budget_ledger"], run_dir=run_dir)
    current = budget["current_run"]
    records_path = run_dir / "calls_summary.json"
    records = read(records_path) if records_path.is_file() else []
    for record in records:
        if read(run_dir / "calls" / record["key"] / record["stage"] / "result.json") != record:
            raise ValueError("ordered calls_summary differs from native-ledger-bound result")
    representation = audit_summary_cost_representation(summary, records, current)
    return {"schema_version": "verifier_variants_complete_independent_audit_v1",
            "source_split": plan["source_split"], "metrics": metrics, "budget": budget,
            "cost_representation_audit": representation,
            "recorded_metrics_match": True, "recorded_costs_match": True,
            "provider_calls_by_audit": 0, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {name: sha(run_dir / name) for name in (
                "plan.json", "summary.json", "predictions.json", "scored_predictions.json")},
            "audit_script_sha256": sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run_dir, ledger_path=args.ledger)
    output = args.output or args.run_dir / "independent_variant_audit.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "coverage": report["metrics"]["coverage"],
                      "recorded_metrics_match": True, "recorded_costs_match": True}))


if __name__ == "__main__":
    main()
