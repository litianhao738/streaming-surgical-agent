"""Account for a CLOSED action-cue trial without loading labels or predictions.

Actual dispatched accounting is distinct from the cost/token equivalent of
deploying each arm separately. Shared H0, Phase and identical-wire review panels
are reused logically, while each actual HTTP request is counted once as paid.
Only standard-library modules are imported; frozen inference code is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ARMS = ("control", "action_cue")
SEATS = ("grok", "qwen", "gpt", "gemini", "deepseek")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(output, relative):
    path = (output / relative).resolve()
    if not path.is_relative_to(output.resolve()):
        raise ValueError("archive path escapes the run directory")
    return path


def load_closed(output):
    """Validate closure and frozen artifacts before reading accounting records."""
    output = Path(output).resolve()
    completion_path = output / "completion.json"
    if not completion_path.is_file():
        raise ValueError("accounting requires completion.json; inference is not closed")
    done = read(completion_path)
    if not done.get("closed_utc") or done.get("fatal_error"):
        raise ValueError("accounting requires closed nonfatal inference")
    ledger = read(output / "budget.json")
    if ledger.get("stopped") is not True or any(c.get("status") == "DISPATCHED" for c in ledger["calls"]):
        raise ValueError("accounting refuses open ledgers or outstanding requests")
    frozen = done.get("inference_artifact_sha256", {})
    if not {"plan.json", "budget.json"} <= set(frozen):
        raise ValueError("closed accounting inputs are not frozen")
    for relative, digest in frozen.items():
        if sha(safe_path(output, relative)) != digest:
            raise ValueError("closed artifact changed: " + relative)
    files = {p.relative_to(output).as_posix() for folder in ("calls", "targets", "request_intents")
             for p in (output / folder).rglob("*.json")}
    if not files <= set(frozen):
        raise ValueError("unfrozen call or target artifact")
    plan = read(output / "plan.json")
    if tuple(plan.get("arms", ())) != ARMS or len(ledger["calls"]) != done.get("post_calls"):
        raise ValueError("unexpected arms or mismatched closed call count")
    keys = [s["key"] for s in plan["selection"]]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate target key")
    calls = ledger["calls"]
    if len({c["index"] for c in calls}) != len(calls) or len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate call index or target/stage/seat")
    allowed = {("h0", "base"), *(("shared_phase", s) for s in SEATS),
               *((a + "_proposal", "base") for a in ARMS),
               *((a + "_review", s) for a in ARMS for s in SEATS)}
    for call in calls:
        if call["target"] not in keys or (call["stage"], call["seat"]) not in allowed:
            raise ValueError("unexpected call target, stage or seat")
        charge = Decimal(str(call["charge"]))
        if not charge.is_finite() or charge < 0:
            raise ValueError("invalid charge")
        folder = f"calls/{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        if read(safe_path(output, folder + "/record.json")) != call:
            raise ValueError("call record differs from ledger")
    for account, occupied in ledger.get("occupied", {}).items():
        carried = Decimal(str(ledger.get("carried_occupied", {}).get(account, "0")))
        amount = sum((Decimal(str(c["charge"])) for c in calls if c["account"] == account), Decimal(0))
        if amount + carried != Decimal(str(occupied)):
            raise ValueError("ledger account arithmetic mismatch")
    records, shared = {}, {}
    for key in keys:
        shared[key] = read(safe_path(output, f"targets/{key}/shared.json"))
        records[key] = {}
        for arm in ARMS:
            path = safe_path(output, f"targets/{key}/{arm}.json")
            if path.is_file():
                records[key][arm] = read(path)
    return plan, ledger, done, records, shared


def token_value(call, field):
    usage = call.get("usage") or {}
    value = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
             if field == "reasoning_tokens" else usage.get(field))
    return value if type(value) is int and value >= 0 else None


def token_totals(calls):
    result = {}
    for field in TOKEN_FIELDS:
        values = [token_value(c, field) for c in calls]
        known = [v for v in values if v is not None]
        result[field] = {"known_sum": sum(known), "calls_with_native_usage": len(known),
                         "calls_missing_native_usage": len(values) - len(known),
                         "complete_total": sum(known) if len(known) == len(values) else None}
    return result


def cost_totals(calls):
    result = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "amount": Decimal(0)}))
    for call in calls:
        entry = result[call["account"]][call.get("charge_kind", "unknown_reserved")]
        entry["calls"] += 1
        entry["amount"] += Decimal(str(call["charge"]))
    return {account: {kind: {"calls": item["calls"], "amount": str(item["amount"])}
                      for kind, item in kinds.items()} for account, kinds in result.items()}


def group_summary(calls):
    return {"call_occurrences": len(calls), "unique_paid_source_calls": len({c["index"] for c in calls}),
            "costs_by_account_and_kind": cost_totals(calls), "native_tokens": token_totals(calls),
            "transport_statuses": dict(Counter(c["status"] for c in calls))}


def grouped(calls, key):
    groups = defaultdict(list)
    for call in calls:
        groups[key(call)].append(call)
    return {name: group_summary(values) for name, values in sorted(groups.items())}


def logical_target_calls(key, arm, actual, records):
    """Resolve the exact paid source for each logically required review panel."""
    own = records.get(key, {}).get(arm, {})
    shared_from = own.get("shared_from")
    review_arm = arm
    if shared_from:
        if shared_from not in ARMS or shared_from == arm:
            raise ValueError("invalid shared panel source")
        source = records.get(key, {}).get(shared_from)
        if not source or source.get("shared_from"):
            raise ValueError("missing, chained or cyclic shared panel source")
        if not own.get("request_fingerprints") or own["request_fingerprints"] != source.get("request_fingerprints"):
            raise ValueError("shared panel is not identical for all review requests")
        if any(c["stage"] == arm + "_review" for c in actual):
            raise ValueError("shared panel also has separately dispatched reviews")
        review_arm = shared_from
    stages = {"h0": "h0", "phase": "shared_phase", "proposal": arm + "_proposal", "review": review_arm + "_review"}
    groups = {role: [c for c in actual if c["stage"] == stage] for role, stage in stages.items()}
    logical = [call for values in groups.values() for call in values]
    if len({c["index"] for c in logical}) != len(logical):
        raise ValueError("paid source counted twice within one logical arm")
    return logical, groups, review_arm


def _duration(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def reconstructed_latency(groups, expected_review_calls=5):
    times = {}
    for role, calls in groups.items():
        values = [_duration(c.get("elapsed_seconds")) for c in calls]
        # An unexecuted branch consumes zero request time; a missing recorded
        # duration for an executed request remains unknown.
        times[role] = None if any(v is None for v in values) else (max(values) if values else 0.0)
    reconstructed = None
    if groups["h0"] and all(v is not None for v in times.values()):
        reconstructed = times["h0"] + max(times["phase"], times["proposal"] + times["review"])
    counts = {role: len(calls) for role, calls in groups.items()}
    return {"seconds": reconstructed, "stage_request_max_seconds": times, "stage_call_counts": counts,
            "expected_review_calls": expected_review_calls,
            "all_deployed_stages_dispatched": counts == {"h0": 1, "phase": 5, "proposal": 1, "review": expected_review_calls},
            "formula": "h0 + max(phase max request, proposal + review max request)"}


def distribution(values):
    values = list(values)
    known = [v for v in values if v is not None]
    return {"targets": len(values), "known_targets": len(known), "missing_targets": len(values) - len(known),
            "known_sum": sum(known), "complete_sum": sum(known) if len(values) == len(known) else None,
            "mean_known": statistics.mean(known) if known else None,
            "median_known": statistics.median(known) if known else None}


def proposal_pairs(keys, calls):
    pairs = []
    for key in keys:
        selected = {a: [c for c in calls if c["target"] == key and c["stage"] == a + "_proposal"] for a in ARMS}
        for arm in ARMS:
            if len(selected[arm]) > 1:
                raise ValueError("multiple proposal attempts for a target")
        values = {a: {field: token_value(selected[a][0], field) if selected[a] else None
                      for field in TOKEN_FIELDS} for a in ARMS}
        pairs.append({"key": key, "native_tokens": values, "deltas": {
            field: values["action_cue"][field] - values["control"][field]
            if all(values[a][field] is not None for a in ARMS) else None for field in TOKEN_FIELDS}})
    summary = {}
    for field in TOKEN_FIELDS:
        valid = [p for p in pairs if p["deltas"][field] is not None]
        summary[field] = {"verified_pairs": len(valid), "missing_pairs": len(pairs) - len(valid),
                          "control_known_paired_total": sum(p["native_tokens"]["control"][field] for p in valid),
                          "action_cue_known_paired_total": sum(p["native_tokens"]["action_cue"][field] for p in valid),
                          "increased_pairs": sum(p["deltas"][field] > 0 for p in valid),
                          "decreased_pairs": sum(p["deltas"][field] < 0 for p in valid),
                          "unchanged_pairs": sum(p["deltas"][field] == 0 for p in valid)}
    return {"pairs": pairs, "summary": summary}


def phase_validity(keys, shared):
    by_seat = {seat: Counter() for seat in SEATS}
    panel_status = Counter()
    targets = []
    for key in keys:
        phase = shared[key].get("phase")
        if phase is None:
            panel_status["NOT_RUN"] += 1
            targets.append({"key": key, "panel_status": "NOT_RUN"})
            continue
        errors = phase.get("errors", {})
        if set(errors) != set(SEATS):
            raise ValueError("shared Phase record lacks five explicit validation outcomes")
        panel_status[phase.get("status", "MISSING_STATUS")] += 1
        for seat, error in errors.items():
            if error is not None and not isinstance(error, str):
                raise ValueError("invalid saved Phase validation error")
            by_seat[seat][error or "VALID"] += 1
        targets.append({"key": key, "panel_status": phase.get("status"), "errors": errors})
    return {"panel_statuses": dict(panel_status), "by_seat": {s: dict(v) for s, v in by_seat.items()},
            "targets": targets, "note": "Saved structural/evidence contract errors only; no GT, phase labels or semantic accuracy inspected."}


def build_report(plan, ledger, done, records, shared):
    keys = [s["key"] for s in plan["selection"]]
    calls = ledger["calls"]
    logical = {arm: [] for arm in ARMS}
    role_groups = {arm: defaultdict(list) for arm in ARMS}
    targets, reused = [], Counter()
    for key in keys:
        actual = [c for c in calls if c["target"] == key]
        row = {"key": key, "arms": {}}
        for arm in ARMS:
            arm_calls, groups, review_arm = logical_target_calls(key, arm, actual, records)
            logical[arm].extend(arm_calls)
            for role, values in groups.items():
                role_groups[arm][role].extend(values)
            reused[arm] += review_arm != arm
            row["arms"][arm] = {**group_summary(arm_calls), "review_paid_source_arm": review_arm,
                "paid_source_call_indices": [c["index"] for c in arm_calls],
                "roles": {role: {"paid_source_call_indices": [c["index"] for c in values], **group_summary(values)}
                          for role, values in groups.items()},
                "reconstructed_latency": reconstructed_latency(groups, expected_review_calls=(
                    0 if records.get(key, {}).get(arm, {}).get("status") == "EMPTY_POOL_UNVERIFIED" else 5))}
        targets.append(row)
    covered = {c["index"] for values in logical.values() for c in values}
    if covered != {c["index"] for c in calls}:
        raise ValueError("a paid call has no corresponding logical deployed stage")
    arms = {arm: {**group_summary(logical[arm]), "by_role": {role: group_summary(values)
                for role, values in role_groups[arm].items()},
                "by_model": grouped(logical[arm], lambda c: c["model"]), "shared_review_panels_reused": reused[arm],
                "reconstructed_latency_seconds": distribution(t["arms"][arm]["reconstructed_latency"]["seconds"] for t in targets),
                "targets_all_deployed_stages_dispatched": sum(t["arms"][arm]["reconstructed_latency"]["all_deployed_stages_dispatched"] for t in targets)}
            for arm in ARMS}
    return {"profile": plan.get("profile"), "targets": len(keys), "closed_utc": done["closed_utc"],
            "actual_dispatched_accounting": {**group_summary(calls),
                "by_stage": grouped(calls, lambda c: c["stage"]),
                "by_model": grouped(calls, lambda c: c["model"]),
                "observed_whole_experiment_elapsed_seconds": done.get("elapsed_seconds")},
            "separate_deployment_logical_equivalent": arms, "target_accounting": targets,
            "proposal_native_token_pairs": proposal_pairs(keys, calls), "shared_phase_validation": phase_validity(keys, shared),
            "interpretation": [
                "Actual accounting counts each dispatched request once. Native charges, estimates and unknown reserves are separate; reserves are not confirmed billed cost.",
                "Each logical arm includes its own H0 and Phase overhead plus its proposal/review panel. Shared paid calls are copied into each logical arm, not billed again.",
                "When all review request fingerprints match, the exact source arm's review usage/charge is attributed to both logical arms, including failures.",
                "Logical-equivalent totals are observed-call reconstructions, not new paid API requests or a prediction of future provider bills.",
                "Missing native usage stays missing. known_sum is only the reported subset; complete_total is null if any relevant call lacks the count.",
                "Reconstructed latency is h0 + max(Phase request maximum, proposal + review request maximum). It excludes local orchestration overhead and is not an end-to-end randomized timing comparison.",
                "Observed request durations include provider queueing and competition from concurrent experiment requests; reuse is not an independent latency observation.",
                "New frames belong to previously used Training videos, not independent new-video generalization. This accounting tool reads no GT or prediction labels."],
            "audit": {"closed_artifact_hashes_verified": True, "all_paid_calls_accounted_for": True,
                      "query_gt_read": False, "prediction_labels_inspected": False,
                      "source_completion_sha256": None}}


def summarize(output):
    output = Path(output).resolve()
    source = load_closed(output)
    completion_hash = sha(output / "completion.json")
    result = build_report(*source)
    result["audit"]["source_completion_sha256"] = completion_hash
    # Guard against a simultaneous mutation before creating the derived artifact.
    load_closed(output)
    if sha(output / "completion.json") != completion_hash:
        raise ValueError("completion changed during accounting")
    result["audit"]["accounting_tool_sha256"] = sha(Path(__file__))
    destination = output / "arm_accounting.json"
    temporary = output / "arm_accounting.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.output)
    print(json.dumps({"targets": report["targets"],
        "actual_dispatched_accounting": report["actual_dispatched_accounting"],
        "logical_arms": report["separate_deployment_logical_equivalent"],
        "proposal_native_token_pairs": report["proposal_native_token_pairs"]["summary"],
        "shared_phase_validation": {k: report["shared_phase_validation"][k] for k in ("panel_statuses", "by_seat")}},
        ensure_ascii=False, indent=2))
