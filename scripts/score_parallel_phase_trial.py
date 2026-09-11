"""Closed-run replay and masked evaluation of parallel Graph R1 plus Phase.

The scorer performs no model calls or credential reads. It loads query labels
only after checking raw requests, branch ledgers, the saved merge and sources.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_parallel_phase_trial import (
    BRANCH_MAX_CALLS,
    BRANCHES,
    STAGES,
    collect_phase,
    run_graph,
    verify_plan,
)
from scripts.run_prior_panel_trial import read, save
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from scripts.score_phase_extension_trial import audit as audit_source
from scripts.score_phase_extension_trial import (
    independent_phase_decision,
    phase_transition,
)
from scripts.score_prior_feedback_continuation import _hashes
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import digest, labels
from surgical_agent.research.verification.semantic_coordinator import item_error

TASKS = ("instrument", "verb", "target", "ivt", "phase")
VERSIONS = ("h0", "previous_graph", "previous_final", "graph_only", "final")
PAIRS = (("h0", "previous_graph"), ("previous_graph", "previous_final"),
         ("h0", "graph_only"), ("h0", "final"), ("previous_graph", "graph_only"),
         ("previous_final", "final"), ("graph_only", "final"))


def _identities(rows):
    return [(r["key"], r["video_id"], r["frame_id"]) for r in rows]


def _without_timing(record):
    return {key: value for key, value in record.items() if key != "timing"}


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_timing(record):
    """Check recorded intervals rather than adding overlapping branch times."""
    timing = record["timing"]
    names = {"graph_start", "graph_end", "graph_seconds", "phase_start", "phase_end",
             "phase_seconds", "join_seconds", "merge_seconds", "parallel_seconds"}
    if set(timing) != names or not all(_finite_nonnegative(v) for v in timing.values()):
        raise ValueError("invalid branch timing envelope")
    for branch in BRANCHES:
        start, end, elapsed = (timing[f"{branch}_{name}"] for name in ("start", "end", "seconds"))
        if end < start or not math.isclose(end - start, elapsed, abs_tol=1e-6):
            raise ValueError("branch interval differs from elapsed time")
        if timing["join_seconds"] + 1e-6 < end:
            raise ValueError("join recorded before a branch completed")
        stages = record[branch]["timing"]
        if not isinstance(stages, dict) or not all(_finite_nonnegative(v) for v in stages.values()):
            raise ValueError("invalid within-branch stage timings")
    if timing["parallel_seconds"] + 1e-6 < timing["join_seconds"] + timing["merge_seconds"]:
        raise ValueError("parallel interval excludes part of its merge")
    if not _finite_nonnegative(record["input_seconds"]):
        raise ValueError("invalid input preparation duration")


def audit(output, adapter):
    """Rebuild both requests independently and verify the Phase-only merge."""
    output = Path(output)
    plan, done, budget = (read(output / f"{name}.json") for name in ("plan", "completion", "budget"))
    if not done.get("closed_utc") or done.get("fatal_error") or budget.get("stopped") is not True:
        raise ValueError("requires closed nonfatal inference")
    if done.get("new_h0_calls") != 0 or done.get("query_gt_not_loaded_during_inference") is not True:
        raise ValueError("cached H0 and isolated inference declarations required")
    for name in ("plan", "initial_state", "predictions", "budget"):
        if sha(output / f"{name}.json") != done[f"{name}_sha256"]:
            raise ValueError("closed snapshot changed")
    _hashes(output, done["inference_artifact_sha256"], "closed inference")
    required = {p.relative_to(output).as_posix() for folder in ("branches", "targets")
                for p in (output / folder).rglob("*.json")}
    if required != set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen or missing branch artifact")
    verify_plan(plan)
    source_plan, previous, _, _, _, _ = audit_source(Path(plan["source_root"]), adapter)
    rows = read(output / "predictions.json")["targets"]
    initials = read(output / "initial_state.json")["targets"]
    expected = _identities(plan["selection"])
    if (len(expected) != 8 or len(set(expected)) != 8
            or len({key for key, _, _ in expected}) != 8
            or len({(video, frame) for _, video, frame in expected}) != 8
            or any(_identities(group) != expected for group in (rows, initials, previous))
            or plan["selection"] != source_plan["selection"]):
        raise ValueError("same eight cached Training targets and order required")
    if any(adapter.entries[row["video_id"]].split is not DatasetSplit.TRAINING for row in rows):
        raise ValueError("Testing or Validation may not enter this development experiment")
    expected_phase = {key: group["phase_short"] for key, group in source_plan["phase_inputs"].items()}
    if plan["phase_inputs"] != expected_phase:
        raise ValueError("parallel Phase must use the frozen short causal inputs")
    ledgers, all_calls = {}, []
    if budget["limits"] != plan["limits"] or budget["branch_limits"] != plan["branch_limits"]:
        raise ValueError("combined budget changed the frozen allowance")
    if set(budget["branch_budget_sha256"]) != set(BRANCHES):
        raise ValueError("both independent branch budgets required")
    for branch in BRANCHES:
        path = output / "branches" / branch / "budget.json"
        if sha(path) != budget["branch_budget_sha256"][branch]:
            raise ValueError("branch ledger differs from root closure")
        ledger = read(path)
        ledgers[branch] = ledger
        calls = ledger["calls"]
        if ledger["limits"] != plan["branch_limits"][branch]:
            raise ValueError("branch allowance differs from frozen plan")
        if any(Decimal(value) != 0 for value in ledger["carried_occupied"].values()):
            raise ValueError("fresh branch ledger must not carry historical model charges")
        accounted = {account: Decimal(0) for account in ledger["occupied"]}
        for call in calls:
            charge = Decimal(call["charge"])
            if not charge.is_finite() or charge < 0 or call["account"] not in accounted:
                raise ValueError("invalid branch account or native charge")
            accounted[call["account"]] += charge
        if any(accounted[account] != Decimal(value) for account, value in ledger["occupied"].items()):
            raise ValueError("branch accounting differs from its call charges")
        if ledger.get("stopped") is not True or len(calls) > BRANCH_MAX_CALLS[branch]:
            raise ValueError("open branch ledger or exceeded branch call cap")
        if any(c["status"] == "DISPATCHED" for c in calls):
            raise ValueError("outstanding branch request")
        if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
            raise ValueError("duplicate call or undeclared retry")
        known = {row["key"] for row in rows}
        for call in calls:
            allowed = ((call["stage"] == STAGES["proposal"] and call["seat"] == "base")
                       or (call["stage"] == STAGES["review"] and call["seat"] in SEATS)) if branch == "graph" else (
                           call["stage"] == STAGES["phase"] and call["seat"] in SEATS)
            if call["target"] not in known or not allowed:
                raise ValueError("unexpected target, stage or family in branch")
        all_calls.extend({"branch": branch, **c} for c in calls)
    if budget["calls"] != all_calls or done["post_calls"] != len(all_calls) or len(all_calls) > 88:
        raise ValueError("combined ledger differs from independent branches")
    if done["statuses"] != dict(Counter(c["status"] for c in all_calls)):
        raise ValueError("completion transport status counts differ")
    records = {}
    for row, initial, old, selected in zip(rows, initials, previous, plan["selection"], strict=True):
        key = row["key"]
        baselines = {"h0": old["h0"], "previous_graph": old["graph_r1"], "previous_final": old["phase_short"]}
        if any(row[name] != value or initial[name] != value for name, value in baselines.items()):
            raise ValueError("cached H0 or previous comparison baseline changed")
        prior_path = Path(plan["original_graph_root"]) / "priors" / f"{row['video_id']}.json"
        if plan["prior_paths_sha256"].get(str(prior_path)) != sha(prior_path):
            raise ValueError("query prior missing from frozen inputs")
        prior = read(prior_path)
        if (prior["excluded_video"] != row["video_id"] or row["video_id"] in prior["fit_videos"]
                or not prior["fit_videos"]
                or any(adapter.entries[video].split is not DatasetSplit.TRAINING for video in prior["fit_videos"])
                or digest({k: v for k, v in prior.items() if k != "table_sha256"}) != prior["table_sha256"]):
            raise ValueError("invalid or contaminated leave-video-out prior")
        record = read(output / "targets" / key / "pipeline.json")
        records[key] = record
        graph_calls = [c for c in ledgers["graph"]["calls"] if c["target"] == key]
        phase_calls = [c for c in ledgers["phase"]["calls"] if c["target"] == key]
        replay_graph_calls = ReplayCalls(output / "branches" / "graph", graph_calls)
        replay_phase_calls = ReplayCalls(output / "branches" / "phase", phase_calls)
        graph = run_graph(replay_graph_calls, build_gemini_base(adapter, selected), selected, initial["h0"], prior)
        phase = collect_phase(replay_phase_calls, plan["phase_inputs"][key])
        if _without_timing(graph) != _without_timing(record["graph"]):
            raise ValueError(f"raw graph replay differs: {key}")
        if _without_timing(phase) != _without_timing(record["phase"]):
            raise ValueError(f"raw Phase replay differs: {key}")
        if (record["graph"]["request_fingerprints"]["proposal"] != plan["wire_fingerprints"][key]["proposal"]
                or record["phase"]["request_fingerprints"] != plan["wire_fingerprints"][key]["phase"]):
            raise ValueError("request fingerprint differs from preflight")
        if len(replay_graph_calls.rows) != len(graph_calls) or len(replay_phase_calls.rows) != len(phase_calls):
            raise ValueError("unconsumed or reused branch response")
        if graph["reviews"] is not None:
            independent_means = {}
            for proposition in graph["pool"]["propositions"]:
                scores = []
                for seat in SEATS:
                    judgment = graph["reviews"][seat]["judgments"].get(proposition["id"])
                    invalid = item_error(judgment, proposition["task"], len(selected["images"]))
                    scores.append(None if invalid else judgment["rating"])
                independent_means[proposition["id"]] = None if None in scores else sum(scores) / 5
            if independent_means != graph["means"]:
                raise ValueError("independent five-seat graph score arithmetic differs")
        decision = independent_phase_decision(graph["prediction"], phase["raw_reviews"],
                                              len(plan["phase_inputs"][key]["images"]))
        final = deepcopy(graph["prediction"])
        final["phase"] = [decision["after"]]
        if (record["phase_decision"] != decision or record["final"] != final
                or row["graph_only"] != graph["prediction"] or row["final"] != final):
            raise ValueError("saved merge differs from independently counted Phase votes")
        if row["graph_only"]["phase"] != row["h0"]["phase"]:
            raise ValueError("four-head graph branch changed Phase")
        if any(row["final"][task] != row["graph_only"][task] for task in TASKS[:4]):
            raise ValueError("parallel merge changed graph interaction predictions")
        for version in VERSIONS:
            if labels(row[version]) != row[version]:
                raise ValueError("invalid five-head final-label prediction")
        validate_timing(record)
    return plan, rows, initials, records, budget, done


def summarize(rows, truth, records):
    truths = {(item["video_id"], item["frame_id"]): item for item in truth}
    metrics, details, phase_cases = {}, [], []
    for version in VERSIONS:
        inputs = [{**truths[row["video_id"], row["frame_id"]], "h0": row["h0"], "h1": None,
                   "final": row[version]} for row in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        for task in TASKS:
            tp = fp = fn = exact = valid = 0
            for row in rows:
                item = truths[row["video_id"], row["frame_id"]]
                if not item["mask"][task]:
                    continue
                predicted, expected = set(row[version][task]), set(item["gt"][task])
                tp += len(predicted & expected)
                fp += len(predicted - expected)
                fn += len(expected - predicted)
                exact += predicted == expected
                valid += 1
            measured = metrics[version]["tasks"][task]
            if (measured["tp"], measured["fp"], measured["fn"], measured["exact_matches"],
                    measured["valid_targets"]) != (tp, fp, fn, exact, valid):
                raise ValueError("independent masked metric counts differ")
    coverage = {task: Counter() for task in TASKS[:4]}
    for row in rows:
        key = row["key"]
        item = truths[row["video_id"], row["frame_id"]]
        record = records[key]
        details.append({"key": key, "graph_status": record["graph"]["status"],
            "phase_status": record["phase"]["status"], "phase_decision": record["phase_decision"],
            "comparisons": {f"{a}_to_{b}": frame_delta(row[a], row[b], item["gt"], item["mask"])
                            for a, b in PAIRS}})
        valid = bool(item["mask"]["phase"])
        expected_phase = item["gt"]["phase"] if valid else None
        phase_cases.append({"key": key, "valid_phase_gt": valid, "gt": expected_phase,
            "predictions": {version: row[version]["phase"] for version in VERSIONS},
            "decision": record["phase_decision"], "errors": record["phase"]["errors"],
            "observations": record["phase"]["raw_reviews"],
            "transitions": {f"{a}_to_{b}": phase_transition(row[a]["phase"], row[b]["phase"], expected_phase, valid)
                            for a, b in PAIRS}})
        pool = record["graph"]["pool"]
        for task in TASKS[:4]:
            if not item["mask"][task]:
                continue
            expected = set(item["gt"][task])
            proposed = {p["label_id"] for p in pool["propositions"] if p["task"] == task}
            coverage[task].update(valid_targets=1, gt_positive=len(expected), pool_true=len(proposed & expected),
                pool_false=len(proposed - expected), missing_from_pool=len(expected - proposed))
    return {"metrics": metrics, "candidate_coverage": coverage,
        "comparisons": {f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details])
                        for a, b in PAIRS},
        "phase_transitions": {f"{a}_to_{b}": dict(Counter(c["transitions"][f"{a}_to_{b}"] for c in phase_cases))
                              for a, b in PAIRS}}, details, phase_cases


def runtime_summary(records, budget, done):
    """Record measured overlap and distinguish serial counterfactual arithmetic."""
    charges = defaultdict(lambda: defaultdict(Decimal))
    branch_charges = defaultdict(lambda: defaultdict(lambda: defaultdict(Decimal)))
    stage_charges = defaultdict(lambda: defaultdict(lambda: defaultdict(Decimal)))
    for call in budget["calls"]:
        charges[call["account"]][call["charge_kind"]] += Decimal(call["charge"])
        branch_charges[call["branch"]][call["account"]][call["charge_kind"]] += Decimal(call["charge"])
        stage_charges[call["stage"]][call["account"]][call["charge_kind"]] += Decimal(call["charge"])
    def encoded(values):
        return {account: {kind: str(value) for kind, value in kinds.items()} for account, kinds in values.items()}
    totals = Counter()
    stages = {branch: Counter() for branch in BRANCHES}
    timing_rows = []
    for key, record in records.items():
        validate_timing(record)
        timing = record["timing"]
        overlap = max(0.0, min(timing["graph_end"], timing["phase_end"])
                      - max(timing["graph_start"], timing["phase_start"]))
        serial_equivalent = timing["graph_seconds"] + timing["phase_seconds"] + timing["merge_seconds"]
        data = {"key": key, "input_seconds": record["input_seconds"], **timing,
            "overlap_seconds": overlap, "serial_equivalent_seconds": serial_equivalent,
            "parallel_increment_over_graph_seconds": timing["parallel_seconds"] - timing["graph_seconds"],
            "time_saved_vs_serial_equivalent_seconds": serial_equivalent - timing["parallel_seconds"]}
        timing_rows.append(data)
        totals.update({key: value for key, value in data.items() if key != "key"})
        for branch in BRANCHES:
            stages[branch].update(record[branch]["timing"])
    serial = totals["serial_equivalent_seconds"]
    graph = totals["graph_seconds"]
    return {"calls": len(budget["calls"]), "new_h0_calls": 0,
        "charges": encoded(charges), "branch_charges": {b: encoded(v) for b, v in branch_charges.items()},
        "stage_charges": {stage: encoded(values) for stage, values in stage_charges.items()},
        "stage_calls": {stage: sum(c["stage"] == stage for c in budget["calls"]) for stage in STAGES.values()},
        "branch_calls": {b: sum(c["branch"] == b for c in budget["calls"]) for b in BRANCHES},
        "statuses": dict(Counter(c["status"] for c in budget["calls"])),
        "branch_transport_statuses": {b: dict(Counter(c["status"] for c in budget["calls"] if c["branch"] == b))
                                      for b in BRANCHES},
        "branch_semantic_statuses": {b: dict(Counter(r[b]["status"] for r in records.values())) for b in BRANCHES},
        "inference_seconds": done["inference_seconds"], "timing_totals": dict(totals),
        "branch_stage_seconds": {b: dict(v) for b, v in stages.items()}, "target_timings": timing_rows,
        "parallel_increment_over_graph_percent": 100 * (totals["parallel_seconds"] - graph) / graph if graph else None,
        "reduction_vs_serial_equivalent_percent": 100 * (serial - totals["parallel_seconds"]) / serial if serial else None,
        "timing_caveat": "Branch intervals are measured in the same concurrent run. Their sum is only a serial-equivalent arithmetic estimate, not an independently executed serial trial. API provider contention and run-to-run variability remain. End-to-end H0 time is excluded because H0 is cached."}


def score(output, adapter):
    output = Path(output)
    plan, rows, _, records, budget, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": row["video_id"], "frame_id": row["frame_id"],
        "h0": row["h0"], "h1": None, "final": row["final"]} for row in rows])
    result, details, phase_cases = summarize(rows, truth, records)
    result.update(profile=plan["profile"], targets=len(rows), limitations=plan.get("limitations", []),
        runtime=runtime_summary(records, budget, done), audit={"raw_outputs_replayed": True,
            "actual_requests_verified": len(budget["calls"]), "phase_votes_independently_checked": True,
            "metric_counts_independently_checked": True, "final_four_heads_equal_fresh_graph": True,
            "fresh_graph_phase_equals_cached_h0": True, "gt_loaded_after_closed_inference": True,
            "training_only_lovo_priors_checked": True})
    for name, value in (("metrics", result), ("frame_deltas", details), ("phase_cases", phase_cases), ("scored_truth", truth)):
        save(output / f"{name}.json", value)
    print(json.dumps({"f1": {v: {task: m["micro_f1"] for task, m in arm["tasks"].items()}
                            for v, arm in result["metrics"].items()},
        "phase_transitions": result["phase_transitions"], "runtime": result["runtime"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
