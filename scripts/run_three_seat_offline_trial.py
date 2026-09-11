"""Frozen-response GPT/Gemini/DeepSeek ablation; never dispatch an API call.

Prepare writes the protocol before scoring. The primary arm changes only the
interaction panel, retaining the original five-seat Phase result. A secondary
arm also reduces Phase to three structurally valid choices and a two-vote rule.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import now, read, save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from scripts.score_parallel_phase_trial import audit as audit_source
from scripts.score_phase_extension_trial import phase_transition
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.phase_extension import phase_choice_error
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.review_normalization import normalize_review
from surgical_agent.research.verification.semantic_coordinator import item_error

PROFILE = "frozen_three_seat_gpt_gemini_deepseek_offline_v1"
SOURCE = ROOT / "artifacts/preflight/parallel_phase_eight_20260909_v1"
SELECTED_SEATS = ("gpt", "gemini", "deepseek")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VERSIONS = ("h0", "five_seat_graph", "five_seat_final", "three_seat_graph",
            "three_seat_four_keep_phase", "three_seat_all")
PAIRS = (("h0", "five_seat_final"), ("h0", "three_seat_four_keep_phase"),
         ("h0", "three_seat_all"), ("five_seat_graph", "three_seat_graph"),
         ("five_seat_final", "three_seat_four_keep_phase"), ("five_seat_final", "three_seat_all"),
         ("three_seat_four_keep_phase", "three_seat_all"))


def three_interaction(h0, record, image_count):
    """Revalidate original evidence, then average exactly three valid items."""
    labels(h0)
    pool = record["pool"]
    if record["raw_reviews"] is None:
        return {"prediction": deepcopy(h0), "status": "INHERITED_NO_REVIEW", "means": None,
                "diagnostics": {}, "normalized": {}, "format_diagnostics": {}}
    reviews, formatting = {}, {}
    for seat in SELECTED_SEATS:
        reviews[seat], formatting[seat] = normalize_review(record["raw_reviews"][seat], pool,
                                                          seat=seat, image_count=image_count)
        if reviews[seat] != record["reviews"][seat]:
            raise ValueError("three-seat normalization changed original evidence rules")
    means, diagnostics = {}, {}
    for proposition in pool["propositions"]:
        pid, scores, invalid = proposition["id"], [], {}
        for seat in SELECTED_SEATS:
            item = reviews[seat]["judgments"].get(pid)
            reason = item_error(item, proposition["task"], image_count)
            if reason:
                invalid[seat] = formatting[seat]["errors"].get(pid, [reason])
            scores.append(None if reason else item["rating"])
        means[pid] = None if invalid else sum(scores) / 3
        diagnostics[pid] = {"scores": dict(zip(SELECTED_SEATS, scores, strict=True)), "invalid": invalid}
    try:
        prediction = panel.select(h0, pool, means, threshold=4.0)
        status = "REPLAYED"
    except (ApiSchemaError, ValueError, TypeError, KeyError):
        prediction, status = deepcopy(h0), "SELECTION_FAILED"
    if prediction["phase"] != h0["phase"]:
        raise ValueError("interaction subset changed Phase")
    return {"prediction": prediction, "status": status, "means": means, "diagnostics": diagnostics,
            "normalized": reviews, "format_diagnostics": formatting}


def three_phase(current, raw, image_count):
    """All three choices valid, at least two equal non-null choices to switch."""
    labels(current)
    prediction = deepcopy(current)
    before = current["phase"][0]
    errors = {seat: phase_choice_error(raw[seat], image_count) for seat in SELECTED_SEATS}
    decision = {"before": before, "after": before, "reason": "INVALID_PANEL",
                "votes": {str(i): 0 for i in range(7)}, "abstentions": 0, "errors": errors}
    if any(errors.values()):
        return prediction, decision
    for seat in SELECTED_SEATS:
        value = raw[seat]["phase_id"]
        if value is None:
            decision["abstentions"] += 1
        else:
            decision["votes"][str(value)] += 1
    winners = [int(phase) for phase, votes in decision["votes"].items() if votes >= 2]
    decision["reason"] = "NO_MAJORITY"
    if winners:
        prediction["phase"] = [winners[0]]
        decision.update(after=winners[0], reason="CURRENT_PHASE_MAJORITY" if winners[0] == before else "MAJORITY_PHASE_SWITCH")
    return prediction, decision


def prepare(output, source):
    if output.exists():
        raise ValueError("new offline experiment directory required")
    # No GT labels or selected-seat metrics are read before protocol freezing.
    old_plan, done = read(source / "plan.json"), read(source / "completion.json")
    if done.get("fatal_error") or not done.get("closed_utc"):
        raise ValueError("source must be a closed nonfatal experiment")
    paths = [source / f"{name}.json" for name in ("plan", "initial_state", "predictions", "budget", "completion")]
    paths += list((source / "branches").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    sources = dict(old_plan["source_sha256"])
    sources[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source.resolve()),
        "selection": old_plan["selection"], "selected_seats": SELECTED_SEATS, "versions": VERSIONS,
        "primary_arm": "three_seat_four_keep_phase", "secondary_arm": "three_seat_all",
        "interaction_rule": "Use original saved H0, candidate pool and raw responses. Reapply existing schema and evidence validation to GPT/Gemini/DeepSeek only. Exactly three valid judgments per candidate required; mean >=4 adds, <=2 removes. Existing panel.select component and final-schema guards unchanged.",
        "phase_primary": "Reuse original five-seat final Phase verbatim, isolating four-head reviewer count.",
        "phase_secondary": "Same saved GPT/Gemini/DeepSeek Phase responses. All three structurally valid, at least two matching non-null choices; otherwise preserve H0 Phase. Never use GT to select a phase.",
        "threshold": 4.0, "phase_votes": 2, "new_api_calls": 0, "refit_or_reproposal": False,
        "source_sha256": sources,
        "archive_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(set(paths))},
        "scope": "Eight previously inspected Training targets; offline fixed-answer counterfactual, not a newly run three-model API experiment or independent held-out validation.",
        "timing_policy": "Estimate retained panel API duration as the maximum of original per-seat elapsed times for each target. Provider contention, scheduling and future latency changes are not measured. No API saving was actually incurred by this offline replay.",
        "failure_policy": "Retain failed replies and all targets in denominators. Never replace invalid evidence with a neutral score or remove failed targets.",
        "gt_policy": "Protocol is saved first; source raw replay and original five-seat equality verified before score_saved loads task-masked GT. Testing is not used."}
    save(output / "plan.json", plan)
    destination = output / "frozen_source" / Path(__file__).relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, destination)
    print(json.dumps({"prepared": str(output), "profile": PROFILE, "seats": SELECTED_SEATS, "new_api_calls": 0}), flush=True)


def verify_plan(output):
    plan = read(output / "plan.json")
    if (plan["profile"] != PROFILE or tuple(plan["selected_seats"]) != SELECTED_SEATS
            or tuple(plan["versions"]) != VERSIONS or plan["threshold"] != 4.0
            or plan["phase_votes"] != 2 or plan["new_api_calls"] != 0):
        raise ValueError("offline protocol changed")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source code changed after protocol freeze")
    for name, expected in plan["archive_sha256"].items():
        if sha(Path(plan["source_root"]) / name) != expected:
            raise ValueError("original model-answer archive changed")
    return plan


def evaluate(rows, truth, records):
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics, deltas, phase_deltas = {}, [], []
    for version in VERSIONS:
        inputs = [{**truths[row["video_id"], row["frame_id"]], "h0": row["h0"], "h1": None,
                   "final": row[version]} for row in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        for task in TASKS:
            tp = fp = fn = exact = valid = 0
            for row in rows:
                target = truths[row["video_id"], row["frame_id"]]
                if not target["mask"][task]:
                    continue
                predicted, expected = set(row[version][task]), set(target["gt"][task])
                tp += len(predicted & expected)
                fp += len(predicted - expected)
                fn += len(expected - predicted)
                exact += predicted == expected
                valid += 1
            measured = metrics[version]["tasks"][task]
            if (measured["tp"], measured["fp"], measured["fn"], measured["exact_matches"],
                    measured["valid_targets"]) != (tp, fp, fn, exact, valid):
                raise ValueError("independent masked counts differ")
    for row in rows:
        target = truths[row["video_id"], row["frame_id"]]
        deltas.append({"key": row["key"], "comparisons": {
            f"{a}_to_{b}": frame_delta(row[a], row[b], target["gt"], target["mask"]) for a, b in PAIRS}})
        valid = bool(target["mask"]["phase"])
        expected_phase = target["gt"]["phase"] if valid else None
        phase_deltas.append({"key": row["key"], "decision": records[row["key"]]["phase_three"],
            "comparisons": {f"{a}_to_{b}": phase_transition(row[a]["phase"], row[b]["phase"], expected_phase, valid)
                            for a, b in PAIRS}})
    return {"metrics": metrics,
        "comparisons": {f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in deltas])
                        for a, b in PAIRS},
        "phase_transitions": {f"{a}_to_{b}": dict(Counter(d["comparisons"][f"{a}_to_{b}"] for d in phase_deltas))
                              for a, b in PAIRS}}, deltas, phase_deltas


def counterfactual_runtime(rows, records, source_budget):
    totals, detail = Counter(), []
    for row in rows:
        target_calls = [c for c in source_budget["calls"] if c["target"] == row["key"]]
        graph = [c for c in target_calls if c["stage"] == "graph_review"]
        phase = [c for c in target_calls if c["stage"] == "phase_review"]
        if len(graph) != 5 or len(phase) != 5:
            raise ValueError("this timing comparison requires all original five dispatches")
        def maximum(calls, seats=None):
            kept = [c["elapsed_seconds"] for c in calls if seats is None or c["seat"] in seats]
            if not kept or any(not math.isfinite(v) or v < 0 for v in kept):
                raise ValueError("missing or invalid historical elapsed time")
            return max(kept)
        durations = {"graph_five_api_max": maximum(graph), "graph_three_api_max": maximum(graph, SELECTED_SEATS),
            "phase_five_api_max": maximum(phase), "phase_three_api_max": maximum(phase, SELECTED_SEATS)}
        timings = records[row["key"]]["source_timing"]
        graph_local = timings["graph_seconds"] - records[row["key"]]["source_graph_review_seconds"]
        phase_local = timings["phase_seconds"] - records[row["key"]]["source_phase_api_seconds"]
        durations["original_parallel_seconds"] = timings["parallel_seconds"]
        durations["primary_parallel_estimate"] = max(graph_local + durations["graph_three_api_max"], timings["phase_seconds"])
        durations["secondary_parallel_estimate"] = max(graph_local + durations["graph_three_api_max"],
                                                      phase_local + durations["phase_three_api_max"])
        totals.update(durations)
        detail.append({"key": row["key"], **durations})
    def charges_for(mode):
        groups = defaultdict(lambda: defaultdict(Decimal))
        kept = []
        for call in source_budget["calls"]:
            if call["stage"] == "graph_review" and call["seat"] not in SELECTED_SEATS:
                continue
            if mode == "secondary" and call["stage"] == "phase_review" and call["seat"] not in SELECTED_SEATS:
                continue
            kept.append(call)
            groups[call["account"]][call["charge_kind"]] += Decimal(call["charge"])
        return {"retained_historical_calls": len(kept), "charges": {
            account: {kind: str(v) for kind, v in values.items()} for account, values in groups.items()}}
    graph5, graph3 = totals["graph_five_api_max"], totals["graph_three_api_max"]
    return {"new_api_calls": 0, "source_calls": len(source_budget["calls"]), "timing_totals": dict(totals),
        "graph_panel_max_reduction_percent": 100 * (graph5 - graph3) / graph5 if graph5 else None,
        "retained_call_counterfactual": {mode: charges_for(mode) for mode in ("primary", "secondary")},
        "target_timings": detail,
        "caveat": "Historical exact-answer counterfactual, not actual three-seat API latency or billing. Removed reviewers can change contention and response variability. unknown_reserved remains a conservative source-ledger reservation; documented provider-declared zero-charge 403s are not billed spend."}


def score(output, adapter):
    if (output / "completion.json").exists():
        raise ValueError("offline comparison already closed; preserve prior result")
    plan = verify_plan(output)
    source_plan, originals, _, source_records, budget, _ = audit_source(Path(plan["source_root"]), adapter)
    if source_plan["selection"] != plan["selection"]:
        raise ValueError("shared sample selection changed")
    rows, records = [], {}
    for original, selected in zip(originals, plan["selection"], strict=True):
        key = original["key"]
        record = source_records[key]
        interaction = three_interaction(original["h0"], record["graph"], len(selected["images"]))
        three_graph = interaction["prediction"]
        primary = deepcopy(three_graph)
        primary["phase"] = deepcopy(original["final"]["phase"])
        secondary, phase_decision = three_phase(three_graph, record["phase"]["raw_reviews"],
                                               len(source_plan["phase_inputs"][key]["images"]))
        if any(primary[t] != three_graph[t] or secondary[t] != three_graph[t] for t in TASKS[:4]):
            raise ValueError("Phase merge changed interaction predictions")
        row = {k: original[k] for k in ("key", "video_id", "frame_id", "h0")}
        row.update(five_seat_graph=original["graph_only"], five_seat_final=original["final"],
                   three_seat_graph=three_graph, three_seat_four_keep_phase=primary, three_seat_all=secondary)
        rows.append(row)
        records[key] = {"interaction_three": interaction, "phase_three": phase_decision,
            "source_timing": record["timing"], "source_graph_review_seconds": record["graph"]["timing"]["review_seconds"],
            "source_phase_api_seconds": record["phase"]["timing"]["api_seconds"]}
    save(output / "predictions.json", {"targets": rows})
    save(output / "review_analysis.json", records)
    # Original five-seat raw replay/equality and this fixed protocol were
    # verified before the first evaluation target or GT label is loaded here.
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["three_seat_four_keep_phase"]} for r in rows])
    result, deltas, phase_deltas = evaluate(rows, truth, records)
    result.update(profile=PROFILE, targets=len(rows), selected_seats=SELECTED_SEATS,
        runtime_counterfactual=counterfactual_runtime(rows, records, budget),
        audit={"original_five_seat_raw_replay_equals_saved": True, "same_h0_pool_and_answers": True,
            "primary_phase_equals_original_five_seat": True, "metric_counts_independently_checked": True,
            "frozen_protocol_before_gt_scoring": True, "new_api_calls": 0}, scope=plan["scope"])
    for name, value in (("metrics", result), ("frame_deltas", deltas), ("phase_deltas", phase_deltas), ("scored_truth", truth)):
        save(output / f"{name}.json", value)
    verify_plan(output)
    paths = list(output.glob("*.json"))
    save(output / "completion.json", {"closed_utc": now(), "new_api_calls": 0,
        "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in paths}})
    print(json.dumps({"f1": {v: {t: value["micro_f1"] for t, value in arm["tasks"].items()}
                            for v, arm in result["metrics"].items()},
        "changes": result["comparisons"], "phase": result["phase_transitions"],
        "runtime": result["runtime_counterfactual"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.source)
    else:
        score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
