"""Offline scoring of a closed, prompt-only action-cue proposer experiment.

The runner's frozen request replay must pass before query labels are loaded.
Failed predictions stay in every applicable denominator. This module makes no
model requests, and its pure summarizer can be tested with synthetic labels.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.semantic_coordinator import item_error

ARMS = ("control", "action_cue")
VERSIONS = ("h0", *ARMS)
TASKS = tuple(TASK_ID_BOUNDS)
TARGET_COUNT = 24
PAIRS = (("h0", "control"), ("h0", "action_cue"), ("control", "action_cue"))
SUCCESS_RULE = (
    "On all 24 frozen Training targets, action_cue must strictly improve Verb "
    "candidate recall and final Verb micro-F1 over control, without reducing "
    "Verb micro-precision or IVT/Instrument/Target micro-F1. No selector failure; "
    "every target attempted; all 24 native proposal prompt-token pairs verified "
    "and none increases. Shared Phase changes are not credited as prompt benefit."
)


def _prediction(row, version):
    # An empty fallback after failed H0 is unavailable, not a valid empty answer.
    return None if row["h0"] is None else row[version]


def _sets(prediction, task):
    return set(prediction[task]) if prediction is not None else set()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _counts(predictions, expected):
    tp = sum(len(p & g) for p, g in zip(predictions, expected, strict=True))
    fp = sum(len(p - g) for p, g in zip(predictions, expected, strict=True))
    fn = sum(len(g - p) for p, g in zip(predictions, expected, strict=True))
    return {"tp": tp, "fp": fp, "fn": fn,
            "micro_precision": (tp / (tp + fp) if tp + fp else 0.0) if expected else None,
            "micro_recall": (tp / (tp + fn) if tp + fn else 0.0) if expected else None,
            "micro_f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            if expected else None}


def independent_counts(rows, truths, metrics):
    """Cross-check pooled TP/FP/FN, P/R/F1, and exact matches independently."""
    manual = {}
    for version in VERSIONS:
        manual[version] = {}
        for task in TASKS:
            valid = [r for r in rows if truths[r["video_id"], r["frame_id"]]["mask"][task]]
            predictions = [_prediction(r, version) for r in valid]
            predicted = [_sets(p, task) for p in predictions]
            expected = [set(truths[r["video_id"], r["frame_id"]]["gt"][task]) for r in valid]
            exact = sum(p is not None and ids == gt for p, ids, gt in
                        zip(predictions, predicted, expected, strict=True))
            counts = {**_counts(predicted, expected), "valid_targets": len(valid),
                      "failed_predictions": sum(p is None for p in predictions),
                      "exact_matches": exact, "exact_set_accuracy": _ratio(exact, len(valid))}
            for key, value in counts.items():
                actual = metrics[version]["tasks"][task][key]
                if actual != value and not (isinstance(value, float) and isinstance(actual, float)
                                           and math.isclose(actual, value, abs_tol=1e-12)):
                    raise ValueError(f"independent metric mismatch: {version}/{task}/{key}")
            manual[version][task] = counts
        valid_rows = [r for r in rows if any(truths[r["video_id"], r["frame_id"]]["mask"].values())]
        exact = sum(_prediction(r, version) is not None and all(
            _sets(_prediction(r, version), t) == set(truths[r["video_id"], r["frame_id"]]["gt"][t])
            for t in TASKS if truths[r["video_id"], r["frame_id"]]["mask"][t]) for r in valid_rows)
        if metrics[version]["all_valid_heads_exact"] != {
                "valid_targets": len(valid_rows), "exact_matches": exact,
                "accuracy": _ratio(exact, len(valid_rows))}:
            raise ValueError("independent all-head exact mismatch")
    return manual


def candidate_coverage(rows, records, truths):
    fields = ("valid_targets", "failed_h0_targets", "gt_positive", "pool_size", "pool_true",
              "pool_false", "missing_from_pool", "selected_true", "selected_false", "fn",
              "fn_inside_pool", "fn_outside_pool", "new_true_candidates", "new_false_candidates")
    totals = {a: {t: dict.fromkeys(fields, 0) for t in TASKS[:-1]} for a in ARMS}
    details = []
    for row in rows:
        gt = truths[row["video_id"], row["frame_id"]]
        for arm in ARMS:
            propositions = (records.get(row["key"], {}).get(arm, {}).get("pool") or {}).get("propositions", [])
            for task in TASKS[:-1]:
                if not gt["mask"][task]:
                    continue
                available = {p["label_id"] for p in propositions if p["task"] == task}
                expected = set(gt["gt"][task])
                predicted = _sets(_prediction(row, arm), task)
                missing = expected - predicted
                novel = available - _sets(row["h0"], task)
                counts = {"valid_targets": 1, "failed_h0_targets": int(row["h0"] is None),
                          "gt_positive": len(expected), "pool_size": len(available),
                          "pool_true": len(available & expected), "pool_false": len(available - expected),
                          "missing_from_pool": len(expected - available),
                          "selected_true": len(predicted & expected), "selected_false": len(predicted - expected),
                          "fn": len(missing), "fn_inside_pool": len(missing & available),
                          "fn_outside_pool": len(missing - available),
                          "new_true_candidates": len(novel & expected),
                          "new_false_candidates": len(novel - expected)}
                for key, value in counts.items():
                    totals[arm][task][key] += value
                details.append({"key": row["key"], "arm": arm, "task": task,
                    "gt": sorted(expected), "pool": sorted(available), "prediction": sorted(predicted),
                    "prediction_available": _prediction(row, arm) is not None,
                    "missing_from_pool": sorted(expected - available),
                    "fn_inside_pool": sorted(missing & available), "fn_outside_pool": sorted(missing - available),
                    "new_true_candidates": sorted(novel & expected), "new_false_candidates": sorted(novel - expected)})
    for arm in ARMS:
        for counts in totals[arm].values():
            counts["pool_recall"] = _ratio(counts["pool_true"], counts["gt_positive"])
            counts["pool_precision"] = _ratio(counts["pool_true"], counts["pool_size"])
    return {"arms": totals, "targets": details,
            "note": "Counts are target-label occurrences, not distinct class counts. Failed H0 stays in coverage denominators."}


def verb_counts(rows, records, truths):
    result = {version: {} for version in VERSIONS}
    valid = [r for r in rows if truths[r["video_id"], r["frame_id"]]["mask"]["verb"]]
    lower, upper = TASK_ID_BOUNDS["verb"]
    for version in VERSIONS:
        for label_id in range(lower, upper + 1):
            predicted = [{label_id} & _sets(_prediction(r, version), "verb") for r in valid]
            expected = [{label_id} & set(truths[r["video_id"], r["frame_id"]]["gt"]["verb"]) for r in valid]
            counts = {"label_id": label_id, "valid_targets": len(valid), **_counts(predicted, expected)}
            counts["gt_positive"] = counts["tp"] + counts["fn"]
            if version in ARMS:
                available = [{p["label_id"] for p in (records.get(r["key"], {}).get(version, {}).get("pool") or {}).get(
                    "propositions", []) if p["task"] == "verb"} for r in valid]
                counts["pool_true"] = sum(bool(g) and label_id in pool for g, pool in zip(expected, available, strict=True))
                counts["pool_recall"] = _ratio(counts["pool_true"], counts["gt_positive"])
                counts["fn_inside_pool"] = sum(bool(g - p) and label_id in pool for p, g, pool in
                                               zip(predicted, expected, available, strict=True))
                counts["fn_outside_pool"] = counts["fn"] - counts["fn_inside_pool"]
            result[version][str(label_id)] = counts
    return result


def review_validity(rows, records):
    result = {a: {"arm_statuses": Counter(), "panels_present": 0, "panels_absent": 0,
                  "shared_panels": 0, "seat_item_counts": {s: Counter() for s in SEATS},
                  "seat_response_counts": {s: Counter() for s in SEATS},
                  "valid_five_seat_candidates": 0, "unavailable_candidate_means": 0} for a in ARMS}
    for row in rows:
        for arm in ARMS:
            record = records.get(row["key"], {}).get(arm, {})
            group = result[arm]
            group["arm_statuses"][record.get("status", row.get("statuses", {}).get(arm, "MISSING_RECORD"))] += 1
            group["shared_panels"] += bool(record.get("shared_from"))
            reviews = record.get("reviews")
            if reviews is None:
                group["panels_absent"] += 1
                continue
            group["panels_present"] += 1
            pool = (record.get("pool") or {}).get("propositions", [])
            ids = {p["id"] for p in pool}
            for seat in SEATS:
                response = reviews.get(seat) if isinstance(reviews, dict) else None
                outer_valid = (isinstance(response, dict) and set(response) == {"judgments"}
                               and isinstance(response["judgments"], dict)
                               and not set(response["judgments"]) - ids)
                judgments = response["judgments"] if outer_valid else {}
                errors = [item_error(judgments.get(p["id"]), p["task"], 3) for p in pool]
                group["seat_item_counts"][seat].update(e or "VALID" for e in errors)
                group["seat_response_counts"][seat]["VALID" if outer_valid and not any(errors) else "INVALID"] += 1
            means = record.get("means") or {}
            group["valid_five_seat_candidates"] += sum(means.get(p["id"]) is not None for p in pool)
            group["unavailable_candidate_means"] += sum(means.get(p["id"]) is None for p in pool)
    return result


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _distribution(values):
    values = [v for v in values if _number(v)]
    return {"n": len(values), "sum": sum(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def accounting(rows, records, ledger, done):
    groups = defaultdict(lambda: {"calls": 0, "charges_by_account_and_kind": defaultdict(lambda: defaultdict(Decimal)),
        "statuses": Counter(), "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
        "calls_with_prompt_tokens": 0, "calls_with_completion_tokens": 0,
        "calls_with_reasoning_tokens": 0, "elapsed_seconds": []})
    for call in ledger["calls"]:
        arm = next((a for a in ARMS if call["stage"].startswith(a + "_")), "shared")
        for name in ("all", "arm:" + arm, "stage:" + call["stage"]):
            group = groups[name]
            group["calls"] += 1
            group["charges_by_account_and_kind"][call["account"]][call.get("charge_kind", "unknown_reserved")] += Decimal(str(call["charge"]))
            group["statuses"][call["status"]] += 1
            usage = call.get("usage") or {}
            for token in ("prompt_tokens", "completion_tokens"):
                if type(usage.get(token)) is int and usage[token] >= 0:
                    group[token] += usage[token]
                    group["calls_with_" + token] += 1
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if type(reasoning) is int and reasoning >= 0:
                group["reasoning_tokens"] += reasoning
                group["calls_with_reasoning_tokens"] += 1
            group["elapsed_seconds"].append(call.get("elapsed_seconds"))
    costs = {name: {**group, "statuses": dict(group["statuses"]),
                   "charges_by_account_and_kind": {a: {kind: str(value) for kind, value in charges.items()}
                                                    for a, charges in group["charges_by_account_and_kind"].items()},
                   "elapsed_seconds": _distribution(group["elapsed_seconds"])} for name, group in groups.items()}
    timing = {}
    for arm in ARMS:
        data = [records.get(row["key"], {}).get(arm, {}) for row in rows]
        timing[arm] = {"proposal_seconds": _distribution(r.get("proposal_seconds") for r in data),
                       "fresh_panel_seconds": _distribution(r.get("panel_seconds") for r in data if not r.get("shared_from")),
                       "shared_panels_excluded": sum(bool(r.get("shared_from")) for r in data)}
    pairs = []
    for row in rows:
        calls = {arm: [c for c in ledger["calls"] if c["target"] == row["key"] and c["stage"] == arm + "_proposal"] for arm in ARMS}
        tokens = {}
        for arm in ARMS:
            value = (calls[arm][0].get("usage") or {}).get("prompt_tokens") if len(calls[arm]) == 1 else None
            tokens[arm] = value if type(value) is int and value >= 0 else None
        verified = all(tokens[a] is not None for a in ARMS)
        pairs.append({"key": row["key"], **tokens, "native_pair_verified": verified,
                      "delta": tokens["action_cue"] - tokens["control"] if verified else None,
                      "does_not_increase": tokens["action_cue"] <= tokens["control"] if verified else None})
    paired = [p for p in pairs if p["native_pair_verified"]]
    return {"dispatched": costs, "timing": timing, "elapsed_seconds": done.get("elapsed_seconds"),
            "post_calls": len(ledger["calls"]),
            "proposal_native_prompt_tokens": {"pairs": pairs, "verified_pairs": len(paired),
                "control_total_verified_pairs": sum(p["control"] for p in paired),
                "action_cue_total_verified_pairs": sum(p["action_cue"] for p in paired),
                "increased_pairs": sum(p["delta"] > 0 for p in paired)},
            "notes": ["Token counts come only from the archived provider usage; missing usage is not zero or a text proxy.",
                      "Accounts/currencies, native charges, conservative estimates and unknown reserves remain separate.",
                      "Shared H0/Phase and identical-pool panels are charged once; arm charges are observed expenses, not equal-work savings.",
                      "Sum of request durations is not wall time; only completion elapsed_seconds measures the whole run."]}


def summarize(plan, rows, records, truth, ledger, done):
    """Calculate only from frozen predictions and supplied post-closure labels."""
    if tuple(plan.get("arms", ())) != ARMS:
        raise ValueError("unexpected experiment arms")
    identities = lambda values: [(r["video_id"], r["frame_id"]) for r in values]
    if identities(rows) != identities(plan["selection"]) or len(set(identities(rows))) != len(rows):
        raise ValueError("predictions do not preserve every planned unique target")
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    if len(truths) != len(truth) or set(truths) != set(identities(rows)):
        raise ValueError("GT must cover exactly the frozen targets")
    for row in rows:
        left, right = (_prediction(row, a) for a in ARMS)
        if left is not None and right is not None and left["phase"] != right["phase"]:
            raise ValueError("shared Phase differs between arms")
    metrics = {}
    for version in VERSIONS:
        data = [{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None,
                 "final": _prediction(r, version)} for r in rows]
        metrics[version] = compute_repair_comparison(data)["arms"]["final"]
    manual = independent_counts(rows, truths, metrics)
    deltas = {left + "_to_" + right: [{"key": r["key"], **frame_delta(
        _prediction(r, left), _prediction(r, right), truths[r["video_id"], r["frame_id"]]["gt"],
        truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows] for left, right in PAIRS}
    comparisons = {name: summarize_deltas(values) for name, values in deltas.items()}
    coverage = candidate_coverage(rows, records, truths)
    validity = review_validity(rows, records)
    runtime = accounting(rows, records, ledger, done)
    control, action = (metrics[a]["tasks"] for a in ARMS)
    def compare(left, right, strict=False):
        return left is not None and right is not None and (left > right if strict else left >= right)
    attempted = {c["target"] for c in ledger["calls"] if c["stage"] == "h0"}
    tokens = runtime["proposal_native_prompt_tokens"]
    checks = {
        "all_24_planned_targets_retained_and_attempted": len(rows) == TARGET_COUNT and attempted == {r["key"] for r in rows},
        "verb_candidate_recall_strictly_increases": compare(coverage["arms"]["action_cue"]["verb"]["pool_recall"],
                                                            coverage["arms"]["control"]["verb"]["pool_recall"], True),
        "verb_f1_strictly_increases": compare(action["verb"]["micro_f1"], control["verb"]["micro_f1"], True),
        "verb_precision_does_not_decrease": compare(action["verb"]["micro_precision"], control["verb"]["micro_precision"]),
        **{task + "_f1_does_not_decrease": compare(action[task]["micro_f1"], control[task]["micro_f1"])
           for task in ("instrument", "target", "ivt")},
        "no_selector_failure": all("SELECTION_FAILED" not in validity[a]["arm_statuses"] and
                                   "SELECTOR_FAILED" not in validity[a]["arm_statuses"] for a in ARMS),
        "all_24_native_proposal_token_pairs_verified": tokens["verified_pairs"] == TARGET_COUNT,
        "no_verified_proposal_prompt_token_increase": bool(tokens["verified_pairs"]) and tokens["increased_pairs"] == 0,
        "inference_closed_without_fatal_error": bool(done.get("closed_utc")) and not done.get("fatal_error"),
    }
    result = {"profile": plan.get("profile"), "targets": len(rows), "metrics": metrics,
              "comparisons": comparisons, "candidate_coverage": coverage["arms"],
              "per_verb": verb_counts(rows, records, truths), "review_validity": validity, "runtime": runtime,
              "predeclared_success": {"rule": plan.get("success_rule", SUCCESS_RULE), "checks": checks,
                                      "confirmed": all(checks.values())},
              "audit": {"all_planned_targets_retained": True, "independent_counts_passed": True,
                        "phase_shared_between_arms": True, "failed_h0_predictions": sum(r["h0"] is None for r in rows)},
              "independent_counts": manual,
              "limitations": ["This is a small Training development experiment, not independent-video generalization or Gate OOF.",
                  "The model generates different candidate pools; coverage improvement alone does not establish final accuracy benefit.",
                  "Both arms share H0 and Phase; H0-to-arm Phase changes cannot be attributed to proposal wording.",
                  "Identical-pool panel reuse is paired reuse, not an independent model or timing replicate.",
                  "Failures and fallback predictions remain scored; native usage may be unavailable after request failures."]}
    return result, deltas, coverage


def score(output, adapter):
    # Import after module initialization so the runner can dispatch to this scorer.
    from scripts.run_action_cue_candidate_trial import audit

    output = Path(output)
    plan, rows, records, ledger, done = audit(output, adapter)
    if not done.get("closed_utc") or ledger.get("stopped") is not True:
        raise ValueError("GT scoring requires closed inference")
    # The first query-label read is below the runner's full frozen replay audit.
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": _prediction(r, "control")} for r in rows])
    result, deltas, coverage = summarize(plan, rows, records, truth, ledger, done)
    result["audit"].update(raw_replay_passed=True, GT_loaded_after_closed_inference=True)
    # Detect concurrent mutation before writing any results; the second audit reads no query GT.
    audit(output, adapter)
    summary = {k: result[k] for k in ("profile", "targets", "metrics", "comparisons", "candidate_coverage",
                                     "predeclared_success", "audit")}
    summary["runtime"] = {k: result["runtime"][k] for k in ("post_calls", "elapsed_seconds", "dispatched")}
    for name, payload in (("metrics", result), ("summary", summary), ("scored_truth", truth),
                          ("frame_deltas", deltas), ("candidate_coverage", coverage)):
        save(output / f"{name}.json", payload)
    print(json.dumps({"targets": len(rows), "f1": {a: {t: m["micro_f1"] for t, m in metrics["tasks"].items()}
        for a, metrics in result["metrics"].items()}, "predeclared_success": result["predeclared_success"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
