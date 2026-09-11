"""Offline attribution of a closed Validation confirmation; never calls an API."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_prior_panel_trial import read, save
from scripts.run_validation_confirmation import decide_joint
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import COMPONENTS

TASKS = ("instrument", "verb", "target", "ivt", "phase")


def label_name(task, label):
    if task == "ivt":
        return " / ".join(_TASK_NAMES[t][v] for t, v in COMPONENTS[label].items())
    return _TASK_NAMES[task][label]


def auc(values):
    positive = [v for v, yes in values if yes]
    negative = [v for v, yes in values if not yes]
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def error_count(prediction, truth):
    return sum(len(set(prediction[t]) ^ set(truth[t] or [])) for t in TASKS)


def subset_metrics(rows, truths):
    result = {}
    for arm in ("h0", "control", "v2.0.0", "v2.0.1"):
        metrics = {}
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                got, gt = set(row["predictions"][arm][task]), set(truths[row["key"]][task] or [])
                tp += len(got & gt)
                fp += len(got - gt)
                fn += len(gt - got)
            metrics[task] = {"f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                              "precision": 100 * tp / (tp + fp) if tp + fp else 0,
                              "errors": fp + fn}
        result[arm] = {"mean_f1": sum(m["f1"] for m in metrics.values()) / 5,
                        "mean_precision": sum(m["precision"] for m in metrics.values()) / 5,
                        "errors": sum(m["errors"] for m in metrics.values()), "tasks": metrics}
    return result


def analyze(output):
    plan = read(output / "plan.json")
    completion = read(output / "completion.json")
    scored = read(output / "metrics.json")
    if completion.get("fatal_error") or scored["targets"] != len(plan["selection"]):
        raise ValueError("closed complete scored confirmation required")
    truths = {f"{r['video_id']}_{r['frame_id']}": r["gt"] for r in read(output / "scored_truth.json")}
    initials = read(output / "initials.json")
    rows = read(output / "predictions.json")["targets"]
    summary, details = {}, []
    for arm in ("control", "v2.0.0", "v2.0.1"):
        coverage = {t: Counter() for t in TASKS[:4]}
        edits = {t: Counter() for t in TASKS}
        misses = {t: Counter() for t in TASKS[:4]}
        false_positives = {t: Counter() for t in TASKS[:4]}
        frame_counts, phase_pairs = Counter(), Counter()
        scores, seat_scores = defaultdict(list), defaultdict(list)
        invalids, format_errors = defaultdict(Counter), defaultdict(Counter)
        nearest_ivt_mismatches = Counter()
        oracle_gain = 0
        for row in rows:
            key = row["key"]
            gt, h0, final = truths[key], row["predictions"]["h0"], row["predictions"][arm]
            record = read(output / "targets" / key / "result.json")
            outcome = record[arm]
            for seat, diagnostics in outcome.get("format_diagnostics", {}).items():
                for errors in diagnostics.get("errors", {}).values():
                    format_errors[seat].update([errors] if isinstance(errors, str) else errors)
            if arm != "control":
                replay = decide_joint(record["joint_raw"], record["joint_pool"], h0,
                                      3 if arm == "v2.0.1" else 5)
                if replay["prediction"] != final or replay["means"] != outcome["means"]:
                    raise ValueError("joint replay mismatch: " + key + "/" + arm)
            means = outcome["means"]
            pool = initials[key]["pool"] if arm == "control" else record["joint_pool"]
            ids = {(p["task"], p["label_id"]): p["id"] for p in pool["propositions"]}
            delta = error_count(final, gt) - error_count(h0, gt)
            category = "helped" if delta < 0 else "harmed" if delta > 0 else "unchanged" if final == h0 else "changed_tie"
            frame_counts[category] += 1
            oracle_gain += max(0, -delta)
            changes, missing_details, false_details = [], [], []
            for task in TASKS:
                truth, before, after = set(gt[task] or []), set(h0[task]), set(final[task])
                for label in sorted(before ^ after):
                    addition = label in after
                    good = (label in truth) == addition
                    kind = ("good" if good else "bad") + ("_add" if addition else "_delete")
                    edits[task][kind] += 1
                    pid = ids.get((task, label))
                    changes.append({"task": task, "label": label, "name": label_name(task, label),
                                    "kind": kind, "mean": means.get(pid),
                                    "seat_scores": outcome["diagnostics"].get(pid, {}).get("scores")})
                if task == "phase":
                    phase_pairs[f"{gt[task][0]}->{final[task][0]}"] += 1
                    continue
                available = {v for t, v in ids if t == task}
                coverage[task].update(gt=len(truth), h0_hits=len(truth & before),
                                      pool_hits=len(truth & available), final_hits=len(truth & after))
                for label in sorted(truth - after):
                    pid = ids.get((task, label))
                    mean = means.get(pid)
                    blocks = []
                    if label in before:
                        reason = "correct_h0_deleted"
                    elif pid is None:
                        reason = "not_in_candidate_pool"
                    elif mean is None:
                        reason = "insufficient_valid_ratings"
                    elif mean < 4:
                        reason = "own_rating_below_add_threshold"
                    elif task == "ivt":
                        blocks = [{"task": t, "label": v, "name": label_name(t, v),
                                   "mean": means.get(ids.get((t, v)))}
                                  for t, v in COMPONENTS[label].items()
                                  if means.get(ids.get((t, v))) is None or means[ids[t, v]] < 4]
                        reason = "component_support_veto" if blocks else "unexplained_selector_miss"
                    else:
                        reason = "unexplained_selector_miss"
                    misses[task][reason] += 1
                    missing_details.append({"task": task, "label": label, "name": label_name(task, label),
                                            "reason": reason, "mean": mean, "component_blocks": blocks,
                                            "seat_scores": outcome["diagnostics"].get(pid, {}).get("scores")})
                for label in sorted(after - truth):
                    pid = ids.get((task, label))
                    reason = "wrong_h0_retained" if label in before else "wrong_candidate_added"
                    false_positives[task][reason] += 1
                    false_details.append({"task": task, "label": label, "name": label_name(task, label),
                                          "reason": reason, "mean": means.get(pid),
                                          "seat_scores": outcome["diagnostics"].get(pid, {}).get("scores")})
                    if task == "ivt" and truth:
                        components = COMPONENTS[label]
                        same_instrument = [v for v in truth if COMPONENTS[v]["instrument"] == components["instrument"]]
                        alternatives = same_instrument or sorted(truth)
                        mismatches = {v: [t for t, c in components.items() if COMPONENTS[v][t] != c]
                                      for v in alternatives}
                        distance = min(map(len, mismatches.values()))
                        closest = {v: m for v, m in mismatches.items() if len(m) == distance}
                        patterns = sorted({"+".join(m) for m in closest.values()})
                        nearest_ivt_mismatches[patterns[0] if len(patterns) == 1 else "ambiguous:" + "|".join(patterns)] += 1
                        false_details[-1]["closest_gt_ivts"] = [{"label": v, "name": label_name("ivt", v),
                                                                   "different_components": m} for v, m in closest.items()]
            for p in pool["propositions"]:
                pid, task = p["id"], p["task"]
                yes = p["label_id"] in (gt[task] or [])
                if means.get(pid) is not None:
                    scores[task].append((means[pid], yes))
                d = outcome["diagnostics"][pid]
                for seat, value in zip(SEATS, d["scores"], strict=True):
                    if value is not None:
                        seat_scores[seat, task].append((value, yes))
                for seat in d.get("invalid", {}):
                    invalids[seat][task] += 1
            details.append({"key": key, "arm": arm, "error_delta": delta, "outcome": category,
                            "h0_errors": error_count(h0, gt), "final_errors": error_count(final, gt),
                            "h0": h0, "final": final, "gt": gt,
                            "changes": changes, "misses": missing_details, "false_positives": false_details,
                            "phase_decision": outcome.get("phase_decision")})
        score_stats = {}
        for task, values in scores.items():
            pos, neg = [v for v, y in values if y], [v for v, y in values if not y]
            score_stats[task] = {"auc": auc(values), "correct_n": len(pos), "incorrect_n": len(neg),
                                 "correct_mean": sum(pos) / len(pos) if pos else None,
                                 "incorrect_mean": sum(neg) / len(neg) if neg else None,
                                 "correct_mean_at_least_4": sum(v >= 4 for v in pos),
                                 "incorrect_mean_at_least_4": sum(v >= 4 for v in neg),
                                 "correct_mean_at_most_2": sum(v <= 2 for v in pos)}
        summary[arm] = {"frame_outcomes": frame_counts, "coverage": coverage, "edits": edits,
                        "miss_causes": misses, "false_positive_causes": false_positives,
                        "rating_discrimination": score_stats,
                        "seat_auc": {f"{s}/{t}": auc(v) for (s, t), v in seat_scores.items()},
                        "invalid_candidate_ratings_by_seat": dict(invalids), "phase_confusion": phase_pairs,
                        "format_errors_by_seat": dict(format_errors),
                        "false_ivt_nearest_gt_component_differences": nearest_ivt_mismatches,
                        "oracle_error_reduction": oracle_gain,
                        "oracle_note": "Retrospective GT oracle; not a trained or deployable Gate."}
    rule_changed = [r["key"] for r in rows if r["predictions"]["v2.0.0"] != r["predictions"]["v2.0.1"]]
    recovered = []
    for row in rows:
        record = read(output / "targets" / row["key"] / "result.json")
        for pid, value in record["v2.0.1"]["means"].items():
            if value is not None and record["v2.0.0"]["means"][pid] is None:
                task, label = pid.rsplit("_", 1)
                recovered.append({"key": row["key"], "candidate": pid, "mean": value,
                                  "correct": int(label) in (truths[row["key"]][task] or [])})
    calls = read(output / "budget.json")["calls"]
    money = defaultdict(lambda: defaultdict(Decimal))
    for call in calls:
        account, kind = call["account"], call["charge_kind"]
        value = Decimal(call["charge"])
        folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        if kind == "unknown_reserved" and (folder / "response.json").exists():
            error = read(folder / "response.json").get("body", {}).get("error", {})
            if isinstance(error, dict) and "No credits were charged" in error.get("message", ""):
                money[account]["explicit_unbilled_reserved"] += value
                continue
        money[account][kind] += value
    prior = read(Path(plan["source_archive"]) / "priors" / f"{plan['video']}.json")
    prior_support = {t: {"valid_frames": p["global"][0]["valid_frames"],
                          "valid_videos": p["global"][0]["valid_videos"],
                          "eligible_classes": sum(r["eligible"] for r in p["global"]),
                          "class_count": len(p["global"])} for t, p in prior["tasks"].items()}
    ivt_prior = {r["id"]: r for r in prior["tasks"]["ivt"]["global"]}
    prior_misses = Counter()
    hint_misses = Counter()
    for d in details:
        if d["arm"] == "v2.0.1":
            for miss in d["misses"]:
                if miss["task"] == "ivt" and miss["reason"] == "not_in_candidate_pool":
                    prior_misses["eligible_in_training_prior" if ivt_prior[miss["label"]]["eligible"]
                                 else "ineligible_in_training_prior"] += 1
                    hints = {r["ivt_id"] for r in initials[d["key"]]["hints"]["packet"]["relations"]}
                    hint_misses["hinted_but_not_proposed" if miss["label"] in hints
                                else "not_hinted_and_not_proposed"] += 1
    failed_targets = sorted({r["target"] for r in calls if not r["status"].startswith("JSON_PARSED")})
    complete_rows = [r for r in rows if r["key"] not in failed_targets]
    report = {"targets": len(rows), "summary": summary, "rule_changed_targets": rule_changed,
              "joint_replay_verified": 2 * len(rows),
              "recovered_means": recovered,
              "prior_support": prior_support, "out_of_pool_ivt_prior_eligibility": prior_misses,
              "out_of_pool_ivt_hint_funnel": hint_misses,
              "accounting": {a: {k: str(v) for k, v in buckets.items()} for a, buckets in money.items()},
              "complete_response_sensitivity": {
                  "targets": len(complete_rows), "excluded_targets": failed_targets,
                  "metrics": subset_metrics(complete_rows, truths),
                  "note": "Secondary matched subset, excluding any target with a failed transport/JSON response in any arm. Primary results keep every target."},
              "api_statuses_by_stage_seat": dict(Counter(f"{r['stage']}/{r['seat']}/{r['status']}" for r in calls)),
              "details": details, "additional_api_calls": 0}
    save(output / "diagnosis.json", report)
    print({"targets": len(rows), "rule_changed_targets": rule_changed,
           "frame_outcomes": {a: dict(s["frame_outcomes"]) for a, s in summary.items()}})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    analyze(parser.parse_args().output)
