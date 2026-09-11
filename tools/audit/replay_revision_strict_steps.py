"""GT-blind single-step attribution on a completed paired repair trial.

Every alternative starts from the ACTUAL round's before state and pool. These
are conditional single-step shadows, not counterfactual closed-loop runs. All
predictions are written before loading saved GT/masks for independent scoring.
"""

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_complete_gt_semantic_trial import normalize_review_wire
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)
from surgical_agent.research.verification.review_normalization import normalize_review
from tools.audit.audit_repair_revision import (
    change_counts,
    indexed,
    metrics,
    read,
    require,
    sha,
)

ARM = "single_proposer_tolerant"
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VARIANTS = ("actual_tolerant_step", "old_schema_step", "old_transport_schema_step", "duplicate_core_compat_step")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sources(source):
    result = {}
    for row in read(source / "budget.json")["calls"]:
        key = row["target"], row["stage"], row["seat"]
        result[key] = source / "calls" / f"{row['index']:03d}_{key[0]}_{key[1]}_{key[2]}"
    path = source / "cache_hits.json"
    for hit in read(path) if path.exists() else []:
        key = hit["target"], hit["stage"], hit["seat"]
        require(key not in result, "cache and paid call collide")
        folder = Path(hit["source"])
        require(sha(folder / "request.json") == hit["request_sha256"], "cached request digest differs")
        require(sha(folder / "response.json") == hit["response_sha256"], "cached response digest differs")
        result[key] = folder
    return result


def parsed_from_saved(folder):
    """Never recover refused, truncated, wrong-provider or failed transport."""
    diagnostic = {"source": str(folder), "eligible_transport": False}
    if folder is None or not (folder / "response.json").exists():
        return None, None, diagnostic
    response, record = read(folder / "response.json"), read(folder / "record.json")
    raw = response["body"]
    diagnostic["response_sha256"] = sha(folder / "response.json")
    if (response["http_status"] != 200 or record["status"] not in {
            "JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED", "SAFE_JSON_REJECTED"}):
        return None, None, diagnostic
    body = read(folder / "request.json")
    require(raw.get("model") == body["model"], "unexpected response model")
    require(raw["choices"][0]["finish_reason"] == "stop" and not raw["choices"][0]["message"].get("refusal"),
            "cannot recover truncation or refusal")
    diagnostic["eligible_transport"] = True
    text = raw["choices"][0]["message"]["content"]
    try:
        old = json.loads(text)
    except (ValueError, TypeError):
        old = None
        diagnostic["old_json_loads_failed"] = True
    else:
        diagnostic["old_json_loads_failed"] = False
    compatible, diagnostic["compatible_parse"] = parse_review_json_compatible(text)
    diagnostic["original_call_status"] = record["status"]
    return old, compatible, diagnostic


def decide(record, reviews, image_count, *, tolerant):
    if tolerant:
        normalized = {seat: normalize_review(reviews[seat], record["pool"], seat=seat, image_count=image_count)[0]
                      for seat in SEATS}
    else:
        normalized = {seat: normalize_review_wire(seat, reviews[seat], record["pool"]) for seat in SEATS}
    if not record["pool"]["propositions"]:
        return deepcopy(record["before"]), {}, {}
    means, diagnostics = panel.aggregate(normalized, record["pool"], image_count=image_count)
    try:
        after = panel.select(record["before"], record["pool"], means)
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        # Same preservation behavior as the actual runner's SELECT_FAILED path.
        after = deepcopy(record["before"])
    return after, means, diagnostics


def edits(before, after):
    return {(task, label, direction) for task in TASKS[:4] for direction, values in (
        ("add", set(after[task]) - set(before[task])),
        ("remove", set(before[task]) - set(after[task]))) for label in values}


def describe_edits(values):
    return [{"task": task, "label_id": label, "direction": direction} for task, label, direction in sorted(values)]


def one_step(key, record, image_count, locations):
    old_transport, compatible, parsing = {}, {}, {}
    for seat in SEATS:
        folder = locations.get((key, f"{ARM}_review_{record['round']}", seat))
        old_transport[seat], compatible[seat], parsing[seat] = parsed_from_saved(folder)
    outputs, means, diagnostics = {}, {}, {}
    inputs = {"actual_tolerant_step": record["raw_reviews"], "old_schema_step": record["raw_reviews"],
              "old_transport_schema_step": old_transport, "duplicate_core_compat_step": compatible}
    for name in VARIANTS:
        outputs[name], means[name], diagnostics[name] = decide(
            record, inputs[name], image_count, tolerant=name in {"actual_tolerant_step", "duplicate_core_compat_step"})
    require(outputs["actual_tolerant_step"] == record["after"], "actual tolerant decision cannot be reproduced")
    actual_edits = edits(record["before"], record["after"])
    comparisons = {}
    for variant in VARIANTS[1:]:
        other_edits = edits(record["before"], outputs[variant])
        recovered = {}
        for pid, actual_mean in means["actual_tolerant_step"].items():
            other_mean = means[variant][pid]
            actual_invalid = diagnostics["actual_tolerant_step"][pid]["invalid"]
            other_invalid = diagnostics[variant][pid]["invalid"]
            if actual_mean != other_mean or actual_invalid != other_invalid:
                recovered[pid] = {"actual_mean": actual_mean, "alternative_mean": other_mean,
                                  "actual_invalid": actual_invalid, "alternative_invalid": other_invalid,
                                  "newly_usable_seats_in_actual": sorted(set(other_invalid) - set(actual_invalid)),
                                  "newly_usable_seats_in_alternative": sorted(set(actual_invalid) - set(other_invalid))}
        comparisons[variant] = {"same_edits": describe_edits(actual_edits & other_edits),
                                "only_actual_edits": describe_edits(actual_edits - other_edits),
                                "only_alternative_edits": describe_edits(other_edits - actual_edits),
                                "changed_evidence": recovered}
        for category in ("only_actual_edits", "only_alternative_edits"):
            for edit in comparisons[variant][category]:
                task, label = edit["task"], edit["label_id"]
                dependencies = {f"{task}_{label}"}
                if task == "ivt":
                    dependencies.update(f"{t}_{c}" for t, c in COMPONENTS[label].items())
                else:
                    dependencies.update(f"ivt_{c}" for output in outputs.values() for c in output["ivt"]
                                        if COMPONENTS[c][task] == label)
                edit["changed_evidence_in_direct_or_component_dependencies"] = sorted(dependencies & recovered.keys())
    return {"key": key, "round": record["round"], "before": record["before"], "outputs": outputs,
            "comparisons": comparisons, "parse_diagnostics": parsing,
            "original_round_status": record["status"], "same_actual_before_and_pool": True}


def assess_edits(items, truth):
    totals = Counter()
    result = []
    for item in items:
        task = item["task"]
        scored = {**item, "valid_gt": truth["mask"][task]}
        if truth["mask"][task]:
            present = item["label_id"] in truth["gt"][task]
            helpful = present if item["direction"] == "add" else not present
            scored["outcome"] = "beneficial" if helpful else "harmful"
            totals[scored["outcome"]] += 1
        else:
            scored["outcome"] = "unscored"
            totals["unscored"] += 1
        result.append(scored)
    return result, dict(totals)


def replay(source, output):
    require(not output.exists(), "new output directory required")
    completion, budget = read(source / "completion.json"), read(source / "budget.json")
    require(budget["stopped"] is True, "source inference must be finished")
    plan, states = read(source / "plan.json"), read(source / "inference_states.json")
    selection = {r["key"]: r for r in plan["selection"]}
    require(set(selection) == set(states), "source states omit selected targets")
    locations = sources(source)
    source_hashes = {str(path): sha(path) for path in (source / "plan.json", source / "inference_states.json", source / "completion.json", source / "budget.json")}
    rounds, steps = {}, []
    for number in range(1, completion["last_snapshot"] + 1):
        snapshot = indexed(read(source / f"round_{number}_predictions.json"))
        require(set(snapshot) == set(selection), "source snapshot omits targets")
        rows = []
        for key, state in states.items():
            record = next((r for r in state["arms"][ARM]["history"] if r["round"] == number), None)
            if record is not None:
                step = one_step(key, record, len(selection[key]["causal_frame_ids"]), locations)
                steps.append(step)
                variants = step["outputs"]
                before = step["before"]
            else:
                final = snapshot[key]["arms"][ARM]["final"]
                variants = dict.fromkeys(VARIANTS, final)
                before = final
            rows.append({"video_id": state["video_id"], "frame_id": state["frame_id"], "h0": state["h0"],
                         "actual_before": before, "actual_step_present": record is not None,
                         "variants": variants, "actual_rounds": snapshot[key]["arms"][ARM]["actual_rounds"]})
        rounds[str(number)] = rows
    # This artifact exists in full before reading a single GT label or score.
    predictions = {"source": str(source), "source_sha256": source_hashes, "new_api_calls": 0,
                   "kind": "conditional single-step shadows, not independent closed loops",
                   "gt_labels_read_before_predictions_saved": False, "rounds": rounds, "steps": steps,
                   "variants": list(VARIANTS), "historical_transport_note": "old json.loads includes its historical last-key-wins behavior; this is diagnostic, not a recommended parser"}
    save(output / "predictions_before_gt.json", predictions)
    frozen_hash = sha(output / "predictions_before_gt.json")
    scored_rounds, attribution = {}, []
    for number, rows in rounds.items():
        truth_path = source / "scores" / f"round_{number}_{ARM}_truth.json"
        truths = indexed(read(truth_path))
        require(set(truths) == {f"{r['video_id']}_{r['frame_id']}" for r in rows}, "truth subset differs from full selection")
        reports = {}
        for variant in VARIANTS:
            scored = []
            for row in rows:
                key = f"{row['video_id']}_{row['frame_id']}"
                truth = truths[key]
                require(row["h0"] == truth["h0"], "GT joining changed frozen H0")
                scored.append({**truth, "final": row["variants"][variant], "actual_before": row["actual_before"],
                               "actual_after": row["variants"]["actual_tolerant_step"]})
            reports[variant] = {"metrics": metrics(scored, "final"),
                                "changes_from_h0": change_counts(scored),
                                "changes_from_actual_before": change_counts(scored, before="actual_before"),
                                "changes_from_actual_after": change_counts(scored, before="actual_after")}
        scored_rounds[number] = {"targets": len(rows), "actual_steps": sum(r["actual_step_present"] for r in rows),
                                  "truth_sha256": sha(truth_path), "h0_metrics": metrics(list(truths.values()), "h0"),
                                  "variants": reports}
        for step in [s for s in steps if str(s["round"]) == number]:
            item = {"key": step["key"], "round": step["round"], "comparisons": {}}
            for variant, comparison in step["comparisons"].items():
                outcomes = {}
                for category in ("same_edits", "only_actual_edits", "only_alternative_edits"):
                    values, counts = assess_edits(comparison[category], truths[step["key"]])
                    outcomes[category] = {"edits": values, "counts": counts}
                item["comparisons"][variant] = outcomes
            attribution.append(item)
    totals = {}
    for variant in VARIANTS[1:]:
        totals[variant] = {}
        for category in ("same_edits", "only_actual_edits", "only_alternative_edits"):
            counts = Counter()
            for item in attribution:
                counts.update(item["comparisons"][variant][category]["counts"])
            totals[variant][category] = dict(counts)
    require(sha(output / "predictions_before_gt.json") == frozen_hash, "predictions changed during GT scoring")
    require(all(sha(Path(path)) == digest for path, digest in source_hashes.items()), "source changed during replay")
    report = {"verified": True, "new_api_calls": 0, "prediction_sha256": frozen_hash,
              "predictions_saved_before_gt_join": True, "kind": predictions["kind"],
              "all_predeclared_targets_in_every_round": True, "best_round_selected_from_gt": False,
              "rounds": scored_rounds, "step_edit_attribution": attribution,
              "step_edit_totals": totals,
              "totals_caveat": "Counts are per-step edit events, not unique final-label improvements; they cannot establish an alternative closed-loop effect."}
    save(output / "report.json", report)
    print(json.dumps({"verified": True, "new_api_calls": 0, "step_edit_totals": totals,
                      "last_snapshot": completion["last_snapshot"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.source, args.output)
