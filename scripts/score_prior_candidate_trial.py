"""Offline score/audit of the frozen two-arm prior-guided proposer trial."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_candidate_trial import ARMS, TASKS, verify
from scripts.run_prior_panel_trial import read, save
from scripts.run_repair_revision_trial import normalize_five
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import make_pool


def validate(output):
    plan, done, ledger = (read(output / name) for name in ("plan.json", "completion.json", "budget.json"))
    if not done.get("closed_utc") or ledger["stopped"] is not True:
        raise ValueError("score requires closed inference")
    for name in ("plan", "predictions", "budget"):
        if sha(output / f"{name}.json") != done[f"{name}_sha256"]:
            raise ValueError("closed snapshot changed")
    verify(plan, output)
    if any(r["status"] == "DISPATCHED" for r in ledger["calls"]):
        raise ValueError("outstanding API call")
    rows = read(output / "predictions.json")["targets"]
    if ([(r["key"], r["video_id"], r["frame_id"]) for r in rows]
            != [(r["key"], r["video_id"], r["frame_id"]) for r in plan["selection"]]
            or any(set(r["arms"]) != set(ARMS) for r in rows)):
        raise ValueError("all planned targets and both arms required")
    if plan["arms"] != list(ARMS):
        raise ValueError("wrong comparison arms")
    return plan, rows, ledger


def runtime(rows, ledger):
    result = {"arms": {}, "paired_panel": [], "costs": {}}
    for arm in ARMS:
        panels = [r["arms"][arm].get("panel_seconds") for r in rows]
        panels = [v for v in panels if v is not None]
        result["arms"][arm] = {"measured_panels": len(panels),
            "median_panel_seconds": statistics.median(panels) if panels else None,
            "max_panel_seconds": max(panels) if panels else None,
            "shared_panels": sum(bool(r["arms"][arm].get("shared_from")) for r in rows),
            "statuses": dict(Counter(r["arms"][arm]["status"] for r in rows))}
    for r in rows:
        left, right = (r["arms"][a].get("panel_seconds") for a in ARMS)
        if left is not None and right is not None and left > 0:
            result["paired_panel"].append({"key": r["key"], "seconds_added": right-left,
                                           "percent_added": (right/left-1)*100})
    pairs = result["paired_panel"]
    result["median_paired_percent_added"] = statistics.median(r["percent_added"] for r in pairs) if pairs else None
    result["median_paired_seconds_added"] = statistics.median(r["seconds_added"] for r in pairs) if pairs else None
    groups = defaultdict(lambda: {"calls": 0, "charges": defaultdict(Decimal), "charge_kinds": Counter(),
                                   "prompt_tokens": 0, "completion_tokens": 0})
    for call in ledger["calls"]:
        group = next((a for a in ARMS if call["stage"].startswith(a+"_")), "shared_h0")
        for name in (group, "all"):
            g = groups[name]
            g["calls"] += 1
            g["charges"][call["account"]] += Decimal(call["charge"])
            g["charge_kinds"][call["charge_kind"]] += 1
            g["prompt_tokens"] += call.get("usage", {}).get("prompt_tokens", 0)
            g["completion_tokens"] += call.get("usage", {}).get("completion_tokens", 0)
    result["costs"] = {name: {**g, "charges": {k: str(v) for k, v in g["charges"].items()},
                                    "charge_kinds": dict(g["charge_kinds"])} for name, g in groups.items()}
    result["cost_note"] = "Shared exact-wire panels counted once; native USD and estimated CNY separate. Per-arm observed expenses exclude shared work; never present them as equal-work causal savings."
    return result


def score(output, adapter):
    plan, rows, ledger = validate(output)
    reports, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["arms"]["control"]["prediction"]} for r in rows])
    truth_by_id = {(r["video_id"], r["frame_id"]): r for r in truth}
    metrics = {"h0": reports["arms"]["h0"]}
    for arm in ARMS:
        data = [{**truth_by_id[r["video_id"], r["frame_id"]], "final": r["arms"][arm]["prediction"]} for r in rows]
        metrics[arm] = compute_repair_comparison(data)["arms"]["final"]
    pairs = [("h0", "control"), ("h0", "prior_graph"), ("control", "prior_graph")]
    details, coverage = [], {a: {t: {"valid_targets": 0, "gt_positive": 0, "pool_true": 0,
                                   "pool_false": 0, "pool_size": 0, "selected_true": 0} for t in TASKS[:4]} for a in ARMS}
    audit = {"replayed_panels": 0, "candidate_pools_rebuilt": 0, "source_and_closed_snapshots_valid": True,
             "invalid_judgments": {a: Counter() for a in ARMS}, "audit_scope": "offline replay and independent score; no extra model calls"}
    for row in rows:
        gt = truth_by_id[row["video_id"], row["frame_id"]]
        predictions = {"h0": row["h0"], **{a: row["arms"][a]["prediction"] for a in ARMS}}
        details.append({"key": row["key"], "comparisons": {f"{l}_to_{r}": frame_delta(predictions[l], predictions[r], gt["gt"], gt["mask"])
                        for l, r in pairs}})
        for arm in ARMS:
            data = row["arms"][arm]
            for task in TASKS[:4]:
                if not gt["mask"][task]:
                    continue
                expected = set(gt["gt"][task])
                pool_ids = {p["label_id"] for p in data.get("pool", {"propositions": []})["propositions"] if p["task"] == task}
                c = coverage[arm][task]
                c["valid_targets"] += 1
                c["gt_positive"] += len(expected)
                c["pool_true"] += len(pool_ids & expected)
                c["pool_false"] += len(pool_ids - expected)
                c["pool_size"] += len(pool_ids)
                c["selected_true"] += len(set((data["prediction"] or {}).get(task, [])) & expected)
            path = output / "targets" / row["key"] / f"{arm}.json"
            if not path.exists():
                continue
            record = read(path)
            if record.get("proposal") is not None and record["status"] != "PROPOSAL_FAILED":
                if make_pool(row["h0"], record["proposal"], make_pool(row["h0"])) != record["pool"]:
                    raise ValueError("candidate pool differs from model proposal union")
                audit["candidate_pools_rebuilt"] += 1
            if "raw" in record:
                normalized, formatting = normalize_five(record["raw"], record["pool"], 3)
                means, diagnostics = panel.aggregate(normalized, record["pool"], image_count=3)
                if normalized != record["reviews"] or means != record["means"] or diagnostics != record["diagnostics"]:
                    raise ValueError("review replay differs")
                predicted = panel.select(row["h0"], record["pool"], means, threshold=4)
                if predicted != data["prediction"]:
                    raise ValueError("repair replay differs")
                audit["replayed_panels"] += 1
                for result in formatting.values():
                    for causes in result.get("errors", {}).values():
                        audit["invalid_judgments"][arm].update(causes if isinstance(causes, list) else [causes])
    for tables in coverage.values():
        for c in tables.values():
            c["pool_recall"] = c["pool_true"]/c["gt_positive"] if c["gt_positive"] else None
    result = {"profile": plan["profile"], "targets": len(rows), "metrics": metrics, "candidate_coverage": coverage,
              "comparisons": {f"{l}_to_{r}": summarize_deltas([d["comparisons"][f"{l}_to_{r}"] for d in details]) for l, r in pairs},
              "runtime": runtime(rows, ledger), "audit": audit,
              "limitations": ["Eight Training time targets, not independent surgery generalization or Gate OOF.",
                              "Paired H0 but different proposals/pools; effect combines Training priors and candidate guidance.",
                              "Unresolved is not a model pass; unchanged fallback remains scored."]}
    validate(output)
    save(output / "metrics.json", result)
    save(output / "frame_deltas.json", details)
    save(output / "scored_truth.json", truth)
    save(output / "audit.json", audit)
    print(json.dumps({"f1": {a: {t: v["micro_f1"] for t, v in m["tasks"].items()} for a, m in metrics.items()},
                      "ivt_candidate_coverage": {a: v["ivt"] for a, v in coverage.items()},
                      "changes": {k: v["categories"] for k, v in result["comparisons"].items()}, "audit": audit}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
