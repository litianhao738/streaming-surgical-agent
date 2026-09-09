"""Offline, closed-run replay and per-head mask-aware paired comparison."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_five_head_repair_trial import (
    REPAIR_STAGE,
    REVIEW_STAGE,
    run_target,
    verify_plan,
)
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_panel_trial import read, save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from scripts.score_prior_feedback_continuation import _hashes, _parsed_call
from scripts.score_visual_repair_trial import audit as audit_source
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.semantic_coordinator import item_error

VERSIONS = ("h0", "graph_r1", "previous_four", "llm_raw", "paper_style", "panel_four_shadow", "panel_five")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
PAIRS = (("h0", "graph_r1"), ("graph_r1", "previous_four"), ("graph_r1", "llm_raw"),
         ("graph_r1", "paper_style"), ("graph_r1", "panel_five"), ("previous_four", "panel_five"),
         ("paper_style", "panel_five"), ("panel_four_shadow", "panel_five"))


class ReplayCalls:
    """Read saved wire/response pairs only; invoking .call never uses a network."""
    def __init__(self, output, calls):
        self.output, self.available, self.rows = output, calls, []
        self.stopped = not bool(calls)

    def call(self, target, stage, seat, body):
        matches = [c for c in self.available if (c["target"], c["stage"], c["seat"]) == (target, stage, seat)]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("duplicate source call")
        call = matches[0]
        folder = self.output / "calls" / f"{call['index']:03d}_{target}_{stage}_{seat}"
        if read(folder / "request.json") != redact_images(body) or read(folder / "record.json") != call:
            raise ValueError("actual request or ledger differs from frozen protocol")
        self.rows.append(call)
        return _parsed_call(self.output, call)


def audit(output, adapter):
    plan, done, ledger = (read(output / f"{n}.json") for n in ("plan", "completion", "budget"))
    if not done.get("closed_utc") or done.get("fatal_error") or ledger.get("stopped") is not True:
        raise ValueError("requires closed nonfatal inference")
    for name in ("plan", "initial_state", "predictions", "budget"):
        if sha(output / f"{name}.json") != done[f"{name}_sha256"]:
            raise ValueError("closed snapshot changed")
    _hashes(output, done["inference_artifact_sha256"], "closed artifacts")
    required = {p.relative_to(output).as_posix() for folder in ("calls", "targets") for p in (output / folder).rglob("*.json")}
    if required != set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen or missing inference artifact")
    verify_plan(plan)
    source_plan, previous, inherited, _, _, _ = audit_source(Path(plan["source_root"]), adapter)
    initials = read(output / "initial_state.json")["targets"]
    rows = read(output / "predictions.json")["targets"]
    identities = lambda group: [(r["key"], r["video_id"], r["frame_id"]) for r in group]
    if (len(rows) != 8 or identities(rows) != identities(initials) or identities(rows) != identities(previous)
            or identities(rows) != identities(plan["selection"]) or plan["selection"] != source_plan["selection"]):
        raise ValueError("eight fixed identities/order/images required")
    calls = ledger["calls"]
    if len(calls) > 48 or len(calls) != done["post_calls"] or any(c["status"] == "DISPATCHED" for c in calls):
        raise ValueError("invalid or outstanding call count")
    if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate call or retry")
    known = {r["key"] for r in rows}
    if any(c["target"] not in known or (c["stage"], c["seat"]) not in
           {(REPAIR_STAGE, "base"), *((REVIEW_STAGE, s) for s in SEATS)} for c in calls):
        raise ValueError("unexpected call")
    records = {}
    for selected, initial, old_initial, old_row, row in zip(plan["selection"], initials, inherited, previous, rows, strict=True):
        if {k: initial[k] for k in old_initial} != old_initial or initial["previous_four"] != old_row["final"]:
            raise ValueError("cached initial state changed")
        prior_paths = [Path(p) for p in plan["prior_files_sha256"] if Path(p).stem == selected["video_id"]]
        if len(prior_paths) != 1 or read(prior_paths[0]) != initial["phase_prior"]:
            raise ValueError("prior differs from declared Training table")
        if row["h0"] != initial["h0"] or row["graph_r1"] != initial["graph_r1"]["prediction"] or row["previous_four"] != initial["previous_four"]:
            raise ValueError("comparison baseline changed")
        key = row["key"]
        record = read(output / "targets" / key / "repair.json")
        records[key] = record
        target_calls = [c for c in calls if c["target"] == key]
        replay_calls = ReplayCalls(output, target_calls)
        base = build_gemini_base(adapter, selected)
        replay = run_target(replay_calls, base, selected, initial)
        for field in record.keys() - {"repair_seconds", "review_seconds"}:
            if record[field] != replay[field]:
                raise ValueError(f"raw-response replay differs: {key} {field}")
        if len(replay_calls.rows) != len(target_calls):
            raise ValueError("unconsumed API response")
        for arm in VERSIONS[3:]:
            if row[arm] != replay[arm]:
                raise ValueError("prediction arm differs from replay")
        for version in VERSIONS:
            if labels(row[version]) != row[version]:
                raise ValueError("invalid final label contract")
        if record["reviews"] is not None:
            independent = {}
            for p in record["compiled"]["pool"]["propositions"]:
                scores = []
                for seat in SEATS:
                    item = record["reviews"][seat]["judgments"].get(p["id"])
                    scores.append(None if item_error(item, p["task"], len(base.images)) else item["rating"])
                independent[p["id"]] = None if None in scores else sum(scores) / 5
            if independent != record["means"]:
                raise ValueError("independent five-seat arithmetic differs")
    return plan, rows, initials, records, ledger, done


def summarize(rows, truth, initials, records):
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics, details = {}, []
    for version in VERSIONS:
        inputs = [{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None, "final": r[version]} for r in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                t = truths[row["video_id"], row["frame_id"]]
                if not t["mask"][task]:
                    continue
                predicted, expected = set(row[version][task]), set(t["gt"][task])
                tp += len(predicted & expected)
                fp += len(predicted - expected)
                fn += len(expected - predicted)
            m = metrics[version]["tasks"][task]
            if (m["tp"], m["fp"], m["fn"]) != (tp, fp, fn):
                raise ValueError("independent metric counts differ")
    coverage = {arm: {t: Counter() for t in TASKS} for arm in ("original_pool", "repair_pool")}
    for row, initial in zip(rows, initials, strict=True):
        t, record = truths[row["video_id"], row["frame_id"]], records[row["key"]]
        details.append({"key": row["key"], "status": record["status"], "phase_decision": record["phase_decision"],
            "comparisons": {f"{a}_to_{b}": frame_delta(row[a], row[b], t["gt"], t["mask"]) for a, b in PAIRS}})
        for arm in coverage:
            pool = initial["graph_r1"]["pool"] if arm == "original_pool" or record["compiled"] is None else record["compiled"]["pool"]
            for task in TASKS:
                if not t["mask"][task]:
                    continue
                expected = set(t["gt"][task])
                candidates = {p["label_id"] for p in pool["propositions"] if p["task"] == task}
                coverage[arm][task].update(valid_targets=1, gt_positive=len(expected), pool_true=len(candidates & expected),
                    pool_false=len(candidates - expected), missing_from_pool=len(expected - candidates))
    return {"metrics": metrics, "candidate_coverage": coverage,
        "comparisons": {f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details]) for a, b in PAIRS}}, details


def score(output, adapter):
    plan, rows, initials, records, ledger, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["panel_five"]} for r in rows])
    result, details = summarize(rows, truth, initials, records)
    charges = defaultdict(lambda: defaultdict(Decimal))
    stage_charges = defaultdict(lambda: defaultdict(Decimal))
    for c in ledger["calls"]:
        charges[c["account"]][c["charge_kind"]] += Decimal(c["charge"])
        stage_charges[c["stage"]][c["account"]] += Decimal(c["charge"])
    result.update(profile=plan["profile"], targets=len(rows), limitations=plan["limitations"],
        runtime={"calls": len(ledger["calls"]), "charges": {a: {k: str(v) for k, v in g.items()} for a, g in charges.items()},
            "stage_charges": {a: {k: str(v) for k, v in g.items()} for a, g in stage_charges.items()},
            "inference_seconds": done["inference_seconds"], "statuses": done["statuses"],
            "target_statuses": dict(Counter(r["status"] for r in records.values()))},
        audit={"raw_outputs_replayed": True, "actual_requests_verified": len(ledger["calls"]),
            "means_and_metric_counts_independently_checked": True, "gt_loaded_after_closed_inference": True})
    for name, value in (("metrics", result), ("frame_deltas", details), ("scored_truth", truth)):
        save(output / f"{name}.json", value)
    print(json.dumps({"f1": {v: {t: m["micro_f1"] for t, m in item["tasks"].items()} for v, item in result["metrics"].items()},
        "comparisons": {k: v["categories"] for k, v in result["comparisons"].items()}, "runtime": result["runtime"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
