"""Closed-run raw-response replay and mask-aware scoring; never calls an LLM."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_panel_trial import read, save
from scripts.run_recent_mean_panel_trial import review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_visual_repair_trial import (
    REPAIR_STAGE,
    REVIEW_STAGE,
    repair_wire,
    verify_plan,
)
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from scripts.score_prior_feedback_continuation import _hashes, _parsed_call
from scripts.score_prior_feedback_continuation import validate as validate_source
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.semantic_coordinator import item_error
from surgical_agent.research.verification.visual_repair import (
    apply_reviewed_repair,
    compile_repair,
)

VERSIONS = ("h0", "graph_r1", "temporary", "final")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
PAIRS = (("h0", "graph_r1"), ("graph_r1", "temporary"), ("graph_r1", "final"), ("temporary", "final"))


def audit(output, adapter):
    plan, done, ledger = (read(output / f"{n}.json") for n in ("plan", "completion", "budget"))
    if not done.get("closed_utc") or done.get("fatal_error") or ledger.get("stopped") is not True:
        raise ValueError("requires a closed, nonfatal run")
    for name in ("plan", "initial_state", "predictions", "budget"):
        if sha(output / f"{name}.json") != done[f"{name}_sha256"]:
            raise ValueError("closed snapshot changed")
    _hashes(output, done["inference_artifact_sha256"], "closed artifacts")
    required = {p.relative_to(output).as_posix() for folder in ("calls", "targets") for p in (output / folder).rglob("*.json")}
    if not required <= set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen call or target artifact")
    verify_plan(plan)
    _, _, source_initials, _ = validate_source(Path(plan["source_root"]))
    initials = read(output / "initial_state.json")["targets"]
    if initials != source_initials:
        raise ValueError("cached H0, prior, candidates or reviews changed")
    rows = read(output / "predictions.json")["targets"]
    identities = lambda group: [(r["key"], r["video_id"], r["frame_id"]) for r in group]
    if len(rows) != 8 or identities(rows) != identities(initials) or identities(rows) != identities(plan["selection"]):
        raise ValueError("eight fixed identities and order required")
    calls = ledger["calls"]
    if len(calls) > 48 or len(calls) != done["post_calls"] or any(c["status"] == "DISPATCHED" for c in calls):
        raise ValueError("invalid or outstanding call count")
    if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate call or retry")
    known_keys = {r["key"] for r in rows}
    if any(c["target"] not in known_keys or (c["stage"], c["seat"]) not in
           {(REPAIR_STAGE, "base"), *((REVIEW_STAGE, s) for s in SEATS)} for c in calls):
        raise ValueError("unexpected stage, seat or target")
    records = {}
    for selected, initial, row in zip(plan["selection"], initials, rows, strict=True):
        old, key = initial["graph_r1"], row["key"]
        record = read(output / "targets" / key / "repair.json")
        records[key] = record
        if row["h0"] != initial["h0"] or row["graph_r1"] != old["prediction"] or record["before"] != old["prediction"]:
            raise ValueError("inherited prediction changed")
        target_calls = [c for c in calls if c["target"] == key]
        base = build_gemini_base(adapter, selected)
        repair_calls = [c for c in target_calls if c["seat"] == "base"]
        repair_raw = _parsed_call(output, repair_calls[0]) if repair_calls else None
        if repair_raw != record["raw_repair"]:
            raise ValueError("repair differs from raw API response")
        expected_temporary = expected_final = deepcopy(old["prediction"])
        expected_compiled = None
        try:
            expected_compiled = compile_repair(old["prediction"], old["pool"], repair_raw, old["issues"], image_count=len(base.images))
        except (ValueError, TypeError, KeyError, ApiSchemaError):
            if record["status"] not in {"REPAIR_FAILED", "BUDGET_STOPPED", "NOT_ATTEMPTED"}:
                raise ValueError("invalid repair claims success") from None
        else:
            expected_temporary = expected_compiled["temporary"]
        if expected_compiled != record["compiled"]:
            raise ValueError("repair compilation differs")
        review_calls = [c for c in target_calls if c["stage"] == REVIEW_STAGE]
        if len(review_calls) != record["review_calls_observed"]:
            raise ValueError("review count mismatch")
        if review_calls and (expected_compiled is None or not expected_compiled["review_pool"]["propositions"]):
            raise ValueError("review without a candidate queue")
        if record["raw_reviews"] is not None:
            raw = {s: None for s in SEATS}
            for call in review_calls:
                raw[call["seat"]] = _parsed_call(output, call)
            if raw != record["raw_reviews"]:
                raise ValueError("review differs from raw response")
            pool = expected_compiled["review_pool"]
            reviews, formatting = normalize_five(raw, pool, len(base.images))
            if reviews != record["reviews"] or formatting != record["format_diagnostics"]:
                raise ValueError("review normalization differs")
            means = {}
            for p in pool["propositions"]:
                scores = []
                for seat in SEATS:
                    item = reviews[seat]["judgments"].get(p["id"])
                    scores.append(None if item_error(item, p["task"], len(base.images)) else item["rating"])
                means[p["id"]] = None if None in scores else sum(scores) / 5
            if means != record["means"]:
                raise ValueError("independent five-seat arithmetic differs")
            if len(review_calls) == 5:
                try:
                    expected_final = apply_reviewed_repair(expected_compiled, means)
                except (ValueError, TypeError, KeyError, ApiSchemaError):
                    if record["status"] != "SELECTION_FAILED":
                        raise ValueError("invalid final selection claims success") from None
                else:
                    if record["status"] != "REVIEWED":
                        raise ValueError("completed review status differs")
            elif record["status"] != "INCOMPLETE_PANEL":
                raise ValueError("incomplete panel status differs")
        elif review_calls:
            raise ValueError("raw reviews missing")
        if (record["temporary"] != expected_temporary or record["prediction"] != expected_final
                or row["temporary"] != expected_temporary or row["final"] != expected_final):
            raise ValueError("temporary or final prediction replay differs")
        for version in VERSIONS:
            if labels(row[version]) != row[version] or row[version]["phase"] != row["h0"]["phase"]:
                raise ValueError("invalid label contract or changed Phase")
        for call in target_calls:
            body = repair_wire(base, selected, initial) if call["seat"] == "base" else review_wire(call["seat"], base, selected, expected_compiled["review_pool"])
            folder = output / "calls" / f"{call['index']:03d}_{key}_{call['stage']}_{call['seat']}"
            if (read(folder / "request.json") != redact_images(body) or read(folder / "record.json") != call
                    or record["request_fingerprints"][call["seat"]] != fingerprint(body)):
                raise ValueError("actual request differs from frozen protocol")
    return plan, rows, initials, records, ledger, done


def summarize(rows, truth, initials, records):
    truth_by_key = {(t["video_id"], t["frame_id"]): t for t in truth}
    details, metrics = [], {}
    for version in VERSIONS:
        inputs = [{**truth_by_key[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None, "final": r[version]} for r in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                t = truth_by_key[row["video_id"], row["frame_id"]]
                if not t["mask"][task]:
                    continue
                predicted, expected = set(row[version][task]), set(t["gt"][task])
                tp += len(predicted & expected)
                fp += len(predicted - expected)
                fn += len(expected - predicted)
            m = metrics[version]["tasks"][task]
            if (m["tp"], m["fp"], m["fn"]) != (tp, fp, fn):
                raise ValueError("independent metric counts differ")
    coverage = {arm: {t: Counter() for t in TASKS[:4]} for arm in ("original_pool", "repair_pool")}
    for row, initial in zip(rows, initials, strict=True):
        truth_row = truth_by_key[row["video_id"], row["frame_id"]]
        record = records[row["key"]]
        details.append({"key": row["key"], "status": record["status"], "comparisons": {
            f"{a}_to_{b}": frame_delta(row[a], row[b], truth_row["gt"], truth_row["mask"]) for a, b in PAIRS}})
        for arm in coverage:
            pool = initial["graph_r1"]["pool"] if arm == "original_pool" or record["compiled"] is None else record["compiled"]["pool"]
            for task in TASKS[:4]:
                if not truth_row["mask"][task]:
                    continue
                expected = set(truth_row["gt"][task])
                candidates = {p["label_id"] for p in pool["propositions"] if p["task"] == task}
                coverage[arm][task].update(valid_targets=1, gt_positive=len(expected), pool_true=len(candidates & expected),
                                           pool_false=len(candidates - expected), missing_from_pool=len(expected - candidates))
    return {"metrics": metrics, "candidate_coverage": coverage,
            "comparisons": {f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details]) for a, b in PAIRS}}, details


def score(output, adapter):
    plan, rows, initials, records, ledger, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
                                      "h0": r["h0"], "h1": None, "final": r["final"]} for r in rows])
    result, details = summarize(rows, truth, initials, records)
    charges = defaultdict(lambda: defaultdict(Decimal))
    for c in ledger["calls"]:
        charges[c["account"]][c["charge_kind"]] += Decimal(c["charge"])
    result.update(profile=plan["profile"], targets=len(rows), limitations=plan["limitations"],
        runtime={"calls": len(ledger["calls"]), "charges": {a: {k: str(v) for k, v in group.items()} for a, group in charges.items()},
                 "inference_seconds": done["inference_seconds"], "statuses": done["statuses"],
                 "target_statuses": dict(Counter(r["status"] for r in records.values()))},
        audit={"raw_outputs_replayed": True, "actual_requests_verified": len(ledger["calls"]),
               "means_and_metric_counts_independently_checked": True, "gt_loaded_after_closed_inference": True})
    for name, value in (("metrics", result), ("frame_deltas", details), ("scored_truth", truth)):
        save(output / f"{name}.json", value)
    print(json.dumps({"f1": {v: {t: m["micro_f1"] for t, m in item["tasks"].items()} for v, item in result["metrics"].items()},
        "comparisons": {k: v["categories"] for k, v in result["comparisons"].items()},
        "coverage": result["candidate_coverage"], "runtime": result["runtime"]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
