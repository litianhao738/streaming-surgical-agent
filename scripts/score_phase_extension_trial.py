"""Offline replay, immutable-four-head checks and masked Phase comparisons.

No inference or credential access occurs here. Query GT is loaded only after
closed snapshots, actual requests, raw responses and selection are verified.
"""
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

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_phase_extension_trial import ARMS, build_inputs, run_panel, verify_plan
from scripts.run_prior_panel_trial import read, save
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_five_head_repair_trial import audit as audit_source
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from scripts.score_prior_feedback_continuation import _hashes
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import labels

TASKS = ("instrument", "verb", "target", "ivt", "phase")
VERSIONS = ("h0", "graph_r1", "previous_five", *ARMS)
PAIRS = (("h0", "graph_r1"), ("graph_r1", "previous_five"),
         *(("graph_r1", arm) for arm in ARMS),
         *(("previous_five", arm) for arm in ARMS), (ARMS[0], ARMS[1]))


def independent_phase_decision(current, raw_reviews, image_count):
    """Recompute single-label votes without using the runtime phase selector."""
    before = current["phase"][0]
    decision = {"before": before, "after": before, "reason": "INVALID_PANEL",
                "votes": {str(p): 0 for p in range(7)}, "abstentions": 0}
    if not isinstance(raw_reviews, dict) or set(raw_reviews) != set(SEATS):
        return decision
    if type(image_count) is not int or not 1 <= image_count <= 3:
        return decision
    for raw in raw_reviews.values():
        if not isinstance(raw, dict) or set(raw) != {"phase_id", "image_indices", "observation"}:
            return decision
        phase, refs, observation = raw["phase_id"], raw["image_indices"], raw["observation"]
        if phase is not None and (type(phase) is not int or not 0 <= phase < 7):
            return decision
        if (not isinstance(refs, list)
                or any(type(i) is not int or not 0 <= i < image_count for i in refs)
                or len(set(refs)) != len(refs)
                or (phase is not None and image_count - 1 not in refs)):
            return decision
        if not isinstance(observation, str) or not observation.strip() or len(observation) > 1000:
            return decision
    for raw in raw_reviews.values():
        if raw["phase_id"] is None:
            decision["abstentions"] += 1
        else:
            decision["votes"][str(raw["phase_id"])] += 1
    winners = [int(phase) for phase, votes in decision["votes"].items() if votes >= 3]
    decision["reason"] = "NO_MAJORITY"
    if winners:
        winner = winners[0]
        decision.update(after=winner,
                        reason="CURRENT_PHASE_MAJORITY" if winner == before else "MAJORITY_PHASE_SWITCH")
    return decision


def _identities(rows):
    return [(r["key"], r["video_id"], r["frame_id"]) for r in rows]


def audit(output, adapter):
    """Verify immutable artifacts, re-create wires and independently count votes."""
    output = Path(output)
    plan, done, ledger = (read(output / f"{n}.json") for n in ("plan", "completion", "budget"))
    if not done.get("closed_utc") or done.get("fatal_error") or ledger.get("stopped") is not True:
        raise ValueError("requires closed nonfatal inference")
    for name in ("plan", "initial_state", "predictions", "budget"):
        if sha(output / f"{name}.json") != done[f"{name}_sha256"]:
            raise ValueError("closed snapshot changed")
    _hashes(output, done["inference_artifact_sha256"], "closed artifacts")
    required = {p.relative_to(output).as_posix() for folder in ("calls", "targets")
                for p in (output / folder).rglob("*.json")}
    if required != set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen or missing inference artifact")
    verify_plan(plan)
    source_plan, previous, _, _, _, _ = audit_source(Path(plan["source_root"]), adapter)
    initials = read(output / "initial_state.json")["targets"]
    rows = read(output / "predictions.json")["targets"]
    expected = _identities(plan["selection"])
    if (len(expected) != 8 or len(set(expected)) != 8
            or len({k for k, _, _ in expected}) != 8
            or len({(v, f) for _, v, f in expected}) != 8
            or any(_identities(group) != expected for group in (initials, rows, previous))
            or plan["selection"] != source_plan["selection"]):
        raise ValueError("all eight cached identities, order and H0 inputs required")
    if any(adapter.entries[r["video_id"]].split is not DatasetSplit.TRAINING for r in rows):
        raise ValueError("only the frozen Training cohort is allowed")
    if build_inputs(adapter, plan["selection"]) != plan["phase_inputs"]:
        raise ValueError("phase images differ from deterministic causal sampling")
    calls = ledger["calls"]
    if (len(calls) > 80 or plan["max_calls"] != 80 or len(calls) != done["post_calls"]
            or any(c["status"] == "DISPATCHED" for c in calls)):
        raise ValueError("invalid or outstanding call count")
    if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate call or undeclared retry")
    if done["statuses"] != dict(Counter(c["status"] for c in calls)):
        raise ValueError("completion transport counts differ")
    known = {r["key"] for r in rows}
    if any(c["target"] not in known or c["stage"] not in ARMS or c["seat"] not in SEATS for c in calls):
        raise ValueError("unexpected target, arm or reviewer")
    records = {}
    for row, initial, old in zip(rows, initials, previous, strict=True):
        key = row["key"]
        baselines = {"h0": old["h0"], "graph_r1": old["graph_r1"], "previous_five": old["panel_five"]}
        if any(row[version] != value or initial[version] != value for version, value in baselines.items()):
            raise ValueError("cached comparison baseline changed")
        records[key] = {}
        if set(plan["phase_inputs"][key]) != set(ARMS):
            raise ValueError("both predeclared Phase arms are required")
        for arm in ARMS:
            selected = plan["phase_inputs"][key][arm]
            if _identities([selected]) != _identities([row]):
                raise ValueError("Phase image target differs from prediction identity")
            frames = selected["causal_frame_ids"]
            if (not frames or any(type(f) is not int for f in frames)
                    or frames != sorted(set(frames)) or frames[-1] != row["frame_id"]
                    or len(frames) != len(selected["images"])):
                raise ValueError("invalid causal Phase image identities")
            record = read(output / "targets" / key / f"{arm}.json")
            records[key][arm] = record
            target_calls = [c for c in calls if c["target"] == key and c["stage"] == arm]
            replay_calls = ReplayCalls(output, target_calls)
            replay = run_panel(replay_calls, selected, initial["graph_r1"], arm)
            if set(record) != set(replay):
                raise ValueError("panel record fields differ from replay")
            for field in record.keys() - {"seconds"}:
                if record[field] != replay[field]:
                    raise ValueError(f"raw response replay differs: {key} {arm} {field}")
            if len(replay_calls.rows) != len(target_calls):
                raise ValueError("unconsumed API response")
            independent = independent_phase_decision(initial["graph_r1"], record["raw_reviews"], len(frames))
            if record["decision"] != independent:
                raise ValueError("independent five-seat Phase decision differs")
            prediction = deepcopy(initial["graph_r1"])
            prediction["phase"] = [independent["after"]]
            if record["prediction"] != prediction or row[arm] != prediction:
                raise ValueError("final prediction differs from independent vote replay")
            if any(row[arm][t] != row["graph_r1"][t] for t in TASKS[:4]):
                raise ValueError("Phase extension changed an interaction head")
        for version in VERSIONS:
            if labels(row[version]) != row[version]:
                raise ValueError("invalid final label contract")
    return plan, rows, initials, records, ledger, done


def phase_transition(before, after, gt, valid):
    """A wrong-to-wrong single-label replacement is never a beneficial deletion."""
    if not valid:
        return "unscored"
    if before == after:
        return "unchanged"
    if after == gt:
        return "improved"
    if before == gt:
        return "harmed"
    return "no_benefit_replacement"


def summarize(rows, truth, records, phase_inputs):
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics, details, phase_cases = {}, [], []
    for version in VERSIONS:
        inputs = [{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None, "final": r[version]}
                  for r in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        for task in TASKS:
            tp = fp = fn = exact = valid = 0
            for row in rows:
                t = truths[row["video_id"], row["frame_id"]]
                if not t["mask"][task]:
                    continue
                predicted, expected = set(row[version][task]), set(t["gt"][task])
                tp += len(predicted & expected)
                fp += len(predicted - expected)
                fn += len(expected - predicted)
                exact += predicted == expected
                valid += 1
            actual = metrics[version]["tasks"][task]
            if (actual["tp"], actual["fp"], actual["fn"], actual["exact_matches"], actual["valid_targets"]) != (tp, fp, fn, exact, valid):
                raise ValueError("independent masked metric counts differ")
    for row in rows:
        t, key = truths[row["video_id"], row["frame_id"]], row["key"]
        details.append({"key": key, "comparisons": {
            f"{a}_to_{b}": frame_delta(row[a], row[b], t["gt"], t["mask"]) for a, b in PAIRS}})
        valid = bool(t["mask"]["phase"])
        gt = t["gt"]["phase"] if valid else None
        case = {"key": key, "valid_phase_gt": valid, "gt": gt,
                "predictions": {version: row[version]["phase"] for version in VERSIONS},
                "transitions": {f"{a}_to_{b}": phase_transition(row[a]["phase"], row[b]["phase"], gt, valid)
                                for a, b in PAIRS}, "arms": {}}
        for arm in ARMS:
            record = records[key][arm]
            case["arms"][arm] = {"status": record["status"], "decision": record["decision"],
                "causal_frame_ids": phase_inputs[key][arm]["causal_frame_ids"],
                "errors": record["errors"], "reviewer_observations": record["raw_reviews"]}
        phase_cases.append(case)
    return {"metrics": metrics,
        "comparisons": {f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details])
                        for a, b in PAIRS},
        "phase_transitions": {f"{a}_to_{b}": dict(Counter(c["transitions"][f"{a}_to_{b}"] for c in phase_cases))
                              for a, b in PAIRS}}, details, phase_cases


def score(output, adapter):
    output = Path(output)
    plan, rows, _, records, ledger, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r[ARMS[-1]]} for r in rows])
    result, details, phase_cases = summarize(rows, truth, records, plan["phase_inputs"])
    charges = defaultdict(lambda: defaultdict(Decimal))
    arm_charges = defaultdict(lambda: defaultdict(lambda: defaultdict(Decimal)))
    for call in ledger["calls"]:
        charges[call["account"]][call["charge_kind"]] += Decimal(call["charge"])
        arm_charges[call["stage"]][call["account"]][call["charge_kind"]] += Decimal(call["charge"])
    result.update(profile=plan["profile"], targets=len(rows), limitations=plan["limitations"],
        runtime={"calls": len(ledger["calls"]),
            "charges": {account: {kind: str(value) for kind, value in kinds.items()} for account, kinds in charges.items()},
            "arm_charges": {arm: {account: {kind: str(value) for kind, value in kinds.items()}
                                  for account, kinds in accounts.items()} for arm, accounts in arm_charges.items()},
            "inference_seconds": done["inference_seconds"], "statuses": done["statuses"],
            "arm_calls": {arm: sum(c["stage"] == arm for c in ledger["calls"]) for arm in ARMS},
            "arm_statuses": {arm: dict(Counter(r[arm]["status"] for r in records.values())) for arm in ARMS},
            "arm_seconds": {arm: sum(r[arm]["seconds"] for r in records.values()) for arm in ARMS}},
        audit={"raw_outputs_replayed": True, "actual_requests_verified": len(ledger["calls"]),
            "votes_and_metric_counts_independently_checked": True,
            "four_heads_equal_graph_r1_for_every_target": True,
            "causal_sampling_independently_recreated": True, "gt_loaded_after_closed_inference": True})
    for name, value in (("metrics", result), ("frame_deltas", details), ("phase_cases", phase_cases), ("scored_truth", truth)):
        save(output / f"{name}.json", value)
    print(json.dumps({"f1": {v: {t: m["micro_f1"] for t, m in item["tasks"].items()}
                                  for v, item in result["metrics"].items()},
        "phase_transitions": result["phase_transitions"], "runtime": result["runtime"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
