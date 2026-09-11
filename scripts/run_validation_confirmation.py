"""Confirm v2.0.1 on the Validation split, once, against H0 and the v1.3.0 default.

Single-use. Every arm shares one cached H0, one graph candidate pool and the same
three images, all collected beforehand by `collect_validation_inputs.py`. The
reviewer roster is the published default one; no roster or prompt is changed
here. Three arms are compared:

  control    v1.3.0: compact four-head mean review plus a blind five-seat Phase vote
  v2.0.0     joint review of four heads and seven Phases, five valid seats required
  v2.0.1     the same joint answers, but an interaction candidate is scored when at
             least three seats are valid; Phase keeps the five-valid rule

v2.0.0 and v2.0.1 read the same paid answers, so the relaxed rule costs nothing
extra and the pair isolates it exactly.

Predeclared success, fixed before any call: v2.0.1 must beat both H0 and control
on mean F1, on mean precision and on total label errors. Anything else is a
failure and is reported as one.

Commands: prepare, preflight (zero API), execute (single-use), score.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_joint_phase_feedback_trial as joint
from scripts import run_split_review_trial as common
from scripts.collect_validation_inputs import validation_bases
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_pool_expansion_trial import aggregate
from scripts.run_prior_panel_trial import now, read, save, truth_row
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.five_head_repair import normalize_five_heads
from surgical_agent.research.verification.phase_extension import phase_choice_error

PROFILE = "validation_confirmation_v201_v1"
VARIANT = "v1"
MIN_VALID = 3
TASKS = joint.TASKS
ARMS = ("control", "v2.0.0", "v2.0.1")
LIMITS = {"openrouter_usd": "2.5", "aliyun_cny": "2", "xai_usd": "0"}
STANDARD = ("v2.0.1 must exceed both H0 and control on mean F1, on mean precision and on total "
            "label errors (fewer). Any other outcome is a failure.")
RUNTIME_FILES = [Path(__file__).resolve(), ROOT / "scripts/run_pool_expansion_trial.py",
                 ROOT / "scripts/collect_validation_inputs.py"]


def decide_joint(raw, pool, h0, min_valid):
    reviews, formats = normalize_five_heads(raw, pool, image_count=3)
    means, diagnostics = aggregate(reviews, pool, min_valid)
    prediction, decision = joint.select_joint(h0, pool, means)
    return {"reviews": reviews, "format_diagnostics": formats, "means": means,
            "diagnostics": diagnostics, "prediction": prediction, "phase_decision": decision}


def run_target(calls, base, selected, initial, index):
    """Control and joint arms, order alternating so neither is always first."""
    key, h0, pool = selected["key"], initial["h0"], initial["pool"]
    record = {"key": key, "h0": h0, "predictions": {"h0": deepcopy(h0)}}

    def control():
        with ThreadPoolExecutor(max_workers=2) as workers:
            graph = workers.submit(joint.panel_call, calls, key, "control_graph",
                                   {s: joint.roster.review_wire(s, base, selected, pool) for s in SEATS})
            phase = workers.submit(joint.panel_call, calls, key, "control_phase",
                                   {s: joint.roster.phase_wire(s, selected) for s in SEATS})
            raw, phase_raw = graph.result(), phase.result()
        reviews, formats = common.normalize_five(raw, pool, 3)
        means, diagnostics = common.panel.aggregate(reviews, pool, image_count=3)
        four = common.panel.select(h0, pool, means, threshold=4)
        final, decision = common.apply_phase_choices(four, phase_raw, 3)
        record["control"] = {"raw": raw, "phase_raw": phase_raw, "means": means,
                             "diagnostics": diagnostics, "format_diagnostics": formats,
                             "phase_decision": decision}
        record["predictions"]["control"] = final

    def joint_arm():
        raw = calls.call(key, "phase_recommendation", "base", joint.phase_proposal(base, selected, initial))
        error = phase_choice_error(raw, 3)
        record["phase_proposal"] = {"raw": raw, "error": error}
        recommendation = None if error else raw["phase_id"]
        joint_pool = joint.joint_pool(pool)
        answers = joint.panel_call(calls, key, "joint_r1",
                                   {s: joint.joint_wire(s, base, selected, joint_pool, h0,
                                                        recommendation, VARIANT) for s in SEATS})
        record["joint_pool"] = joint_pool
        # One paid panel, two admission rules; the pair isolates the relaxed rule.
        for arm, min_valid in (("v2.0.0", len(SEATS)), ("v2.0.1", MIN_VALID)):
            outcome = decide_joint(answers, joint_pool, h0, min_valid)
            record[arm] = {k: v for k, v in outcome.items() if k != "reviews"}
            record["predictions"][arm] = outcome["prediction"]
        record["joint_raw"] = answers

    for action in ((control, joint_arm) if index % 2 == 0 else (joint_arm, control)):
        action()
    return record


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    upstream_plan = read(source / "plan.json")
    done = read(source / "completion.json")
    if done.get("fatal_error"):
        raise ValueError("a clean upstream collection is required")
    initials = {}
    for selected in upstream_plan["selection"]:
        key = selected["key"]
        relative = f"targets/{key}/result.json"
        if sha(source / relative) != done["hashes"][relative]:
            raise ValueError("upstream result seal mismatch: " + key)
        data = read(source / relative)
        initials[key] = {k: deepcopy(data[k]) for k in ("h0", "pool", "hints")}
        audit = data["hints"]["audit"]
        if audit["excluded_video"] != selected["video_id"] or selected["video_id"] in audit["fit_videos"]:
            raise ValueError("query video leaked into its own hints: " + key)
    save(output / "initials.json", initials)
    for path in RUNTIME_FILES:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    plan = {"profile": PROFILE, "created_utc": now(), "source_archive": str(source.resolve()),
            "source_plan_sha256": sha(source / "plan.json"), "video": upstream_plan["video"],
            "selection": upstream_plan["selection"], "arms": list(ARMS), "variant": VARIANT,
            "min_valid_interaction": MIN_VALID, "min_valid_phase": len(SEATS),
            "models": joint.roster.MODELS, "families": joint.roster.FAMILIES,
            "transports": joint.roster.TRANSPORTS, "limits": LIMITS,
            "credential_root": str(ROOT.resolve()),
            "max_calls": 16 * len(upstream_plan["selection"]), "automatic_retries": 0,
            "initials_sha256": sha(output / "initials.json"), "predeclared_standard": STANDARD,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_FILES},
            "policy": ("One cached H0, graph pool and image set per target, shared by every arm. "
                       "Published roster and prompts unchanged. v2.0.0 and v2.0.1 read the same paid "
                       "answers. No retries, no GT in any request, no default change."),
            "limitations": [
                "One Validation video: a new surgery for this mechanism, not cross-video generalization.",
                "Validation is spent by this run; it can no longer serve as unseen data afterwards.",
                "Upstream H0 and candidates are collected once and shared, so H0 variance is not measured.",
                "Control is re-run here, so its numbers are not the archived Training ones."]}
    save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "video": plan["video"],
                      "targets": len(plan["selection"]), "max_calls": plan["max_calls"]}), flush=True)


def verify_plan(plan, output):
    if plan["profile"] != PROFILE or plan["models"] != joint.roster.MODELS:
        raise ValueError("profile or roster mismatch")
    if plan["initials_sha256"] != sha(output / "initials.json"):
        raise ValueError("initial state changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("runtime source changed: " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            if sha(image["path"]) != image["sha256"]:
                raise ValueError("frozen image changed")


class MockCalls:
    def __init__(self, pool_of):
        self.pool_of, self.rows, self.stopped = pool_of, [], False
        self.lock = __import__("threading").RLock()

    def call(self, target, stage, seat, body):
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        if stage == "phase_recommendation":
            return {"phase_id": 1, "image_indices": [2], "observation": "Offline preflight answer."}
        if stage == "control_phase":
            return {"phase_id": 1, "image_indices": [2], "observation": "Offline preflight answer."}
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [],
                "observation": "Offline preflight answer."}
        return {"judgments": {p["id"]: dict(item) for p in self.pool_of(stage)["propositions"]}}


def preflight(output, adapter):
    plan = read(output / "plan.json")
    verify_plan(plan, output)
    initials = read(output / "initials.json")
    bases = validation_bases(adapter, plan)
    fingerprints, checks = {}, []
    for index, selected in enumerate(plan["selection"]):
        key = selected["key"]
        initial, base = initials[key], bases[key]
        pool, jpool = initial["pool"], joint.joint_pool(initial["pool"])
        bodies = {"control_graph": {s: joint.roster.review_wire(s, base, selected, pool) for s in SEATS},
                  "control_phase": {s: joint.roster.phase_wire(s, selected) for s in SEATS},
                  "joint_r1": {s: joint.joint_wire(s, base, selected, jpool, initial["h0"], 1, VARIANT)
                               for s in SEATS}}
        for stage, group in bodies.items():
            for body in group.values():
                packet = json.loads(body["messages"][0]["content"][0]["text"])
                if {"gt", "ground_truth"} & packet.keys():
                    raise ValueError("request must not carry ground truth")
            fingerprints.setdefault(key, {})[stage] = {s: fingerprint(b) for s, b in group.items()}
        record = run_target(MockCalls(lambda stage, jp=jpool, p=pool: jp if stage == "joint_r1" else p),
                            base, selected, initial, index)
        checks.append({"key": key, "arms": sorted(record["predictions"]),
                       "pool": len(pool["propositions"]), "joint_pool": len(jpool["propositions"])})
    save(output / "wire_fingerprints.json", fingerprints)
    save(output / "offline_preflight.json", {"checked_utc": now(), "api_calls": 0, "targets": checks})
    print(json.dumps({"preflight": str(output), "api_calls": 0, "targets": len(checks),
                      "arms": sorted({a for c in checks for a in c["arms"]})}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no paid overwrite")
    verify_plan(plan, output)
    initials = read(output / "initials.json")
    preflight_report = read(output / "offline_preflight.json")
    if [s["key"] for s in preflight_report["targets"]] != [s["key"] for s in plan["selection"]]:
        raise ValueError("complete offline preflight required")
    bases = validation_bases(adapter, plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    rows, start, fatal = [], perf_counter(), None
    with joint.credential_context(plan), joint.roster.lightweight_protocol():
        calls = joint.roster.GLMCalls(output, limits={a: Decimal(v) for a, v in plan["limits"].items()},
                                      rates=joint.roster.RATES, providers=joint.roster.PROVIDERS,
                                      max_calls=plan["max_calls"], reasoning_seats=("gemini",))
        try:
            for index, selected in enumerate(plan["selection"]):
                if calls.stopped:
                    break
                key = selected["key"]
                record = run_target(calls, bases[key], selected, initials[key], index)
                save(output / "targets" / key / "result.json", record)
                rows.append({"key": key, "video_id": selected["video_id"],
                             "frame_id": selected["frame_id"], "predictions": record["predictions"]})
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": key, "calls": len(calls.rows),
                                  "occupied": {k: str(v) for k, v in calls.occupied.items()}}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            save(output / "predictions.json", {"targets": rows})
            save(output / "completion.json", {
                "closed_utc": now(), "fatal_error": fatal, "calls": len(calls.rows),
                "statuses": dict(Counter(r["status"] for r in calls.rows)),
                "occupied": {k: str(v) for k, v in calls.occupied.items()},
                "seconds": perf_counter() - start, "gt_not_loaded_during_inference": True})


def score(output, adapter):
    plan = read(output / "plan.json")
    if not (output / "completion.json").exists():
        raise ValueError("inference must be closed before ground truth is read")
    rows = read(output / "predictions.json")["targets"]
    done = read(output / "completion.json")
    if (done.get("fatal_error") or done["calls"] != plan["max_calls"]
            or [r["key"] for r in rows] != [s["key"] for s in plan["selection"]]):
        raise ValueError("complete declared cohort and call count required before confirmation scoring")
    wanted = {(r["video_id"], r["frame_id"]) for r in rows}
    truth = {}
    for video in {v for v, _ in wanted}:
        for resolved in adapter.iter_video(video):
            item = truth_row(resolved)
            if (item["video_id"], item["frame_id"]) in wanted:
                truth[f"{item['video_id']}_{item['frame_id']}"] = item
    save(output / "scored_truth.json", [truth[r["key"]] for r in rows])
    metrics = {}
    for name in ("h0", *ARMS):
        entry, f1s, precisions = {}, [], []
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                item = truth[row["key"]]
                if not item["mask"].get(task):
                    continue
                got, gt = set(row["predictions"][name][task]), set(item["gt"][task] or [])
                tp, fp, fn = tp + len(got & gt), fp + len(got - gt), fn + len(gt - got)
            f1 = round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None
            precision = round(100 * tp / (tp + fp), 2) if tp + fp else None
            entry[task] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "precision": precision}
            f1s.append(f1 or 0)
            precisions.append(precision or 0)
        entry["mean_f1"] = round(sum(f1s) / len(f1s), 2)
        entry["mean_precision"] = round(sum(precisions) / len(precisions), 2)
        entry["errors"] = sum(entry[t]["fp"] + entry[t]["fn"] for t in TASKS)
        metrics[name] = entry
    candidate = metrics["v2.0.1"]
    passed = all(candidate["mean_f1"] > metrics[b]["mean_f1"]
                 and candidate["mean_precision"] > metrics[b]["mean_precision"]
                 and candidate["errors"] < metrics[b]["errors"] for b in ("h0", "control"))
    deltas = {b: {"mean_f1": round(candidate["mean_f1"] - metrics[b]["mean_f1"], 2),
                  "mean_precision": round(candidate["mean_precision"] - metrics[b]["mean_precision"], 2),
                  "errors": candidate["errors"] - metrics[b]["errors"]} for b in ("h0", "control")}
    save(output / "metrics.json", {"scored_utc": now(), "targets": len(rows), "metrics": metrics,
                                   "predeclared_standard": STANDARD, "passed": passed, "deltas": deltas})
    print(f"{'arm':<10}" + "".join(f"{t:>11}" for t in TASKS) + f"{'meanF1':>9}{'meanP':>9}{'errors':>8}")
    for name, entry in metrics.items():
        print(f"{name:<10}" + "".join(f"{entry[t]['f1']:>11}" for t in TASKS)
              + f"{entry['mean_f1']:>9}{entry['mean_precision']:>9}{entry['errors']:>8}")
    print("\ndeltas of v2.0.1:", json.dumps(deltas))
    print("PREDECLARED STANDARD PASSED:", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    dataset = joint.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.source, dataset)
    elif args.command == "preflight":
        preflight(args.output, dataset)
    elif args.command == "execute":
        execute(args.output, dataset)
    else:
        score(args.output, dataset)
