"""Confirm the unified five-head candidate on fresh Validation targets, once.

Design under test (`prior_gated_joint.py`): every head goes propose -> five-seat
verify -> Python admit.

  four heads  graph proposal -> five compact reviews -> frozen mean selector ->
              prior gate (veto < 0.01, add >= 0.70, H0 Phase bucket; frozen)
  Phase       one visual Phase recommendation (no prior hints) -> the same five
              seats rate the four-head pool and all seven Phases jointly ->
              unique best >= 4 that beats the current Phase replaces it

Arms scored on the same H0, pool, images and saved answers:

  h0                        frozen initial prediction
  control                   v1.3.0 default: compact four heads + blind Phase majority
  gated_control             prior-gated confirmation candidate, Phase frozen to H0
  gated_control_jointphase  gated_control four heads + jointly verified Phase   <- primary
  gated_joint_jointphase    gate over the joint panel's four-head means + joint Phase (one panel)

Leakage: reviewers and the recommender see H0 and the raw pool only (asserted on
every request in preflight and execute); the recommender gets no prior hints;
the prior excludes the query video; targets are fresh VID110 frames farther than
`gap` raw frames from every image used by any earlier VID110 experiment; GT is
read only by `score` after `completion.json` exists. Gate and Phase thresholds
are frozen from earlier work and not tuned here.

Predeclared standard, fixed before any call: gated_control_jointphase must have
four heads identical to gated_control (structural), mean F1 above and total
label errors below h0, control and gated_control, mean precision above h0 and
control, Verb and IVT F1 no lower than control, and must not break a Phase that
H0 had right. gated_joint_jointphase is reported, not judged.

Commands: prepare (zero API), preflight (zero API), execute (single-use paid), score.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import collect_validation_inputs as collector
from scripts import run_joint_phase_feedback_trial as joint
from scripts import run_prior_gated_confirmation as gated
from scripts import run_split_review_trial as common
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_openrouter_gemini_h0_trial import gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_panel_trial import now, read, save, truth_row
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.research.retrieval.prior_candidates import (
    retrieve_candidate_hints,
)
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
    make_pool,
)
from surgical_agent.research.verification.prior_gated_joint import (
    ARMS,
    PHASE_THRESHOLD,
    PRIMARY,
    VERSION,
    assert_ungated_request,
    decide,
    joint_pool,
)

PROFILE = "prior_gated_joint_phase_validation_confirmation_v1"
TASKS = joint.TASKS
GATE = dict(gated.GATE)
JOINT_PROMPT_VARIANT = "v1"
PRIOR_SOURCE = gated.PRIOR_SOURCE
USED_PLANS = [*gated.USED_PLANS, ROOT / "artifacts/preflight/prior_gated_vid110_confirm_20260911_v1/plan.json"]
LIMITS = {"openrouter_usd": "3", "aliyun_cny": "2", "xai_usd": "0"}
STAGES = {"h0": 1, "proposal": 1, "phase_recommendation": 1, "control_graph": 5, "control_phase": 5, "joint_r1": 5}
CALLS_PER_TARGET = sum(STAGES.values())  # 18
STANDARD = ("gated_control_jointphase must keep the four heads of gated_control byte-for-byte, exceed h0, "
            "control and gated_control on mean F1, have fewer total label errors than all three, exceed h0 "
            "and control on mean precision, have Verb F1 and IVT F1 no lower than control, and break no Phase "
            "that H0 had right. Any other outcome is a failure and is reported as one.")
RUNTIME_FILES = [Path(__file__).resolve(), ROOT / "scripts/run_prior_gated_confirmation.py",
                 ROOT / "scripts/run_joint_phase_feedback_trial.py", ROOT / "scripts/collect_validation_inputs.py",
                 ROOT / "src/surgical_agent/research/verification/prior_gated_joint.py",
                 ROOT / "src/surgical_agent/research/verification/prior_gated_repair.py",
                 ROOT / "src/surgical_agent/research/verification/recent_mean_panel.py",
                 ROOT / "src/surgical_agent/research/verification/five_head_repair.py",
                 ROOT / "src/surgical_agent/research/verification/phase_extension.py",
                 ROOT / "src/surgical_agent/research/retrieval/prior_candidates.py",
                 ROOT / f"src/surgical_agent/research/verification/prompts/joint_phase_feedback_{JOINT_PROMPT_VARIANT}.txt"]


def used_frames():
    used = set()
    for path in USED_PLANS:
        for s in read(path)["selection"]:
            used.update(s["causal_frame_ids"])
    return used


def phase_recommendation_wire(base, selected, h0, pool, hints):
    """Visual Phase recommendation: H0 and raw pool as context, no prior relation hints."""
    body = joint.phase_proposal(base, selected, {"h0": h0, "pool": pool, "hints": hints})
    packet = joint.packet_of(body)
    packet.pop("candidate_relation_hints", None)
    return joint.put_packet(body, packet)


def joint_review_wires(base, selected, h0, pool, recommendation):
    jpool = joint_pool(pool)
    return {s: joint.joint_wire(s, base, selected, jpool, h0, recommendation, JOINT_PROMPT_VARIANT) for s in SEATS}


def check_requests(h0, bodies):
    for body in bodies:
        assert_ungated_request(joint.packet_of(body), h0)


def prepare(output, adapter, video, count, gap):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    if adapter.entries[video].split is not DatasetSplit.VALIDATION:
        raise ValueError("this confirmation is for the Validation split")
    prior = read(PRIOR_SOURCE)
    if prior["excluded_video"] != video or video in prior["fit_videos"]:
        raise ValueError("prior must exclude the query video")
    used = used_frames()
    selection = gated.choose(adapter, video, count, gap, used)
    if {s["frame_id"] for s in selection} & used:
        raise ValueError("fresh selection overlaps used frames")
    save(output / "priors" / f"{video}.json", prior)
    for path in RUNTIME_FILES:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    plan = {"profile": PROFILE, "created_utc": now(), "video": video, "selection": selection,
            "gap": gap, "used_frames_excluded": sorted(used), "used_plans": [str(p) for p in USED_PLANS],
            "selection_rule": (f"{count} time quantiles of {video}; nearest sample with three real causal frames, "
                               f"all five task masks valid, and every image id more than {gap} raw frames from "
                               "every image of every earlier VID110 experiment and of already chosen targets. "
                               "Masks only, no label values."),
            "arms": list(ARMS), "candidate": PRIMARY, "version": VERSION,
            "gate": {**GATE, "version": gated.GATE_VERSION}, "phase_threshold": PHASE_THRESHOLD,
            "joint_prompt_variant": JOINT_PROMPT_VARIANT,
            "threshold_selection": ("Gate fixed on 136 Training targets by tools/audit/prior_gated_replay.py; Phase "
                                    "rule fixed by phase_extension.phase_apply (2026-09-09). Nothing tuned here."),
            "models": joint.roster.MODELS, "families": joint.roster.FAMILIES, "transports": joint.roster.TRANSPORTS,
            "limits": LIMITS, "credential_root": str(ROOT.resolve()), "stages": STAGES,
            "max_calls": CALLS_PER_TARGET * len(selection), "automatic_retries": 0,
            "prior_sha256": sha(output / "priors" / f"{video}.json"), "prior_source": str(PRIOR_SOURCE),
            "predeclared_standard": STANDARD,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_FILES},
            "policy": ("One Gemini H0, one graph proposal, one visual Phase recommendation without prior hints, "
                       "one five-seat compact review, one five-seat blind Phase vote and one five-seat joint review "
                       "per target. Every request describes H0 and the raw pool only; the gate runs after all "
                       "answers are saved. No retries, no GT in any request, no default change."),
            "leakage_controls": [
                "assert_ungated_request on every review and recommendation packet (preflight and execute).",
                "Phase recommender receives no candidate_relation_hints; gate and Phase share no prior table.",
                "Prior table is leave-query-video-out; excluded_video re-checked at load.",
                "Gate bucket is the H0 Phase, so the verified Phase cannot feed back into the four heads.",
                "Fresh frames > gap from all earlier VID110 experiment images; GT read only in score."],
            "limitations": [
                "Same Validation video as the earlier 48 VID110 targets; new frames, not a new surgery.",
                "The joint Phase rule fixed 1 and broke 0 Phase on each of two archived batches; expected effect is small.",
                "The gate encodes Training annotation conventions, not new visual evidence.",
                "H0 is regenerated here, so H0 variance is not separated from mechanism effect."]}
    save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "targets": len(selection), "max_calls": plan["max_calls"],
                      "frames": [s["frame_id"] for s in selection], "api_calls": 0}), flush=True)


def verify_plan(plan, output):
    if plan["profile"] != PROFILE or plan["models"] != joint.roster.MODELS or plan["version"] != VERSION:
        raise ValueError("profile, roster or version mismatch")
    if plan["gate"] != {**GATE, "version": gated.GATE_VERSION} or plan["phase_threshold"] != PHASE_THRESHOLD:
        raise ValueError("frozen thresholds changed")
    if plan["prior_sha256"] != sha(output / "priors" / f"{plan['video']}.json"):
        raise ValueError("prior changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("runtime source changed: " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            if sha(image["path"]) != image["sha256"]:
                raise ValueError("frozen image changed")


class LedgerCalls(joint.roster.GLMCalls):
    """Same transport; the budget file write retries on the Windows file-lock error.

    Added after the first execution was interrupted by PermissionError on
    budget.json with three panels in flight (archive v1, recovered by
    resume_prior_gated_joint_confirmation.py). Local file retry only; no API retry.
    """

    def persist(self):
        for attempt in range(40):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(.05)


def decide_record(h0, pool, compact_raw, phase_raw, joint_raw, prior, gate):
    return decide(h0, pool, prior, compact_raw=compact_raw, joint_raw=joint_raw, phase_raw=phase_raw, gate=gate,
                  apply_phase_choices=common.apply_phase_choices, normalize_compact=common.normalize_five)


def run_target(calls, base, selected, prior, gate):
    key, video = selected["key"], selected["video_id"]
    timing = {}
    stamp = perf_counter()
    raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
    if raw is None:
        raise ValueError("H0 call failed; preserve the failure")
    h0 = gated.h0_from_raw(raw)
    timing["h0"] = perf_counter() - stamp
    pool = make_pool(h0)
    hints = retrieve_candidate_hints(h0, prior, video_id=video)
    stamp = perf_counter()
    proposal = calls.call(key, "proposal", "base", proposal_wire(base, selected, h0, pool, hints["packet"]))
    timing["proposal"] = perf_counter() - stamp
    if proposal is not None:
        try:
            pool = make_pool(h0, proposal, pool)
        except ValueError:
            proposal = {"invalid": proposal}
    compact_wires = {s: joint.roster.review_wire(s, base, selected, pool) for s in SEATS}
    phase_wires = {s: joint.roster.phase_wire(s, selected) for s in SEATS}
    recommendation_wire = phase_recommendation_wire(base, selected, h0, pool, hints)
    check_requests(h0, [*compact_wires.values(), recommendation_wire])
    record = {}

    def joint_branch():
        stamp = perf_counter()
        rec = calls.call(key, "phase_recommendation", "base", recommendation_wire)
        error = joint.phase_choice_error(rec, 3)
        record["phase_recommendation"] = {"raw": rec, "error": error}
        wires = joint_review_wires(base, selected, h0, pool, None if error else rec["phase_id"])
        check_requests(h0, wires.values())
        record["joint_raw"] = joint.panel_call(calls, key, "joint_r1", wires)
        record["timing_joint_branch"] = perf_counter() - stamp

    stamp = perf_counter()
    with ThreadPoolExecutor(max_workers=3) as workers:
        graph = workers.submit(joint.panel_call, calls, key, "control_graph", compact_wires)
        phase = workers.submit(joint.panel_call, calls, key, "control_phase", phase_wires)
        branch = workers.submit(joint_branch)
        review_raw, phase_raw = graph.result(), phase.result()
        branch.result()
    timing["panels_parallel"] = perf_counter() - stamp
    timing["joint_branch"] = record.pop("timing_joint_branch")
    predictions, detail = decide_record(h0, pool, review_raw, phase_raw, record["joint_raw"], prior, gate)
    return {"key": key, "h0_raw": raw, "h0": h0, "hints": hints, "proposal_raw": proposal, "pool": pool,
            "review_raw": review_raw, "phase_raw": phase_raw, **record, **detail,
            "predictions": predictions, "timing_seconds": timing}


class MockCalls:
    """Zero-API stand-in: exercises every wire, schema, leakage check and decision path."""

    def __init__(self):
        self.rows = []

    def call(self, target, stage, seat, body):
        self.rows.append({"target": target, "stage": stage, "seat": seat, "model": body.get("model")})
        packet = json.loads(body["messages"][0]["content"][0]["text"]) if stage != "h0" else {}
        if {"gt", "ground_truth", "labels"} & packet.keys():
            raise ValueError("request must not carry ground truth")
        if stage == "h0":
            return {"schema_version": FINAL_ONLY_SCHEMA_VERSION, "instrument": {"selected_ids": [0, 2]},
                    "verb": {"selected_ids": [1, 2]}, "target": {"selected_ids": [0, 1]},
                    "ivt": {"selected_ids": [17, 59]}, "phase": {"selected_id": 3}}
        if stage == "proposal":
            return {"instrument": [], "verb": [], "target": [], "ivt": [60]}
        if stage == "phase_recommendation":
            if "candidate_relation_hints" in packet:
                raise ValueError("prior hints reached the Phase recommender")
            return {"phase_id": 1, "image_indices": [2], "observation": "Offline preflight answer."}
        if stage == "control_phase":
            return {"phase_id": 3, "image_indices": [2], "observation": "Offline preflight answer."}
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [],
                "observation": "Offline preflight answer."}
        judgments = {p["id"]: dict(item) for p in packet["propositions"]}
        if stage == "joint_r1":
            for pid in judgments:
                if pid.startswith("phase_"):
                    supported = pid == "phase_1"
                    judgments[pid] = {"rating": 5 if supported else 1,
                                      "finding": "MATCH" if supported else "REFUTED",
                                      "scope": "WHOLE_FRAME", "image_indices": [2],
                                      "observation": "Offline preflight answer."}
        return {"judgments": judgments}


def preflight(output, adapter):
    plan = read(output / "plan.json")
    verify_plan(plan, output)
    prior = read(output / "priors" / f"{plan['video']}.json")
    bases = collector.validation_bases(adapter, plan)
    checks, fingerprints = [], {}
    for selected in plan["selection"]:
        key = selected["key"]
        calls = MockCalls()
        record = run_target(calls, bases[key], selected, prior, plan["gate"])
        if len(calls.rows) != CALLS_PER_TARGET or dict(Counter(r["stage"] for r in calls.rows)) != STAGES:
            raise ValueError("unexpected call count in preflight")
        pool, h0 = record["pool"], record["h0"]
        wires = {"control_graph": {s: joint.roster.review_wire(s, bases[key], selected, pool) for s in SEATS},
                 "control_phase": {s: joint.roster.phase_wire(s, selected) for s in SEATS},
                 "joint_r1": joint_review_wires(bases[key], selected, h0, pool, 1),
                 "phase_recommendation": {"base": phase_recommendation_wire(bases[key], selected, h0, pool,
                                                                            record["hints"])}}
        fingerprints[key] = {stage: {s: fingerprint(b) for s, b in group.items()} for stage, group in wires.items()}
        p = record["predictions"]
        # fixture: hook/dissect/cystic_plate in phase 3 is vetoed, 60 admitted, joint Phase 3 -> 1
        if record["gate_log_control"]["vetoed"] != [59] or record["gate_log_control"]["prior_added"] != [60]:
            raise ValueError("gate did not behave as specified on the preflight fixture")
        if p[PRIMARY]["phase"] != [1] or p["gated_control"]["phase"] != [3] or p["control"]["phase"] != [3]:
            raise ValueError("Phase admission did not behave as specified on the preflight fixture")
        if any(p[PRIMARY][t] != p["gated_control"][t] for t in TASKS[:4]):
            raise ValueError("primary arm changed the gated four heads")
        checks.append({"key": key, "arms": sorted(p), "pool": len(pool["propositions"]), "calls": len(calls.rows),
                       PRIMARY: p[PRIMARY], "phase_decision": record["phase_decision"]["reason"]})
    save(output / "wire_fingerprints.json", fingerprints)
    save(output / "offline_preflight.json", {"checked_utc": now(), "api_calls": 0, "targets": checks})
    print(json.dumps({"preflight": str(output), "api_calls": 0, "targets": len(checks),
                      "calls_per_target": CALLS_PER_TARGET}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no paid overwrite")
    verify_plan(plan, output)
    preflight_report = read(output / "offline_preflight.json")
    if [s["key"] for s in preflight_report["targets"]] != [s["key"] for s in plan["selection"]]:
        raise ValueError("complete offline preflight required")
    prior = read(output / "priors" / f"{plan['video']}.json")
    bases = collector.validation_bases(adapter, plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    rows, start, fatal = [], perf_counter(), None
    with joint.credential_context(plan), joint.roster.lightweight_protocol():
        calls = LedgerCalls(output, limits={a: Decimal(v) for a, v in plan["limits"].items()},
                            rates=joint.roster.RATES, providers=joint.roster.PROVIDERS,
                            max_calls=plan["max_calls"], reasoning_seats=("gemini",))
        try:
            for selected in plan["selection"]:
                if calls.stopped:
                    break
                key = selected["key"]
                record = run_target(calls, bases[key], selected, prior, plan["gate"])
                save(output / "targets" / key / "result.json", record)
                rows.append({"key": key, "video_id": selected["video_id"], "frame_id": selected["frame_id"],
                             "predictions": record["predictions"], "timing_seconds": record["timing_seconds"]})
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": key, "calls": len(calls.rows), "timing": record["timing_seconds"],
                                  "phase": record["phase_decision"]["reason"],
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


def replay_check(output, plan, rows, prior):
    """Every saved arm prediction must follow from the saved raw answers alone."""
    for row in rows:
        r = read(output / "targets" / row["key"] / "result.json")
        predictions, _ = decide_record(r["h0"], r["pool"], r["review_raw"], r["phase_raw"], r["joint_raw"],
                                       prior, plan["gate"])
        if predictions != row["predictions"] or predictions != r["predictions"]:
            raise ValueError("raw-answer replay differs from saved predictions: " + row["key"])


def phase_edits(rows, truth, arm):
    c = Counter()
    for row in rows:
        item, h0, pred = truth[row["key"]], row["predictions"]["h0"], row["predictions"][arm]
        if item["mask"].get("phase") and pred["phase"] != h0["phase"]:
            gt = item["gt"]["phase"][0]
            c["fixed" if pred["phase"][0] == gt else ("broken" if h0["phase"][0] == gt else "wrong_to_wrong")] += 1
    return dict(sorted(c.items()))


def score(output, adapter):
    plan = read(output / "plan.json")
    if not (output / "completion.json").exists():
        raise ValueError("inference must be closed before ground truth is read")
    rows = read(output / "predictions.json")["targets"]
    done = read(output / "completion.json")
    if (done.get("fatal_error") or done["calls"] != plan["max_calls"]
            or [r["key"] for r in rows] != [s["key"] for s in plan["selection"]]):
        raise ValueError("complete declared cohort and call count required before confirmation scoring")
    prior = read(output / "priors" / f"{plan['video']}.json")
    replay_check(output, plan, rows, prior)
    for row in rows:
        if any(row["predictions"][PRIMARY][t] != row["predictions"]["gated_control"][t] for t in TASKS[:4]):
            raise ValueError("primary arm four heads differ from gated_control: " + row["key"])
    wanted = {(r["video_id"], r["frame_id"]) for r in rows}
    truth = {}
    for video in {v for v, _ in wanted}:
        for resolved in adapter.iter_video(video):
            item = truth_row(resolved)
            if (item["video_id"], item["frame_id"]) in wanted:
                truth[f"{item['video_id']}_{item['frame_id']}"] = item
    save(output / "scored_truth.json", [truth[r["key"]] for r in rows])
    metrics = gated.metrics_for(rows, truth, ARMS)
    c = metrics[PRIMARY]
    edits = {a: phase_edits(rows, truth, a) for a in ARMS if a != "h0"}
    passed = (all(c["mean_f1"] > metrics[b]["mean_f1"] and c["errors"] < metrics[b]["errors"]
                  for b in ("h0", "control", "gated_control"))
              and all(c["mean_precision"] > metrics[b]["mean_precision"] for b in ("h0", "control"))
              and (c["verb"]["f1"] or 0) >= (metrics["control"]["verb"]["f1"] or 0)
              and (c["ivt"]["f1"] or 0) >= (metrics["control"]["ivt"]["f1"] or 0)
              and edits[PRIMARY].get("broken", 0) == 0)
    deltas = {b: {"mean_f1": round(c["mean_f1"] - metrics[b]["mean_f1"], 2),
                  "mean_precision": round(c["mean_precision"] - metrics[b]["mean_precision"], 2),
                  "errors": c["errors"] - metrics[b]["errors"]} for b in ("h0", "control", "gated_control")}
    reasons = Counter(read(output / "targets" / r["key"] / "result.json")["phase_decision"]["reason"] for r in rows)
    timing = {k: round(sum(r["timing_seconds"][k] for r in rows) / len(rows), 2) for k in rows[0]["timing_seconds"]}
    save(output / "metrics.json", {"scored_utc": now(), "targets": len(rows), "metrics": metrics,
                                   "predeclared_standard": STANDARD, "passed": passed, "deltas": deltas,
                                   "phase_edits_vs_h0": edits, "phase_decision_reasons": dict(reasons),
                                   "label_edits": {a: gated.edits(rows, truth, a) for a in ARMS if a != "h0"},
                                   "raw_replayed": True, "mean_timing_seconds": timing})
    print(f"{'arm':<26}" + "".join(f"{t[:5]:>8}" for t in TASKS) + f"{'meanF1':>8}{'meanP':>8}{'err':>6}")
    for name, entry in metrics.items():
        print(f"{name:<26}" + "".join(f"{(entry[t]['f1'] if entry[t]['f1'] is not None else 0):>8}" for t in TASKS)
              + f"{entry['mean_f1']:>8}{entry['mean_precision']:>8}{entry['errors']:>6}")
    print("\ndeltas of", PRIMARY, json.dumps(deltas))
    print("phase edits vs h0:", json.dumps(edits), " reasons:", dict(reasons), " mean timing:", timing)
    print("PREDECLARED STANDARD PASSED:", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", default="VID110")
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--gap", type=int, default=175)
    args = parser.parse_args()
    dataset = joint.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"),
                                                       causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, dataset, args.video, args.targets, args.gap)
    elif args.command == "preflight":
        preflight(args.output, dataset)
    elif args.command == "execute":
        execute(args.output, dataset)
    else:
        score(args.output, dataset)
