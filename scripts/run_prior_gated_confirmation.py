"""Confirm prior-gated IVT admission on fresh Validation targets, once.

Design under test (no new model calls beyond the v1.3.0 graph branch):

  H0 (Gemini, cached recipe) -> graph hints -> single proposal -> five compact
  reviewers -> frozen mean selector -> Python prior gate:
      veto a selected non-null IVT whose leave-video-out, H0-phase-conditioned
      Training rate is below 0.01; admit a pool IVT whose rate is >= 0.70 with
      its components; Phase frozen to H0.

Arms scored on the same H0, pool, images and reviewer answers:

  h0               frozen initial prediction
  control          v1.3.0 default: mean panel four heads + blind five-seat Phase vote
  control_h0phase  v1.3.0 four heads, Phase frozen to H0 (drops five Phase calls)
  gated_h0         prior gate on H0 alone (zero reviewer calls)
  gated_control    prior gate on the v1.3.0 four heads, Phase frozen to H0  <- candidate

Targets are new VID110 frames whose three causal images lie more than `gap` raw
frames from every image already used by the earlier VID110 experiments; masks
only, no label values. Gate thresholds were fixed on 136 Training targets by
`tools/audit/prior_gated_replay.py` and are not tuned here.

Predeclared standard, fixed before any call: gated_control must exceed both H0
and control on five-head mean F1 and mean precision, have fewer total label
errors than both, and have Verb F1 and IVT F1 no lower than control.

Commands: prepare (zero API), preflight (zero API), execute (single-use paid), score.
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

from scripts import collect_validation_inputs as collector
from scripts import run_joint_phase_feedback_trial as joint
from scripts import run_split_review_trial as common
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_openrouter_gemini_h0_trial import gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_panel_trial import now, read, save, truth_row
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import (
    FINAL_ONLY_SCHEMA_VERSION,
    validate_final_only,
)
from surgical_agent.research.retrieval.prior_candidates import (
    retrieve_candidate_hints,
)
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
    make_pool,
)
from surgical_agent.research.verification.prior_gated_repair import (
    VERSION as GATE_VERSION,
)
from surgical_agent.research.verification.prior_gated_repair import (
    select_prior_gated,
)
from surgical_agent.research.verification.prior_panel import labels

PROFILE = "prior_gated_validation_confirmation_v1"
TASKS = joint.TASKS
ARMS = ("h0", "control", "control_h0phase", "gated_h0", "gated_control")
CANDIDATE = "gated_control"
GATE = {"veto_rate": 0.01, "add_rate": 0.7, "prune": []}
PRIOR_SOURCE = ROOT / "artifacts/preflight/validation_vid110_inputs_20260911_v2/priors/VID110.json"
USED_PLANS = [ROOT / "artifacts/preflight/validation_vid110_inputs_20260911_v1/plan.json",
              ROOT / "artifacts/preflight/validation_vid110_inputs_20260911_v2/plan.json"]
LIMITS = {"openrouter_usd": "1.5", "aliyun_cny": "1", "xai_usd": "0"}
CALLS_PER_TARGET = 12  # h0 + proposal + 5 graph reviews + 5 blind Phase votes
STANDARD = ("gated_control must exceed both h0 and control on mean F1 and on mean precision, have fewer "
            "total label errors than both, and have Verb F1 and IVT F1 no lower than control. Any other "
            "outcome is a failure and is reported as one.")
RUNTIME_FILES = [Path(__file__).resolve(), ROOT / "scripts/collect_validation_inputs.py",
                 ROOT / "src/surgical_agent/research/verification/prior_gated_repair.py",
                 ROOT / "src/surgical_agent/research/verification/recent_mean_panel.py",
                 ROOT / "src/surgical_agent/research/retrieval/prior_candidates.py",
                 ROOT / "tools/audit/prior_gated_replay.py"]


def used_frames():
    used = set()
    for path in USED_PLANS:
        for s in read(path)["selection"]:
            used.update(s["causal_frame_ids"])
    return used


def choose(adapter, video, count, gap, used):
    """Time quantiles; masks only; every image farther than `gap` from used or chosen images."""
    samples = list(adapter.iter_inference_video(video))
    masks = {r.inference.target_frame_id: truth_row(r)["mask"] for r in adapter.iter_video(video)}
    blocked, selection = set(used), []
    for numerator in range(1, count + 1):
        anchor = samples[len(samples) * numerator // (count + 1)].target_frame_id
        eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                    and all(masks.get(s.target_frame_id, {}).get(t, False) for t in TASKS)
                    and all(abs(f - known) > gap for f in s.causal_frame_ids for known in blocked)]
        if not eligible:
            raise ValueError(f"no metadata-eligible fresh sample for {video} at quantile {numerator}")
        s = min(eligible, key=lambda x: (abs(x.target_frame_id - anchor), x.target_frame_id))
        selection.append({"key": f"{video}_{s.target_frame_id}", "video_id": video,
                          "frame_id": s.target_frame_id, "anchor_frame_id": anchor,
                          "causal_frame_ids": list(s.causal_frame_ids),
                          "gt_availability_only": masks[s.target_frame_id],
                          "images": [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                                     for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]})
        blocked.update(s.causal_frame_ids)
    return selection


def prepare(output, adapter, video, count, gap):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    if adapter.entries[video].split is not DatasetSplit.VALIDATION:
        raise ValueError("this confirmation is for the Validation split")
    prior = read(PRIOR_SOURCE)
    if prior["excluded_video"] != video or video in prior["fit_videos"]:
        raise ValueError("prior must exclude the query video")
    used = used_frames()
    selection = choose(adapter, video, count, gap, used)
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
                               "every image of the earlier VID110 experiments and of already chosen targets. "
                               "Masks only, no label values."),
            "arms": list(ARMS), "candidate": CANDIDATE, "gate": {**GATE, "version": GATE_VERSION},
            "gate_selection": "Fixed on 136 pooled Training targets by tools/audit/prior_gated_replay.py; not tuned here.",
            "models": joint.roster.MODELS, "families": joint.roster.FAMILIES, "transports": joint.roster.TRANSPORTS,
            "limits": LIMITS, "credential_root": str(ROOT.resolve()),
            "max_calls": CALLS_PER_TARGET * len(selection), "automatic_retries": 0,
            "prior_sha256": sha(output / "priors" / f"{video}.json"), "prior_source": str(PRIOR_SOURCE),
            "predeclared_standard": STANDARD,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_FILES},
            "policy": ("One Gemini H0, one graph proposal and one five-seat compact review per target, shared "
                       "by every arm; blind Phase votes only feed the control arm. Gate is Python over saved "
                       "answers. No retries, no GT in any request, no default change."),
            "limitations": [
                "Same Validation video as the earlier 32-target check; new frames, not a new surgery.",
                "The gate encodes Training annotation conventions, not new visual evidence.",
                "Gate thresholds were selected on Training archives already analysed many times.",
                "H0 is regenerated here, so H0 variance is not separated from gate effect."]}
    save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "targets": len(selection), "max_calls": plan["max_calls"],
                      "frames": [s["frame_id"] for s in selection], "api_calls": 0}), flush=True)


def verify_plan(plan, output):
    if plan["profile"] != PROFILE or plan["models"] != joint.roster.MODELS:
        raise ValueError("profile or roster mismatch")
    if plan["prior_sha256"] != sha(output / "priors" / f"{plan['video']}.json"):
        raise ValueError("prior changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("runtime source changed: " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            if sha(image["path"]) != image["sha256"]:
                raise ValueError("frozen image changed")


def h0_from_raw(raw):
    validate_final_only(raw)
    return labels({t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS})


def decide(h0, pool, review_raw, phase_raw, prior, gate):
    """All arm predictions from one panel; no transport, no GT."""
    reviews, formats = common.normalize_five(review_raw, pool, 3)
    means, diagnostics = common.panel.aggregate(reviews, pool, image_count=3)
    four = common.panel.select(h0, pool, means, threshold=4)
    control, phase_decision = common.apply_phase_choices(four, phase_raw, 3)
    kw = {"phase": h0["phase"][0], "veto_rate": gate["veto_rate"], "add_rate": gate["add_rate"],
          "prune": tuple(gate["prune"])}
    gated_h0, log_h0 = select_prior_gated(h0, pool, None, prior, **kw)
    gated_control, log_control = select_prior_gated(h0, pool, means, prior, **kw)
    predictions = {"h0": deepcopy(h0), "control": control, "control_h0phase": labels({**four, "phase": h0["phase"]}),
                   "gated_h0": gated_h0, "gated_control": gated_control}
    return predictions, {"means": means, "diagnostics": diagnostics, "format_diagnostics": formats,
                         "phase_decision": phase_decision, "gate_log_h0": log_h0, "gate_log_control": log_control}


def run_target(calls, base, selected, prior, gate):
    key, video = selected["key"], selected["video_id"]
    timing = {}
    stamp = perf_counter()
    raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
    if raw is None:
        raise ValueError("H0 call failed; preserve the failure")
    h0 = h0_from_raw(raw)
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
    stamp = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as workers:
        graph = workers.submit(joint.panel_call, calls, key, "control_graph",
                               {s: joint.roster.review_wire(s, base, selected, pool) for s in SEATS})
        phase = workers.submit(joint.panel_call, calls, key, "control_phase",
                               {s: joint.roster.phase_wire(s, selected) for s in SEATS})
        review_raw, phase_raw = graph.result(), phase.result()
    timing["panels_parallel"] = perf_counter() - stamp
    predictions, detail = decide(h0, pool, review_raw, phase_raw, prior, gate)
    return {"key": key, "h0_raw": raw, "h0": h0, "hints": hints, "proposal_raw": proposal, "pool": pool,
            "review_raw": review_raw, "phase_raw": phase_raw, **detail, "predictions": predictions,
            "timing_seconds": timing}


class MockCalls:
    """Zero-API stand-in: exercises every wire, schema and decision path."""

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
        if stage == "control_phase":
            return {"phase_id": 3, "image_indices": [2], "observation": "Offline preflight answer."}
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [],
                "observation": "Offline preflight answer."}
        return {"judgments": {p["id"]: dict(item) for p in packet["propositions"]}}


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
        if len(calls.rows) != CALLS_PER_TARGET:
            raise ValueError("unexpected call count in preflight")
        pool = record["pool"]
        wires = {"control_graph": {s: joint.roster.review_wire(s, bases[key], selected, pool) for s in SEATS},
                 "control_phase": {s: joint.roster.phase_wire(s, selected) for s in SEATS}}
        fingerprints[key] = {stage: {s: fingerprint(b) for s, b in group.items()} for stage, group in wires.items()}
        # the mock H0 carries hook/dissect/cystic_plate in phase 3: the gate must veto it and admit 60
        if record["gate_log_control"]["vetoed"] != [59] or record["gate_log_control"]["prior_added"] != [60]:
            raise ValueError("gate did not behave as specified on the preflight fixture")
        checks.append({"key": key, "arms": sorted(record["predictions"]), "pool": len(pool["propositions"]),
                       "calls": len(calls.rows), "gated_control": record["predictions"]["gated_control"]})
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
        calls = joint.roster.GLMCalls(output, limits={a: Decimal(v) for a, v in plan["limits"].items()},
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


def metrics_for(rows, truth, arms):
    metrics = {}
    for name in arms:
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
    return metrics


def edits(rows, truth, arm):
    c = Counter()
    for row in rows:
        item, h0, pred = truth[row["key"]], row["predictions"]["h0"], row["predictions"][arm]
        for task in TASKS[:4]:
            if not item["mask"].get(task):
                continue
            gt = set(item["gt"][task] or [])
            for label in set(pred[task]) - set(h0[task]):
                c[f"{task}_add_{'good' if label in gt else 'bad'}"] += 1
            for label in set(h0[task]) - set(pred[task]):
                c[f"{task}_del_{'good' if label not in gt else 'bad'}"] += 1
        if item["mask"].get("phase") and pred["phase"] != h0["phase"]:
            gt = item["gt"]["phase"][0]
            c["phase_fixed" if pred["phase"][0] == gt else ("phase_broken" if h0["phase"][0] == gt else "phase_wrong_to_wrong")] += 1
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
    wanted = {(r["video_id"], r["frame_id"]) for r in rows}
    truth = {}
    for video in {v for v, _ in wanted}:
        for resolved in adapter.iter_video(video):
            item = truth_row(resolved)
            if (item["video_id"], item["frame_id"]) in wanted:
                truth[f"{item['video_id']}_{item['frame_id']}"] = item
    save(output / "scored_truth.json", [truth[r["key"]] for r in rows])
    metrics = metrics_for(rows, truth, ARMS)
    c = metrics[CANDIDATE]
    passed = (all(c["mean_f1"] > metrics[b]["mean_f1"] and c["mean_precision"] > metrics[b]["mean_precision"]
                  and c["errors"] < metrics[b]["errors"] for b in ("h0", "control"))
              and (c["verb"]["f1"] or 0) >= (metrics["control"]["verb"]["f1"] or 0)
              and (c["ivt"]["f1"] or 0) >= (metrics["control"]["ivt"]["f1"] or 0))
    deltas = {b: {"mean_f1": round(c["mean_f1"] - metrics[b]["mean_f1"], 2),
                  "mean_precision": round(c["mean_precision"] - metrics[b]["mean_precision"], 2),
                  "errors": c["errors"] - metrics[b]["errors"]} for b in ("h0", "control")}
    per_target = Counter()
    for row in rows:
        item = truth[row["key"]]
        errs = [sum(len(set(row["predictions"][arm][t]) ^ set(item["gt"][t] or [])) for t in TASKS if item["mask"].get(t))
                for arm in (CANDIDATE, "control")]
        d = errs[0] - errs[1]
        per_target["better" if d < 0 else ("worse" if d > 0 else "same")] += 1
    timing = {k: round(sum(r["timing_seconds"][k] for r in rows) / len(rows), 2) for k in rows[0]["timing_seconds"]}
    save(output / "metrics.json", {"scored_utc": now(), "targets": len(rows), "metrics": metrics,
                                   "predeclared_standard": STANDARD, "passed": passed, "deltas": deltas,
                                   "per_target_vs_control": dict(per_target),
                                   "edits": {a: edits(rows, truth, a) for a in ARMS if a != "h0"},
                                   "mean_timing_seconds": timing})
    print(f"{'arm':<17}" + "".join(f"{t[:5]:>8}" for t in TASKS) + f"{'meanF1':>8}{'meanP':>8}{'err':>6}")
    for name, entry in metrics.items():
        print(f"{name:<17}" + "".join(f"{(entry[t]['f1'] if entry[t]['f1'] is not None else 0):>8}" for t in TASKS)
              + f"{entry['mean_f1']:>8}{entry['mean_precision']:>8}{entry['errors']:>6}")
    print("\ndeltas of gated_control:", json.dumps(deltas))
    print("per-target vs control:", dict(per_target), " mean timing:", timing)
    print("PREDECLARED STANDARD PASSED:", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", default="VID110")
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--gap", type=int, default=175)
    args = parser.parse_args()
    dataset = joint.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, dataset, args.video, args.targets, args.gap)
    elif args.command == "preflight":
        preflight(args.output, dataset)
    elif args.command == "execute":
        execute(args.output, dataset)
    else:
        score(args.output, dataset)
