"""Frozen 13-call mainline: original four-head panel plus joint Phase; no blind vote.

CLI: replay, prepare, preflight, execute, score, freeze (see mainline_release).
No Gate training data is collected. Historical experiment runners are preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_prior_gated_joint_confirmation as old
from surgical_agent.research.verification.serialized_ledger import SerializedLedgerMixin

VERSION = "prior-gated-joint-mainline-v1.0.0"
STAGES = {"h0": 1, "proposal": 1, "phase_recommendation": 1, "control_graph": 5, "joint_r1": 5}
CALLS_PER_TARGET = sum(STAGES.values())
PRIMARY = "gated_control_jointphase"
ARMS = ("h0", "gated_control", PRIMARY)


class MainlineCalls(SerializedLedgerMixin, old.joint.roster.GLMCalls):
    """Same model transports, single-writer budget persistence."""


def run_target(calls, base, selected, prior, gate):
    key, video = selected["key"], selected["video_id"]
    if prior["excluded_video"] != video or video in prior["fit_videos"]:
        raise ValueError("query video present in prior")
    timing = {}
    stamp = perf_counter()
    raw = calls.call(key, "h0", "base", old.gemini_h0_wire(base))
    h0 = old.gated.h0_from_raw(raw)
    timing["h0"] = perf_counter() - stamp
    pool = old.make_pool(h0)
    hints = old.retrieve_candidate_hints(h0, prior, video_id=video)
    stamp = perf_counter()
    proposal = calls.call(key, "proposal", "base", old.proposal_wire(base, selected, h0, pool, hints["packet"]))
    timing["proposal"] = perf_counter() - stamp
    if proposal is not None:
        try:
            pool = old.make_pool(h0, proposal, pool)
        except ValueError:
            proposal = {"invalid": proposal}
    compact = {s: old.joint.roster.review_wire(s, base, selected, pool) for s in old.SEATS}
    recommendation = old.phase_recommendation_wire(base, selected, h0, pool, hints)
    old.check_requests(h0, [*compact.values(), recommendation])
    record = {}

    def joint_branch():
        stamp = perf_counter()
        rec = calls.call(key, "phase_recommendation", "base", recommendation)
        error = old.joint.phase_choice_error(rec, 3)
        record["phase_recommendation"] = {"raw": rec, "error": error}
        wires = old.joint_review_wires(base, selected, h0, pool, None if error else rec["phase_id"])
        old.check_requests(h0, wires.values())
        record["joint_raw"] = old.joint.panel_call(calls, key, "joint_r1", wires)
        record["joint_branch_seconds"] = perf_counter() - stamp

    stamp = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as workers:
        graph = workers.submit(old.joint.panel_call, calls, key, "control_graph", compact)
        phase = workers.submit(joint_branch)
        review_raw = graph.result()
        phase.result()
    timing["panels_parallel"] = perf_counter() - stamp
    timing["joint_branch"] = record.pop("joint_branch_seconds")
    predictions, detail = old.decide_record(h0, pool, review_raw, None, record["joint_raw"], prior, gate)
    return {"key": key, "h0_raw": raw, "h0": h0, "hints": hints, "proposal_raw": proposal,
            "pool": pool, "review_raw": review_raw, **record, **detail,
            "predictions": {a: predictions[a] for a in ARMS}, "timing_seconds": timing}


def replay(source, output):
    from scripts.resume_prior_gated_joint_confirmation import ResumeCalls

    if output.exists():
        raise ValueError("use a fresh replay output")
    plan = old.read(source / "plan.json")
    saved = old.read(source / "predictions.json")["targets"]
    adapter = old.joint.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases = old.collector.validation_bases(adapter, plan)
    calls = ResumeCalls(output, source, plan, offline=True)
    prior = old.read(source / "priors" / f"{plan['video']}.json")
    results = []
    with old.joint.roster.lightweight_protocol():
        for selected, row in zip(plan["selection"], saved, strict=True):
            if selected["key"] != row["key"]:
                raise ValueError("target identity mismatch")
            record = run_target(calls, bases[row["key"]], selected, prior, plan["gate"])
            for arm in ARMS:
                if json.dumps(record["predictions"][arm], sort_keys=True) != json.dumps(row["predictions"][arm], sort_keys=True):
                    raise ValueError("prediction byte mismatch: " + row["key"])
            if any(record["predictions"][PRIMARY][t] != row["predictions"]["gated_control"][t]
                   for t in old.TASKS[:4]):
                raise ValueError("four heads changed")
            results.append({"key": row["key"], "all_three_arms_equal": True, "four_heads_equal": True})
    if len(calls.used) != len(saved) * CALLS_PER_TARGET or any(k[1] == "control_phase" for k in calls.used):
        raise ValueError("unexpected replay request set")
    report = {"version": VERSION, "source": str(source), "api_calls": 0,
              "request_identity_checks": len(calls.used), "targets": results,
              "prediction_serialization": "canonical JSON, preserving array order",
              "source_predictions_sha256": old.sha(source / "predictions.json")}
    old.save(output / "replay.json", report)
    print(json.dumps({"replayed_targets": len(results), "requests_matched": len(calls.used), "api_calls": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "prepare", "preflight", "execute", "score", "freeze"))
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/prior_gated_joint_vid110_confirm_20260911_v1_resume1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.source, args.output)
    else:
        from scripts import prior_gated_mainline_release as release
        getattr(release, args.command)(args.output)


if __name__ == "__main__":
    main()
