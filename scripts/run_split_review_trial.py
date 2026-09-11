"""Paired compact joint/split review on eight archived Training targets."""
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

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_compact_verifier_trial import hydrate
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_new_training_verb_guard_trial import InferenceOnlyAdapter
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_feedback_continuation import metadata_preflight
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2
from scripts.run_repair_revision_trial import normalize_five
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification import split_review as split
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.phase_extension import apply_phase_choices

SOURCE = ROOT / "artifacts/preflight/compact_verifier_paired_20260909_v2_continued"
ARMS = ("control", "split")
STAGES = ("control", "components", "relations")
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
MAX_CALLS = 120
RULE = ("Eight previously inspected Training targets, development only. Same H0, pool, compact baseline, "
        "cached compact Phase, models and mean4 admission. Equal-head mean F1 must improve; IVT F1 and "
        "mean Precision must not fall; summed FP+FN must decrease. Report individual regressions and "
        "failures, do not promote automatically or tune after scoring. No retries or new candidates.")


def same(a, b):
    if a != b:
        raise ValueError("frozen protocol mismatch")


def verify(output):
    plan = read(output / "plan.json")
    same((plan["profile"], plan["limits"], plan["max_calls"], plan["rule"]), (split.PROFILE, LIMITS, MAX_CALLS, RULE))
    for name, digest in plan["sources"].items():
        same(sha(ROOT / name), digest)
    for name, digest in plan["inputs"].items():
        same(sha(output / name), digest)
    for row in plan["selection"]:
        for image in row["images"]:
            same(sha(image["path"]), image["sha256"])
    return plan


def bodies(output, selected, adapter):
    base = build_gemini_base(InferenceOnlyAdapter(adapter), selected)
    data = [im.content for im in base.images]
    return {stage: {s: hydrate(read(output / "requests" / selected["key"] / f"{stage}_{s}.json"), data)
        for s in SEATS} for stage in STAGES}


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    old, done = read(SOURCE / "plan.json"), read(SOURCE / "completion.json")
    if done["fatal_error"] or not read(SOURCE / "budget.json")["stopped"]:
        raise ValueError("source must be closed")
    for name, digest in done["hashes"].items():
        same(sha(SOURCE / name), digest)
    selection, initials = old["selection"], read(SOURCE / "initial_state.json")
    same(len(selection), 8)
    for selected in selection:
        key = selected["key"]
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        record = read(SOURCE / "targets" / key / "compact.json")
        initial = initials[key]
        initial["phase_raw"] = record["raw"]["phase"]
        same(apply_phase_choices(record["graph"], initial["phase_raw"], 3)[0], record["final"])
        for seat in SEATS:
            body = read(SOURCE / "requests" / key / f"compact_graph_{seat}.json")
            packet = json.loads(body["messages"][0]["content"][0]["text"])
            same(packet["propositions"], initial["pool"]["propositions"])
            same(body["model"], MODELS[seat])
            same(next(iter(packet)), "academic_context")
            save(output / "requests" / key / f"control_{seat}.json", body)
            for group in split.GROUPS:
                save(output / "requests" / key / f"{group}_{seat}.json", split.split_wire(body, group))
        bodies(output, selected, adapter)
    save(output / "initial_state.json", initials)
    save(output / "metadata.json", metadata_preflight())
    # Complete runtime snapshot, including ontology resources omitted by older packages.
    deps = {*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "DEFAULT_PIPELINE_VERSION.json",
        ROOT / "configs/perception/joint_openrouter_h0.yaml", ROOT / "tests/unit/test_split_review.py"}
    for path in sorted(deps):
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    inputs = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts]
    save(output / "plan.json", {"profile": split.PROFILE, "created_utc": now(), "selection": selection,
        "models": MODELS, "providers": PROVIDERS, "rates": RATES_V2, "limits": LIMITS,
        "max_calls": MAX_CALLS, "rule": RULE, "source_completion_sha256": sha(SOURCE / "completion.json"),
        "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(deps)},
        "inputs": {p.relative_to(output).as_posix(): sha(p) for p in inputs},
        "concurrency": {"control": 5, "split": 10, "targets": 1},
        "phase": "same cached compact five-seat answers in both arms; no new Phase costs or timing",
        "scope": "candidate review only; no H0/proposal calls, no Tracker/Gate, no new candidate discovery"})
    verify(output)
    print(json.dumps({"prepared": True, "targets": [s["key"] for s in selection], "max_calls": MAX_CALLS,
        "limits": LIMITS, "plan_sha256": sha(output / "plan.json")}), flush=True)


class BoundCalls(TimedCalls):
    def __init__(self, output):
        super().__init__(output, previous_budget={k: "0" for k in LIMITS},
            limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2, providers=PROVIDERS,
            max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.attempted = set()

    def persist(self):
        # Windows may briefly lock the ledger: only retry the local atomic write.
        for attempt in range(20):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)

    def call(self, target, stage, seat, body):
        same(redact_images(body), read(self.output / "requests" / target / f"{stage}_{seat}.json"))
        with self.lock:
            identity = (target, stage, seat)
            if identity in self.attempted:
                raise ValueError("no retry")
            self.attempted.add(identity)
        return super().call(target, stage, seat, body)


def run_arm(calls, key, initial, wire, arm):
    stages = ("control",) if arm == "control" else tuple(split.GROUPS)
    jobs = [(g, s) for g in stages for s in SEATS]
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        values = list(executor.map(lambda x: calls.call(key, x[0], x[1], wire[x[0]][x[1]]), jobs))
    raw = {g: {s: v for (group, s), v in zip(jobs, values, strict=True) if g == group} for g in stages}
    request_seconds = perf_counter() - start
    pool = initial["pool"]
    if arm == "control":
        reviews, formatting = normalize_five(raw["control"], pool, 3)
        means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
        result = {"prediction": panel.select(initial["h0"], pool, means, threshold=4),
            "means": means, "diagnostics": diagnostics, "formatting": formatting, "reviews": reviews}
    else:
        result = split.combine(initial["h0"], pool, raw)
    result["final"], result["phase_decision"] = apply_phase_choices(result["prediction"], initial["phase_raw"], 3)
    result.update(raw=raw, request_seconds=request_seconds, seconds=perf_counter() - start)
    return result


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls, initials = BoundCalls(output), read(output / "initial_state.json")
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
        "h0": initials[s["key"]]["h0"], **{a: apply_phase_choices(initials[s["key"]]["h0"],
            initials[s["key"]]["phase_raw"], 3)[0] for a in ARMS}} for s in plan["selection"]]
    start, fatal = perf_counter(), None
    try:
        for index, (selected, row) in enumerate(zip(plan["selection"], rows, strict=True)):
            wire = bodies(output, selected, adapter)
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
                if calls.stopped:
                    continue
                result = run_arm(calls, row["key"], initials[row["key"]], wire, arm)
                save(output / "targets" / row["key"] / f"{arm}.json", result)
                row[arm] = result["final"]
                save(output / "predictions.json", rows)
                print(json.dumps({"target": row["key"], "arm": arm, "post_calls": len(calls.rows),
                    "seconds": result["seconds"], "stopped": calls.stopped}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", rows)
        try:
            verify(output)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            files = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
            save(output / "completion.json", {"fatal_error": fatal, "closed_utc": now(),
                "seconds": perf_counter() - start, "post_calls": len(calls.rows),
                "hashes": {p.relative_to(output).as_posix(): sha(p) for p in files}})


def score(output, adapter):
    plan = verify(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal inference required")
    for name, digest in done["hashes"].items():
        same(sha(output / name), digest)
    calls = ledger["calls"]
    same(len(calls), done["post_calls"])
    same(len(calls), len({(c["target"], c["stage"], c["seat"]) for c in calls}))
    for account, limit in LIMITS.items():
        same(sum(Decimal(c["charge"]) for c in calls if c["account"] == account), Decimal(ledger["occupied"][account]))
        if Decimal(ledger["occupied"][account]) > Decimal(limit):
            raise ValueError("budget exceeded")
    replay, initials, rows = ReplayCalls(output, calls), read(output / "initial_state.json"), read(output / "predictions.json")
    records = {}
    for selected, row in zip(plan["selection"], rows, strict=True):
        same((row["key"], row["video_id"], row["frame_id"], row["h0"]),
             (selected["key"], selected["video_id"], selected["frame_id"], initials[selected["key"]]["h0"]))
        wire = bodies(output, selected, adapter)
        for arm in ARMS:
            path = output / "targets" / row["key"] / f"{arm}.json"
            if not path.exists():
                same(row[arm], apply_phase_choices(row["h0"], initials[row["key"]]["phase_raw"], 3)[0])
                continue
            fresh = run_arm(replay, row["key"], initials[row["key"]], wire, arm)
            stored = read(path)
            same({k: v for k, v in fresh.items() if k not in ("seconds", "request_seconds")},
                 {k: v for k, v in stored.items() if k not in ("seconds", "request_seconds")})
            same(row[arm], fresh["final"])
            records[row["key"], arm] = stored
        same(row["control"]["phase"], row["split"]["phase"])
    same(len(replay.rows), len(calls))
    _, truth = score_saved(adapter, [{**r, "h1": None, "final": r["split"]} for r in rows])
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    metrics = {a: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[a]} for r in rows])["arms"]["final"] for a in ("h0", *ARMS)}
    deltas = {a: [{"key": r["key"], **frame_delta(r[a], r["split"], truths[r["video_id"], r["frame_id"]]["gt"],
        truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows] for a in ("h0", "control")}
    summaries = {a: summarize_deltas(ds) for a, ds in deltas.items()}
    totals = {}
    for arm in ARMS:
        own = [c for c in calls if (c["stage"] == "control") == (arm == "control")]
        totals[arm] = {"post_calls": len(own), "statuses": dict(Counter(c["status"] for c in own)),
            "costs": {acc: {kind: str(sum(Decimal(c["charge"]) for c in own if c["account"] == acc and c["charge_kind"] == kind))
                for kind in ("native", "conservative_estimate", "unknown_reserved")} for acc in LIMITS},
            "review_wall_seconds": sum(r["request_seconds"] for (key, a), r in records.items() if a == arm),
            "valid_candidates": sum(sum(v is not None for v in r["means"].values()) for (key, a), r in records.items() if a == arm),
            "reported_prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) or 0 for c in own),
            "missing_usage_calls": sum("prompt_tokens" not in c.get("usage", {}) for c in own)}
    before, after = metrics["control"]["tasks"], metrics["split"]["tasks"]
    checks = {"all_calls": len(calls) == MAX_CALLS,
        "mean_f1_improved": sum(h["micro_f1"] for h in after.values()) > sum(h["micro_f1"] for h in before.values()),
        "mean_precision_not_decreased": sum(h["micro_precision"] for h in after.values()) >= sum(h["micro_precision"] for h in before.values()),
        "ivt_f1_not_decreased": after["ivt"]["micro_f1"] >= before["ivt"]["micro_f1"],
        "net_errors_reduced": summaries["control"]["net_errors_removed"] > 0}
    report = {"metrics": metrics, "changes": summaries, "totals": totals, "checks": checks,
        "success": all(checks.values()), "raw_replayed": True, "elapsed_seconds": done["seconds"],
        "scope": RULE, "timing_scope": "review-only wall time; cached H0/proposal/Phase excluded; max5 vs max10 concurrency"}
    save(output / "metrics.json", report)
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", deltas)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    globals()[args.command](args.output, adapter)
