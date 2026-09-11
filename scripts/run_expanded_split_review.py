"""Frozen 32-target comparison: shared fresh H0/proposal/Phase, compact joint vs split."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_compact_parallel_repair as compact
from scripts import run_split_review_trial as old
from scripts.prepare_final_only_gate_training import masks_only
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_new_training_verb_guard_trial import validate_prior
from scripts.run_openrouter_gemini_h0_trial import gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_semantic_candidate_trial import PROPOSER
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import make_pool

PROFILE = "expanded_training_compact_split_confirmation_v1"
VIDEOS = ("VID103", "VID23", "VID31", "VID96")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
LIMITS = {"openrouter_usd": "3", "xai_usd": "3", "aliyun_cny": "2"}
MAX_CALLS = 704
GAP = 175
SOURCE = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1"
RULE = ("Thirty-two new time/mask-selected Training targets in the only four videos with interaction GT. Same fresh H0, proposal pool, "
        "and independent compact Phase in both arms. Original compact control and frozen component/relation "
        "split wire; original mean4 admission. No parameter search or post-score sample replacement. "
        "Mean F1 must improve vs control, mean Precision and IVT F1 must not decrease, and FP+FN must decrease. "
        "Report H0 comparison, all heads, paired errors, failures, costs and timing. Not unseen-video proof.")


def history():
    seed = ROOT / "artifacts/preflight/default_improvement_inputs_20260909_v1_history.json"
    prior = old.read(seed)
    excluded = {v: set(prior["excluded_targets_by_video"].get(v, [])) for v in VIDEOS}
    hashes = {str(seed): old.sha(seed)}
    def visit(value, video=None):
        if isinstance(value, dict):
            video = value.get("video_id", video)
            if video in excluded:
                excluded[video].update(value[k] for k in ("frame_id", "target_frame_id") if type(value.get(k)) is int)
                excluded[video].update(f for f in value.get("causal_frame_ids", []) if type(f) is int)
            for child in value.values():
                visit(child, video)
        elif isinstance(value, list):
            for child in value:
                visit(child, video)
    for folder, directories, names in os.walk(ROOT / "artifacts/preflight"):
        directories[:] = [d for d in directories if d not in {"frozen_source", "calls", "requests", "views", "targets", "clean_checkout"}]
        for name in ("plan.json", "selection.json", "selection_manifest.json", "pilot_selection.json"):
            if name in names:
                path = Path(folder) / name
                visit(old.read(path))
                hashes[str(path)] = old.sha(path)
    return excluded, hashes


def choose(adapter, excluded):
    selection = []
    for video in VIDEOS:
        if adapter.entries[video].split is not old.DatasetSplit.TRAINING:
            raise ValueError("Training only")
        samples = list(adapter.iter_inference_video(video))
        masks = {r.inference.target_frame_id: masks_only(r) for r in adapter.iter_video(video)}
        used = set(excluded[video])
        for numerator in range(1, 9):
            anchor = samples[len(samples) * numerator // 9].target_frame_id
            eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                and masks.get(s.target_frame_id, {}).get("verb", False)
                and masks.get(s.target_frame_id, {}).get("ivt", False)
                and all(abs(f - known) > GAP for f in s.causal_frame_ids for known in used)]
            if not eligible:
                raise ValueError("no historical-separated sample, no silent replacement")
            s = min(eligible, key=lambda item: (abs(item.target_frame_id - anchor), item.target_frame_id))
            selection.append({"key": f"{video}_{s.target_frame_id}", "video_id": video, "frame_id": s.target_frame_id,
                "anchor_frame_id": anchor, "quantile": f"{numerator}/9", "causal_frame_ids": list(s.causal_frame_ids),
                "gt_availability_only": masks[s.target_frame_id],
                "images": [{"frame_id": f, "path": str(p), "sha256": old.sha(p)}
                    for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]})
            used.update(s.causal_frame_ids)
    return selection


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    inventory = {}
    for video, entry in sorted(adapter.entries.items()):
        if entry.split is old.DatasetSplit.TRAINING:
            counts = Counter()
            for resolved in adapter.iter_video(video):
                counts.update({task: int(valid) for task, valid in masks_only(resolved).items()})
            inventory[video] = dict(counts)
    if {v for v, counts in inventory.items() if counts.get("ivt", 0) and counts.get("verb", 0)} != set(VIDEOS):
        raise ValueError("interaction GT availability changed: review sampling before paid dispatch")
    excluded, hashes = history()
    selection = choose(adapter, excluded)
    old.save(output / "training_mask_inventory.json", inventory)
    old.save(output / "history.json", {"excluded_targets_by_video": {v: sorted(fs) for v, fs in excluded.items()}, "source_sha256": hashes})
    view, source_plan = old.InferenceOnlyAdapter(adapter), old.read(SOURCE / "plan.json")
    for video in VIDEOS:
        path = SOURCE / "priors" / f"{video}.json"
        old.same(old.sha(path), source_plan["prior_sha256"][video])
        prior = old.read(path)
        validate_prior(prior, video, view)
        old.save(output / "priors" / f"{video}.json", prior)
    for row in selection:
        base = old.build_gemini_base(view, row)
        row["h0_fingerprint"] = fingerprint(gemini_h0_wire(base))
        row["phase_fingerprints"] = {s: fingerprint(compact.phase_wire(s, row)) for s in old.SEATS}
    old.save(output / "metadata.json", old.metadata_preflight())
    deps = {*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "DEFAULT_PIPELINE_VERSION.json",
        ROOT / "configs/perception/joint_openrouter_h0.yaml", ROOT / "tests/unit/test_expanded_split_review.py", ROOT / "tools/audit/audit_expanded_split_review.py", ROOT / "tools/audit/preflight_expanded_split_review.py"}
    deps = sorted(p for p in deps if p.is_file())
    for path in deps:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    inputs = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts]
    old.save(output / "plan.json", {"profile": PROFILE, "created_utc": old.now(), "selection": selection,
        "rule": RULE, "limits": LIMITS, "max_calls": MAX_CALLS, "models": old.MODELS, "proposer": PROPOSER,
        "providers": old.PROVIDERS, "h0": "fresh OpenRouter Gemini, frozen original joint prediction prompt",
        "selection_rule": "nearest eligible 1/9 through 8/9 targets in each video; all three image identities >175 raw frames from historical/planned identities; masks only; 250-frame pilot infeasible for VID96 before API/GT scoring",
        "scope": "same-video new-target check; no Tracker/Gate or candidate/threshold changes",
        "sources": {p.relative_to(ROOT).as_posix(): old.sha(p) for p in deps},
        "inputs": {p.relative_to(output).as_posix(): old.sha(p) for p in inputs}})
    verify(output)
    print(json.dumps({"prepared": True, "targets": [s["key"] for s in selection], "max_calls": MAX_CALLS,
        "limits": LIMITS, "plan_sha256": old.sha(output / "plan.json")}), flush=True)


def verify(output):
    plan = old.read(output / "plan.json")
    old.same((plan["profile"], plan["rule"], plan["limits"], plan["max_calls"]), (PROFILE, RULE, LIMITS, MAX_CALLS))
    old.same(plan["models"], old.MODELS)
    old.same(plan["proposer"], PROPOSER)
    for names, root in ((plan["sources"], ROOT), (plan["inputs"], output)):
        for name, digest in names.items():
            old.same(old.sha(root / name), digest)
    for row in plan["selection"]:
        for image in row["images"]:
            old.same(old.sha(image["path"]), image["sha256"])
    return plan


class BoundCalls(old.TimedCalls):
    def __init__(self, output):
        super().__init__(output, limits={a: Decimal(v) for a, v in LIMITS.items()}, rates=old.RATES_V2,
            providers=old.PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.attempted = set()
        self.known = {s["key"] for s in old.read(output / "plan.json")["selection"]}

    def persist(self):
        for attempt in range(20):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)

    def call(self, target, stage, seat, body):
        allowed = {(s, r) for s in ("control", "components", "relations", "phase") for r in old.SEATS} | {("h0", "base"), ("proposal", "base")}
        with self.lock:
            ident = (target, stage, seat)
            if target not in self.known or (stage, seat) not in allowed or ident in self.attempted:
                raise ValueError("undeclared or repeated request")
            self.attempted.add(ident)
            old.save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json", old.redact_images(body))
        value = super().call(target, stage, seat, body)
        with self.lock:
            record = next((r for r in self.rows if (r["target"], r["stage"], r["seat"]) == ident), {})
            if record.get("exception_type") == "ProxyError":
                self.stopped = True
                self.persist()
                old.save(self.output / "network_circuit_breaker.json", {"target": target, "stage": stage, "seat": seat})
        return value


def collect_phase(calls, selected):
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        raw = dict(zip(old.SEATS, pool.map(lambda s: calls.call(selected["key"], "phase", s, compact.phase_wire(s, selected)), old.SEATS), strict=True))
    return {"raw": raw, "seconds": perf_counter() - start}


def run_case(calls, base, selected, prior, index):
    start = perf_counter()
    raw = calls.call(selected["key"], "h0", "base", gemini_h0_wire(base))
    record = {"h0_raw": raw, "h0": None, "arms": {}, "predictions": {a: None for a in ("h0", *old.ARMS)}, "status": "H0_FAILED"}
    try:
        validate_final_only(raw)
        h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
    except (ApiSchemaError, ValueError, TypeError, KeyError):
        return record
    record.update(h0=h0, status="READY")
    pool = make_pool(h0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        phase_future = executor.submit(collect_phase, calls, selected)
        hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
        record["hints"] = hints
        proposal = calls.call(selected["key"], "proposal", "base", proposal_wire(base, selected, h0, pool, hints["packet"]))
        record["proposal_raw"] = proposal
        try:
            if proposal is None:
                raise ValueError("missing proposal")
            pool = make_pool(h0, proposal, pool)
        except (ApiSchemaError, ValueError, TypeError, KeyError):
            record["status"] = "PROPOSAL_FAILED"
        phase = phase_future.result()
    record.update(pool=pool, phase=phase)
    initial = {"h0": h0, "pool": pool, "phase_raw": phase["raw"]}
    fallback = old.apply_phase_choices(h0, phase["raw"], 3)[0]
    record["predictions"] = {"h0": h0, **{a: deepcopy(fallback) for a in old.ARMS}}
    if record["status"] == "READY" and pool["propositions"]:
        control = {s: compact.review_wire(s, base, selected, pool) for s in old.SEATS}
        wire = {"control": control}
        empty = []
        for group in old.split.GROUPS:
            if old.split.subpool(pool, group)["propositions"]:
                wire[group] = {s: old.split.split_wire(b, group) for s, b in control.items()}
            else:
                empty.append(group)
                wire[group] = {s: None for s in old.SEATS}
        class SkipEmpty:
            def call(self, target, stage, seat, body):
                return {"judgments": {}} if stage in empty else calls.call(target, stage, seat, body)
        for arm in (old.ARMS if index % 2 == 0 else old.ARMS[::-1]):
            result = old.run_arm(SkipEmpty(), selected["key"], initial, wire, arm)
            record["arms"][arm] = result
            record["predictions"][arm] = result["final"]
    record["seconds"] = perf_counter() - start
    return record


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as file:
        file.write(old.sha(output / "plan.json"))
    calls = BoundCalls(output)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"], **{a: None for a in ("h0", *old.ARMS)}} for s in plan["selection"]]
    start, fatal = perf_counter(), None
    try:
        for index, (s, row) in enumerate(zip(plan["selection"], rows, strict=True)):
            if calls.stopped:
                break
            base = old.build_gemini_base(old.InferenceOnlyAdapter(adapter), s)
            old.same(fingerprint(gemini_h0_wire(base)), s["h0_fingerprint"])
            result = run_case(calls, base, s, old.read(output / "priors" / f"{s['video_id']}.json"), index)
            old.save(output / "targets" / s["key"] / "result.json", result)
            row.update(result["predictions"])
            old.save(output / "predictions.json", rows)
            print(json.dumps({"target": s["key"], "completed": index + 1, "calls": len(calls.rows), "status": result["status"], "stopped": calls.stopped}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        old.save(output / "predictions.json", rows)
        try:
            verify(output)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            files = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
            old.save(output / "completion.json", {"fatal_error": fatal, "closed_utc": old.now(), "elapsed_seconds": perf_counter() - start,
                "post_calls": len(calls.rows), "hashes": {p.relative_to(output).as_posix(): old.sha(p) for p in files}})


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in ("seconds", "request_seconds")}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def score(output, adapter):
    plan, done, ledger = verify(output), old.read(output / "completion.json"), old.read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal run required")
    for name, digest in done["hashes"].items():
        old.same(old.sha(output / name), digest)
    calls = ledger["calls"]
    old.same(len(calls), len({(c["target"], c["stage"], c["seat"]) for c in calls}))
    for a, limit in LIMITS.items():
        old.same(sum(Decimal(c["charge"]) for c in calls if c["account"] == a), Decimal(ledger["occupied"][a]))
        if Decimal(ledger["occupied"][a]) > Decimal(limit):
            raise ValueError("budget exceeded")
    replay, rows, records = old.ReplayCalls(output, calls), old.read(output / "predictions.json"), []
    for index, (s, row) in enumerate(zip(plan["selection"], rows, strict=True)):
        old.same((row["key"], row["video_id"], row["frame_id"]), (s["key"], s["video_id"], s["frame_id"]))
        path = output / "targets" / s["key"] / "result.json"
        if not path.exists():
            old.same([row[a] for a in ("h0", *old.ARMS)], [None] * 3)
            continue
        base = old.build_gemini_base(old.InferenceOnlyAdapter(adapter), s)
        result = run_case(replay, base, s, old.read(output / "priors" / f"{s['video_id']}.json"), index)
        stored = old.read(path)
        old.same(strip_timing(result), strip_timing(stored))
        for a in ("h0", *old.ARMS):
            old.same(row[a], result["predictions"][a])
        records.append(stored)
    old.same(len(replay.rows), len(calls))
    _, truth = old.score_saved(adapter, [{**r, "h1": None, "final": r["split"]} for r in rows])
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    metrics = {a: old.compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None, "final": r[a]} for r in rows])["arms"]["final"] for a in ("h0", *old.ARMS)}
    deltas = {f"{a}_to_{b}": [{"key": r["key"], **old.frame_delta(r[a], r[b], truths[r["video_id"], r["frame_id"]]["gt"],
        truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows] for a, b in (("h0", "control"), ("h0", "split"), ("control", "split"))}
    changes = {k: old.summarize_deltas(v) for k, v in deltas.items()}
    totals = {}
    for group, stages in {"shared": ("h0", "proposal", "phase"), "control": ("control",), "split": ("components", "relations")}.items():
        own = [c for c in calls if c["stage"] in stages]
        totals[group] = {"calls": len(own), "statuses": dict(Counter(c["status"] for c in own)),
            "costs": {a: {kind: str(sum(Decimal(c["charge"]) for c in own if c["account"] == a and c["charge_kind"] == kind))
                for kind in ("native", "conservative_estimate", "unknown_reserved")} for a in LIMITS}}
        if group != "shared":
            totals[group]["review_seconds"] = sum(r["arms"].get(group, {}).get("request_seconds", 0) for r in records)
    before, after = metrics["control"]["tasks"], metrics["split"]["tasks"]
    checks = {"all_targets_with_arms": len(records) == len(plan["selection"]) == 32 and all(set(r["arms"]) == set(old.ARMS) for r in records),
        "mean_f1_improved": sum(h["micro_f1"] for h in after.values()) > sum(h["micro_f1"] for h in before.values()),
        "mean_precision_not_decreased": sum(h["micro_precision"] for h in after.values()) >= sum(h["micro_precision"] for h in before.values()),
        "ivt_f1_not_decreased": after["ivt"]["micro_f1"] >= before["ivt"]["micro_f1"],
        "net_errors_reduced": changes["control_to_split"]["net_errors_removed"] > 0}
    report = {"metrics": metrics, "changes": changes, "totals": totals, "checks": checks, "success": all(checks.values()),
        "raw_replayed": True, "elapsed_seconds": done["elapsed_seconds"], "scope": RULE}
    old.save(output / "scored_truth.json", truth)
    old.save(output / "frame_deltas.json", deltas)
    old.save(output / "metrics.json", report)
    print(json.dumps({"checks": checks, "f1": {a: {t: h["micro_f1"] for t, h in m["tasks"].items()} for a, m in metrics.items()}, "totals": totals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adapter = old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    globals()[args.command](args.output, adapter)
