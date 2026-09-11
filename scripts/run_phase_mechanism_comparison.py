"""Same frozen 32 targets: Phase wording contrasts and historical five-head repair."""
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

from scripts import run_expanded_split_review as source_runner
from scripts import run_five_head_repair_trial as legacy
from scripts import run_phase_extension_trial as phase

old = source_runner.old
SOURCE = ROOT / "artifacts/preflight/expanded_split_review_32_20260910_v1"
PROFILE = "phase_wording_and_legacy_five_head_same32_v1"
ARMS = ("compact_repeat", "original_phase", "revised_phase", "legacy_panel")
PHASE_ARMS = ARMS[:3]
VERSIONS = ("h0", "cached_default", "graph_before_phase", *ARMS, "legacy_paper_style", "legacy_llm_raw")
MAX_CALLS = 672
LIMITS = {"openrouter_usd": "5", "xai_usd": "5", "aliyun_cny": "3"}
TEMPLATE = ROOT / "src/surgical_agent/research/verification/prompts/phase_hypothesis_review_v1.txt"
RULE = ("All 32 already inspected Training targets, unchanged identities/images/H0/graph results/LOVO priors. "
        "Fresh compact Phase control, original Phase wording and H0-hypothesis revised Phase wording use "
        "identical single-choice schema, models and three-of-five valid-response selection. The revised arm "
        "adds untrusted H0 Phase as input; therefore it tests wording plus hypothesis visibility, not text length alone. "
        "Historical five-head visual rewrite and seven-Phase mean-score selection reuse exact historical functions "
        "on the cached current graph-R1 predecessor, not the old eight-target upstream run. "
        "Rotate four-arm order per target; five requests parallel per panel; no automatic retries. "
        "Close/replay all calls before GT scoring. Report all heads, harmful edits, failures, time/cost and per-video results. "
        "Revised Phase must beat fresh compact AND H0 in Phase F1 to show net semantic repair benefit; "
        "all-keep is not success. Already inspected development set, no independent generalization or automatic promotion.")


def revised_wire(body, h0):
    out = deepcopy(body)
    packet = json.loads(out["messages"][0]["content"][0]["text"])
    pid = h0["phase"][0]
    if type(pid) is not int or not 0 <= pid < 7:
        raise ValueError("invalid Phase hypothesis")
    packet["instructions"] = TEMPLATE.read_text(encoding="utf-8").strip().replace(
        "{current_image_index}", str(packet["current_image_index"]))
    packet["reference_phase_hypothesis"] = {"phase_id": pid,
        "phase_name": phase.PHASE_GUIDE[pid]["name"], "status": "Untrusted model hypothesis; not visual evidence."}
    out["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return out


def phase_body(arm, seat, selected, initial):
    if arm == "compact_repeat":
        return source_runner.compact.phase_wire(seat, selected)
    body = phase.phase_wire(seat, selected)
    return revised_wire(body, initial["h0"]) if arm == "revised_phase" else body


def adapt_initial(record, prior):
    control = record["arms"]["control"]
    prediction = deepcopy(control["prediction"])
    old.same(prediction["phase"], record["h0"]["phase"])
    return {"h0": deepcopy(record["h0"]), "cached_default": deepcopy(control["final"]),
        "hints": deepcopy(record["hints"]), "phase_prior": deepcopy(prior),
        "graph_r1": {"prediction": prediction, "pool": deepcopy(record["pool"]),
            "raw": deepcopy(control["raw"]["control"]), "reviews": deepcopy(control["reviews"]),
            "issues": old.panel.unresolved(prediction, record["pool"], control["means"], control["diagnostics"])}}


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    previous = source_runner.verify(SOURCE)
    done, ledger = old.read(SOURCE / "completion.json"), old.read(SOURCE / "budget.json")
    if done["fatal_error"] or not ledger["stopped"] or len(previous["selection"]) != 32:
        raise ValueError("closed 32-target source required")
    for name, value in done["hashes"].items():
        old.same(old.sha(SOURCE / name), value)
    # Confirm that the reused legacy mechanism is byte-identical to its old snapshot.
    legacy_root = ROOT / "artifacts/preflight/five_head_repair_eight_20260909_v1/frozen_source"
    for name in ("scripts/run_five_head_repair_trial.py", "src/surgical_agent/research/verification/five_head_repair.py",
                 "src/surgical_agent/research/verification/repair_feedback_v2.py"):
        old.same(old.sha(ROOT / name), old.sha(legacy_root / name))
    selected = deepcopy(previous["selection"])
    inputs, source_paths = {}, [SOURCE / n for n in ("plan.json", "completion.json", "predictions.json")]
    view = old.InferenceOnlyAdapter(adapter)
    for row in selected:
        if adapter.entries[row["video_id"]].split is not old.DatasetSplit.TRAINING:
            raise ValueError("Training only")
        prior_path = SOURCE / "priors" / f"{row['video_id']}.json"
        record_path = SOURCE / "targets" / row["key"] / "result.json"
        record, prior = old.read(record_path), old.read(prior_path)
        source_runner.validate_prior(prior, row["video_id"], view)
        initial = adapt_initial(record, prior)
        base = old.build_gemini_base(view, row)
        for arm in PHASE_ARMS:
            for seat in old.SEATS:
                body = phase_body(arm, seat, row, initial)
                old.save(output / "preflight_requests" / row["key"] / f"{arm}_{seat}.json", old.redact_images(body))
        old.save(output / "preflight_requests" / row["key"] / "legacy_repair.json",
                 old.redact_images(legacy.repair_wire(base, row, initial)))
        inputs[row["key"]] = initial
        source_paths.extend((prior_path, record_path))
    old.save(output / "initial_state.json", inputs)
    old.save(output / "metadata.json", old.metadata_preflight())
    deps = {*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "DEFAULT_PIPELINE_VERSION.json",
        ROOT / "configs/perception/joint_openrouter_h0.yaml",
        ROOT / "tests/unit/test_phase_mechanism_comparison.py", ROOT / "tools/audit/audit_phase_mechanism_comparison.py",
        ROOT / "tools/audit/preflight_phase_mechanism_comparison.py"}
    deps = sorted(p for p in deps if p.is_file())
    for path in deps:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    local = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts]
    old.save(output / "plan.json", {"profile": PROFILE, "rule": RULE, "created_utc": old.now(), "selection": selected,
        "models": old.MODELS, "proposer": legacy.PROPOSER, "providers": old.PROVIDERS, "max_calls": MAX_CALLS, "limits": LIMITS,
        "sources": {p.relative_to(ROOT).as_posix(): old.sha(p) for p in deps},
        "inputs": {p.relative_to(output).as_posix(): old.sha(p) for p in local},
        "source_archive": {str(p.resolve()): old.sha(p) for p in set(source_paths)}})
    verify(output)
    print(json.dumps({"prepared": True, "targets": len(selected), "max_calls": MAX_CALLS, "limits": LIMITS,
        "plan_sha256": old.sha(output / "plan.json")}), flush=True)


def verify(output):
    plan = old.read(output / "plan.json")
    old.same((plan["profile"], plan["rule"], plan["models"], plan["proposer"], plan["limits"], plan["max_calls"]),
        (PROFILE, RULE, old.MODELS, legacy.PROPOSER, LIMITS, MAX_CALLS))
    for entries, root in ((plan["sources"], ROOT), (plan["inputs"], output), (plan["source_archive"], Path())):
        for name, digest in entries.items():
            old.same(old.sha(root / name), digest)
    for row in plan["selection"]:
        for im in row["images"]:
            old.same(old.sha(im["path"]), im["sha256"])
    return plan


class Calls(source_runner.BoundCalls):
    def __init__(self, output):
        old.TimedCalls.__init__(self, output, limits={a: Decimal(v) for a, v in LIMITS.items()}, rates=old.RATES_V2,
            providers=old.PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.known = {r["key"] for r in old.read(output / "plan.json")["selection"]}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        allowed = {(stage, seat) for stage in (*PHASE_ARMS, legacy.REVIEW_STAGE) for seat in old.SEATS} | {(legacy.REPAIR_STAGE, "base")}
        with self.lock:
            identity = (target, stage, seat)
            if target not in self.known or (stage, seat) not in allowed or identity in self.attempted:
                raise ValueError("undeclared or duplicate request")
            self.attempted.add(identity)
            old.save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json", old.redact_images(body))
        result = old.TimedCalls.call(self, target, stage, seat, body)
        with self.lock:
            rec = next((r for r in self.rows if (r["target"], r["stage"], r["seat"]) == identity), {})
            if rec.get("exception_type") == "ProxyError":
                self.stopped = True
                self.persist()
        return result


def run_case(calls, base, selected, initial, index, skipped_arms=()):
    before = initial["graph_r1"]["prediction"]
    predictions = {"h0": deepcopy(initial["h0"]), "cached_default": deepcopy(initial["cached_default"]),
        "graph_before_phase": deepcopy(before), **{a: deepcopy(before) for a in (*ARMS, "legacy_paper_style", "legacy_llm_raw")}}
    branches, skipped = {}, []
    order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
    for arm in order:
        start = perf_counter()
        skip = calls.stopped or arm in skipped_arms
        if skip:
            skipped.append(arm)
        if arm in PHASE_ARMS:
            raw = {s: None for s in old.SEATS}
            if not skip:
                with ThreadPoolExecutor(max_workers=5) as pool:
                    raw = dict(zip(old.SEATS, pool.map(lambda s, arm=arm: calls.call(selected["key"], arm, s,
                        phase_body(arm, s, selected, initial)), old.SEATS), strict=True))
            prediction, decision = phase.apply_phase_choices(before, raw, 3)
            branches[arm] = {"raw": raw, "prediction": prediction, "decision": decision, "seconds": perf_counter() - start}
            predictions[arm] = prediction
        else:
            if skip:
                result = legacy.empty_record(initial)
                result["status"] = "BUDGET_STOPPED"
            else:
                result = legacy.run_target(calls, base, selected, initial)
            branches[arm] = result
            predictions.update(legacy_panel=result["panel_five"], legacy_paper_style=result["paper_style"], legacy_llm_raw=result["llm_raw"])
    return {"predictions": predictions, "branches": branches, "order": list(order), "skipped_arms": skipped}


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "seconds" and not k.endswith("_seconds")}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as handle:
        handle.write(old.sha(output / "plan.json"))
    calls, initials = Calls(output), old.read(output / "initial_state.json")
    calls.persist()
    rows = [{"key": r["key"], "video_id": r["video_id"], "frame_id": r["frame_id"],
        **{a: deepcopy(initials[r["key"]]["graph_r1"]["prediction"]) for a in VERSIONS},
        "h0": deepcopy(initials[r["key"]]["h0"]), "cached_default": deepcopy(initials[r["key"]]["cached_default"])} for r in plan["selection"]]
    start, fatal = perf_counter(), None
    try:
        for index, (selected, row) in enumerate(zip(plan["selection"], rows, strict=True)):
            if calls.stopped:
                break
            base = old.build_gemini_base(old.InferenceOnlyAdapter(adapter), selected)
            result = run_case(calls, base, selected, initials[selected["key"]], index)
            old.save(output / "targets" / selected["key"] / "result.json", result)
            row.update(result["predictions"])
            old.save(output / "predictions.json", rows)
            print(json.dumps({"completed": index + 1, "target": selected["key"], "calls": len(calls.rows),
                "legacy_status": result["branches"].get("legacy_panel", {}).get("status", "NOT_IN_THIS_TRIAL"), "stopped": calls.stopped}), flush=True)
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
            old.save(output / "completion.json", {"fatal_error": fatal, "elapsed_seconds": perf_counter() - start,
                "closed_utc": old.now(), "calls": len(calls.rows), "hashes": {p.relative_to(output).as_posix(): old.sha(p) for p in files}})


def score(output, adapter):
    plan, done, ledger = verify(output), old.read(output / "completion.json"), old.read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal run required")
    for name, value in done["hashes"].items():
        old.same(old.sha(output / name), value)
    calls = ledger["calls"]
    old.same(len(calls), len({(r["target"], r["stage"], r["seat"]) for r in calls}))
    for account, limit in LIMITS.items():
        cost = sum(Decimal(r["charge"]) for r in calls if r["account"] == account)
        old.same(cost, Decimal(ledger["occupied"][account]))
        if cost > Decimal(limit):
            raise ValueError("budget exceeded")
    replay, initials = old.ReplayCalls(output, calls), old.read(output / "initial_state.json")
    rows, records = old.read(output / "predictions.json"), []
    for index, (selected, row) in enumerate(zip(plan["selection"], rows, strict=True)):
        old.same((row["key"], row["video_id"], row["frame_id"]), (selected["key"], selected["video_id"], selected["frame_id"]))
        path = output / "targets" / row["key"] / "result.json"
        if not path.exists():
            continue
        base = old.build_gemini_base(old.InferenceOnlyAdapter(adapter), selected)
        stored = old.read(path)
        result = run_case(replay, base, selected, initials[row["key"]], index, stored["skipped_arms"])
        old.same(without_timing(result), without_timing(stored))
        for arm in VERSIONS:
            old.same(result["predictions"][arm], row[arm])
        records.append(old.read(path))
    old.same(len(replay.rows), len(calls))
    _, truth = old.score_saved(adapter, [{**r, "h1": None, "final": r["revised_phase"]} for r in rows])
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    metrics = {arm: old.compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[arm]} for r in rows])["arms"]["final"] for arm in VERSIONS}
    pairs = [("h0", a) for a in VERSIONS if a != "h0"]
    if "compact_repeat" in VERSIONS:
        pairs += [("compact_repeat", a) for a in ARMS[1:]] + [("cached_default", "compact_repeat")]
    deltas = {f"{a}_to_{b}": [{"key": r["key"], **old.frame_delta(r[a], r[b], truths[r["video_id"], r["frame_id"]]["gt"],
        truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows] for a, b in pairs}
    totals = {}
    for arm in ARMS:
        stages = (arm,) if arm in PHASE_ARMS else (legacy.REPAIR_STAGE, legacy.REVIEW_STAGE)
        own = [c for c in calls if c["stage"] in stages]
        totals[arm] = {"calls": len(own), "statuses": dict(Counter(c["status"] for c in own)),
            "costs": {a: {kind: str(sum(Decimal(c["charge"]) for c in own if c["account"] == a and c["charge_kind"] == kind))
                for kind in ("native", "conservative_estimate", "unknown_reserved")} for a in LIMITS},
            "seconds": sum(r["branches"][arm]["seconds"] if arm in PHASE_ARMS else
                r["branches"][arm]["repair_seconds"] + r["branches"][arm]["review_seconds"] for r in records)}
    phase_f1 = {a: m["tasks"]["phase"]["micro_f1"] for a, m in metrics.items()}
    report = {"metrics": metrics, "changes": {k: old.summarize_deltas(v) for k, v in deltas.items()},
        "totals": totals, "raw_replayed": True, "elapsed_seconds": done["elapsed_seconds"],
        "revised_phase_success": len(records) == 32 and phase_f1["revised_phase"] > max(phase_f1.get("compact_repeat", phase_f1["h0"]), phase_f1["h0"]),
        "scope": RULE}
    old.save(output / "scored_truth.json", truth)
    old.save(output / "frame_deltas.json", deltas)
    old.save(output / "metrics.json", report)
    print(json.dumps({"phase_f1": phase_f1, "revised_phase_success": report["revised_phase_success"], "totals": totals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adapter = old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    globals()[args.command](args.output, adapter)
