"""One authorized supplemental attempt for each of the two missing Sol targets."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import compare_mainline_backbones as trial

old, main = trial.old, trial.main
SOURCE = ROOT / "artifacts/preflight/mainline_backbone_training6_20260911_v2"
COMPOSITE = ROOT / "artifacts/research/mainline_backbone_training6_comparison_20260911_v1"
PREVIOUS = ROOT / "artifacts/preflight/mainline_sol_supplement_20260911_v1"
OUTPUT = ROOT / "artifacts/preflight/mainline_sol_supplement_20260911_v1_resume1"
KEYS = {"VID103_18501", "VID23_28551"}


class ContinueCalls:
    def __init__(self, previous, current):
        self.previous, self.current = previous, current

    def call(self, target, stage, seat, body):
        if (target, stage, seat) in self.previous.rows:
            return self.previous.call(target, stage, seat, body)
        return self.current.call(target, stage, seat, body)


def target(calls, base, selected, plan):
    row = {k: selected[k] for k in ("key", "video_id", "frame_id")}
    started = perf_counter()
    try:
        record = main.run_target(calls, base, selected,
            old.read(SOURCE / "priors" / f"{selected['video_id']}.json"), plan["gate"])
    except (ValueError, TypeError, KeyError, trial.ApiSchemaError) as exc:
        row.update(status="TARGET_FAILED", error_type=type(exc).__name__,
                   predictions={a: None for a in main.ARMS})
    else:
        row.update(status="PREDICTED", predictions=record["predictions"], timing=record["timing_seconds"])
    row["seconds"] = perf_counter() - started
    return row


def run():
    if OUTPUT.exists():
        raise ValueError("Fresh archive required; this script must not be rerun automatically")
    plan = trial.verify(SOURCE)
    bases = trial.bases_for(plan)
    selected = [s for s in plan["selection"] if s["key"] in KEYS]
    original = old.read(SOURCE / "sol/predictions.json")["targets"]
    assert {r["key"] for r in original if r["status"] == "TARGET_FAILED"} == KEYS
    unchanged = {name: old.sha(ROOT / name) for name in
                 ("DEFAULT_PIPELINE_VERSION.json", "BEST_PIPELINE_VERSION.json", "TRAINING_BASE_MODEL_SELECTION.json")}
    prior_calls = old.read(SOURCE / "sol/budget.json")["calls"]
    checks = []
    with old.joint.roster.lightweight_protocol():
        for s in selected:
            mock = trial.Mock("sol")
            target(mock, bases[s["key"]], s, plan)
            assert Counter(r["stage"] for r in mock.rows) == Counter(main.STAGES)
            previous = next(r for r in prior_calls if r["target"] == s["key"] and r["stage"] == "h0")
            assert previous["http_status"] == 403
            folder = SOURCE / "sol/calls" / f"{previous['index']:03d}_{s['key']}_h0_base"
            assert mock.wires["h0", "base"] == old.read(folder / "request.json")
            checks.append({"target": s["key"], "calls_mocked": len(mock.rows), "original_h0_wire_identical": True})
    old.save(OUTPUT / "plan.json", {"created_utc": old.now(), "source": str(SOURCE),
        "source_plan_sha256": old.sha(SOURCE / "plan.json"), "selection": selected,
        "policy": "User-authorized second attempt; one per missing target, no automatic retries, unchanged route/images/prompts. Successful original four preserved. Original first-attempt ranking remains historical.",
        "max_calls": 18, "limits": trial.LIMITS, "reused_archive": str(PREVIOUS),
        "unchanged_root_hashes": unchanged, "gate_data_collected": False})
    old.save(OUTPUT / "preflight.json", checks)
    shutil.copyfile(__file__, OUTPUT / "frozen_supplement.py")
    print(json.dumps({"preflight": checks}), flush=True)
    rows, started = [], perf_counter()
    with old.joint.credential_context(plan), old.joint.roster.lightweight_protocol():
        calls = trial.ModelCalls(OUTPUT / "sol", "sol", plan)
        calls.max_calls = 18
        bridge = ContinueCalls(trial.Replay(PREVIOUS / "sol", "sol"), calls)
        try:
            for s in selected:
                row = target(bridge, bases[s["key"]], s, plan)
                rows.append(row)
                old.save(OUTPUT / "sol/predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "status": row["status"], "seconds": row["seconds"]}), flush=True)
        finally:
            calls.stopped = True
            try:
                calls.persist()
            finally:
                calls.close_ledger()
    elapsed = perf_counter() - started
    old.save(OUTPUT / "completion.json", {"seconds": elapsed, "gate_data_collected": False,
        "evidence_sha256": {str(p.relative_to(OUTPUT)): old.sha(p) for p in OUTPUT.rglob("*.json")}})
    replay = trial.Replay(OUTPUT / "sol", "sol")
    prior_replay = trial.Replay(PREVIOUS / "sol", "sol")
    replay_bridge = ContinueCalls(prior_replay, replay)
    with old.joint.roster.lightweight_protocol():
        for s, row in zip(selected, rows, strict=True):
            rerun = target(replay_bridge, bases[s["key"]], s, plan)
            assert rerun["status"] == row["status"] and rerun["predictions"] == row["predictions"]
    assert replay.used == set(replay.rows)
    assert prior_replay.used == set(prior_replay.rows)
    merged = old.read(COMPOSITE / "predictions.json")
    replacements = {r["key"]: r for r in rows}
    merged["sol"] = [replacements.get(r["key"], r) for r in original]
    truth = old.read(COMPOSITE / "scored_truth.json")
    results = {name: {"completed": sum(r["status"] == "PREDICTED" for r in rs),
                     "metrics": trial.release.metrics(rs, truth)} for name, rs in merged.items()}
    charges, failures = defaultdict(Decimal), []
    all_calls = [(archive, c) for archive in (PREVIOUS, OUTPUT)
                 for c in old.read(archive / "sol/budget.json")["calls"]]
    assert len({(c['target'], c['stage'], c['seat']) for _, c in all_calls}) == len(all_calls)
    for archive, c in all_calls:
        folder = archive / "sol/calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        response = old.read(folder / "response.json") if (folder / "response.json").exists() else {}
        message = response.get("body", {}).get("error", {}).get("message", "")
        no_charge = "No credits were charged" in message
        charges[c["account"] + "|" + ("provider_reported_no_charge" if no_charge else c["charge_kind"])] += Decimal("0") if no_charge else Decimal(c["charge"])
        if c["status"] not in trial.VALID:
            failures.append({"target": c["target"], "stage": c["stage"], "seat": c["seat"],
                             "http_status": c.get("http_status"), "message": message})
    assert all(old.sha(ROOT / name) == digest for name, digest in unchanged.items())
    report = {"models": results, "new_calls": len(all_calls), "resume_calls": len(replay.rows),
              "new_seconds": elapsed + old.read(PREVIOUS / "completion.json")["seconds"],
              "new_charges": {k: str(v) for k, v in charges.items()}, "failures": failures,
              "replayed_requests": len(replay.used) + len(prior_replay.used), "root_pointers_unchanged": True,
              "note": "Sol second-attempt composite; Gemini/Qwen original attempts. Six development targets only.",
              "gate_data_collected": False}
    old.save(OUTPUT / "predictions_composite.json", merged)
    old.save(OUTPUT / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
