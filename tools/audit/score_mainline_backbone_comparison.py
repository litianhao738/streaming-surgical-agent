"""Zero-API composite scoring of the paired run and Qwen identity recovery."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import compare_mainline_backbones as trial
from scripts import recover_qwen_backbone_alias as recovery

old, main = trial.old, trial.main


class QwenReplay(trial.Replay):
    def __init__(self, folder):
        super().__init__(folder, "qwen")
        self.cached = old.read(folder / "verified_cached_h0.json")
        self.h0_uses = set()

    def call(self, target, stage, seat, body):
        if stage == "h0":
            self.h0_uses.add(target)
            return deepcopy(self.cached[target]["raw"])
        key = target, stage, seat
        row = self.rows.get(key)
        if row is None:
            return None
        self.used.add(key)
        folder = self.folder / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
        wire = recovery.wire(body) if seat == "base" else body
        if old.read(folder / "request.json") != old.joint.roster.transport.redact_images(wire):
            raise ValueError("Qwen dependent request mismatch")
        if row["status"] not in trial.VALID:
            return None
        raw = old.read(folder / "response.json")["body"]
        if raw["model"] != wire["model"]:
            raise ValueError("Qwen replay identity mismatch")
        content = raw["choices"][0]["message"]["content"]
        return json.loads(content) if seat == "base" else trial.parse_review_json(content)[0]


def accounting(locations):
    rows, charges, stages, uncharged = [], defaultdict(Decimal), defaultdict(Decimal), []
    for folder in locations:
        for c in old.read(folder / "budget.json")["calls"]:
            record = dict(c)
            record["source_archive"] = str(folder)
            path = folder / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}" / "response.json"
            amount, kind = Decimal(c["charge"]), c["charge_kind"]
            if path.exists():
                raw = old.read(path)["body"]
                error = raw.get("error") or {}
                if isinstance(error, dict) and "No credits were charged" in error.get("message", ""):
                    amount, kind = Decimal(0), "provider_reported_no_charge"
                    uncharged.append({"target": c["target"], "stage": c["stage"], "seat": c["seat"],
                                      "reserved_usd": c["charge"], "response_sha256": old.sha(path)})
            charges[c["account"] + "|" + kind] += amount
            stages[c["stage"] + "|" + c["account"] + "|" + kind] += amount
            rows.append(record)
    ids = [(c["target"], c["stage"], c["seat"]) for c in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("repeated API request across archives")
    return {"calls": len(rows), "charges": {k: str(v) for k, v in charges.items()},
            "stage_charges": {k: str(v) for k, v in stages.items()},
            "statuses": dict(Counter(c["status"] for c in rows)), "provider_reported_free": uncharged,
            "h0_seconds_sum": sum(c["elapsed_seconds"] for c in rows if c["stage"] == "h0"),
            "model_ids": dict(Counter(c["model"] for c in rows)),
            "issues": [{k: c.get(k) for k in ("target", "stage", "seat", "status", "http_status", "exception_type")}
                       for c in rows if c["status"] not in trial.VALID]}


def run(source, recovered, output):
    if output.exists():
        raise ValueError("fresh score archive required")
    _recovery_plan, plan = recovery.verify(recovered)
    done = old.read(recovered / "completion.json")
    if done["fatal_error"]:
        raise ValueError("recovery did not complete")
    for rel, digest in done["evidence_sha256"].items():
        if old.sha(recovered / rel) != digest:
            raise ValueError("recovery evidence changed")
    bases = trial.bases_for(plan)
    qrows = old.read(recovered / "predictions.json")["targets"]
    if [r["key"] for r in qrows] != [s["key"] for s in plan["selection"]]:
        raise ValueError("all six targets required")
    replay = QwenReplay(recovered)
    with old.joint.roster.lightweight_protocol():
        for row, selected in zip(qrows, plan["selection"], strict=True):
            if row["status"] != "PREDICTED":
                continue
            record = main.run_target(replay, bases[row["key"]], selected,
                     old.read(source / "priors" / f"{selected['video_id']}.json"), plan["gate"])
            if record["predictions"] != row["predictions"]:
                raise ValueError("Qwen raw replay changed prediction")
    if replay.used != set(replay.rows) or len(replay.h0_uses) != 6:
        raise ValueError("incomplete recovery replay")
    # All inference is now closed. Original scorer also replays Sol and Gemini raw responses.
    trial.score(source)
    truth = old.read(source / "scored_truth.json")
    original = old.read(source / "comparison.json")
    groups = {n: old.read(source / n / "predictions.json")["targets"] for n in ("sol", "gemini")}
    groups["qwen"] = qrows
    results = {}
    for name, rows in groups.items():
        metrics = trial.release.metrics(rows, truth)
        ledger = accounting([source / name, recovered] if name == "qwen" else [source / name])
        if name == "qwen":
            for issue in ledger["issues"]:
                if issue["stage"] == "h0":
                    issue["resolution"] = "RECOVERED_VERIFIED_ALIAS_NO_RETRY"
            ledger["verified_alias_h0_recoveries"] = 6
        ledger["unresolved_issue_count"] = sum("resolution" not in item for item in ledger["issues"])
        seconds = sum(r["seconds"] for r in rows)
        valid_rows = [r for r in rows if r["status"] == "PREDICTED"]
        changes = Counter()
        for row in valid_rows:
            h, f, gt = row["predictions"]["h0"], row["predictions"][main.PRIMARY], truth[row["key"]]["gt"]
            for task in old.TASKS:
                before, after, expected = set(h[task]), set(f[task]), set(gt[task])
                changes["beneficial_labels"] += len((after - before) & expected) + len((before - after) - expected)
                changes["harmful_labels"] += len((before - after) & expected) + len((after - before) - expected)
            if h["phase"] != f["phase"]:
                changes["phase_fixed" if f["phase"] == gt["phase"] else "phase_broken" if h["phase"] == gt["phase"] else "phase_wrong_to_wrong"] += 1
        results[name] = {"metrics": metrics, "accounting": ledger, "targets_complete": len(valid_rows),
                         "eligible": len(valid_rows) == 6, "seconds_sum": seconds,
                         "mean_seconds_per_complete_target": sum(r["seconds"] for r in valid_rows) / len(valid_rows) if valid_rows else None,
                         "timing_method": "Original H0 HTTP latency plus later repair wall time; reconstructed, not one continuous paired run" if name == "qwen" else "Measured target wall time with three backbones running in parallel",
                         "edits": dict(changes)}
        results[name]["phase_accuracy_percent"] = {
            arm: 100 * sum(r["predictions"][arm] is not None and
                           r["predictions"][arm]["phase"] == truth[r["key"]]["gt"]["phase"] for r in rows) / len(rows)
            for arm in main.ARMS}
    def rank(name):
        r = results[name]
        m = r["metrics"][main.PRIMARY]
        occupancy = old.read(recovered / "budget.json")["occupied"]["openrouter_usd"] if name == "qwen" else old.read(source / name / "budget.json")["occupied"]["openrouter_usd"]
        return (-m["mean_f1"], -m["ivt"]["f1"], -m["phase"]["f1"], Decimal(occupancy), r["seconds_sum"])
    ranking = sorted((n for n in groups if results[n]["eligible"]), key=rank)
    common = set.intersection(*({r["key"] for r in rows if r["status"] == "PREDICTED"} for rows in groups.values()))
    adapter = old.common.CholecTrack20DatasetAdapter(trial.release.DATASET, causal_window_size=3)
    inventory = {v: sum(1 for _ in adapter.iter_inference_video(v)) for v in trial.COHORT}
    report = {"schema_version": "mainline_backbone_comparison_with_alias_recovery_v1", "created_utc": old.now(),
              "mechanism": main.VERSION, "targets": 6, "selection": [s["key"] for s in plan["selection"]],
              "models": results, "ranking": ranking, "recommendation": ranking[0] if ranking else None,
              "selection_rule": trial.STANDARD["rank"], "total_api_calls": original["calls"] + done["calls"],
              "api_execution_wall_seconds_sum": original["wall_seconds"] + done["seconds"],
              "common_success_subset": {"count": len(common), "targets": sorted(common),
                   "scope": "Diagnostic only; not the predeclared ranking denominator",
                   "metrics": {n: trial.release.metrics([r for r in rows if r["key"] in common], truth) for n, rows in groups.items()}},
              "training_video_inventory": inventory, "four_video_total_targets": sum(inventory.values()),
              "testing_used": False, "gate_collection_started": False,
              "source": str(source), "qwen_recovery": str(recovered),
              "evidence_sha256": {"source_plan": old.sha(source / "plan.json"), "source_completion": old.sha(source / "completion.json"),
                                  "recovery_plan": old.sha(recovered / "plan.json"), "recovery_completion": old.sha(recovered / "completion.json"),
                                  "scored_truth": old.sha(source / "scored_truth.json"), "audit_source": old.sha(Path(__file__))},
              "limitations": ["Six Training targets are model-selection development evidence, not independent validation.",
                              "Sol failures are upstream moderation refusals, not scored visual judgments; all six targets remain in headline metrics.",
                              "Qwen alias returned verified -0902 snapshot. Original ledger remains unchanged; six valid H0 responses are recovered without API repetition.",
                              "Qwen timing is reconstructed and runs under different concurrent load; do not claim a precise speed ranking.",
                              "All four Training videos have influenced development; a future Gate evaluation needs an untouched evaluation split."]}
    old.save(output / "comparison.json", report)
    old.save(output / "predictions.json", groups)
    old.save(output / "scored_truth.json", truth)
    print(json.dumps({"recommendation": report["recommendation"], "calls": report["total_api_calls"],
          "inventory": inventory, "models": {n: {"complete": r["targets_complete"], "h0": r["metrics"]["h0"]["mean_f1"],
          "final": r["metrics"][main.PRIMARY]["mean_f1"], "ivt": r["metrics"][main.PRIMARY]["ivt"]["f1"],
          "phase": r["metrics"][main.PRIMARY]["phase"]["f1"], "seconds": r["seconds_sum"], "charges": r["accounting"]["charges"]}
          for n, r in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.recovery.resolve(), args.output.resolve())
