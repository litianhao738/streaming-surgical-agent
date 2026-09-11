"""Independent integer metrics, actual-roster checks and readable trial tables."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def audit(output):
    metrics = read(output / "metrics.json")
    rows, truths = read(output / "predictions.json"), read(output / "scored_truth.json")
    truth = {(r["video_id"], r["frame_id"]): r for r in truths}
    plan, budget = read(output / "plan.json"), read(output / "budget.json")
    checked = 0
    for arm, report in metrics["metrics"].items():
        for task, recorded in report["tasks"].items():
            counts = Counter()
            for row in rows:
                gt = truth[row["video_id"], row["frame_id"]]
                if not gt["mask"][task]:
                    continue
                expected, predicted = set(gt["gt"][task]), set(row[arm][task])
                counts.update(tp=len(expected & predicted), fp=len(predicted - expected), fn=len(expected - predicted),
                    valid_targets=1, exact_matches=int(expected == predicted))
            assert all(counts[k] == recorded[k] for k in counts), (arm, task, counts)
            den = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            assert abs(recorded["micro_f1"] - (2 * counts["tp"] / den if den else 0)) < 1e-10
            checked += 1
    for call in budget["calls"]:
        expected = "google/gemini-3.8-flash" if call["seat"] == "base" else plan["models"][call["seat"]]
        assert call["model"] == expected
        assert call["account"] == ("aliyun_cny" if call["seat"] == "qwen" else "openrouter_usd")
        folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        request = read(folder / "request.json")
        assert request["model"] == expected
    for account in plan["limits"]:
        total = sum(Decimal(c["charge"]) for c in budget["calls"] if c["account"] == account)
        assert total == Decimal(budget["occupied"][account]) <= Decimal(plan["limits"][account])
    phase_edits = {}
    for arm in metrics["metrics"]:
        counts = Counter()
        for row in rows:
            gt = truth[row["video_id"], row["frame_id"]]
            if not gt["mask"]["phase"]:
                continue
            before, after, expected = row["h0"]["phase"], row[arm]["phase"], gt["gt"]["phase"]
            counts["corrected" if before != expected and after == expected else
                   "harmed" if before == expected and after != expected else
                   "wrong_to_wrong" if before != after else "unchanged"] += 1
        phase_edits[arm] = dict(counts)
    table = ["# Joint Phase feedback trial", "", "F1 (%); arithmetic mean of five task micro-F1 scores.", "",
        "| Arm | I | V | T | IVT | Phase | Mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    tasks = ("instrument", "verb", "target", "ivt", "phase")
    for arm, report in metrics["metrics"].items():
        values = [report["tasks"][t]["micro_f1"] for t in tasks]
        table.append("| " + " | ".join([arm, *map(pct, values), pct(sum(values) / 5)]) + " |")
    completion = read(output / "completion.json")
    interrupted = completion.get("recovered_interrupted_run", False)
    timing = completion.get("elapsed_time_note", "API timestamp span lower bound (interrupted run)") if interrupted else "Actual paired execution"
    table += ["", "Phase changes relative to shared H0:", "", "```json", json.dumps(phase_edits, indent=2), "```",
        "", "Predeclared success: " + json.dumps(metrics["success"]), "", "Costs: " + json.dumps(metrics["costs"]),
        "", f"{timing}: {metrics['elapsed_seconds']:.2f} s; {metrics['calls']} calls. Cached upstream excluded."]
    (output / "result_tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    result = {"independent_head_tables_verified": checked, "request_models_verified": True,
        "ledger_verified": True, "phase_changes": phase_edits,
        "metrics_sha256": hashlib.sha256((output / "metrics.json").read_bytes()).hexdigest()}
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
