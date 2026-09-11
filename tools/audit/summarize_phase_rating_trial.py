"""Offline comparison tables and valid-panel sensitivity for the Phase rating trial."""
import hashlib
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/preflight/phase_simple_ratings_same32_20260910_v1"
PREVIOUS = ROOT / "artifacts/preflight/phase_hypothesis_instance_same32_20260910_v2"
TASKS = ("instrument", "verb", "target", "ivt", "phase")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return str((Decimal(str(value))*100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    completion, metrics, audit = (read(OUTPUT / n) for n in ("completion.json", "metrics.json", "independent_audit.json"))
    assert completion["fatal_error"] is None and metrics["raw_replayed"]
    assert audit["metrics_sha256"] == hashlib.sha256((OUTPUT / "metrics.json").read_bytes()).hexdigest()
    for name, digest in completion["hashes"].items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
    details = read(OUTPUT / "phase_rating_audit.json")
    rows = {r["key"]: r for r in read(OUTPUT / "predictions.json")}
    old_rows = {r["key"]: r for r in read(PREVIOUS / "predictions.json")}
    truths = {(r["video_id"], r["frame_id"]): r for r in read(OUTPUT / "scored_truth.json")}
    old_truths = {(r["video_id"], r["frame_id"]): r for r in read(PREVIOUS / "scored_truth.json")}
    assert set(rows) == set(old_rows)
    for key, row in rows.items():
        assert row["h0"] == old_rows[key]["h0"] and row["graph_before_phase"] == old_rows[key]["graph_before_phase"]
        identity = row["video_id"], row["frame_id"]
        assert all(truths[identity][k] == old_truths[identity][k] for k in ("gt", "mask"))
    previous = read(PREVIOUS / "metrics.json")
    variants = {"H0": metrics["metrics"]["h0"], "四头修复＋Phase保留H0": metrics["metrics"]["graph_before_phase"],
        "历史长提示v2（跨次参考）": previous["metrics"]["revised_phase"],
        "本次简短单选": metrics["metrics"]["compact_repeat"], "本次简短评分": metrics["metrics"]["revised_phase"]}
    selected = set(details["compact_repeat"]["valid_targets"]) & set(details["revised_phase"]["valid_targets"])
    sensitivity = {}
    for arm in ("h0", "compact_repeat", "revised_phase"):
        good = sum(rows[k][arm]["phase"] == truths[rows[k]["video_id"], rows[k]["frame_id"]]["gt"]["phase"]
            for k in selected if truths[rows[k]["video_id"], rows[k]["frame_id"]]["mask"]["phase"])
        valid = sum(truths[rows[k]["video_id"], rows[k]["frame_id"]]["mask"]["phase"] for k in selected)
        sensitivity[arm] = {"valid_targets": valid, "correct": good, "phase_f1": good/valid if valid else None}
    costs = {"native_usd": Decimal(0), "estimated_cny": Decimal(0), "unknown_reserved_usd": Decimal(0),
        "unknown_reserved_cny": Decimal(0)}
    for total in metrics["totals"].values():
        for account, amounts in total["costs"].items():
            if account.endswith("usd"):
                costs["native_usd"] += Decimal(amounts["native"])
                costs["unknown_reserved_usd"] += Decimal(amounts["unknown_reserved"])
            else:
                costs["estimated_cny"] += Decimal(amounts["conservative_estimate"])
                costs["unknown_reserved_cny"] += Decimal(amounts["unknown_reserved"])
    calls = read(OUTPUT / "budget.json")["calls"]
    switches = {}
    for arm in ("compact_repeat", "revised_phase"):
        cases = []
        for key, row in rows.items():
            if row[arm]["phase"] != row["h0"]["phase"]:
                branch = read(OUTPUT / "targets" / key / "result.json")["branches"][arm]
                cases.append({"key": key, "h0": row["h0"]["phase"], "final": row[arm]["phase"],
                    "gt": truths[row["video_id"], row["frame_id"]]["gt"]["phase"], "decision": branch["decision"]})
        switches[arm] = cases
    report = {"phase_f1": {a: m["tasks"]["phase"]["micro_f1"] for a, m in variants.items()},
        "mean_f1": {a: sum(m["tasks"][t]["micro_f1"] for t in TASKS)/5 for a, m in variants.items()},
        "phase_switches": switches, "phase_changes": audit["phase_switches_vs_h0"],
        "valid_panel_sensitivity": sensitivity, "valid_panel_keys": sorted(selected),
        "costs": {k: str(v) for k, v in costs.items()}, "elapsed_seconds": metrics["elapsed_seconds"],
        "calls": len(calls), "statuses": dict(Counter(c["status"] for c in calls)),
        "predeclared_success": metrics["revised_phase_success"], "phase_validation": details,
        "metrics_sha256": audit["metrics_sha256"],
        "previous_metrics_sha256": hashlib.sha256((PREVIOUS / "metrics.json").read_bytes()).hexdigest()}
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = []
    for field, title in (("micro_f1", "F1"), ("micro_precision", "Precision"), ("exact_set_accuracy", "集合Accuracy")):
        lines += [f"### {title}（%）", "", "| 版本 | Instrument | Verb | Target | IVT | Phase |", "|---|---:|---:|---:|---:|---:|"]
        for name, m in variants.items():
            lines.append("| "+name+" | "+" | ".join(pct(m["tasks"][t][field]) for t in TASKS)+" |")
        lines.append("")
    (OUTPUT / "result_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("phase_switches", "phase_validation", "valid_panel_keys")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
