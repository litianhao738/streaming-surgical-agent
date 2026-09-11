"""Offline, mask-aware summaries for closed short/context Phase rating arms."""
import hashlib
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/preflight/phase_temporal_ratings_same32_20260910_v2"
TASKS = ("instrument", "verb", "target", "ivt", "phase")


def read(name):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def pct(value):
    return str((Decimal(str(value))*100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    done, metrics, audit, checks = [read(n) for n in ("completion.json", "metrics.json", "independent_audit.json", "temporal_rating_audit.json")]
    assert done["fatal_error"] is None and metrics["raw_replayed"]
    assert audit["metrics_sha256"] == hashlib.sha256((OUTPUT / "metrics.json").read_bytes()).hexdigest()
    for name, digest in done["hashes"].items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
    plan, rows = read("plan.json"), read("predictions.json")
    truths = {(r["video_id"], r["frame_id"]): r for r in read("scored_truth.json")}
    valid = set(checks["compact_repeat"]["valid_targets"]) & set(checks["revised_phase"]["valid_targets"])
    full = {r["key"] for r in plan["selection"] if [f-r["frame_id"] for f in r["phase_context"]["causal_frame_ids"]] == [-750, -250, 0]}
    subset_metrics = {}
    for name, keys in (("both_valid", valid), ("exact30", full), ("exact30_both_valid", full & valid)):
        values = {}
        for arm in ("h0", "compact_repeat", "revised_phase"):
            selected = [r for r in rows if r["key"] in keys and truths[r["video_id"], r["frame_id"]]["mask"]["phase"]]
            good = sum(r[arm]["phase"] == truths[r["video_id"], r["frame_id"]]["gt"]["phase"] for r in selected)
            values[arm] = {"valid_targets": len(selected), "correct": good, "phase_f1": good/len(selected) if selected else None}
        subset_metrics[name] = values
    calls = read("budget.json")["calls"]
    costs = {"native_usd": Decimal(0), "estimated_cny": Decimal(0), "unknown_reserved_usd": Decimal(0), "unknown_reserved_cny": Decimal(0)}
    for c in calls:
        if c["account"].endswith("usd"):
            costs["unknown_reserved_usd" if c["charge_kind"] == "unknown_reserved" else "native_usd"] += Decimal(c["charge"])
        else:
            costs["unknown_reserved_cny" if c["charge_kind"] == "unknown_reserved" else "estimated_cny"] += Decimal(c["charge"])
    changes = {}
    for arm in ("compact_repeat", "revised_phase"):
        cases = []
        for r in rows:
            if r[arm]["phase"] == r["h0"]["phase"]:
                continue
            branch = read(f"targets/{r['key']}/result.json")["branches"][arm]
            cases.append({"key": r["key"], "h0": r["h0"]["phase"], "final": r[arm]["phase"],
                "gt": truths[r["video_id"], r["frame_id"]]["gt"]["phase"], "decision": branch["decision"]})
        changes[arm] = cases
    report = {"phase_f1": {a: m["tasks"]["phase"]["micro_f1"] for a, m in metrics["metrics"].items()},
        "phase_changes": audit["phase_switches_vs_h0"], "phase_cases": changes, "subsets": subset_metrics,
        "validity": checks, "costs": {k: str(v) for k, v in costs.items()}, "elapsed_seconds": metrics["elapsed_seconds"],
        "calls": len(calls), "statuses": dict(Counter(c["status"] for c in calls)),
        "failures": [{k: c.get(k) for k in ("target", "stage", "seat", "status", "http_status")} for c in calls if c["status"] != "JSON_PARSED"],
        "success": metrics["revised_phase_success"], "metrics_sha256": audit["metrics_sha256"]}
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = []
    arms = {"h0": "H0", "compact_repeat": "本次短历史评分", "revised_phase": "本次较长历史评分"}
    for field, title in (("micro_f1", "F1"), ("micro_precision", "Precision"), ("exact_set_accuracy", "集合Accuracy")):
        lines += [f"### {title}（%）", "", "| 版本 | Instrument | Verb | Target | IVT | Phase |", "|---|---:|---:|---:|---:|---:|"]
        for arm, label in arms.items():
            m = metrics["metrics"][arm]
            lines.append("| "+label+" | "+" | ".join(pct(m["tasks"][t][field]) for t in TASKS)+" |")
        lines.append("")
    (OUTPUT / "result_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("phase_cases", "validity")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
