"""Summarize two closed, replayed Phase experiments without inference calls."""
import hashlib
import json
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from surgical_agent.research.verification.phase_extension import phase_choice_error

FIRST = ROOT / "artifacts/preflight/phase_mechanism_same32_20260910_v1"
SECOND = ROOT / "artifacts/preflight/phase_hypothesis_instance_same32_20260910_v2"
TASKS = ("instrument", "verb", "target", "ivt", "phase")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    archives = {}
    for name, folder in (("first", FIRST), ("format_v2", SECOND)):
        completion, metrics = read(folder / "completion.json"), read(folder / "metrics.json")
        assert not completion["fatal_error"] and metrics["raw_replayed"]
        assert read(folder / "independent_audit.json")["independent_head_tables_verified"] in (20, 45)
        for rel, digest in completion["hashes"].items():
            assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == digest
        archives[name] = {"folder": folder, "metrics": metrics,
            "predictions": {r["key"]: r for r in read(folder / "predictions.json")},
            "truth": {(r["video_id"], r["frame_id"]): r for r in read(folder / "scored_truth.json")}}
    first, second = archives.values()
    assert set(first["predictions"]) == set(second["predictions"])
    assert set(first["truth"]) == set(second["truth"])
    for key, value in first["truth"].items():
        # Scorer rows also carry the last scored arm's `final`; that is not GT.
        for field in ("video_id", "frame_id", "gt", "mask", "h0"):
            assert value[field] == second["truth"][key][field]
    for key in first["predictions"]:
        for field in ("h0", "cached_default", "graph_before_phase"):
            assert first["predictions"][key][field] == second["predictions"][key][field]
    specifications = [(a, first, a) for a in first["metrics"]["metrics"]]
    specifications += [("revised_phase_format_v2", second, "revised_phase")]
    report = {"archives": {n: {"path": str(a["folder"]),
        "metrics_sha256": hashlib.sha256((a["folder"] / "metrics.json").read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((a["folder"] / "plan.json").read_bytes()).hexdigest()}
        for n, a in archives.items()}, "results": {}, "phase_validation": {}, "paired_valid_sensitivity": {}}
    valid_targets = {}
    for label, archive, arm in specifications:
        metric = archive["metrics"]["metrics"][arm]
        switches = Counter()
        for key, row in archive["predictions"].items():
            truth = archive["truth"][row["video_id"], row["frame_id"]]
            if truth["mask"]["phase"]:
                before, after, gt = row["h0"]["phase"], row[arm]["phase"], truth["gt"]["phase"]
                switches["unchanged" if before == after else "corrected" if after == gt
                    else "harmed" if before == gt else "wrong_to_wrong"] += 1
        report["results"][label] = {"tasks": metric["tasks"],
            "mean_f1": sum(metric["tasks"][t]["micro_f1"] for t in TASKS) / 5,
            "phase_changes_vs_h0": dict(switches)}
        if arm not in ("compact_repeat", "original_phase", "revised_phase"):
            continue
        reasons, errors, valid = Counter(), Counter(), set()
        for key in archive["predictions"]:
            branch = read(archive["folder"] / "targets" / key / "result.json")["branches"][arm]
            reasons[branch["decision"]["reason"]] += 1
            bad = {s: phase_choice_error(raw, 3) for s, raw in branch["raw"].items()}
            for seat, error in bad.items():
                if error:
                    raw = branch["raw"][seat]
                    kind = "missing_response" if raw is None else "parsed_but_invalid"
                    errors[f"{seat}:{error}:{kind}"] += 1
            if not any(bad.values()):
                valid.add(key)
        valid_targets[label] = valid
        report["phase_validation"][label] = {"decision_counts": dict(reasons), "errors": dict(errors),
            "valid_panels": len(valid), "invalid_targets": sorted(set(archive["predictions"]) - valid)}
    for label, archive, arm in specifications:
        if label not in valid_targets or label == "compact_repeat":
            continue
        selected = sorted(valid_targets[label] & valid_targets["compact_repeat"])
        values = {}
        for comparison, source, field in (("h0", first, "h0"), ("compact_repeat", first, "compact_repeat"),
                                           (label, archive, arm)):
            good = 0
            for key in selected:
                row = source["predictions"][key]
                gt = source["truth"][row["video_id"], row["frame_id"]]["gt"]["phase"]
                good += row[field]["phase"] == gt
            values[comparison] = {"correct": good, "targets": len(selected),
                "phase_f1": good / len(selected) if selected else None}
        report["paired_valid_sensitivity"][label] = {"keys": selected, "results": values,
            "note": "Both panels pass structural validation; post-outcome subset is a sensitivity check, not primary selection."}
    totals = {"calls": 0, "elapsed_seconds": 0, "native_usd": Decimal(0),
        "estimated_cny": Decimal(0), "unknown_reserved_usd": Decimal(0), "unknown_reserved_cny": Decimal(0)}
    for archive in archives.values():
        totals["elapsed_seconds"] += archive["metrics"]["elapsed_seconds"]
        for item in archive["metrics"]["totals"].values():
            totals["calls"] += item["calls"]
            for account, cost in item["costs"].items():
                if account.endswith("usd"):
                    totals["native_usd"] += Decimal(cost["native"])
                    totals["unknown_reserved_usd"] += Decimal(cost["unknown_reserved"])
                    assert Decimal(cost["conservative_estimate"]) == 0
                else:
                    totals["estimated_cny"] += Decimal(cost["conservative_estimate"])
                    totals["unknown_reserved_cny"] += Decimal(cost["unknown_reserved"])
                    assert Decimal(cost["native"]) == 0
    report["combined_cost_time"] = {k: str(v) if isinstance(v, Decimal) else v for k, v in totals.items()}
    (SECOND / "combined_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = []
    for field, title in (("micro_f1", "F1"), ("micro_precision", "Precision"), ("exact_set_accuracy", "集合 Accuracy")):
        lines += [f"### {title}（%）", "", "| 版本 | Instrument | Verb | Target | IVT | Phase |", "|---|---:|---:|---:|---:|---:|"]
        for label, result in report["results"].items():
            lines.append("| " + label + " | " + " | ".join(percent(result['tasks'][t][field]) for t in TASKS) + " |")
        lines.append("")
    (SECOND / "result_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase_validation": report["phase_validation"],
        "phase_f1": {a: r["tasks"]["phase"]["micro_f1"] for a, r in report["results"].items()},
        "combined_cost_time": report["combined_cost_time"],
        "paired_valid_sensitivity": report["paired_valid_sensitivity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
