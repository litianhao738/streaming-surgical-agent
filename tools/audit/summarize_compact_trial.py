"""Summarize a scored compact-prompt trial; no model calls."""
import argparse
import json
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output):
    report, ledger = read(output / "metrics.json"), read(output / "budget.json")
    calls = ledger["calls"]
    pairs = report["input_token_pairs"]
    comparable = [p for p in pairs if p["nonincreasing"] is not None]
    result = {"success": report["success"], "checks": report["success_checks"],
              "f1": {a: {t: round(100 * v["micro_f1"], 2) for t, v in m["tasks"].items()}
                     for a, m in report["metrics"].items()},
              "comparable_input_pairs": len(comparable), "missing_input_pairs": len(pairs) - len(comparable),
              "increased_input_pairs": sum(p["nonincreasing"] is False for p in pairs),
              "strictly_decreased_input_pairs": sum(p["prompt_tokens"]["compact"] < p["prompt_tokens"]["control"] for p in comparable),
              "branch_input_tokens": {}, "reconstructed_parallel_api_seconds": {},
              "costs_full_160": {a: report["totals"][a]["costs"] for a in ("control", "compact")},
              "cumulative_occupied_including_v1_and_unknown_reserves": ledger["occupied"],
              "comparisons": report["comparisons"],
              "validity": {a: {k: report["totals"][a][k] for k in ("valid_graph_items", "valid_phase_responses", "statuses")}
                           for a in ("control", "compact")}}
    for branch in ("graph", "phase"):
        group = [p for p in comparable if p["branch"] == branch]
        totals = {a: sum(p["prompt_tokens"][a] for p in group) for a in ("control", "compact")}
        result["branch_input_tokens"][branch] = {"pairs": len(group), **totals,
                     "reduction_percent": round(100 * (1 - totals["compact"] / totals["control"]), 2)}
    for arm in ("control", "compact"):
        own = [c for c in calls if c["stage"].startswith(arm + "_")]
        elapsed = sum(max(c["elapsed_seconds"] for c in own if c["target"] == key)
                      for key in {c["target"] for c in own})
        result["reconstructed_parallel_api_seconds"][arm] = round(elapsed, 3)
        result["costs_full_160"][arm]["native_usd_total"] = str(sum(Decimal(v["native"])
                    for k, v in report["totals"][arm]["costs"].items() if k.endswith("usd")))
    result["per_seat_input_tokens"] = {}
    for seat in {p["seat"] for p in pairs}:
        group = [p for p in comparable if p["seat"] == seat]
        result["per_seat_input_tokens"][seat] = {"pairs": len(group),
               **{a: sum(p["prompt_tokens"][a] for p in group) for a in ("control", "compact")}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.output)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
