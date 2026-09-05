"""Read-only paired comparisons; writes a separate analysis artifact."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surgical_agent.artifacts.manifest import atomic_write_json

TASKS = ("instrument", "verb", "target", "ivt", "phase")


def analyze(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    rows = {(r["arm"], r["frame"]): r for r in summary["frames"]}
    fids = sorted({fid for _, fid in rows})
    paired = {}
    for before, after in (("A", "B"), ("B", "C"), ("A", "C")):
        comparison = {}
        for task in TASKS:
            common = [fid for fid in fids if rows[before, fid]["exact"][task] is not None
                      and rows[after, fid]["exact"][task] is not None]
            comparison[task] = {"n": len(common),
                "before_correct": sum(rows[before, fid]["exact"][task] for fid in common),
                "after_correct": sum(rows[after, fid]["exact"][task] for fid in common),
                "gained_frames": [fid for fid in common if not rows[before, fid]["exact"][task] and rows[after, fid]["exact"][task]],
                "lost_frames": [fid for fid in common if rows[before, fid]["exact"][task] and not rows[after, fid]["exact"][task]]}
        paired[f"{before}_to_{after}"] = comparison
    closure = {}
    for arm in ("A", "B", "C"):
        available = [r for r in summary["frames"] if r["arm"] == arm and r["closure"] is not None]
        closure[arm] = {"n": len(available),
            "component_inclusion_pass": sum(r["closure"]["ivt_components_in_selected"] for r in available),
            "exact_projection_pass": sum(r["closure"]["exact_projection"] for r in available),
            "failed_inclusion_frames": [r["frame"] for r in available if not r["closure"]["ivt_components_in_selected"]]}
    stratified = {}
    for arm in ("A", "B", "C"):
        stratified[arm] = {}
        for name in ("null_only", "mixed", "nonnull_only"):
            chosen = []
            for fid in fids:
                row = rows[arm, fid]
                gt = row["gt"]["ivt"]
                category = "null_only" if all(i >= 94 for i in gt) else "nonnull_only" if all(i < 94 for i in gt) else "mixed"
                if category == name:
                    chosen.append(row)
            stratified[arm][name] = {t: {"scored": sum(r["exact"][t] is not None for r in chosen),
                                         "correct": sum(r["exact"][t] is True for r in chosen)} for t in TASKS}
    value = {"paired": paired, "closure": closure, "stratified": stratified,
             "failures": [r for r in results if r["status"] != "OK"]}
    atomic_write_json(directory / "paired_analysis.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
