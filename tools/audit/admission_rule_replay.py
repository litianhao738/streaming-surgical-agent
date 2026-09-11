"""Replay alternative panel-aggregation rules over closed archives.

Offline only: no API call, no dataset read, no archive write, no default
changed. The published selector `recent_mean_panel.select` is called unchanged;
only the per-candidate mean handed to it is recomputed, so any difference comes
from the aggregation rule and not from a reimplemented admission rule.

Rules compared
  current          mean of five seats, blocked unless all five are valid
  median5          median of five seats, blocked unless all five are valid
  relaxed4_mean    mean of the valid seats when at least four are valid
  relaxed4_median  median of the valid seats when at least four are valid
  relaxed3_mean    mean of the valid seats when at least three are valid
  seats3_median    median of a fixed three-seat subset, all three valid

`current` is expected to reproduce the archive's own four-head prediction; that
check is reported per round and is the correctness guard for everything else.
Phase is carried over from the archive unchanged, so this isolates the four
interaction heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS

HEADS = ("instrument", "verb", "target", "ivt")
SUBSETS = tuple(combinations(range(len(SEATS)), 3))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_rounds(node, path=""):
    """Every archived review round that kept its pool, means and diagnostics."""
    found = []
    if isinstance(node, dict):
        if all(k in node for k in ("pool", "means", "diagnostics")) and isinstance(
                node.get("pool"), dict) and isinstance(node.get("diagnostics"), dict):
            found.append((path or "root", node))
        for key, value in node.items():
            found += find_rounds(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += find_rounds(value, f"{path}[{i}]")
    return found


def aggregate(scores, rule):
    """None means the rule refuses to score this candidate, which keeps H0."""
    if not isinstance(scores, list) or len(scores) != len(SEATS):
        return None
    valid = [s for s in scores if s is not None]
    if rule == "current":
        return mean(scores) if len(valid) == len(SEATS) else None
    if rule == "median5":
        return median(scores) if len(valid) == len(SEATS) else None
    if rule == "relaxed4_mean":
        return mean(valid) if len(valid) >= 4 else None
    if rule == "relaxed4_median":
        return median(valid) if len(valid) >= 4 else None
    if rule == "relaxed3_mean":
        return mean(valid) if len(valid) >= 3 else None
    if rule.startswith("seats3_median:"):
        picked = [scores[i] for i in map(int, rule.split(":")[1].split(","))]
        return median(picked) if all(s is not None for s in picked) else None
    raise ValueError("unknown rule " + rule)


BLOCKS = ("ivt_delete_only", "delete_only")


def rules():
    base = ["current", "median5", "relaxed4_mean", "relaxed4_median", "relaxed3_mean"]
    # A "+block" suffix withholds the score of every candidate H0 did not hold,
    # which the frozen selector reads as "keep": additions become impossible,
    # deletions are untouched. ivt_delete_only blocks new relations only.
    blocked = [f"{b}+{block}" for b in ("current", "relaxed3_mean") for block in BLOCKS]
    return base + blocked + [f"seats3_median:{','.join(map(str, c))}" for c in SUBSETS]


def blocked(rule, proposition, h0):
    block = rule.partition("+")[2]
    if not block:
        return False
    novel = proposition["label_id"] not in set(h0[proposition["task"]])
    return novel and (block == "delete_only" or proposition["task"] == "ivt")


def counts_for(prediction, truth):
    out = Counter()
    for head in HEADS:
        if not truth["mask"].get(head):
            continue
        gt, got = set(truth["gt"][head] or []), set(prediction[head])
        out[f"{head}_tp"] += len(gt & got)
        out[f"{head}_fp"] += len(got - gt)
        out[f"{head}_fn"] += len(gt - got)
    return out


def replay(archive):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    totals, faithful, rounds_seen = {}, Counter(), Counter()
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        row = truth.get(key)
        if not row:
            continue
        result = read(path)
        h0 = result.get("h0")
        if not isinstance(h0, dict):
            continue
        for name, node in find_rounds(result):
            pool = {"propositions": [p for p in node["pool"]["propositions"]
                                     if p.get("task") in HEADS]}
            if not pool["propositions"]:
                continue
            base = {t: list(h0[t]) for t in HEADS}
            base["phase"] = list(h0.get("phase", [0]))
            archived = node.get("prediction")
            for rule in rules():
                base_rule = rule.partition("+")[0]
                means = {p["id"]: None if blocked(rule, p, h0) else aggregate(
                    (node["diagnostics"].get(p["id"]) or {}).get("scores"), base_rule)
                    for p in pool["propositions"]}
                try:
                    out = panel.select(base, pool, means, threshold=4.0)
                except (ValueError, TypeError, KeyError):
                    continue
                bucket = totals.setdefault((name, rule), Counter())
                bucket.update(counts_for(out, row))
                bucket["targets"] += 1
                if rule == "current" and isinstance(archived, dict):
                    rounds_seen[name] += 1
                    faithful[name] += all(
                        sorted(out[h]) == sorted(archived.get(h, [])) for h in HEADS)
    return {"archive": root.name, "totals": totals,
            "faithfulness": {k: f"{faithful[k]}/{rounds_seen[k]}" for k in rounds_seen}}


def f1(c, head):
    tp, fp, fn = c[f"{head}_tp"], c[f"{head}_fp"], c[f"{head}_fn"]
    return round(200 * tp / (2 * tp + fp + fn), 2) if (2 * tp + fp + fn) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--round", default=None, help="only this archived round name")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    merged, faith = {}, {}
    for archive in args.archives:
        try:
            r = replay(archive)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        faith[r["archive"]] = r["faithfulness"]
        for (name, rule), c in r["totals"].items():
            if args.round and name != args.round:
                continue
            merged.setdefault(rule, Counter()).update(c)

    if not merged:
        print("no replayable round found")
        return
    print("faithfulness of `current` vs archived prediction (matched/rounds):")
    for a, f in faith.items():
        if f:
            print(f"   {a}: {f}")
    print(f"\n{'rule':<32}" + "".join(f"{h:>11}" for h in HEADS)
          + f"{'meanP':>8}{'IVT tp/fp':>11}{'sumFP+FN':>10}{'items':>7}")
    named = [r for r in merged if not r.startswith("seats3")]
    for rule in named:
        c = merged[rule]
        errs = sum(c[f"{h}_fp"] + c[f"{h}_fn"] for h in HEADS)
        precision = [100 * c[f"{h}_tp"] / (c[f"{h}_tp"] + c[f"{h}_fp"])
                     for h in HEADS if c[f"{h}_tp"] + c[f"{h}_fp"]]
        print(f"{rule:<32}" + "".join(f"{f1(c, h):>11}" for h in HEADS)
              + f"{sum(precision) / len(precision):>8.2f}"
              + f"{str(c['ivt_tp']) + '/' + str(c['ivt_fp']):>11}{errs:>10}{c['targets']:>7}")
    subs = {r: merged[r] for r in merged if r.startswith("seats3")}
    if subs:
        print(f"\nthree-seat medians ({len(subs)} subsets), per head F1 spread:")
        for h in HEADS:
            vals = sorted(f1(c, h) for c in subs.values())
            print(f"   {h:<11} min={vals[0]:<7} median={vals[len(vals)//2]:<7} max={vals[-1]:<7}"
                  f" | five-seat current={f1(merged['current'], h)}")
        best = max(subs, key=lambda r: sum(f1(subs[r], h) or 0 for h in HEADS))
        print(f"   (best post-hoc subset {best.split(':')[1]} — selection on the same data, "
              f"not a validated choice)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"faithfulness": faith,
             "totals": {r: dict(c) for r, c in merged.items()},
             "f1": {r: {h: f1(c, h) for h in HEADS} for r, c in merged.items()}},
            indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
