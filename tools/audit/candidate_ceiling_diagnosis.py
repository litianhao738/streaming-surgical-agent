"""Offline split of repair loss into a coverage ceiling and a review rejection.

Reads only closed experiment archives and the Training ground truth they already
scored. No API call, no dataset read, no prediction is regenerated and no
archive is written. For every head each missed GT label becomes either
NOT_PROPOSED, meaning it never entered any candidate pool that run, so no
reviewer could have admitted it, or REJECTED, meaning a pool held it and the
panel still did not select it. The pool's ceiling recall is the best recall any
perfect reviewer could have reached on that run.

Phase is reported separately because it is single-label and its pool is fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

HEADS = ("instrument", "verb", "target", "ivt")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk_pools(node, found):
    """Every candidate pool the run ever built, at any nesting depth."""
    if isinstance(node, dict):
        propositions = node.get("propositions")
        if isinstance(propositions, list) and all(
                isinstance(p, dict) and {"task", "label_id"} <= p.keys() for p in propositions):
            found.append(propositions)
        for value in node.values():
            walk_pools(value, found)
    elif isinstance(node, list):
        for value in node:
            walk_pools(value, found)
    return found


def reviewed_labels(result):
    """Union over every round: was this label ever put in front of a reviewer?"""
    out = {h: set() for h in HEADS}
    for propositions in walk_pools(result, []):
        for p in propositions:
            if p["task"] in out and type(p["label_id"]) is int:
                out[p["task"]].add(p["label_id"])
    return out


def labels_of(prediction, head):
    value = prediction.get(head)
    if isinstance(value, dict):  # tolerate a raw five-head wire payload
        value = value.get("selected_ids", [value.get("selected_id")])
    return set(value or [])


def decompose(archive):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    rows, arms = [], set()
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        if key not in truth:
            continue
        result = read(path)
        predictions = result.get("predictions")
        if not isinstance(predictions, dict):
            continue
        arms.update(predictions)
        rows.append({"key": key, "truth": truth[key], "pool": reviewed_labels(result),
                     "predictions": predictions})
    if not rows:
        raise ValueError("no scored target with predictions in " + str(root))
    report = {"archive": root.name, "targets": len(rows), "arms": sorted(arms), "heads": {}}
    for head in HEADS:
        scored = [r for r in rows if r["truth"]["mask"].get(head)]
        gt_total = sum(len(set(r["truth"]["gt"][head])) for r in scored)
        in_pool = sum(len(set(r["truth"]["gt"][head]) & r["pool"][head]) for r in scored)
        entry = {"scored_targets": len(scored), "gt_labels": gt_total,
                 "gt_in_pool": in_pool,
                 "ceiling_recall": round(100 * in_pool / gt_total, 2) if gt_total else None,
                 "never_proposed": gt_total - in_pool, "arms": {}}
        for arm in sorted(arms):
            hit = missed_absent = missed_rejected = predicted = 0
            for r in scored:
                if arm not in r["predictions"]:
                    continue
                gt = set(r["truth"]["gt"][head])
                pool, pred = r["pool"][head], labels_of(r["predictions"][arm], head)
                predicted += len(pred)
                hit += len(gt & pred)
                missed_absent += len(gt - pool)
                missed_rejected += len((gt & pool) - pred)
            entry["arms"][arm] = {
                "recall": round(100 * hit / gt_total, 2) if gt_total else None,
                "predicted_labels": predicted, "correct": hit,
                "missed_never_proposed": missed_absent, "missed_rejected_by_review": missed_rejected}
        report["heads"][head] = entry
    report["phase"] = phase_report(rows)
    return report


def phase_report(rows):
    scored = [r for r in rows if r["truth"]["mask"].get("phase")]
    arms = sorted({a for r in scored for a in r["predictions"]})
    out = {"scored_targets": len(scored), "arms": {}}
    for arm in arms:
        correct = changed = broke = fixed = 0
        for r in scored:
            if arm not in r["predictions"]:
                continue
            gt = set(r["truth"]["gt"]["phase"])
            pred = labels_of(r["predictions"][arm], "phase")
            base = labels_of(r["predictions"].get("h0", {}), "phase")
            correct += bool(pred & gt)
            if base and pred != base:
                changed += 1
                fixed += bool(pred & gt) and not (base & gt)
                broke += bool(base & gt) and not (pred & gt)
        out["arms"][arm] = {"correct": correct, "of": len(scored), "changed_vs_h0": changed,
                            "fixed": fixed, "broke": broke}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = [decompose(a) for a in args.archives]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
