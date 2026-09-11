"""Do confidence-free features predict whether repair will change anything?

Offline over closed archives. No API call, no dataset read, no archive write.

The label is whether the repaired prediction differs from H0 on any head. It is
derived from the two archived predictions alone, so no ground truth is used and
every archived target is usable, including the six Training videos that carry
no verb/target/IVT supervision.

Evaluation is leave-one-video-out: a fold never scores a video it was fitted
on. A cost Gate is only worth building if it beats the trivial rule of always
running repair, so the always-run baseline is reported beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.gate.final_only_features import (
    HEADS,
    feature_names,
    structural,
    with_proposal,
)
from tools.audit.candidate_ceiling_diagnosis import labels_of, read

REPAIR_CALLS = 6


def find_rounds(node, path=""):
    found = []
    if isinstance(node, dict):
        if isinstance(node.get("pool"), dict) and "means" in node:
            found.append((path or "root", node))
        for key, value in node.items():
            found += find_rounds(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += find_rounds(value, f"{path}[{i}]")
    return found


def collect(archives, stage):
    rows = []
    for archive in archives:
        root = Path(archive)
        target_dir = root / "targets"
        if not target_dir.is_dir():
            continue
        for path in sorted(target_dir.glob("*/result.json")):
            key = path.parent.name
            result = read(path)
            h0 = result.get("h0")
            predictions = result.get("predictions")
            if not isinstance(h0, dict) or not isinstance(predictions, dict):
                continue
            arm = next((a for a in ("joint_r1", "split", "graph", "control", "prior_graph")
                        if a in predictions and a != "h0"), None)
            if arm is None:
                continue
            changed = any(labels_of(predictions[arm], head) != labels_of(h0, head)
                          for head in (*HEADS, "phase"))
            hints = result.get("hints")
            proposal = result.get("proposal_raw") or result.get("proposal")
            features = (with_proposal(h0, proposal, hints) if stage == "with_proposal"
                        else structural(h0, hints))
            rows.append({"archive": root.name, "key": key, "video": key.split("_")[0],
                         "changed": int(changed), "features": features})
    return rows


def fit_predict(train, test, names):
    """Small logistic regression; falls back to the base rate if sklearn is absent."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        rate = sum(r["changed"] for r in train) / max(len(train), 1)
        return [rate] * len(test)
    x_train = [[r["features"].get(n, 0) for n in names] for r in train]
    y_train = [r["changed"] for r in train]
    if len(set(y_train)) < 2:
        rate = sum(y_train) / max(len(y_train), 1)
        return [rate] * len(test)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000, C=0.5).fit(scaler.transform(x_train), y_train)
    x_test = [[r["features"].get(n, 0) for n in names] for r in test]
    return list(model.predict_proba(scaler.transform(x_test))[:, 1])


def auc(pairs):
    positives = [s for s, y in pairs if y == 1]
    negatives = [s for s, y in pairs if y == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


def evaluate(rows, names):
    videos = sorted({r["video"] for r in rows})
    scored = []
    for held in videos:
        train = [r for r in rows if r["video"] != held]
        test = [r for r in rows if r["video"] == held]
        if not train or not test:
            continue
        for r, s in zip(test, fit_predict(train, test, names), strict=True):
            scored.append((s, r["changed"], r))
    pairs = [(s, y) for s, y, _ in scored]
    result = {"targets": len(rows), "videos": len(videos),
              "change_rate_pct": round(100 * sum(r["changed"] for r in rows) / len(rows), 2),
              "held_out_auc": auc(pairs)}
    best = None
    for cut in [i / 100 for i in range(5, 100, 5)]:
        skipped = [(s, y) for s, y in pairs if s < cut]
        if not skipped:
            continue
        missed = sum(y for _, y in skipped)
        saved = len(skipped) * REPAIR_CALLS
        entry = {"threshold": cut, "skipped_frames": len(skipped),
                 "calls_saved": saved, "changes_missed": missed,
                 "skipped_pct": round(100 * len(skipped) / len(pairs), 1),
                 "missed_pct_of_all_changes": round(
                     100 * missed / max(sum(y for _, y in pairs), 1), 1)}
        if missed == 0 and (best is None or entry["calls_saved"] > best["calls_saved"]):
            best = entry
        result.setdefault("sweep", []).append(entry)
    result["best_lossless_skip"] = best
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--stage", choices=("structural", "with_proposal"), default="structural")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = collect(args.archives, args.stage)
    if len(rows) < 20:
        print(f"only {len(rows)} usable targets; not enough to evaluate")
        return
    names = feature_names(args.stage == "with_proposal")
    result = evaluate(rows, names)
    result["stage"] = args.stage
    result["features"] = names
    print(f"stage={args.stage}  targets={result['targets']}  videos={result['videos']}")
    print(f"  frames where repair changed something: {result['change_rate_pct']}%")
    print(f"  leave-one-video-out AUC: {result['held_out_auc']}  (0.5 = no signal)")
    print(f"  per-video counts: {dict(Counter(r['video'] for r in rows))}")
    best = result["best_lossless_skip"]
    if best:
        print(f"  largest lossless skip: {best['skipped_frames']} frames "
              f"({best['skipped_pct']}%), {best['calls_saved']} calls saved, 0 changes missed")
    else:
        print("  no threshold skips frames without missing a change")
    for e in result.get("sweep", []):
        if e["threshold"] in (0.2, 0.3, 0.4, 0.5):
            print(f"    cut={e['threshold']}: skip {e['skipped_pct']}% "
                  f"({e['calls_saved']} calls), miss {e['changes_missed']} changes "
                  f"({e['missed_pct_of_all_changes']}% of all)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
