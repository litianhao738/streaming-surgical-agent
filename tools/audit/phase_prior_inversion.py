"""Can the existing graph prior be inverted into a Phase posterior?

Offline over closed archives, their leave-query-video-out priors and their
already-scored Training ground truth. No API call, no dataset read, no archive
write, no default changed.

The frozen prior stores P(class present | phase) for instrument, verb, target
and IVT. This inverts it with a Bernoulli naive Bayes over the labels a run
actually predicted, giving P(phase | predicted labels), and asks whether the
argmax would beat the phase the pipeline already produced.

Naive Bayes treats classes as conditionally independent given the phase, which
they are not — co-occurring instruments and their relations are correlated, so
the posterior is overconfident. Accuracy here is therefore a screening signal,
never a calibrated probability.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.audit.candidate_ceiling_diagnosis import labels_of, read

HEAD_SETS = {
    "instrument": ("instrument",),
    "ivt": ("ivt",),
    "components": ("instrument", "verb", "target"),
    "all": ("instrument", "verb", "target", "ivt"),
}
EPS = 1e-4
PHASES = range(7)


def phase_buckets(prior, head):
    """Phases the prior actually observed for this head, with their frame counts."""
    table = prior["tasks"][head]["phase"]
    out = {}
    for p in PHASES:
        rows = table.get(str(p))
        if rows and rows[0].get("valid_frames"):
            out[p] = rows
    return out


def posterior(prior, predicted, heads):
    """log P(phase | labels) up to a constant, or None when the prior is unusable."""
    marginal = phase_buckets(prior, "instrument")
    if not marginal:
        return None
    total = sum(rows[0]["valid_frames"] for rows in marginal.values())
    scores = {}
    for p, rows in marginal.items():
        logp = math.log(rows[0]["valid_frames"] / total)
        usable = False
        for head in heads:
            buckets = phase_buckets(prior, head)
            if p not in buckets:
                continue
            usable = True
            observed = set(predicted.get(head) or [])
            for row in buckets[p]:
                rate = min(max(row["rate"], EPS), 1 - EPS)
                logp += math.log(rate if row["id"] in observed else 1 - rate)
        if usable:
            scores[p] = logp
    return scores or None


def evaluate(archives, arm, heads, margin):
    rows, kinds = [], Counter()
    for archive in archives:
        root = Path(archive)
        truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
        priors = {}
        for path in sorted((root / "targets").glob("*/result.json")):
            key = path.parent.name
            row = truth.get(key)
            if not row or not row["mask"].get("phase"):
                continue
            result = read(path)
            predictions = result.get("predictions")
            if not isinstance(predictions, dict) or arm not in predictions:
                continue
            video = row["video_id"]
            if video not in priors:
                p = root / "priors" / f"{video}.json"
                if not p.is_file():
                    break
                priors[video] = read(p)
            prior = priors[video]
            if prior["excluded_video"] != video or video in prior["fit_videos"]:
                raise ValueError("query video leaked into its own prior: " + key)
            predicted = {h: sorted(labels_of(predictions[arm], h)) for h in HEAD_SETS["all"]}
            scores = posterior(prior, predicted, heads)
            if not scores:
                continue
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            top, second = ordered[0], (ordered[1] if len(ordered) > 1 else (None, -math.inf))
            base = next(iter(labels_of(predictions[arm], "phase")), None)
            gt = next(iter(row["gt"]["phase"]), None)
            confident = (top[1] - second[1]) >= margin
            final = top[0] if confident else base
            kinds[("override" if final != base else "keep")] += 1
            if final != base:
                kinds["override_correct" if final == gt else "override_wrong"] += 1
                if base == gt:
                    kinds["broke_a_correct_phase"] += 1
            rows.append({"key": key, "gt": gt, "pipeline": base, "prior_argmax": top[0],
                         "margin": round(top[1] - second[1], 3), "final": final})
    if not rows:
        return None
    n = len(rows)
    return {"arm": arm, "heads": "+".join(heads), "margin": margin, "targets": n,
            "pipeline_acc": round(100 * sum(r["pipeline"] == r["gt"] for r in rows) / n, 2),
            "prior_only_acc": round(100 * sum(r["prior_argmax"] == r["gt"] for r in rows) / n, 2),
            "combined_acc": round(100 * sum(r["final"] == r["gt"] for r in rows) / n, 2),
            "overrides": kinds["override"], "override_correct": kinds["override_correct"],
            "override_wrong": kinds["override_wrong"],
            "broke_a_correct_phase": kinds["broke_a_correct_phase"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--arm", default="h0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = []
    print(f"arm={args.arm}")
    print(f"{'heads':<26}{'margin':>7}{'n':>5}{'pipeline':>10}{'prior only':>12}"
          f"{'combined':>10}{'ovr':>6}{'ok':>4}{'bad':>5}{'broke':>7}")
    for name, heads in HEAD_SETS.items():
        for margin in (0.0, 2.0, 5.0, 10.0):
            r = evaluate(args.archives, args.arm, heads, margin)
            if not r:
                continue
            results.append(r)
            print(f"{name:<26}{margin:>7}{r['targets']:>5}{r['pipeline_acc']:>10}"
                  f"{r['prior_only_acc']:>12}{r['combined_acc']:>10}{r['overrides']:>6}"
                  f"{r['override_correct']:>4}{r['override_wrong']:>5}{r['broke_a_correct_phase']:>7}")
    if args.output and results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "rows"} for r in results]
            + [{"detail_for": results[0]["heads"], "rows": results[0]["rows"]}],
            indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
