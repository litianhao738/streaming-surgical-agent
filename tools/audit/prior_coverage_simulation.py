"""How far would widening the leave-video-out prior candidates raise the ceiling?

Offline over closed archives, their already-scored Training ground truth and the
leave-query-video-out priors those runs already froze. No API call, no dataset
read, no prediction is regenerated, no archive is written. Nothing here selects a
label: it only counts how many correct labels a wider candidate pool would have
put in front of the reviewers, which is the ceiling any reviewer could reach.

The prior is ranked by its own frozen rate, phase-conditioned on the H0 phase
where that bucket is eligible, otherwise global. The query video is excluded from
its own prior by construction; that exclusion is re-checked here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.verification.prior_panel import COMPONENTS
from tools.audit.candidate_ceiling_diagnosis import read, reviewed_labels

HEADS = ("verb", "target")
DEPTHS = (0, 1, 2, 3, 5, 8, 10)


def ranked_prior(prior, head, phase):
    """Eligible IDs by descending frozen rate; phase bucket first when eligible."""
    table = prior["tasks"][head]
    buckets = []
    phase_rows = table["phase"].get(str(phase)) or []
    if any(r["eligible"] for r in phase_rows):
        buckets.append(phase_rows)
    buckets.append(table["global"])
    order, seen = [], set()
    for rows in buckets:
        for row in sorted((r for r in rows if r["eligible"]),
                          key=lambda r: (-r["rate"], r["id"])):
            if row["id"] not in seen:
                seen.add(row["id"])
                order.append(row["id"])
    return order


def simulate(archive):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    priors, rows = {}, []
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        row = truth.get(key)
        if not row:
            continue
        result = read(path)
        video = row["video_id"]
        if video not in priors:
            priors[video] = read(root / "priors" / f"{video}.json")
        prior = priors[video]
        if prior["excluded_video"] != video or video in prior["fit_videos"]:
            raise ValueError("query video leaked into its own prior: " + key)
        phase = (result.get("h0") or {}).get("phase", [None])[0]
        rows.append({"key": key, "truth": row, "pool": reviewed_labels(result),
                     "ranked": {h: ranked_prior(prior, h, phase) for h in HEADS}})
    report = {"archive": root.name, "targets": len(rows), "depths": {}}
    for depth in DEPTHS:
        entry = {}
        for head in HEADS:
            scored = [r for r in rows if r["truth"]["mask"].get(head)]
            total = sum(len(set(r["truth"]["gt"][head])) for r in scored)
            covered = added = 0
            for r in scored:
                gt = set(r["truth"]["gt"][head])
                widened = r["pool"][head] | set(r["ranked"][head][:depth])
                covered += len(gt & widened)
                added += len(widened) - len(r["pool"][head])
            entry[head] = {"gt": total, "in_pool": covered,
                           "ceiling_pct": round(100 * covered / total, 2) if total else None,
                           "extra_candidates": added}
        entry["ivt"] = ivt_ceiling(rows, depth)
        report["depths"][str(depth)] = entry
    return report


def ivt_ceiling(rows, depth):
    """An IVT is reachable only when its own three components are reachable."""
    scored = [r for r in rows if r["truth"]["mask"].get("ivt")]
    total = covered = 0
    for r in scored:
        widened = {h: r["pool"][h] | set(r["ranked"][h][:depth]) for h in HEADS}
        widened["instrument"] = r["pool"]["instrument"]
        for ivt in set(r["truth"]["gt"]["ivt"]):
            total += 1
            parts = COMPONENTS[ivt]
            covered += ivt in r["pool"]["ivt"] or all(
                parts[h] in widened[h] for h in ("instrument", "verb", "target"))
    return {"gt": total, "reachable": covered,
            "ceiling_pct": round(100 * covered / total, 2) if total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = [simulate(a) for a in args.archives]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for r in reports:
        print(f"{r['archive']} ({r['targets']} targets)")
        print(f"{'added/head':>11}{'verb ceil':>11}{'target ceil':>13}{'IVT ceil':>10}{'extra cands':>13}")
        for depth, e in r["depths"].items():
            extra = e["verb"]["extra_candidates"] + e["target"]["extra_candidates"]
            print(f"{depth:>11}{e['verb']['ceiling_pct']:>11}{e['target']['ceiling_pct']:>13}"
                  f"{e['ivt']['ceiling_pct']:>10}{extra:>13}")


if __name__ == "__main__":
    main()
