"""Is the correct anatomy already named in the reviewers' own prose?

Offline over closed archives and their already-scored Training ground truth. No
API call, no dataset read, no archive write, no default changed.

The panel rates a fixed candidate list and a threshold turns those ratings into
labels. This measures a different thing: how often the ground-truth Target or
Verb is *mentioned* anywhere in the reviewers' free-text observations for that
frame, whether or not any candidate carrying that label was admitted.

Mention recall above admitted recall means the information reached the response
and the rate-then-threshold stage discarded it. Mention recall at or below the
pool ceiling means an extraction-style reviewer has nothing extra to offer.
Mentions are lexical and unscoped: a name can appear inside a denial, so this
is an upper bound on what extraction could recover, never a predicted score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from tools.audit.candidate_ceiling_diagnosis import read, reviewed_labels

HEADS = ("target", "verb")
# Surface forms a reviewer plausibly writes; deliberately generous, since the
# question is whether the information is present at all.
ALIASES = {
    "target": {
        0: ["gallbladder", "gall bladder"], 1: ["cystic plate"], 2: ["cystic duct"],
        3: ["cystic artery"], 4: ["cystic pedicle"], 5: ["blood vessel", "vessel"],
        6: ["fluid", "blood pool", "irrigation"], 7: ["abdominal wall", "abdominal cavity", "peritoneal wall"],
        8: ["liver", "liver bed", "hepatic"], 9: ["adhesion"], 10: ["omentum", "omental"],
        11: ["peritoneum", "peritoneal"], 12: ["gut", "bowel", "intestine"],
        13: ["specimen bag", "retrieval bag", "endobag"], 14: []},
    "verb": {
        0: ["grasp", "grip", "hold"], 1: ["retract", "traction", "pull"], 2: ["dissect", "dissection", "separat"],
        3: ["coagulat", "cauter", "electrocaut"], 4: ["clip", "clipping"], 5: ["cut", "divid", "transect"],
        6: ["aspirat", "suction", "suck"], 7: ["irrigat", "flush"], 8: ["pack"], 9: []},
}


def observations(node):
    """Every observation string in a round, from either published review shape."""
    out = []
    reviews = node.get("reviews")
    if isinstance(reviews, dict):
        for value in reviews.values():
            judgments = value.get("judgments") if isinstance(value, dict) else None
            if isinstance(judgments, dict):
                for item in judgments.values():
                    if isinstance(item, dict) and isinstance(item.get("observation"), str):
                        out.append(item["observation"])
    return out


def mentioned(text, head):
    lowered = re.sub(r"[_\-]+", " ", text.lower())
    found = set()
    for label, forms in ALIASES[head].items():
        if any(form in lowered for form in forms):
            found.add(label)
    return found


def analyse(archive, round_name):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    stats = {h: Counter() for h in HEADS}
    per_target = []
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        row = truth.get(key)
        if not row:
            continue
        result = read(path)
        node = result.get(round_name)
        if not isinstance(node, dict):
            continue
        texts = observations(node)
        if not texts:
            continue
        blob = " || ".join(texts)
        pool = reviewed_labels(result)
        prediction = node.get("prediction") or {}
        entry = {"target": key, "observations": len(texts)}
        for head in HEADS:
            if not row["mask"].get(head):
                continue
            gt = set(row["gt"][head] or [])
            if not gt:
                continue
            says = mentioned(blob, head)
            admitted = set(prediction.get(head) or [])
            s = stats[head]
            s["gt"] += len(gt)
            s["mentioned"] += len(gt & says)
            s["admitted"] += len(gt & admitted)
            s["in_pool"] += len(gt & pool[head])
            s["mentioned_not_admitted"] += len((gt & says) - admitted)
            s["mentioned_not_even_in_pool"] += len((gt & says) - pool[head])
            s["wrong_mentions"] += len(says - gt)
            entry[head] = {"gt": sorted(gt), "mentioned": sorted(says),
                           "admitted": sorted(admitted & set(range(len(_TASK_NAMES[head]))))}
        per_target.append(entry)
    return {"archive": root.name, "round": round_name,
            "stats": {h: dict(c) for h, c in stats.items()}, "per_target": per_target}


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--round", default="joint_r1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports, merged = [], {h: Counter() for h in HEADS}
    for archive in args.archives:
        try:
            r = analyse(archive, args.round)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        if not any(r["stats"][h] for h in HEADS):
            continue
        reports.append(r)
        for h in HEADS:
            merged[h].update(r["stats"][h])
    if not reports:
        print("no archive carried observation text for round " + args.round)
        return
    print(f"round={args.round}  archives={len(reports)}")
    print(f"{'head':<9}{'GT':>5}{'admitted':>11}{'in pool':>10}{'mentioned':>12}"
          f"{'ment.not adm.':>15}{'ment.not pooled':>17}{'wrong ment.':>13}")
    for head in HEADS:
        c = merged[head]
        print(f"{head:<9}{c['gt']:>5}"
              f"{str(c['admitted']) + ' (' + str(pct(c['admitted'], c['gt'])) + '%)':>11}"
              f"{str(c['in_pool']) + ' (' + str(pct(c['in_pool'], c['gt'])) + '%)':>10}"
              f"{str(c['mentioned']) + ' (' + str(pct(c['mentioned'], c['gt'])) + '%)':>12}"
              f"{c['mentioned_not_admitted']:>15}{c['mentioned_not_even_in_pool']:>17}"
              f"{c['wrong_mentions']:>13}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"round": args.round, "merged": {h: dict(c) for h, c in merged.items()},
             "archives": reports}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
