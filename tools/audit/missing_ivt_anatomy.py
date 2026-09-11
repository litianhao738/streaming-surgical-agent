"""Why a correct IVT never reached a reviewer: assembly gap or perception gap.

Offline over closed archives and their already-scored Training ground truth. No
API call, no dataset read, no archive write. Every GT IVT that never entered a
candidate pool is classified by whether its own instrument, verb and target were
separately available in that target's pools:

  ASSEMBLY_GAP    all three components were in the pools, the triplet was not
  PARTIAL_GAP     some component was present, at least one was missing
  PERCEPTION_GAP  no component was present

An ASSEMBLY_GAP is reachable without better vision, by proposing the triplet a
model already has the parts for. The other two are not.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS
from tools.audit.candidate_ceiling_diagnosis import read, reviewed_labels

NULL_IVT_FROM = 94


def name_of(ivt):
    parts = COMPONENTS[ivt]
    return "/".join(_TASK_NAMES[t][parts[t]] for t in ("instrument", "verb", "target"))


def anatomy(archive):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    kinds, rows, null_missing = Counter(), [], 0
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        if key not in truth or not truth[key]["mask"].get("ivt"):
            continue
        pool = reviewed_labels(read(path))
        for ivt in sorted(set(truth[key]["gt"]["ivt"]) - pool["ivt"]):
            parts = COMPONENTS[ivt]
            present = {t: parts[t] in pool[t] for t in ("instrument", "verb", "target")}
            kind = ("ASSEMBLY_GAP" if all(present.values())
                    else "PERCEPTION_GAP" if not any(present.values()) else "PARTIAL_GAP")
            kinds[kind] += 1
            null_missing += ivt >= NULL_IVT_FROM
            rows.append({"target": key, "ivt": ivt, "name": name_of(ivt), "kind": kind,
                         "components_in_pool": present})
    total = sum(kinds.values())
    return {"archive": root.name, "never_proposed_ivt": total,
            "null_relation_share": null_missing,
            "kinds": {k: {"count": kinds[k], "share_pct": round(100 * kinds[k] / total, 2) if total else None}
                      for k in ("ASSEMBLY_GAP", "PARTIAL_GAP", "PERCEPTION_GAP")},
            "missing_component_counts": dict(Counter(
                t for r in rows for t, ok in r["components_in_pool"].items() if not ok).most_common()),
            "examples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = [anatomy(a) for a in args.archives]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for r in reports:
        print(f"{r['archive']}: {r['never_proposed_ivt']} never-proposed IVT, "
              f"{r['null_relation_share']} null-relation, "
              + ", ".join(f"{k}={v['count']} ({v['share_pct']}%)" for k, v in r["kinds"].items())
              + f" | missing components: {r['missing_component_counts']}")


if __name__ == "__main__":
    main()
