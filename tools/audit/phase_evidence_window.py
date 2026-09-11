"""Does the causal window carry the evidence a Phase decision needs?

Offline over Training ground truth and closed archives. No API call, no
prediction is regenerated, no archive is written. Testing is never opened.

For every scored target this measures where the target frame sits inside its own
ground-truth phase segment: how far the nearest boundary is, and whether the
three supplied images even span more than one phase. A phase error on a frame
deep inside a long segment is not a missing-evidence problem; a phase error on a
frame beside a boundary may be. The correlation is descriptive, not causal.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_prior_panel_trial import truth_row
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from tools.audit.candidate_ceiling_diagnosis import labels_of, read

FPS = 25
WINDOW = (-50, -25, 0)


def phase_timeline(adapter, video):
    """Ordered (frame_id, phase) for every mask-valid annotated frame."""
    rows = []
    for resolved in adapter.iter_video(video):
        row = truth_row(resolved)
        if row["mask"].get("phase"):
            rows.append((row["frame_id"], row["gt"]["phase"][0]))
    return sorted(rows)


def boundary_distance(timeline, frame):
    """Frames to the nearest neighbouring-phase sample, and the local segment."""
    index = {f: i for i, (f, _) in enumerate(timeline)}
    if frame not in index:
        return None
    i = index[frame]
    phase = timeline[i][1]
    left = i
    while left > 0 and timeline[left - 1][1] == phase:
        left -= 1
    right = i
    while right + 1 < len(timeline) and timeline[right + 1][1] == phase:
        right += 1
    before = frame - timeline[left][0] if left > 0 else None
    after = timeline[right][0] - frame if right + 1 < len(timeline) else None
    finite = [d for d in (before, after) if d is not None]
    return {"phase": phase, "frames_since_segment_start": before, "frames_to_segment_end": after,
            "nearest_boundary_frames": min(finite) if finite else None,
            "segment_frames": timeline[right][0] - timeline[left][0],
            "window_phases": sorted({p for f, p in timeline
                                     if f in {frame + offset for offset in WINDOW}})}


def analyse(archive, adapter, arm):
    root = Path(archive)
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(root / "scored_truth.json")}
    timelines, rows = {}, []
    for path in sorted((root / "targets").glob("*/result.json")):
        key = path.parent.name
        row = truth.get(key)
        if not row or not row["mask"].get("phase"):
            continue
        predictions = read(path).get("predictions") or {}
        if arm not in predictions:
            continue
        video = row["video_id"]
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        if video not in timelines:
            timelines[video] = phase_timeline(adapter, video)
        geometry = boundary_distance(timelines[video], row["frame_id"])
        if geometry is None:
            continue
        gt = set(row["gt"]["phase"])
        rows.append({"target": key, "correct": bool(labels_of(predictions[arm], "phase") & gt),
                     "predicted": sorted(labels_of(predictions[arm], "phase")),
                     "gt_phase": sorted(gt), **geometry})
    return summarize(root.name, arm, rows)


def summarize(archive, arm, rows):
    def bucket(row):
        near = row["nearest_boundary_frames"]
        if near is None:
            return "segment_edge_of_video"
        return "<=1s_from_boundary" if near <= FPS else "<=4s_from_boundary" if near <= 4 * FPS else "deep_inside"

    groups = {}
    for row in rows:
        b = bucket(row)
        groups.setdefault(b, []).append(row)
    return {"archive": archive, "arm": arm, "targets": len(rows),
            "single_phase_window": sum(len(r["window_phases"]) <= 1 for r in rows),
            "multi_phase_window": sum(len(r["window_phases"]) > 1 for r in rows),
            "by_distance": {b: {"targets": len(v), "correct": sum(r["correct"] for r in v),
                                "accuracy_pct": round(100 * sum(r["correct"] for r in v) / len(v), 2)}
                            for b, v in sorted(groups.items())},
            "confusions": dict(Counter(f"{r['gt_phase'][0]}->{r['predicted'][0] if r['predicted'] else 'none'}"
                                       for r in rows if not r["correct"]).most_common()),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--arm", default="h0")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    reports = [analyse(a, adapter, args.arm) for a in args.archives]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for r in reports:
        print(f"{r['archive']} [{r['arm']}] n={r['targets']} "
              f"single-phase-window={r['single_phase_window']} multi={r['multi_phase_window']}")
        for b, v in r["by_distance"].items():
            print(f"   {b:<24} {v['correct']}/{v['targets']}  {v['accuracy_pct']}%")
        print(f"   confusions gt->pred: {r['confusions']}")


if __name__ == "__main__":
    main()
