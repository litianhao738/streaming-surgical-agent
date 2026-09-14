"""Zero-call check for scheme-4 output modules v2.2 (docs/SCHEME4_OUTPUT_MODULES_V22_2026-09-14.md).

v2.1 replay must equal the original scheme-4 replay frame by frame; v2.2 must keep routing and calls,
and is compared with v2.1 overall, per head, per video, and by module activity.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
TASKS = ("instrument", "verb", "target", "ivt", "phase")
OLD = ROOT / "artifacts/preflight/tracker_scheme4_default_all_20260914_r1/predictions.jsonl"
V21 = ROOT / "artifacts/preflight/scheme4_output_v21_parity_20260914/predictions.jsonl"
V22 = ROOT / "artifacts/preflight/scheme4_output_v22_all_20260914/predictions.jsonl"
ROWS = ROOT / "artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json"


def load(path):
    return {d["key"]: d for d in (json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip())}


rows = {r["sample_id"]: r for r in json.loads(ROWS.read_text("utf-8"))}
old, v21, v22 = load(OLD), load(V21), load(V22)
if not set(old) == set(v21) == set(v22):
    raise ValueError("replay inventories differ")
keys = sorted(old)
videos = sorted({rows[k]["video_id"] for k in keys})


def same(a, b):
    return all(sorted(a[t]) == sorted(b[t]) for t in TASKS)


def score(preds, video=None):
    tot = np.zeros((5, 3), int)
    for k in keys:
        if video and rows[k]["video_id"] != video:
            continue
        for j, t in enumerate(TASKS):
            a, b = set(preds[k]["prediction"][t]), set(rows[k]["gt"][t])
            tot[j] += (len(a & b), len(a - b), len(b - a))
    tp, fp, fn = tot.T
    f1 = np.divide(200 * tp, 2 * tp + fp + fn, out=np.full(5, 100.0), where=(2 * tp + fp + fn) != 0)
    return round(float(f1.mean()), 3), int((fp + fn).sum()), {t: round(float(f1[j]), 2) for j, t in enumerate(TASKS)}


def errors(pred, k):
    return sum(len(set(pred[t]) ^ set(rows[k]["gt"][t])) for t in TASKS)


parity = sum(not same(old[k]["prediction"], v21[k]["prediction"]) for k in keys)
routing = sum((old[k]["gate_action"], old[k]["logical_calls"]) != (v22[k]["gate_action"], v22[k]["logical_calls"]) for k in keys)
print(f"v2.1 vs original scheme-4 prediction differences: {parity}; v2.2 routing/call differences: {routing}")
for tag, preds in (("v2.1", v21), ("v2.2", v22)):
    print(tag, score(preds), " ".join(f"{v}:{score(preds, v)[:2]}" for v in videos))
a, b = score(v22), score(v21)
ok = a[0] > b[0] and a[1] < b[1] and all(
    score(v22, v)[0] >= score(v21, v)[0] and score(v22, v)[1] <= score(v21, v)[1] for v in videos)
print(f"v2.2 - v2.1: F1 {a[0] - b[0]:+.3f}, errors {a[1] - b[1]:+d}, every video not worse on both: {ok}")
activity = Counter()
for k in keys:
    log = v22[k]["output_modules"]
    for name in ("excluded_tracker_classes", "removed_null_ivts", "removed_null_verb", "removed_null_target"):
        activity[name] += bool(log.get(name))
print("v2.2 module activity (frames):", dict(activity))
changed = [k for k in keys if not same(v21[k]["prediction"], v22[k]["prediction"])]
better = sum(errors(v22[k]["prediction"], k) < errors(v21[k]["prediction"], k) for k in changed)
worse = sum(errors(v22[k]["prediction"], k) > errors(v21[k]["prediction"], k) for k in changed)
print(f"frames changed: {len(changed)}; better {better}, worse {worse}")
