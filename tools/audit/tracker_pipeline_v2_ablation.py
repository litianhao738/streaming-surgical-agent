"""Zero-call ablation chain for Tracker pipeline v2 (docs/HANDOFF_TRACKER_PIPELINE_V2_2026-09-14.md).

6,059 first-pass Training rows, default PGP routes (nested outer, retention90) and recorded costs.
F  Tracker fusion: instrument := frozen OOF Tracker set at t (score >= 0.35); drop IVTs whose instrument
   is absent.
V  Single-function verb rules: add verb x when a detected instrument has leave-one-video-out
   P(x in frame | instrument in frame) >= 0.8 (support >= 20), grasp/retract excluded.
S  Causal phase filter: majority phase over this row and past rows within 10 s (declared before the
   first run; 5/20/30 s are sensitivity only). The nested window choice is in tracker_pipeline_v2_probe.py.
Review options on Gate-routed frames: whole (default), interaction-only, phase-only; plus no review and
review every frame. Calls are exact; USD for partial reviews subtracts the skipped branch's per-seat list
prices from the recorded whole-review USD.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.gate import mainline_training as strict
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS

TR = ROOT / "artifacts/training/tracker_clip_v2_oof5_20260906/oof"
GD = ROOT / "artifacts/training/gate/pgp_ambiguity_assessment_20260913_r2"
ROWS = ROOT / "artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json"
OUT = ROOT / "artifacts/research/tracker_pipeline_v2_ablation_20260914.json"
TASKS = strict.TASKS
INTER = ("instrument", "verb", "target", "ivt")
G, R = _TASK_NAMES["verb"].index("grasp"), _TASK_NAMES["verb"].index("retract")
AMB_V = {G, R}
AMB_IVT = {c for c, comp in COMPONENTS.items() if comp["verb"] in AMB_V}

rows = json.loads(ROWS.read_text("utf-8"))
idx_map = json.loads((TR / "index.json").read_text("utf-8"))["video_to_artifact"]
n = len(rows)
vid = np.array([r["video_id"] for r in rows])
fid = np.array([r["frame_id"] for r in rows])
videos = sorted(set(vid))
TRUTH = [strict._truth(r["gt"], r["mask"]) for r in rows]

TRK = {}
for v in videos:
    frames = json.loads((TR / idx_map[v]).read_text("utf-8"))["videos"][v]["frames"]
    for rec in (frames.values() if isinstance(frames, dict) else frames):
        TRK[(v, int(rec["frame_id"]))] = {int(d["instrument_id"]) for d in rec["tracks"] if d["score"] >= 0.35}


def amb_full(r):
    out = {t: set(r["final_labels"][t]) for t in TASKS}
    out["verb"] = (out["verb"] - AMB_V) | (set(r["cheap_labels"]["verb"]) & AMB_V)
    out["ivt"] = (out["ivt"] - AMB_IVT) | (set(r["cheap_labels"]["ivt"]) & AMB_IVT)
    return out


pred = np.load(GD / "predictions.npz", allow_pickle=True)
pos = {s: i for i, s in enumerate(pred["ids"].tolist())}
order = np.array([pos[r["sample_id"]] for r in rows])
ROUTE = pred["qwen_hgb_route"][order].astype(bool)
COSTS = pred["costs"][order]
KC, KJ = pred["depths"][order].T
CHEAP = [{t: set(r["cheap_labels"][t]) for t in TASKS} for r in rows]
AMB = [amb_full(r) for r in rows]

VPRIOR = {}
for held in videos:
    pres, vco = Counter(), Counter()
    for i, r in enumerate(rows):
        if vid[i] != held:
            for k in r["gt"]["instrument"]:
                pres[k] += 1
                vco.update((k, x) for x in r["gt"]["verb"])
    VPRIOR[held] = {(k, x) for (k, x), c in vco.items() if pres[k] >= 20 and c / pres[k] >= 0.8 and x not in AMB_V}

SEQ = defaultdict(list)
for i in np.lexsort((fid, vid)):
    SEQ[vid[i]].append(i)


def build(route, review="whole", fusion=True, verb_rules=True, smooth=None):
    outs = []
    for i in range(n):
        src = {t: CHEAP[i] for t in TASKS}
        if route[i] and review in ("whole", "interaction"):
            src.update({t: AMB[i] for t in INTER})
        if route[i] and review in ("whole", "phase"):
            src["phase"] = AMB[i]
        lab = {t: set(src[t][t]) for t in TASKS}
        ts = TRK[(vid[i], int(fid[i]))]
        if fusion:
            lab["instrument"] = set(ts)
            lab["ivt"] = {c for c in lab["ivt"] if COMPONENTS[c]["instrument"] in ts}
        if verb_rules:
            lab["verb"] |= {x for k, x in VPRIOR[vid[i]] if k in ts}
        outs.append(lab)
    if smooth:
        raw = [next(iter(o["phase"]), None) for o in outs]
        for seq in SEQ.values():
            for j, i in enumerate(seq):
                win = [raw[k] for k in seq[max(0, j - smooth // 25 + 1):j + 1] if fid[i] - fid[k] <= smooth and raw[k] is not None]
                if win:
                    top = Counter(win).most_common()
                    best = [p for p, c in top if c == top[0][1]]
                    outs[i]["phase"] = {raw[i] if raw[i] in best else best[0]}
    return outs


def quality(outputs, subset=None):
    tot = np.zeros((5, 3), dtype=np.int64)
    for i, lab in enumerate(outputs):
        if subset is None or subset[i]:
            for j, t in enumerate(TASKS):
                c = strict._counts(set(lab[t]), TRUTH[i][t])
                tot[j] += (c["tp"], c["fp"], c["fn"])
    tp, fp, fn = tot.T
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones_like(tp, dtype=float), where=den != 0) * 100
    return {"f1": round(float(f1.mean()), 4), "errors": int((fp + fn).sum()),
            **{t: round(float(f1[j]), 2) for j, t in enumerate(TASKS)}}


USD_C = {"qwen": .000263, "gpt": .001698, "gemini": .002721, "grok": 0.0, "deepseek": 0.0}
USD_J = {"qwen": .000390, "gpt": .002854, "gemini": .004462, "grok": 0.0, "deepseek": 0.0}
ORDER = ["qwen", "gpt", "gemini", "grok", "deepseek"]
PH_USD = np.array([.003483 + sum(USD_J[s] for s in ORDER[:k]) for k in KJ])
IN_USD = np.array([sum(USD_C[s] for s in ORDER[1:k]) for k in KC])


def cost(route, review):
    if review == "none":
        return int(COSTS[:, 0, 0, 0].sum()), round(float(COSTS[:, 0, 0, 1].sum()), 2)
    wc, wu = COSTS[:, 1, 1, 0], COSTS[:, 1, 1, 1]
    c = {"whole": wc, "interaction": wc - 1 - KJ, "phase": wc - np.maximum(KC - 1, 0)}[review]
    u = {"whole": wu, "interaction": wu - PH_USD, "phase": wu - IN_USD}[review]
    return int(np.where(route, c, COSTS[:, 1, 0, 0]).sum()), round(float(np.where(route, u, COSTS[:, 1, 0, 1]).sum()), 2)


NONE, ALL = np.zeros(n, bool), np.ones(n, bool)
CONFIGS = {
    "A0 default PGP": (build(ROUTE, fusion=False, verb_rules=False), cost(ROUTE, "whole")),
    "A1 default + S": (build(ROUTE, fusion=False, verb_rules=False, smooth=250), cost(ROUTE, "whole")),
    "A2a default + F": (build(ROUTE, verb_rules=False), cost(ROUTE, "whole")),
    "A2 default + F + V": (build(ROUTE), cost(ROUTE, "whole")),
    "A3 default + F + V + S": (build(ROUTE, smooth=250), cost(ROUTE, "whole")),
    "B1 interaction-only review + F + V + S": (build(ROUTE, review="interaction", smooth=250), cost(ROUTE, "interaction")),
    "B2 phase-only review + F + V + S": (build(ROUTE, review="phase", smooth=250), cost(ROUTE, "phase")),
    "C0 no review (cheap)": (build(NONE, fusion=False, verb_rules=False), cost(NONE, "none")),
    "C0a no review + F": (build(NONE, verb_rules=False), cost(NONE, "none")),
    "C1 no review + F + V + S": (build(NONE, smooth=250), cost(NONE, "none")),
    "D0 review every frame + F + V + S": (build(ALL, smooth=250), cost(ALL, "whole")),
    "A3 with S = 5 s [sensitivity]": (build(ROUTE, smooth=125), cost(ROUTE, "whole")),
    "A3 with S = 20 s [sensitivity]": (build(ROUTE, smooth=500), cost(ROUTE, "whole")),
    "A3 with S = 30 s [sensitivity]": (build(ROUTE, smooth=750), cost(ROUTE, "whole")),
}
report = {"rows": n, "api_calls": 0,
          "verb_rules_per_fold": {v: sorted((_TASK_NAMES["instrument"][k], _TASK_NAMES["verb"][x]) for k, x in s)
                                  for v, s in VPRIOR.items()},
          "configs": {}}
for name, (outs, (calls, usd)) in CONFIGS.items():
    m = {"all": quality(outs), "per_video": {v: quality(outs, vid == v) for v in videos}, "calls": calls, "usd": usd}
    report["configs"][name] = m
    print(f"{name:42s} F1 {m['all']['f1']:.2f} err {m['all']['errors']:6d} calls {calls:6d} usd {usd:7.2f} "
          + " ".join(f"{t}={m['all'][t]}" for t in TASKS))
    print(" " * 43 + "  ".join(f"{v}: {m['per_video'][v]['f1']:.2f}/{m['per_video'][v]['errors']}" for v in videos))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
