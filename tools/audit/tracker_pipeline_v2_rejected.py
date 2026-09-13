"""Zero-call record of Tracker pipeline v2 alternatives that were tested and not adopted.

Same rows, default PGP routes and costs as tracker_pipeline_v2_ablation.py. Rule used when exploring a
component: pooled F1 up, errors down, and every video not worse on both.
insertion   output fusion (instrument := tracker set, drop IVTs whose instrument is absent) on each base;
            tracker-disagreement review triggers; prior-based IVT add for tracker instruments without an IVT
            (leave-one-video-out top IVT per (instrument, phase), P >= 0.5); Gate change-label rate before and
            after fusion; reviewer instrument edits vs Tracker.
priors      instrument -> verb / target frame co-occurrence priors (leave-one-video-out, P >= 0.8; 0.9 is a
            sensitivity check), with and without grasp/retract, on default + fusion.
components  on REF = default + fusion + single-function verb rules: Tracker temporal smoothing over t, t-1 s and
            t-2 s (majority / fill / confirm), instrument -> phase prior (P >= 0.8), 10 s phase filter.
GT only for evaluation and leave-one-video-out priors.
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
OUT = ROOT / "artifacts/research/tracker_pipeline_v2_rejected_20260914.json"
TASKS = strict.TASKS
NAMES = _TASK_NAMES
G, R = NAMES["verb"].index("grasp"), NAMES["verb"].index("retract")
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


def tset(i, mode="none"):
    v, f = vid[i], int(fid[i])
    cur, p1, p2 = TRK[(v, f)], TRK.get((v, f - 25), set()), TRK.get((v, f - 50), set())
    if mode == "majority":
        return {k for k in cur | p1 | p2 if (k in cur) + (k in p1) + (k in p2) >= 2}
    if mode == "fill":
        return cur | (p1 & p2)
    if mode == "confirm":
        return cur & (p1 | p2)
    return set(cur)


def amb_full(r):
    out = {t: set(r["final_labels"][t]) for t in TASKS}
    out["verb"] = (out["verb"] - AMB_V) | (set(r["cheap_labels"]["verb"]) & AMB_V)
    out["ivt"] = (out["ivt"] - AMB_IVT) | (set(r["cheap_labels"]["ivt"]) & AMB_IVT)
    return out


def copy(lab):
    return {t: set(lab[t]) for t in TASKS}


def fuse(lab, ts):
    out = copy(lab)
    out["instrument"] = set(ts)
    out["ivt"] = {c for c in out["ivt"] if COMPONENTS[c]["instrument"] in ts}
    return out


pred = np.load(GD / "predictions.npz", allow_pickle=True)
pos = {s: i for i, s in enumerate(pred["ids"].tolist())}
order = np.array([pos[r["sample_id"]] for r in rows])
ROUTE = pred["qwen_hgb_route"][order].astype(bool)
COSTS = pred["costs"][order]
H0 = [{t: set(r["h0_labels"][t]) for t in TASKS} for r in rows]
CHEAP = [{t: set(r["cheap_labels"][t]) for t in TASKS} for r in rows]
AMB = [amb_full(r) for r in rows]
DEFAULT = [AMB[i] if ROUTE[i] else CHEAP[i] for i in range(n)]
NONE, ALL = np.zeros(n, bool), np.ones(n, bool)
SEQ = defaultdict(list)
for i in np.lexsort((fid, vid)):
    SEQ[vid[i]].append(i)


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


def report(outputs, route=None, probe=1):
    out = {"all": quality(outputs), "per_video": {v: quality(outputs, vid == v) for v in videos}}
    if route is not None:
        c = COSTS[np.arange(n), probe, route.astype(int)].sum(axis=0)
        out.update(calls=int(c[0]), usd=round(float(c[1]), 2), reviewed=int(route.sum()))
    return out


def adopt(m, ref):
    return bool(m["all"]["f1"] > ref["all"]["f1"] and m["all"]["errors"] < ref["all"]["errors"]
                and all(m["per_video"][v]["f1"] >= ref["per_video"][v]["f1"]
                        and m["per_video"][v]["errors"] <= ref["per_video"][v]["errors"] for v in videos))


CO, IVTP = {}, {}
for held in videos:
    pres, tot = Counter(), Counter()
    co = {"verb": Counter(), "target": Counter(), "phase": Counter()}
    cnt = defaultdict(Counter)
    for i, r in enumerate(rows):
        if vid[i] == held:
            continue
        ph = next(iter(r["gt"]["phase"]), -1)
        if r["mask"]["verb"] and r["mask"]["target"]:
            for k in r["gt"]["instrument"]:
                pres[k] += 1
                for head, counter in co.items():
                    counter.update((k, x) for x in r["gt"][head])
        if r["mask"]["ivt"]:
            for k in r["gt"]["instrument"]:
                tot[(k, ph)] += 1
                for c in r["gt"]["ivt"]:
                    if COMPONENTS[c]["instrument"] == k:
                        cnt[(k, ph)][c] += 1
    CO[held] = {head: {key: c / pres[key[0]] for key, c in co[head].items() if pres[key[0]] >= 20} for head in co}
    IVTP[held] = {key: (cnt[key].most_common(1)[0][0], cnt[key].most_common(1)[0][1] / tot[key])
                  for key in cnt if tot[key] >= 20}


def add_ivt(outputs):
    new = []
    for i, lab in enumerate(outputs):
        lab = copy(lab)
        covered = {COMPONENTS[x]["instrument"] for x in lab["ivt"]}
        ph = next(iter(lab["phase"]), -1)
        for inst in lab["instrument"] - covered:
            cand = IVTP[vid[i]].get((inst, ph))
            if cand is not None and cand[1] >= 0.5:
                c = cand[0]
                lab["ivt"].add(c)
                lab["verb"].add(COMPONENTS[c]["verb"])
                lab["target"].add(COMPONENTS[c]["target"])
        new.append(lab)
    return new


def add_prior(outputs, heads, th, skip_amb):
    new = []
    for i, lab in enumerate(outputs):
        lab = copy(lab)
        for head in heads:
            for (k, x), p in CO[vid[i]][head].items():
                if k in lab["instrument"] and p >= th and not (skip_amb and head == "verb" and x in AMB_V):
                    lab[head].add(x)
        new.append(lab)
    return new


# ---------------- insertion ----------------
TS = [tset(i) for i in range(n)]
FD = [fuse(DEFAULT[i], TS[i]) for i in range(n)]
FCH = [fuse(CHEAP[i], TS[i]) for i in range(n)]
FA = [fuse(AMB[i], TS[i]) for i in range(n)]
ins = {"h0": report(H0), "h0+fusion": report([fuse(H0[i], TS[i]) for i in range(n)]),
       "cheap": report(CHEAP, NONE, 0), "cheap+fusion": report(FCH, NONE, 0),
       "default": report(DEFAULT, ROUTE), "default+fusion": report(FD, ROUTE),
       "review_all": report(AMB, ALL, 0), "review_all+fusion": report(FA, ALL, 0)}
disagree = np.array([TS[i] != CHEAP[i]["instrument"] for i in range(n)])
uncovered = np.array([bool(TS[i] - {COMPONENTS[x]["instrument"] for x in CHEAP[i]["ivt"]}) for i in range(n)])
for name, extra in (("disagree", disagree), ("uncovered", uncovered)):
    rt = ROUTE | extra
    ins[f"default_or_{name}_trigger+fusion"] = report([fuse(AMB[i] if rt[i] else CHEAP[i], TS[i]) for i in range(n)], rt)
    ins[f"only_{name}_trigger+fusion"] = report([fuse(AMB[i] if extra[i] else CHEAP[i], TS[i]) for i in range(n)], extra, 0)
ins["default+fusion+ivt_prior_add"] = report(add_ivt(FD), ROUTE)
ins["cheap+fusion+ivt_prior_add"] = report(add_ivt(FCH), NONE, 0)
ref_ins = ins["default+fusion"]
ins["adopt_vs_default+fusion"] = {k: adopt(m, ref_ins) for k, m in ins.items()
                                   if isinstance(m, dict) and "all" in m and k.startswith(("default_or", "default+fusion+"))}
ins["trigger_rates"] = {"disagree": round(float(disagree.mean()), 4), "uncovered": round(float(uncovered.mean()), 4),
                        "default_route": round(float(ROUTE.mean()), 4)}
ins["gate_change_label_rate"] = {
    "before_fusion": round(float(np.mean([any(AMB[i][t] != CHEAP[i][t] for t in TASKS) for i in range(n)])), 4),
    "after_fusion": round(float(np.mean([any(FA[i][t] != FCH[i][t] for t in TASKS) for i in range(n)])), 4)}
edited = [i for i in range(n) if AMB[i]["instrument"] != CHEAP[i]["instrument"]]
ins["reviewer_instrument_edits"] = {
    "frames": len(edited),
    "errors_cheap": int(sum(len(CHEAP[i]["instrument"] ^ TRUTH[i]["instrument"]) for i in edited)),
    "errors_reviewer": int(sum(len(AMB[i]["instrument"] ^ TRUTH[i]["instrument"]) for i in edited)),
    "errors_tracker": int(sum(len(TS[i] ^ TRUTH[i]["instrument"]) for i in edited))}

# ---------------- priors ----------------
pri = {"rules_p_ge_0.8": sorted({(head, NAMES["instrument"][k], NAMES[head][x]) for held in videos
                                 for head in ("verb", "target") for (k, x), p in CO[held][head].items() if p >= 0.8}),
       "none": report(FD, ROUTE),
       "verb_all": report(add_prior(FD, ("verb",), 0.8, False), ROUTE),
       "verb_nonamb": report(add_prior(FD, ("verb",), 0.8, True), ROUTE),
       "target": report(add_prior(FD, ("target",), 0.8, False), ROUTE),
       "verb_nonamb+target": report(add_prior(FD, ("verb", "target"), 0.8, True), ROUTE),
       "verb_all+target": report(add_prior(FD, ("verb", "target"), 0.8, False), ROUTE),
       "sensitivity_th0.9_verb_nonamb+target": report(add_prior(FD, ("verb", "target"), 0.9, True), ROUTE)}
pri["adopt_vs_none"] = {k: adopt(m, pri["none"]) for k, m in pri.items() if isinstance(m, dict) and "all" in m}


# ---------------- components on REF ----------------
def build_ref(smooth="none", phase=()):
    outs = []
    for i in range(n):
        ts = tset(i, smooth)
        lab = fuse(DEFAULT[i], ts)
        lab["verb"] |= {x for (k, x), p in CO[vid[i]]["verb"].items() if k in ts and p >= 0.8 and x not in AMB_V}
        if "inst" in phase:
            hits = [(p, x) for (k, x), p in CO[vid[i]]["phase"].items() if k in ts and p >= 0.8]
            if hits:
                lab["phase"] = {max(hits)[1]}
        outs.append(lab)
    if "smooth" in phase:
        raw = [next(iter(o["phase"]), None) for o in outs]
        for seq in SEQ.values():
            for j, i in enumerate(seq):
                win = [raw[k] for k in seq[max(0, j - 9):j + 1] if fid[i] - fid[k] <= 250 and raw[k] is not None]
                if win:
                    top = Counter(win).most_common()
                    best = [p for p, c in top if c == top[0][1]]
                    outs[i]["phase"] = {raw[i] if raw[i] in best else best[0]}
    return outs


comp = {"REF": report(build_ref(), ROUTE),
        "tracker_smoothing_majority": report(build_ref(smooth="majority"), ROUTE),
        "tracker_smoothing_fill": report(build_ref(smooth="fill"), ROUTE),
        "tracker_smoothing_confirm": report(build_ref(smooth="confirm"), ROUTE),
        "instrument_to_phase_prior": report(build_ref(phase=("inst",)), ROUTE),
        "phase_filter_10s": report(build_ref(phase=("smooth",)), ROUTE),
        "instrument_to_phase_prior+phase_filter_10s": report(build_ref(phase=("inst", "smooth")), ROUTE)}
comp["phase_rules_p_ge_0.8"] = {v: sorted((NAMES["instrument"][k], NAMES["phase"][x], round(p, 2))
                                          for (k, x), p in CO[v]["phase"].items() if p >= 0.8) for v in videos}
comp["adopt_vs_REF"] = {k: adopt(m, comp["REF"]) for k, m in comp.items() if isinstance(m, dict) and "all" in m}

result = {"rows": n, "api_calls": 0, "insertion": ins, "priors": pri, "components": comp}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
for section in ("insertion", "priors", "components"):
    print(f"== {section}")
    for k, m in result[section].items():
        if isinstance(m, dict) and "all" in m:
            extra = {x: m[x] for x in ("calls", "usd", "reviewed") if x in m}
            print(f"  {k:44s} F1 {m['all']['f1']:.2f} err {m['all']['errors']} {extra}")
        else:
            print(f"  {k:44s} {m}")
