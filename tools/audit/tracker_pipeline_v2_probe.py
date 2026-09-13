"""Zero-call evidence for Tracker pipeline v2 (docs/HANDOFF_TRACKER_PIPELINE_V2_2026-09-14.md).

Plan: PGP Gate v2 + interaction-only review + zero-cost output modules, on the 6,059 first-pass
Training rows and the default PGP cost table.

T  Tracker fusion: instrument := frozen OOF Tracker set (score >= 0.35); drop IVTs whose instrument is
   absent. Plus single-function verb rules: add verb x when a detected instrument has
   leave-one-video-out P(x in frame | instrument in frame) >= 0.8 (support >= 20), grasp/retract excluded.
S  Causal phase filter on the cheap phase: majority over this row and past rows within the window
   (tie keeps the current phase). Window chosen by nested leave-one-video-out from the grid declared
   before the first run (raw, 5/10/20/30/60 s, online HMM). The grid extended to 180 s is a sensitivity
   check only.
Gate v2: GT-free label = interaction review changes the T-fused interaction heads. HGB on the default
   42 features (Qwen probe) or on the 43 post-cheap features (probe-free costs). Nested
   leave-one-video-out threshold: cheapest verify fraction that keeps >= 90% of the review-all F1 gain
   over no review, with errors <= no review and every inner video not worse than no review.
Cost-fidelity: learned Gate vs random within-video selection at equal verify fractions.

Calls are exact (formula checked against recorded costs). USD of interaction-only review subtracts the
Phase branch's per-seat list prices from the recorded whole-review USD. GT is used only for evaluation,
leave-one-video-out priors and selection on fitting videos.
"""
import json
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.gate import mainline_training as strict
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS

TR = ROOT / "artifacts/training/tracker_clip_v2_oof5_20260906/oof"
GD = ROOT / "artifacts/training/gate/pgp_ambiguity_assessment_20260913_r2"
ROWS = ROOT / "artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json"
OUT = ROOT / "artifacts/research/tracker_pipeline_v2_probe_20260914.json"
TASKS = strict.TASKS
INTER = ("instrument", "verb", "target", "ivt")
G, R = _TASK_NAMES["verb"].index("grasp"), _TASK_NAMES["verb"].index("retract")
AMB_V = {G, R}
AMB_IVT = {c for c, comp in COMPONENTS.items() if comp["verb"] in AMB_V}
DECLARED_GRID = (5, 10, 20, 30, 60)
EXTENDED_GRID = (5, 10, 20, 30, 60, 90, 120, 180)
K = 7

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

VPRIOR = {}
for held in videos:
    pres, vco = Counter(), Counter()
    for i, r in enumerate(rows):
        if vid[i] != held:
            for k in r["gt"]["instrument"]:
                pres[k] += 1
                vco.update((k, x) for x in r["gt"]["verb"])
    VPRIOR[held] = {(k, x) for (k, x), c in vco.items() if pres[k] >= 20 and c / pres[k] >= 0.8 and x not in AMB_V}


def amb_full(r):
    out = {t: set(r["final_labels"][t]) for t in TASKS}
    out["verb"] = (out["verb"] - AMB_V) | (set(r["cheap_labels"]["verb"]) & AMB_V)
    out["ivt"] = (out["ivt"] - AMB_IVT) | (set(r["cheap_labels"]["ivt"]) & AMB_IVT)
    return out


def apply_t(lab, i):
    out = {t: set(lab[t]) for t in INTER}
    ts = TRK[(vid[i], int(fid[i]))]
    out["instrument"] = set(ts)
    out["ivt"] = {c for c in out["ivt"] if COMPONENTS[c]["instrument"] in ts}
    out["verb"] |= {x for k, x in VPRIOR[vid[i]] if k in ts}
    return out


FC = [apply_t(r["cheap_labels"], i) for i, r in enumerate(rows)]
FR = [apply_t(amb_full(r), i) for i, r in enumerate(rows)]
y = np.array([any(FC[i][t] != FR[i][t] for t in INTER) for i in range(n)], int)


def inter_counts(outputs):
    m = np.zeros((n, 4, 3), dtype=np.int64)
    for i, lab in enumerate(outputs):
        for j, t in enumerate(INTER):
            c = strict._counts(set(lab[t]), TRUTH[i][t])
            m[i, j] = (c["tp"], c["fp"], c["fn"])
    return m


CI_C, CI_R = inter_counts(FC), inter_counts(FR)

# ---------------- phase filters ----------------
SEQ = defaultdict(list)
for i in np.lexsort((fid, vid)):
    SEQ[vid[i]].append(i)
OBS = [next(iter(r["cheap_labels"]["phase"]), None) for r in rows]
GTP = [next(iter(r["gt"]["phase"]), None) if r["mask"]["phase"] else None for r in rows]


def window(sec):
    out = list(OBS)
    for seq in SEQ.values():
        for j, i in enumerate(seq):
            win = [OBS[k] for k in seq[max(0, j - sec + 1):j + 1] if fid[i] - fid[k] <= 25 * sec and OBS[k] is not None]
            if win:
                top = Counter(win).most_common()
                best = [p for p, c in top if c == top[0][1]]
                out[i] = OBS[i] if OBS[i] in best else best[0]
    return out


def hmm_params(fit):
    trans, emit, pi = np.ones((K, K)), np.ones((K, K)), np.ones(K)
    for v in fit:
        seq = SEQ[v]
        if GTP[seq[0]] is not None:
            pi[GTP[seq[0]]] += 1
        for a, b in pairwise(seq):
            if GTP[a] is not None and GTP[b] is not None and fid[b] - fid[a] == 25:
                trans[GTP[a], GTP[b]] += 1
        for i in seq:
            if GTP[i] is not None and OBS[i] is not None:
                emit[GTP[i], OBS[i]] += 1
    return trans / trans.sum(1, keepdims=True), emit / emit.sum(1, keepdims=True), pi / pi.sum()


def hmm_filter(targets, params):
    trans, emit, pi = params
    powers = [np.eye(K), trans]
    for _ in range(60):
        powers.append(powers[-1] @ trans)
    out = {}
    for v in targets:
        alpha, prev = None, None
        for i in SEQ[v]:
            e = emit[:, OBS[i]] if OBS[i] is not None else np.ones(K)
            if alpha is None:
                alpha = pi * e
            else:
                steps = int(min(60, max(1, round((fid[i] - fid[prev]) / 25))))
                alpha = (alpha @ powers[steps]) * e
            alpha /= alpha.sum()
            out[i] = int(alpha.argmax())
            prev = i
    return out


def phase_counts(pred):
    m = np.zeros((n, 1, 3), dtype=np.int64)
    for i, p in enumerate(pred):
        c = strict._counts({p} if p is not None else set(), TRUTH[i]["phase"])
        m[i, 0] = (c["tp"], c["fp"], c["fn"])
    return m


def f1_of(c):
    tp, fp, fn = c.sum(axis=0).T
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones_like(tp, dtype=float), where=den != 0) * 100
    return round(float(f1.mean()), 4), int((fp + fn).sum())


def hmm_lovo_counts(pool):
    pred = list(OBS)
    for u in pool:
        for i, p in hmm_filter([u], hmm_params([w for w in pool if w != u])).items():
            pred[i] = p
    return phase_counts(pred)


def phase_selection(grid):
    fixed = {"raw": phase_counts(OBS), **{f"w{s}": phase_counts(window(s)) for s in grid}}
    nested, choice = np.zeros((n, 1, 3), dtype=np.int64), {}
    for v in videos:
        fit = [w for w in videos if w != v]
        fm = np.isin(vid, fit)
        scores = {m: f1_of(c[fm])[0] for m, c in fixed.items()}
        scores["hmm"] = f1_of(hmm_lovo_counts(fit)[fm])[0]
        best = max(scores, key=scores.get)
        choice[v] = best
        held = vid == v
        if best == "hmm":
            pred = hmm_filter([v], hmm_params(fit))
            nested[held] = phase_counts([pred.get(i, OBS[i]) for i in range(n)])[held]
        else:
            nested[held] = fixed[best][held]
    lovo = {m: f1_of(c)[0] for m, c in {**fixed, "hmm": hmm_lovo_counts(videos)}.items()}
    return {"lovo_phase_f1": lovo, "nested_choice": choice, "nested_phase_f1": f1_of(nested)[0]}, nested, fixed


phase_main, NESTED_PHASE, FIXED_PHASE = phase_selection(DECLARED_GRID)
phase_ext, _, _ = phase_selection(EXTENDED_GRID)

# ---------------- Gate v2 for interaction-only review ----------------
pred = np.load(GD / "predictions.npz", allow_pickle=True)
pos = {s: i for i, s in enumerate(pred["ids"].tolist())}
order = np.array([pos[r["sample_id"]] for r in rows])
OLD_ROUTE = pred["qwen_hgb_route"][order].astype(bool)
COSTS = pred["costs"][order]
KC, KJ = pred["depths"][order].T
X42 = pred["X"][order].astype(float)
XB = np.array([[r["features_postcheap"][k] for k in sorted(r["features_postcheap"])] for r in rows], float)
USD_J = {"qwen": .000390, "gpt": .002854, "gemini": .004462, "grok": 0.0, "deepseek": 0.0}
ORDER = ["qwen", "gpt", "gemini", "grok", "deepseek"]
PH_USD = np.array([.003483 + sum(USD_J[s] for s in ORDER[:k]) for k in KJ])
CALL_CHECK = {"probe_whole": int((COSTS[:, 1, 1, 0] != 4 + np.maximum(KC - 1, 0) + KJ).sum()),
              "probe_free_whole": int((COSTS[:, 0, 1, 0] != 3 + KC + KJ).sum()),
              "probe_skip_not_3": int((COSTS[:, 1, 0, 0] != 3).sum())}
CALL = {1: (COSTS[:, 1, 0, 0], 3 + np.maximum(KC - 1, 0)), 0: (COSTS[:, 0, 0, 0], COSTS[:, 0, 1, 0] - 1 - KJ)}
USD = {1: (COSTS[:, 1, 0, 1], COSTS[:, 1, 1, 1] - PH_USD), 0: (COSTS[:, 0, 0, 1], COSTS[:, 0, 1, 1] - PH_USD)}
W10 = FIXED_PHASE["w10"]


def evaluate(route, probe, ph=W10, idx=None):
    idx = np.arange(n) if idx is None else idx
    a = route[idx]
    c = np.concatenate([np.where(a[:, None, None], CI_R[idx], CI_C[idx]), ph[idx]], axis=1)
    base = np.concatenate([CI_C[idx], ph[idx]], axis=1)
    per = {v: f1_of(c[vid[idx] == v]) for v in sorted(set(vid[idx]))}
    ok = all(per[v][0] >= f1_of(base[vid[idx] == v])[0] and per[v][1] <= f1_of(base[vid[idx] == v])[1] for v in per)
    calls = np.where(a, CALL[probe][1][idx], CALL[probe][0][idx]).sum()
    usd = np.where(a, USD[probe][1][idx], USD[probe][0][idx]).sum()
    missed = int(sum(y[i] and not a_ for i, a_ in zip(idx, a, strict=True)))
    changes = int(y[idx].sum())
    return {"f1": f1_of(c)[0], "errors": f1_of(c)[1], "per_video": per, "every_video_not_worse_than_no_review": ok,
            "calls": int(calls), "usd": round(float(usd), 2), "reviewed": int(a.sum()),
            "fidelity": round(1 - missed / len(idx), 4), "changes_caught": round(1 - missed / max(1, changes), 4)}


def fit_predict(X, fit, target):
    p = y[fit].mean()
    w = np.where(y[fit] == 1, 0.5 / p, 0.5 / (1 - p))
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=3407)
    return m.fit(X[fit], y[fit], sample_weight=w).predict_proba(X[target])[:, 1]


def nested(X, probe):
    route, folds = np.zeros(n, bool), {}
    for v in videos:
        inner = np.where(vid != v)[0]
        s = np.zeros(n)
        for u in videos:
            if u != v:
                s[vid == u] = fit_predict(X, np.where((vid != v) & (vid != u))[0], np.where(vid == u)[0])
        none, allr = evaluate(np.zeros(n, bool), 0, idx=inner), evaluate(np.ones(n, bool), probe, idx=inner)
        chosen = None
        for frac in np.arange(0.05, 1.0001, 0.05):
            thr = float(np.quantile(s[inner], 1 - frac))
            cand = np.zeros(n, bool)
            cand[inner] = s[inner] >= thr
            e = evaluate(cand, probe, idx=inner)
            if (e["every_video_not_worse_than_no_review"] and cand[inner].any() and e["errors"] <= none["errors"]
                    and e["f1"] - none["f1"] >= 0.9 * max(0.0, allr["f1"] - none["f1"])):
                chosen = (round(float(frac), 2), thr)
                break
        held = np.where(vid == v)[0]
        if chosen is None:
            folds[v] = "INFEASIBLE_NO_REVIEW"
            continue
        route[held] = fit_predict(X, inner, held) >= chosen[1]
        folds[v] = {"inner_fraction": chosen[0], "outer_fraction": round(float(route[held].mean()), 3)}
    return route, folds


r42, f42 = nested(X42, 1)
rb, fb = nested(XB, 0)
NONE, ALL = np.zeros(n, bool), np.ones(n, bool)
ROUTES = {"no_review": (NONE, 0), "review_every_frame": (ALL, 0), "old_default_gate_routes": (OLD_ROUTE, 1),
          "gate_v2_qwen_probe_42": (r42, 1), "gate_v2_probe_free_43": (rb, 0)}
gate = {}
for ph_name, ph in (("phase_w10", W10), ("phase_nested_declared_grid", NESTED_PHASE)):
    none, allr = evaluate(NONE, 0, ph=ph), evaluate(ALL, 0, ph=ph)
    block = {}
    for name, (rt, probe) in ROUTES.items():
        e = evaluate(rt, probe, ph=ph)
        e["f1_gain_retention"] = round((e["f1"] - none["f1"]) / (allr["f1"] - none["f1"]), 4)
        e["error_reduction_retention"] = round((none["errors"] - e["errors"]) / (none["errors"] - allr["errors"]), 4)
        block[name] = e
    gate[ph_name] = block

s42, sb = np.zeros(n), np.zeros(n)
for v in videos:
    s42[vid == v] = fit_predict(X42, np.where(vid != v)[0], np.where(vid == v)[0])
    sb[vid == v] = fit_predict(XB, np.where(vid != v)[0], np.where(vid == v)[0])
rng = np.random.default_rng(3407)
curve = []
for frac in (0.1, 0.2, 0.3, 0.5, 0.7):
    row = {"verify_fraction": frac}
    for name, s, probe in (("learned_qwen_probe", s42, 1), ("learned_probe_free", sb, 0)):
        rt = np.zeros(n, bool)
        for v in videos:
            ii = np.where(vid == v)[0]
            rt[ii[np.argsort(-s[ii])[:round(frac * len(ii))]]] = True
        e = evaluate(rt, probe)
        row[name] = {k: e[k] for k in ("fidelity", "changes_caught", "f1", "errors", "calls", "usd")}
    draws = []
    for _ in range(50):
        rt = np.zeros(n, bool)
        for v in videos:
            ii = np.where(vid == v)[0]
            rt[rng.choice(ii, round(frac * len(ii)), replace=False)] = True
        e = evaluate(rt, 0)
        draws.append([e["fidelity"], e["changes_caught"], e["f1"], e["errors"], e["calls"], e["usd"]])
    d = np.array(draws)
    row["random_mean_of_50"] = dict(zip(("fidelity", "changes_caught", "f1", "errors", "calls", "usd"),
                                        np.round(d.mean(0), 4).tolist(), strict=True))
    row["random_fidelity_sd"] = round(float(d[:, 0].std()), 4)
    curve.append(row)

report = {"rows": n, "api_calls": 0, "call_formula_mismatches": CALL_CHECK,
          "interaction_change_label_rate": round(float(y.mean()), 4), "interaction_change_frames": int(y.sum()),
          "verb_rules_per_fold": {v: sorted((_TASK_NAMES["instrument"][k], _TASK_NAMES["verb"][x]) for k, x in s)
                                  for v, s in VPRIOR.items()},
          "phase_declared_grid": phase_main, "phase_extended_grid_sensitivity": phase_ext,
          "gate_folds": {"gate_v2_qwen_probe_42": f42, "gate_v2_probe_free_43": fb},
          "lovo_auc": {"gate_v2_qwen_probe_42": round(float(roc_auc_score(y, s42)), 4),
                       "gate_v2_probe_free_43": round(float(roc_auc_score(y, sb)), 4)},
          "gate": gate, "cost_fidelity_curve": curve}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("call_formula_mismatches", "interaction_change_label_rate", "verb_rules_per_fold",
                                         "phase_declared_grid", "phase_extended_grid_sensitivity", "lovo_auc")}, default=str))
for ph_name, block in gate.items():
    for name, e in block.items():
        print(f"[{ph_name}] {name:26s} F1 {e['f1']:.4f} err {e['errors']} calls {e['calls']} usd {e['usd']} "
              f"reviewed {e['reviewed']} retention F1 {e['f1_gain_retention']} err {e['error_reduction_retention']} "
              f"caught {e['changes_caught']} ok {e['every_video_not_worse_than_no_review']}")
for row in curve:
    print(row)
