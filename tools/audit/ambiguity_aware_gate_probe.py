"""Zero-call: probe-guided "when to verify" on top of ambiguity-aware verification.

Data: Codex's 6,059 first-pass rows and the per-seat scores saved in each target's
result.json. Ambiguity-aware output = frozen full output with every edit to verb
grasp/retract, or to an IVT whose verb is grasp/retract, reverted to cheap.

Exact stop (Qwen first): ambiguous items are settled from the start; a present item
is settled once it can no longer be refuted (5-seat mean <= 2), an absent one once it
can no longer be supported (mean >= 4); a seat invalid on an item voids it. The Phase
panel stops once no alternative can still beat the current phase. Every early stop is
checked against the ambiguity-aware output (the script raises on any disagreement).

Gate label (GT-free): the ambiguity-aware output differs from cheap. Features: 43
post-cheap features, optionally + 10 features from the Qwen compact-panel probe
(non-ambiguous items only). Models: LogisticRegression and HistGradientBoosting,
leave-one-video-out. Policies verify the top fraction by score (descriptive curve)
or use a nested threshold (outer leave-one-video-out; inner leave-one-video-out on the
other three videos picks the cheapest threshold meeting Codex's development criteria:
pooled F1 >= max(cheap, original full), errors <= min(cheap, original full), every
video not worse than cheap on F1 and errors, and some verification). GT is used only
for evaluation and threshold selection on fitting videos.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.research.gate import mainline_training as strict
from surgical_agent.research.verification.candidate_coordinator import (
    _TASK_NAMES,
    SEATS,
)
from surgical_agent.research.verification.prior_panel import COMPONENTS

OFF = ROOT / "artifacts/training/gate/full_official_reviewers_20260912_v1"
ROWS = ROOT / "artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json"
OUT = ROOT / "artifacts/research/ambiguity_aware_gate_probe_20260913.json"
TASKS, FOUR = strict.TASKS, ("instrument", "verb", "target", "ivt")
G, R = _TASK_NAMES["verb"].index("grasp"), _TASK_NAMES["verb"].index("retract")
AMB_V = {G, R}
AMB_IVT = {c for c, comp in COMPONENTS.items() if comp["verb"] in AMB_V}
USD_C = {"qwen": .000263, "gpt": .001698, "gemini": .002721, "grok": 0.0, "deepseek": 0.0}
USD_J = {"qwen": .000390, "gpt": .002854, "gemini": .004462, "grok": 0.0, "deepseek": 0.0}
BASE = {"h0": .004366, "proposal": .003300, "phase_rec": .003483}
FULL_USD = sum(BASE.values()) + sum(USD_C.values()) + sum(USD_J.values())
ORDER = ["qwen", "gpt", "gemini", "grok", "deepseek"]
IDX = [SEATS.index(s) for s in ORDER]
QI = SEATS.index("qwen")

rows = json.loads(ROWS.read_text("utf-8"))
n = len(rows)
vid = np.array([r["video_id"] for r in rows])
videos = sorted(set(vid))


def amb_item(task, label):
    return (task == "verb" and label in AMB_V) or (task == "ivt" and label in AMB_IVT)


def ambiguity_aware(cheap, full):
    out = {t: sorted(full[t]) for t in TASKS}
    out["verb"] = sorted((set(full["verb"]) - AMB_V) | (set(cheap["verb"]) & AMB_V))
    out["ivt"] = sorted((set(full["ivt"]) - AMB_IVT) | (set(cheap["ivt"]) & AMB_IVT))
    return out


def key(lab):
    return tuple(tuple(sorted(lab[t])) for t in TASKS)


def void(inv, q):
    return any(SEATS[i] in inv for i in q)


def settled(sc, inv, present, amb, q):
    if amb or void(inv, q):
        return True
    r, s = 5 - len(q), sum(sc[i] for i in q)
    return (s + r) / 5 > 2 if present else (s + 5 * r) / 5 < 4


def phase_settled(ph, old, q):
    if void(ph[old][1], q):
        return True
    r = 5 - len(q)
    min_old = (sum(ph[old][0][i] for i in q) + r) / 5
    for p, (sc, inv) in enumerate(ph):
        if p != old and not void(inv, q):
            best = (sum(sc[i] for i in q) + 5 * r) / 5
            if best >= 4 and best > min_old:
                return False
    return True


amb_out, seats, usd_seats, violations, probe = [], np.zeros(n), np.zeros(n), 0, []
for i, r in enumerate(rows):
    res = json.loads((OFF / "targets" / r["sample_id"] / "result.json").read_text("utf-8"))
    h0 = r["h0_labels"]
    comp = [(res["diagnostics"][p["id"]]["scores"], set(res["diagnostics"][p["id"]]["invalid"]),
             p["label_id"] in h0[p["task"]], amb_item(p["task"], p["label_id"])) for p in res["pool"]["propositions"]]
    ph = [(res["joint_diagnostics"][f"phase_{p}"]["scores"], set(res["joint_diagnostics"][f"phase_{p}"]["invalid"]))
          for p in range(7)]
    old = h0["phase"][0]
    a = ambiguity_aware(r["cheap_labels"], r["final_labels"])
    amb_out.append(a)
    kc = next((k for k in range(5) if all(settled(s, v, pr, am, IDX[:k]) for s, v, pr, am in comp)), 5)
    kj = next((k for k in range(5) if phase_settled(ph, old, IDX[:k])), 5)
    if (kc < 5 and any(sorted(a[t]) != sorted(r["cheap_labels"][t]) for t in FOUR)) or (
            kj < 5 and a["phase"] != r["cheap_labels"]["phase"]):
        violations += 1
    seats[i] = kc + kj
    usd_seats[i] = sum(USD_C[s] for s in ORDER[:kc]) + sum(USD_J[s] for s in ORDER[:kj])
    valid = [(sc[QI], pr) for sc, inv, pr, am in comp if not am and "qwen" not in inv]
    pres, absn = [s for s, pr in valid if pr], [s for s, pr in valid if not pr]
    nonamb = [c for c in comp if not c[3]]
    probe.append([len(nonamb), sum("qwen" in c[1] for c in nonamb),
                  min(pres) if pres else 5, sum(s <= 2 for s in pres), sum(s <= 3 for s in pres),
                  max(absn) if absn else 1, sum(s >= 4 for s in absn), sum(s >= 5 for s in absn),
                  sum(settled(sc, inv, pr, am, [QI]) for sc, inv, pr, am in comp) / max(1, len(comp)),
                  int(all(settled(sc, inv, pr, am, [QI]) for sc, inv, pr, am in comp))])
if violations:
    raise ValueError(f"exact stop disagreed with the ambiguity-aware output on {violations} frames")
PROBE_NAMES = ["qwen_nonamb_items", "qwen_invalid_nonamb", "qwen_present_min", "qwen_present_le2", "qwen_present_le3",
               "qwen_absent_max", "qwen_absent_ge4", "qwen_absent_ge5", "qwen_settled_fraction", "qwen_all_settled"]
PQ = np.array(probe, float)
XB = np.array([[r["features_postcheap"][k] for k in sorted(r["features_postcheap"])] for r in rows], float)
rule = np.array([bool(r["features_postcheap"]["proposal_rule"]) for r in rows])
y = np.array([key(a) != key(r["cheap_labels"]) for a, r in zip(amb_out, rows, strict=True)]).astype(int)


def counts(outputs):
    m = np.zeros((n, 5, 3), dtype=np.int64)
    for i, (r, lab) in enumerate(zip(rows, outputs, strict=True)):
        truth = strict._truth(r["gt"], r["mask"])
        for j, t in enumerate(TASKS):
            c = strict._counts(set(lab[t]), truth[t])
            m[i, j] = (c["tp"], c["fp"], c["fn"])
    return m


CC, CA, CF = counts([r["cheap_labels"] for r in rows]), counts(amb_out), counts([r["final_labels"] for r in rows])


def pooled(c):
    tp, fp, fn = c.sum(axis=0).T
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones_like(tp, dtype=float), where=den != 0)
    return [round(float(f1.mean() * 100), 4), int((fp + fn).sum())]


def evaluate(verify, probe_all, idx=None):
    idx = np.arange(n) if idx is None else idx
    v_ = verify[idx]
    c = np.where(v_[:, None, None], CA[idx], CC[idx])
    vids = vid[idx]
    per = {v: pooled(c[vids == v]) for v in sorted(set(vids))}
    base = {v: pooled(CC[idx][vids == v]) for v in per}
    ok = all(per[v][0] >= base[v][0] and per[v][1] <= base[v][1] for v in per)
    k = int(v_.sum())
    if probe_all:  # every frame: H0 + proposal + Qwen compact; verified frames add Phase rec + remaining seats
        seat = np.where(v_, seats[idx], 1)
        usd = np.where(v_, BASE["h0"] + BASE["proposal"] + BASE["phase_rec"] + usd_seats[idx],
                       BASE["h0"] + BASE["proposal"] + USD_C["qwen"])
        calls = 2 * len(idx) + k + int(seat.sum())
    else:
        seat = np.where(v_, seats[idx], 0)
        usd = np.where(v_, BASE["h0"] + BASE["proposal"] + BASE["phase_rec"] + usd_seats[idx],
                       BASE["h0"] + rule[idx] * BASE["proposal"])
        calls = len(idx) + int((rule[idx] & ~v_).sum()) + 2 * k + int(seat.sum())
    return {"verify_fraction": round(float(v_.mean()), 4), "logical_calls": calls,
            "seat_share": round(float(seat.sum() / (10 * len(idx))), 4), "usd_share": round(float(usd.mean() / FULL_USD), 4),
            "all": pooled(c), "per_video": per, "every_video_not_worse_than_cheap": ok,
            "missed_changes": int((y[idx].astype(bool) & ~v_).sum())}


def fit_predict(X, fit, pred, model):
    if model == "lr":
        sc = StandardScaler().fit(X[fit])
        m = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=2000, random_state=3407)
        return m.fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[pred]))[:, 1]
    pos = y[fit].mean()
    w = np.where(y[fit] == 1, 0.5 / pos, 0.5 / (1 - pos))
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=3407)
    return m.fit(X[fit], y[fit], sample_weight=w).predict_proba(X[pred])[:, 1]


def lovo(X, model):
    s = np.zeros(n)
    for v in videos:
        s[vid == v] = fit_predict(X, np.where(vid != v)[0], np.where(vid == v)[0], model)
    return s


def nested(X, model, probe_all, criterion="codex"):
    """criterion 'codex': pooled F1 >= max(cheap, original full) and errors <= min(cheap, original full).
    criterion 'retention90': keep >= 90% of the ambiguity-aware full's F1 gain over cheap, errors <= cheap.
    Both also require every inner video not worse than cheap and some verification; cheapest feasible wins."""
    verify, folds = np.zeros(n, bool), {}
    for v in videos:
        inner = np.where(vid != v)[0]
        inner_scores = np.zeros(n)
        for u in videos:
            if u != v:
                inner_scores[vid == u] = fit_predict(X, np.where((vid != v) & (vid != u))[0], np.where(vid == u)[0], model)
        cheap_ref, full_ref, amb_ref = pooled(CC[inner]), pooled(CF[inner]), pooled(CA[inner])
        chosen = None
        for frac in np.arange(0.05, 1.0001, 0.05):
            thr = float(np.quantile(inner_scores[inner], 1 - frac))
            cand = np.zeros(n, bool)
            cand[inner] = inner_scores[inner] >= thr
            e = evaluate(cand, probe_all, inner)
            if criterion == "codex":
                quality = e["all"][0] >= max(cheap_ref[0], full_ref[0]) and e["all"][1] <= min(cheap_ref[1], full_ref[1])
            else:
                quality = (e["all"][0] - cheap_ref[0] >= 0.9 * max(0.0, amb_ref[0] - cheap_ref[0])
                           and e["all"][1] <= cheap_ref[1])
            if e["every_video_not_worse_than_cheap"] and cand[inner].any() and quality:
                chosen = (round(float(frac), 2), thr, e)
                break
        held = np.where(vid == v)[0]
        outer_scores = fit_predict(X, inner, held, model)
        if chosen is None:
            folds[v] = {"status": "INFEASIBLE_KEEP_CHEAP"}
        else:
            verify[held] = outer_scores >= chosen[1]
            folds[v] = {"status": "FEASIBLE", "inner_verify_fraction": chosen[0], "threshold": chosen[1],
                        "inner_all": chosen[2]["all"], "outer_verify_fraction": round(float(verify[held].mean()), 4)}
    result = evaluate(verify, probe_all)
    cheap_all, full_all, amb_all = pooled(CC), pooled(CF), pooled(CA)
    retention = (result["all"][0] - cheap_all[0]) / (amb_all[0] - cheap_all[0])
    result["gain_retention_vs_ambiguity_full"] = round(float(retention), 4)
    result["passes_codex_criteria"] = bool(
        result["every_video_not_worse_than_cheap"] and verify.any()
        and result["all"][0] >= max(cheap_all[0], full_all[0]) and result["all"][1] <= min(cheap_all[1], full_all[1]))
    result["passes_retention90"] = bool(
        result["every_video_not_worse_than_cheap"] and verify.any() and retention >= 0.9 and result["all"][1] <= cheap_all[1])
    return result, folds


report = {"rows": n, "api_calls": 0, "exact_stop_violations": violations, "probe_feature_names": PROBE_NAMES,
          "intervene_frames": int(y.sum()), "intervene_by_video": {v: int(y[vid == v].sum()) for v in videos},
          "reference": {"cheap": evaluate(np.zeros(n, bool), False),
                        "original_full": {"all": pooled(CF), "per_video": {v: pooled(CF[vid == v]) for v in videos},
                                          "logical_calls": 13 * n},
                        "ambiguity_aware_full_exact_stop": evaluate(np.ones(n, bool), False),
                        "oracle_verify_only_changes": evaluate(y.astype(bool), False)},
          "auc": {}, "curves": [], "nested": {}, "nested_retention90": {}}
variants = {"base_lr": (XB, "lr", False), "base_hgb": (XB, "hgb", False),
            "qwen_lr": (np.hstack([XB, PQ]), "lr", True), "qwen_hgb": (np.hstack([XB, PQ]), "hgb", True)}
scores = {}
for name, (X, model, probe_all) in variants.items():
    s = lovo(X, model)
    scores[name] = s
    report["auc"][name] = {"all": round(roc_auc_score(y, s), 3),
                           **{v: round(roc_auc_score(y[vid == v], s[vid == v]), 3) for v in videos}}
    report["nested"][name] = dict(zip(("outer", "folds"), nested(X, model, probe_all), strict=True))
    report["nested_retention90"][name] = dict(zip(("outer", "folds"), nested(X, model, probe_all, "retention90"), strict=True))
rng = random.Random(3407)
for frac in (.05, .1, .15, .2, .3, .5):
    k = round(frac * n)
    row = {"verify_fraction": frac}
    for name, (_, _, probe_all) in variants.items():
        m = np.zeros(n, bool)
        m[np.argsort(-scores[name])[:k]] = True
        row[name] = evaluate(m, probe_all)
        if name == "qwen_hgb":
            per_video_k = {v: int(m[vid == v].sum()) for v in videos}
    rm = np.zeros(n, bool)
    for v in videos:
        rm[rng.sample(list(np.where(vid == v)[0]), per_video_k[v])] = True
    row["random_within_video_same_counts_as_qwen_hgb"] = evaluate(rm, False)
    report["curves"].append(row)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
print({k: report[k] for k in ("rows", "exact_stop_violations", "intervene_frames", "intervene_by_video")})
for k, v in report["reference"].items():
    print(k, {x: v.get(x) for x in ("verify_fraction", "logical_calls", "seat_share", "usd_share", "all")})
print("AUC:", report["auc"])
for section in ("nested", "nested_retention90"):
    for name, res in report[section].items():
        o = res["outer"]
        print(section.upper(), name, {x: o[x] for x in ("verify_fraction", "logical_calls", "usd_share", "all",
                                                         "every_video_not_worse_than_cheap", "gain_retention_vs_ambiguity_full",
                                                         "passes_codex_criteria", "passes_retention90")},
              o["per_video"], {v: (f["status"], f.get("inner_verify_fraction"), f.get("outer_verify_fraction"))
                               for v, f in res["folds"].items()})
for row in report["curves"]:
    print("--- verify", row["verify_fraction"])
    for k, v in row.items():
        if k != "verify_fraction":
            print(f"   {k:44s} calls={v['logical_calls']} seat={v['seat_share']:.3f} usd={v['usd_share']:.3f} "
                  f"all={v['all']} ok={v['every_video_not_worse_than_cheap']} missed={v['missed_changes']}")
