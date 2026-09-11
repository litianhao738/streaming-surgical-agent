"""Zero-API replay of prior-gated IVT admission over archived panels.

Development cohorts are Training archives that saved the four-head candidate
pool, the five-seat rating means (or raw answers) and the leave-query-video-out
prior tables. The grid of (veto_rate, add_rate, prune) is swept on the pooled
development cohorts against the predeclared standard; the chosen setting is then
scored once on the Validation VID110 archive. Phase is frozen to H0 in every
prior-gated arm (Phase repair is net harmful on three of four cohorts).

Bases the gate is applied on top of:
  h0        no panel at all: gate the frozen H0 (zero review calls)
  control   v1.3.0 four heads (five compact reviews)
  joint     v2.0.x joint four heads where the archive has them

Only archives' own `scored_truth.json` is read for scoring.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_split_review_trial as common
from scripts.run_pool_expansion_trial import aggregate as joint_aggregate
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
)
from surgical_agent.research.verification.five_head_repair import (
    normalize_five_heads,
)
from surgical_agent.research.verification.prior_gated_repair import (
    select_prior_gated,
)
from surgical_agent.research.verification.prior_panel import labels

TASKS = ("instrument", "verb", "target", "ivt", "phase")
FOUR = TASKS[:4]
PRIOR_DIRS = {"VID110": "artifacts/preflight/validation_vid110_inputs_20260911_v2/priors"}
for _v in ("VID103", "VID23", "VID31", "VID96"):
    PRIOR_DIRS[_v] = "artifacts/preflight/joint_phase_independent_fresh16_inputs_20260910_v3/priors"
DEV = ["artifacts/preflight/joint_phase_feedback_dev8_20260910_v3",
       "artifacts/preflight/joint_phase_feedback_fresh16_20260910_v2",
       "artifacts/preflight/joint_phase_confirm16_independent_20260910_v1",
       "artifacts/preflight/expanded_split_review_32_20260910_v1",
       "artifacts/preflight/default_improvement_confirm_20260909_v1",
       "artifacts/preflight/new_training_verb_guard_20260909_v1",
       "artifacts/preflight/disputed_relation_20260909_v1"]
VALIDATION = ["artifacts/preflight/validation_vid110_confirm_20260911_v2_resume1"]
VETO = (None, 0.005, 0.01, 0.02, 0.05)
ADD = (None, 0.5, 0.6, 0.7)
PRUNE = ((), ("verb", "target"))
STANDARD = ("On pooled development targets the gated arm must have mean F1 above and fewer total "
            "label errors than both the same-base ungated arm and the v1.3.0 control, with no head F1 "
            "below the lower of those two. Ties broken by fewer errors then higher mean F1. Validation "
            "VID110 is scored once afterwards with the chosen setting and is not used for selection.")

_priors = {}


def prior_for(video):
    if video not in _priors:
        p = read(ROOT / PRIOR_DIRS[video] / f"{video}.json")
        if p["excluded_video"] != video or video in p["fit_videos"]:
            raise ValueError("query video leaked into its own prior: " + video)
        _priors[video] = p
    return _priors[video]


def four_pool(pool):
    return {"propositions": [p for p in pool["propositions"] if p["task"] != "phase"]}


def load_targets(archive):
    """Normalize the archive layouts into key/h0/pool/control means/joint means/archived preds."""
    archive = Path(archive)
    truth = {f"{t['video_id']}_{t['frame_id']}": t for t in read(archive / "scored_truth.json")}
    initials = read(archive / "initials.json") if (archive / "initials.json").is_file() else {}
    rows = []
    for folder in sorted((archive / "targets").iterdir()):
        if not (folder / "result.json").is_file():
            continue
        r = read(folder / "result.json")
        key = r.get("key") or folder.name
        row = {"key": key, "archive": archive.name, "h0": labels(r["h0"]), "jmeans": None, "jpool": None,
               "archived": {}}
        if "joint_raw" in r:  # validation layout
            pool = initials[key]["pool"]
            reviews, _ = common.normalize_five(r["control"]["raw"], pool, 3)
            row["cmeans"], _ = common.panel.aggregate(reviews, pool, image_count=3)
            jreviews, _ = normalize_five_heads(r["joint_raw"], r["joint_pool"], image_count=3)
            row["jmeans"], _ = joint_aggregate(jreviews, r["joint_pool"], len(SEATS))
            row["jpool"] = four_pool(r["joint_pool"])
            row["archived"] = {"control": r["predictions"]["control"], "joint": r["predictions"]["v2.0.0"]}
        elif "joint_r1" in r:  # joint-phase training layout
            pool = r["initial_pool"]
            reviews, _ = common.normalize_five(r["control"]["raw"], pool, 3)
            row["cmeans"], _ = common.panel.aggregate(reviews, pool, image_count=3)
            jreviews, _ = normalize_five_heads(r["joint_r1"]["raw"], r["joint_r1"]["pool"], image_count=3)
            row["jmeans"], _ = joint_aggregate(jreviews, r["joint_r1"]["pool"], len(SEATS))
            row["jpool"] = four_pool(r["joint_r1"]["pool"])
            row["archived"] = {"control": r["predictions"]["control"], "joint": r["predictions"]["joint_r1"]}
        elif "arms" in r and "pool" in r:  # expanded split layout
            pool, row["cmeans"] = r["pool"], r["arms"]["control"]["means"]
            row["archived"] = {"control": r["predictions"]["control"]}
        elif "arms" in r:  # default improvement layout
            pool, row["cmeans"] = r["arms"]["control"]["pool"], r["arms"]["control"]["means"]
            row["archived"] = {"control": r["predictions"]["control"]}
        elif "graph" in r:  # verb guard / disputed relation layout
            pool, row["cmeans"] = r["graph"]["pool"], r["graph"]["means"]
            row["archived"] = {"control": r["graph"]["prediction"]}
        else:
            raise ValueError("unknown archive layout: " + str(folder))
        row["pool"] = four_pool(pool)
        row["truth"] = truth[key]
        rows.append(row)
    return rows


def with_phase(four, phase):
    return labels({**four, "phase": [phase]})


def arms_for(row, veto, add, prune):
    """All arm predictions for one target under one gate setting."""
    h0, phase, prior = row["h0"], row["h0"]["phase"][0], prior_for(row["key"].split("_")[0])
    kw = {"phase": phase, "veto_rate": veto, "add_rate": add, "prune": prune}
    out = {"h0": h0}
    cmeans = row["cmeans"] if isinstance(row["cmeans"], dict) else None  # failed panel -> H0 four heads
    control4 = common.panel.select(h0, row["pool"], cmeans, threshold=4) if cmeans else labels(h0)
    out["control"] = labels(row["archived"]["control"])  # archived v1.3.0 incl. its Phase vote
    out["control_h0phase"] = with_phase(control4, phase)
    out["gated_h0"], out["log_h0"] = select_prior_gated(h0, row["pool"], None, prior, **kw)
    out["gated_control"], out["log_control"] = select_prior_gated(h0, row["pool"], cmeans, prior, **kw)
    if row["jmeans"] is not None:
        sub = row["jpool"]
        jm = {p["id"]: row["jmeans"][p["id"]] for p in sub["propositions"]}
        joint4 = common.panel.select(h0, sub, jm, threshold=4)
        out["joint_h0phase"] = with_phase(joint4, phase)
        out["gated_joint"], out["log_joint"] = select_prior_gated(h0, sub, jm, prior, **kw)
    return out


def score(rows, preds, arms):
    metrics = {}
    for arm in arms:
        entry, f1s, precisions = {}, [], []
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                p = preds[row["key"]].get(arm)
                if p is None or not row["truth"]["mask"].get(task):
                    continue
                got, gt = set(p[task]), set(row["truth"]["gt"][task] or [])
                tp, fp, fn = tp + len(got & gt), fp + len(got - gt), fn + len(gt - got)
            f1 = round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None
            precision = round(100 * tp / (tp + fp), 2) if tp + fp else None
            entry[task] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "precision": precision}
            f1s.append(f1 or 0)
            precisions.append(precision or 0)
        entry["mean_f1"] = round(sum(f1s) / len(f1s), 2)
        entry["mean_precision"] = round(sum(precisions) / len(precisions), 2)
        entry["errors"] = sum(entry[t]["fp"] + entry[t]["fn"] for t in TASKS)
        metrics[arm] = entry
    return metrics


def edits(rows, preds, arm):
    c = Counter()
    for row in rows:
        p = preds[row["key"]].get(arm)
        if p is None:
            continue
        gt, mask, h0 = row["truth"]["gt"], row["truth"]["mask"], row["h0"]
        for task in FOUR:
            if not mask.get(task):
                continue
            truth = set(gt[task] or [])
            for label in set(p[task]) - set(h0[task]):
                c[f"{task}_add_{'good' if label in truth else 'bad'}"] += 1
            for label in set(h0[task]) - set(p[task]):
                c[f"{task}_del_{'good' if label not in truth else 'bad'}"] += 1
    return dict(sorted(c.items()))


def passes(metrics, arm, references):
    m = metrics[arm]
    refs = [metrics[r] for r in references if r in metrics]
    if not all(m["mean_f1"] > r["mean_f1"] and m["errors"] < r["errors"] for r in refs):
        return False
    return all((m[t]["f1"] or 0) >= min((r[t]["f1"] or 0) for r in refs) for t in TASKS)


def table(metrics, arms):
    head = f"{'arm':<18}" + "".join(f"{t[:5]:>8}" for t in TASKS) + f"{'meanF1':>8}{'meanP':>8}{'err':>6}"
    lines = [head]
    for arm in arms:
        if arm not in metrics:
            continue
        e = metrics[arm]
        lines.append(f"{arm:<18}" + "".join(f"{(e[t]['f1'] if e[t]['f1'] is not None else 0):>8}" for t in TASKS)
                     + f"{e['mean_f1']:>8}{e['mean_precision']:>8}{e['errors']:>6}")
    return "\n".join(lines)


def run(archives, veto, add, prune, dedupe=True):
    rows, preds, seen, fidelity = [], {}, set(), Counter()
    for archive in archives:
        for row in load_targets(ROOT / archive):
            if dedupe and row["key"] in seen:
                continue
            seen.add(row["key"])
            out = arms_for(row, veto, add, prune)
            if "joint" in row["archived"] and "joint_h0phase" in out:
                # fidelity of the recomputed joint four heads against the archived joint prediction
                a = labels(row["archived"]["joint"])
                fidelity["joint_match" if all(a[t] == out["joint_h0phase"][t] for t in FOUR) else "JOINT_MISMATCH"] += 1
            c = labels(row["archived"]["control"])
            fidelity["control_match" if all(c[t] == out["control_h0phase"][t] for t in FOUR) else "CONTROL_MISMATCH"] += 1
            rows.append(row)
            preds[row["key"]] = out
    return rows, preds, dict(fidelity)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/prior_gated_replay_20260911.json")
    args = parser.parse_args()
    arms = ("h0", "control", "control_h0phase", "gated_h0", "gated_control", "joint_h0phase", "gated_joint")
    report = {"replayed_utc": now(), "api_calls": 0, "standard": STANDARD, "dev_archives": DEV,
              "validation_archives": VALIDATION, "grid": {"veto": VETO, "add": ADD, "prune": PRUNE}, "dev": {}}
    dev_results = []
    for veto, add, prune in product(VETO, ADD, PRUNE):
        rows, preds, fid = run(DEV, veto, add, prune)
        m = score(rows, preds, arms)
        setting = {"veto": veto, "add": add, "prune": list(prune)}
        passing = {arm: passes(m, arm, (base, "control")) for arm, base in
                   (("gated_h0", "h0"), ("gated_control", "control_h0phase"), ("gated_joint", "joint_h0phase"))}
        dev_results.append({"setting": setting, "metrics": m, "fidelity": fid, "passing": passing,
                            "edits": {a: edits(rows, preds, a) for a in ("gated_h0", "gated_control", "gated_joint")}})
    report["dev"]["targets"] = len(rows)
    report["dev"]["results"] = dev_results
    base = dev_results[0]["metrics"]
    print(f"DEV pooled ({len(rows)} unique targets), fidelity {dev_results[0]['fidelity']}")
    print(table(base, ("h0", "control", "control_h0phase", "joint_h0phase")))
    print("\nsetting                      gated_h0            gated_control       gated_joint")
    for res in dev_results:
        s = res["setting"]
        cells = []
        for arm in ("gated_h0", "gated_control", "gated_joint"):
            e = res["metrics"][arm]
            cells.append(f"{e['mean_f1']:>6}/{e['errors']:>4}{'*' if res['passing'][arm] else ' '}")
        print(f"veto={s['veto']!s:<6} add={s['add']!s:<5} prune={'VT' if s['prune'] else '--'}   " + "   ".join(cells))
    # choose: among settings where gated_control passes, fewest errors then highest mean F1
    candidates = [(r["metrics"]["gated_control"]["errors"], -r["metrics"]["gated_control"]["mean_f1"], i)
                  for i, r in enumerate(dev_results) if r["passing"]["gated_control"]]
    chosen = dev_results[min(candidates)[2]] if candidates else None
    report["dev"]["chosen"] = chosen["setting"] if chosen else None
    print("\nCHOSEN (gated_control):", chosen["setting"] if chosen else None)
    if chosen:
        s = chosen["setting"]
        vrows, vpreds, vfid = run(VALIDATION, s["veto"], s["add"], tuple(s["prune"]))
        vm = score(vrows, vpreds, arms)
        report["validation"] = {"targets": len(vrows), "setting": s, "fidelity": vfid, "metrics": vm,
                                "edits": {a: edits(vrows, vpreds, a) for a in ("gated_h0", "gated_control", "gated_joint")},
                                "passing": {arm: passes(vm, arm, (b, "control")) for arm, b in
                                            (("gated_h0", "h0"), ("gated_control", "control_h0phase"),
                                             ("gated_joint", "joint_h0phase"))},
                                "logs": {r["key"]: {k: v for k, v in vpreds[r["key"]].items() if k.startswith("log_")}
                                         for r in vrows}}
        print(f"\nVALIDATION VID110 ({len(vrows)} targets) with chosen setting, fidelity {vfid}")
        print(table(vm, arms))
        print("passes:", report["validation"]["passing"])
        print("edits gated_control:", report["validation"]["edits"]["gated_control"])
    save(args.output, report)
    print("\nsaved", args.output)


if __name__ == "__main__":
    main()
