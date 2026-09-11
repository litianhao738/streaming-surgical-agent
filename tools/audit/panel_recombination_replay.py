"""Offline recombination of the two already-paid review panels; zero API calls.

Every joint-Phase archive holds, for the same target, the v1.3.0 control panel
(five compact four-head reviews plus five blind Phase votes) and the v2.0.x joint
panel (five reviews of the four heads and all seven Phases). Both were paid for.
This tool replays alternative Python decision rules over those saved answers:

  h0                    frozen initial prediction
  control               v1.3.0: control four heads + blind Phase majority (fidelity check)
  joint_v200            v2.0.0: joint four heads + joint Phase rule, five valid seats
  joint_v201            v2.0.1: same answers, interaction items need three valid seats
  control4_h0phase      control four heads, Phase frozen to H0 (6 calls per target)
  control4_jointphase   control four heads, Phase from the joint rule
  joint4_h0phase        joint four heads, Phase frozen to H0
  consensus             add only when both panels support, delete only when both refute
  union                 add when either panel supports, delete when either refutes
  pooled10              one ten-rating mean over both panels, at least six valid ratings
  control_noqwen        control panel without the Qwen seat (four seats, all valid)
  joint_noqwen          joint panel without the Qwen seat
  phase_vote4           control four heads, blind Phase switch needs four matching votes
  phase_agree           control four heads, Phase switch only when the blind majority and
                        the joint rule name the same new Phase

Rules other than `h0`, `control`, `joint_v200`, `joint_v201` never change what
was sent to any model; they only change how Python reads the saved answers. The
final five-head JSON shape is unchanged.

Selection is predeclared in `SELECTION_RULE` and evaluated on the Training
archives only; the Validation archive is scored afterwards as a single held-out
check. No ground truth is read except the archives' own `scored_truth.json`.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_joint_phase_feedback_trial as joint
from scripts import run_split_review_trial as common
from scripts.run_pool_expansion_trial import aggregate as joint_aggregate
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
)
from surgical_agent.research.verification.five_head_repair import (
    normalize_five_heads,
)
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    phase_apply,
    phase_choice_error,
)
from surgical_agent.research.verification.prior_panel import labels

TASKS = ("instrument", "verb", "target", "ivt", "phase")
FOUR = TASKS[:4]
RULES = ("h0", "control", "joint_v200", "joint_v201", "control4_h0phase", "control4_jointphase",
         "joint4_h0phase", "consensus", "union", "pooled10", "control_noqwen", "joint_noqwen",
         "phase_vote4", "phase_agree")
BASELINES = ("control", "joint_v201")
SELECTION_RULE = ("On the pooled Training archives a candidate rule must have mean F1 above both "
                  "baselines (control, joint_v201), fewer total label errors than both, and no head F1 "
                  "below the lower baseline value for that head. Among passing rules choose the fewest "
                  "errors, then the highest mean F1. The Validation archive is scored once afterwards "
                  "and is not used for selection.")
TRAINING = ["artifacts/preflight/joint_phase_feedback_dev8_20260910_v3",
            "artifacts/preflight/joint_phase_feedback_fresh16_20260910_v2",
            "artifacts/preflight/joint_phase_confirm16_independent_20260910_v1"]
VALIDATION = ["artifacts/preflight/validation_vid110_confirm_20260911_v2_resume1"]


def four_pool(pool):
    return {"propositions": [deepcopy(p) for p in pool["propositions"] if p["task"] != "phase"]}


def load_targets(archive):
    archive = Path(archive)
    truth = {f"{t['video_id']}_{t['frame_id']}": t for t in read(archive / "scored_truth.json")}
    initials = read(archive / "initials.json") if (archive / "initials.json").is_file() else {}
    rows = []
    for folder in sorted((archive / "targets").iterdir()):
        r = read(folder / "result.json")
        key = r.get("key") or folder.name
        if "initial_pool" in r:
            pool, jpool, jraw = r["initial_pool"], r["joint_r1"]["pool"], r["joint_r1"]["raw"]
            archived = {"control": r["predictions"]["control"], "joint_v200": r["predictions"]["joint_r1"]}
        else:
            pool, jpool, jraw = initials[key]["pool"], r["joint_pool"], r["joint_raw"]
            archived = {"control": r["predictions"]["control"], "joint_v200": r["predictions"]["v2.0.0"],
                        "joint_v201": r["predictions"]["v2.0.1"]}
        rows.append({"key": key, "archive": archive.name, "h0": labels(r["h0"]), "pool": pool,
                     "jpool": jpool, "control_raw": r["control"]["raw"], "phase_raw": r["control"]["phase_raw"],
                     "joint_raw": jraw, "archived": archived, "truth": truth[key]})
    return rows


def seat_scores(diagnostics, pid):
    """Per-seat ratings aligned with SEATS; None where that seat's item was invalid."""
    return dict(zip(SEATS, diagnostics[pid]["scores"], strict=True))


def mean_of(values, need):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if len(valid) >= need else None


def panels(row):
    pool, jpool = row["pool"], row["jpool"]
    reviews, _ = common.normalize_five(row["control_raw"], pool, 3)
    cmeans, cdiag = common.panel.aggregate(reviews, pool, image_count=3)
    jreviews, _ = normalize_five_heads(row["joint_raw"], jpool, image_count=3)
    jmeans5, jdiag = joint_aggregate(jreviews, jpool, len(SEATS))
    jmeans3, _ = joint_aggregate(jreviews, jpool, 3)
    return {"cmeans": cmeans, "cscores": {p["id"]: seat_scores(cdiag, p["id"]) for p in pool["propositions"]},
            "jmeans5": jmeans5, "jmeans3": jmeans3,
            "jscores": {p["id"]: seat_scores(jdiag, p["id"]) for p in jpool["propositions"]}}


def phase_means_from(jmeans):
    means = {f"phase_{i}": jmeans[f"phase_{i}"] for i in range(7)}
    if any(v is None for v in means.values()):
        means = {k: None for k in means}
    return means


def blind_votes(phase_raw, seats, need):
    """Independent single-label votes from the given seats; all must be valid."""
    if any(phase_choice_error(phase_raw[s], 3) for s in seats):
        return None
    votes = Counter(phase_raw[s]["phase_id"] for s in seats if phase_raw[s]["phase_id"] is not None)
    if not votes:
        return None
    phase, count = votes.most_common(1)[0]
    return phase if count >= need else None


def with_phase(four, phase):
    out = deepcopy(four)
    out["phase"] = [phase]
    return out


def combine(row, p, how):
    """Build one means dict over the four-head pool from the two panels."""
    pool_ids = [q["id"] for q in row["pool"]["propositions"]]
    present = {f"{t}_{c}" for t in FOUR for c in row["h0"][t]}
    out = {}
    for pid in pool_ids:
        c, j = p["cmeans"][pid], p["jmeans5"].get(pid)
        if how == "pooled10":
            values = list(p["cscores"][pid].values()) + list(p["jscores"].get(pid, {}).values())
            out[pid] = mean_of(values, 6)
            continue
        both = [v for v in (c, j) if v is not None]
        if how == "consensus":
            # additions need both panels >= 4, deletions need both <= 2
            if c is None or j is None:
                out[pid] = None
            else:
                out[pid] = max(c, j) if pid in present else min(c, j)
        elif how == "union":
            if not both:
                out[pid] = None
            else:
                out[pid] = min(both) if pid in present else max(both)
        else:
            raise ValueError(how)
    return out


def decide(row, p):
    h0, pool, jpool, phase_raw = row["h0"], row["pool"], row["jpool"], row["phase_raw"]
    sub = four_pool(jpool)
    preds = {"h0": deepcopy(h0)}
    control4 = common.panel.select(h0, pool, p["cmeans"], threshold=4)
    preds["control"], control_phase = apply_phase_choices(control4, phase_raw, 3)
    preds["joint_v200"], _ = joint.select_joint(h0, jpool, p["jmeans5"])
    preds["joint_v201"], _ = joint.select_joint(h0, jpool, p["jmeans3"])
    joint4 = common.panel.select(h0, sub, {q["id"]: p["jmeans5"][q["id"]] for q in sub["propositions"]}, threshold=4)
    jphase = phase_means_from(p["jmeans5"])
    preds["control4_h0phase"] = with_phase(control4, h0["phase"][0])
    preds["control4_jointphase"], joint_phase_decision = phase_apply(control4, jphase)
    preds["joint4_h0phase"] = with_phase(joint4, h0["phase"][0])
    for how in ("consensus", "union", "pooled10"):
        four = common.panel.select(h0, pool, combine(row, p, how), threshold=4)
        preds[how], _ = phase_apply(four, jphase)
    seats = tuple(s for s in SEATS if s != "qwen")
    cm4 = {pid: mean_of([p["cscores"][pid][s] for s in seats], 4) for pid in p["cmeans"]}
    four = common.panel.select(h0, pool, cm4, threshold=4)
    vote = blind_votes(phase_raw, seats, 3)
    preds["control_noqwen"] = with_phase(four, vote if vote is not None else h0["phase"][0])
    jm4 = {pid: mean_of([p["jscores"][pid][s] for s in seats], 4) for pid in p["jmeans5"]}
    preds["joint_noqwen"], _ = joint.select_joint(h0, jpool, jm4)
    vote4 = blind_votes(phase_raw, SEATS, 4)
    preds["phase_vote4"] = with_phase(control4, vote4 if vote4 is not None else h0["phase"][0])
    majority = control_phase["after"] if control_phase["after"] != control_phase["before"] else None
    agreed = majority if majority is not None and joint_phase_decision["after"] == majority else h0["phase"][0]
    preds["phase_agree"] = with_phase(control4, agreed)
    return {k: labels(v) for k, v in preds.items()}


def score(rows, predictions):
    metrics = {}
    for rule in RULES:
        entry, f1s, precisions = {}, [], []
        for task in TASKS:
            tp = fp = fn = 0
            for row in rows:
                item = row["truth"]
                if not item["mask"].get(task):
                    continue
                got, gt = set(predictions[row["key"]][rule][task]), set(item["gt"][task] or [])
                tp, fp, fn = tp + len(got & gt), fp + len(got - gt), fn + len(gt - got)
            f1 = round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None
            precision = round(100 * tp / (tp + fp), 2) if tp + fp else None
            entry[task] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "precision": precision}
            f1s.append(f1 or 0)
            precisions.append(precision or 0)
        entry["mean_f1"] = round(sum(f1s) / len(f1s), 2)
        entry["mean_precision"] = round(sum(precisions) / len(precisions), 2)
        entry["errors"] = sum(entry[t]["fp"] + entry[t]["fn"] for t in TASKS)
        metrics[rule] = entry
    return metrics


def edits(rows, predictions):
    """Label-level changes relative to H0, judged against GT where the task is masked in."""
    out = {}
    for rule in RULES:
        c = Counter()
        for row in rows:
            gt, mask, h0, pred = row["truth"]["gt"], row["truth"]["mask"], row["h0"], predictions[row["key"]][rule]
            for task in FOUR:
                if not mask.get(task):
                    continue
                truth = set(gt[task] or [])
                for label in set(pred[task]) - set(h0[task]):
                    c["good_add" if label in truth else "bad_add"] += 1
                for label in set(h0[task]) - set(pred[task]):
                    c["good_delete" if label not in truth else "bad_delete"] += 1
            if mask.get("phase") and pred["phase"] != h0["phase"]:
                truth = gt["phase"][0]
                c["phase_fixed" if pred["phase"][0] == truth else
                  ("phase_broken" if h0["phase"][0] == truth else "phase_wrong_to_wrong")] += 1
        out[rule] = dict(c)
    return out


def passes(metrics, rule):
    m, base = metrics[rule], [metrics[b] for b in BASELINES]
    if rule in ("h0", *BASELINES):
        return False
    if not all(m["mean_f1"] > b["mean_f1"] and m["errors"] < b["errors"] for b in base):
        return False
    return all((m[t]["f1"] or 0) >= min((b[t]["f1"] or 0) for b in base) for t in TASKS)


def table(metrics):
    head = f"{'rule':<20}" + "".join(f"{t[:5]:>8}" for t in TASKS) + f"{'meanF1':>8}{'meanP':>8}{'err':>6}"
    lines = [head]
    for rule in RULES:
        e = metrics[rule]
        lines.append(f"{rule:<20}" + "".join(f"{(e[t]['f1'] if e[t]['f1'] is not None else 0):>8}" for t in TASKS)
                     + f"{e['mean_f1']:>8}{e['mean_precision']:>8}{e['errors']:>6}")
    return "\n".join(lines)


def run(archives):
    rows, predictions, fidelity = [], {}, Counter()
    for archive in archives:
        for row in load_targets(ROOT / archive):
            preds = decide(row, panels(row))
            for name, archived in row["archived"].items():
                fidelity["match" if labels(archived) == preds[name] else f"MISMATCH:{name}"] += 1
            rows.append(row)
            predictions[row["key"]] = preds
    return rows, predictions, dict(fidelity)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/panel_recombination_replay_20260911.json")
    args = parser.parse_args()
    report = {"replayed_utc": now(), "api_calls": 0, "selection_rule": SELECTION_RULE, "rules": RULES,
              "training_archives": TRAINING, "validation_archives": VALIDATION, "cohorts": {}}
    trows, tpreds, tfid = run(TRAINING)
    per_cohort = {}
    for archive in TRAINING:
        subset = [r for r in trows if r["archive"] == Path(archive).name]
        per_cohort[Path(archive).name] = {"targets": len(subset), "metrics": score(subset, tpreds)}
    tmetrics = score(trows, tpreds)
    passing = [r for r in RULES if passes(tmetrics, r)]
    chosen = sorted(passing, key=lambda r: (tmetrics[r]["errors"], -tmetrics[r]["mean_f1"]))
    report["cohorts"]["training_pooled"] = {"targets": len(trows), "fidelity": tfid, "metrics": tmetrics,
                                            "edits": edits(trows, tpreds), "per_archive": per_cohort,
                                            "passing_rules": passing, "chosen": chosen[0] if chosen else None}
    print(f"TRAINING pooled ({len(trows)} targets), fidelity {tfid}")
    print(table(tmetrics))
    print("passing:", passing, "chosen:", chosen[0] if chosen else None)
    vrows, vpreds, vfid = run(VALIDATION)
    vmetrics = score(vrows, vpreds)
    report["cohorts"]["validation_vid110"] = {"targets": len(vrows), "fidelity": vfid, "metrics": vmetrics,
                                             "edits": edits(vrows, vpreds),
                                             "passes_same_standard": [r for r in RULES if passes(vmetrics, r)]}
    print(f"\nVALIDATION VID110 ({len(vrows)} targets), fidelity {vfid}")
    print(table(vmetrics))
    print("rules passing the same standard on VID110:", report["cohorts"]["validation_vid110"]["passes_same_standard"])
    report["predictions"] = {"training": {k: v for k, v in tpreds.items()}, "validation": vpreds}
    save(args.output, report)
    print("\nsaved", args.output)


if __name__ == "__main__":
    main()
