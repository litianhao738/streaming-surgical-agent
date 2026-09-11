"""Zero-API replay: prior-gated four heads combined with the archived joint Phase decision.

Reads only archives that saved both the v1.3.0 compact panel and the joint
five-head panel (raw answers) plus their own `scored_truth.json`. Reproduces the
evidence quoted in `prior_gated_joint.py`:

  * how much a different Phase bucket could change the gated four heads
    (H0 Phase vs blind vote vs joint Phase vs ground-truth Phase as an upper
    bound that is never available at inference);
  * five arms with the gate's Phase taken from H0 or from the joint panel.

Development: 40 Training targets (VID103/23/31/96). Validation: 32 archived
VID110 targets, already analysed by the gate work; reported as a check, never
as a selection basis. Nothing is tuned here; thresholds are the frozen ones.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools/audit")]

import prior_gated_replay as base

from scripts.run_prior_panel_trial import now, save
from surgical_agent.research.verification.prior_gated_joint import (
    VERSION,
    decide_phase,
)
from surgical_agent.research.verification.prior_gated_repair import (
    select_prior_gated,
)
from surgical_agent.research.verification.prior_panel import labels

GATE = {"veto_rate": 0.01, "add_rate": 0.7, "prune": ()}
DEV = [a for a in base.DEV if "joint_phase" in a]
OUTPUT = ROOT / "artifacts/research/prior_gated_joint_replay_20260911.json"


def joint_rows(archives):
    rows, seen = [], set()
    for archive in archives:
        for row in base.load_targets(ROOT / archive):
            if row["key"] in seen or row["jmeans"] is None:
                continue
            seen.add(row["key"])
            rows.append(row)
    return rows


def gate(row, means, pool, phase):
    prior = base.prior_for(row["key"].split("_")[0])
    out, _ = select_prior_gated(row["h0"], pool, means, prior, phase=phase, **GATE)
    return out


def four_errors(pred, row):
    return sum(len(set(pred[t]) ^ set(row["truth"]["gt"][t] or [])) for t in base.FOUR if row["truth"]["mask"].get(t))


def phase_sensitivity(rows):
    """Four-head label errors of gated_control when the gate bucket comes from each Phase source."""
    totals, changed = Counter(), Counter()
    for row in rows:
        cmeans = row["cmeans"] if isinstance(row["cmeans"], dict) else None
        sources = {"h0_phase": row["h0"]["phase"][0], "blind_vote": labels(row["archived"]["control"])["phase"][0],
                   "joint_phase": labels(row["archived"]["joint"])["phase"][0],
                   "ground_truth_phase_upper_bound": row["truth"]["gt"]["phase"][0]}
        errors = {k: four_errors(gate(row, cmeans, row["pool"], p), row) for k, p in sources.items()}
        totals.update(errors)
        for k in sources:
            if k != "h0_phase" and errors[k] != errors["h0_phase"]:
                changed[f"{k}_{'better' if errors[k] < errors['h0_phase'] else 'worse'}"] += 1
    return {"four_head_errors_by_gate_phase_source": dict(totals), "targets_changed_vs_h0_phase": dict(changed)}


def arms(row):
    h0p = row["h0"]["phase"][0]
    cmeans = row["cmeans"] if isinstance(row["cmeans"], dict) else None
    gated_control = gate(row, cmeans, row["pool"], h0p)
    sub = row["jpool"]
    jm = {p["id"]: row["jmeans"][p["id"]] for p in sub["propositions"]}
    gated_joint = gate(row, jm, sub, h0p)
    jphase, decision = decide_phase(row["h0"], row["jmeans"])
    return {"h0": row["h0"], "control": labels(row["archived"]["control"]), "joint_r1": labels(row["archived"]["joint"]),
            "gated_control": gated_control,
            "gated_control_jointphase": labels({**gated_control, "phase": jphase}),
            "gated_joint_jointphase": labels({**gated_joint, "phase": jphase})}, decision


def phase_edits(rows, preds, arm):
    c = Counter()
    for row in rows:
        gt, h0, pred = row["truth"]["gt"]["phase"][0], row["h0"]["phase"][0], preds[row["key"]][arm]["phase"][0]
        if pred != h0:
            c["fixed" if pred == gt else ("broken" if h0 == gt else "wrong_to_wrong")] += 1
    return dict(c)


def cohort(rows):
    preds, reasons = {}, Counter()
    for row in rows:
        preds[row["key"]], decision = arms(row)
        reasons[decision["reason"]] += 1
        archived_phase = labels(row["archived"]["joint"])["phase"]
        if preds[row["key"]]["gated_control_jointphase"]["phase"] != archived_phase:
            raise ValueError("joint Phase decision does not reproduce the archived joint prediction: " + row["key"])
    names = list(next(iter(preds.values())))
    metrics = base.score(rows, preds, names)
    return {"targets": len(rows), "videos": sorted({r["key"].split("_")[0] for r in rows}),
            "metrics": metrics, "phase_edits_vs_h0": {a: phase_edits(rows, preds, a) for a in names if a != "h0"},
            "phase_decision_reasons": dict(reasons), "phase_sensitivity": phase_sensitivity(rows),
            "joint_phase_reproduced_from_means": len(rows)}


def main():
    dev = cohort(joint_rows(DEV))
    validation = cohort(joint_rows(base.VALIDATION))
    report = {"replayed_utc": now(), "api_calls": 0, "version": VERSION, "gate": {**GATE, "prune": list(GATE["prune"])},
              "dev_archives": DEV, "validation_archives": base.VALIDATION, "dev": dev, "validation": validation,
              "note": ("Validation rows are the 32 VID110 targets already analysed by the gate work: a check, not a "
                       "selection basis. The ground-truth Phase column is an upper bound never available at inference.")}
    save(OUTPUT, report)
    for name, c in (("dev", dev), ("validation", validation)):
        print(f"\n== {name}: {c['targets']} targets {c['videos']}")
        print(base.table(c["metrics"], list(c["metrics"])))
        print("phase edits vs h0:", c["phase_edits_vs_h0"])
        print("phase sensitivity:", c["phase_sensitivity"])
    print("\nsaved", OUTPUT)


if __name__ == "__main__":
    main()
