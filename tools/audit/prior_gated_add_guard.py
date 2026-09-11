"""T2b: fixed >=3 compact-mean add guard, evaluated once on archived data only."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).parent)]

import prior_gated_replay as replay

from surgical_agent.research.verification.prior_gated_repair import select_prior_gated

OUTPUT = ROOT / "artifacts/research/prior_gated_add_guard_20260911_v1.json"
FRESH = ("prior_gated_vid110_confirm_20260911_v1", "prior_gated_joint_vid110_confirm_20260911_v1_resume1")


def rows_for():
    train, _, _ = replay.run(replay.DEV, .01, .7, ())
    val, _, _ = replay.run(replay.VALIDATION, .01, .7, ())
    for name in FRESH:
        folder = ROOT / "artifacts/preflight" / name
        truth = {f"{r['video_id']}_{r['frame_id']}": r for r in replay.read(folder / "scored_truth.json")}
        for saved in replay.read(folder / "predictions.json")["targets"]:
            r = replay.read(folder / "targets" / saved["key"] / "result.json")
            reviews, _ = replay.common.normalize_five(r["review_raw"], r["pool"], 3)
            means, _ = replay.common.panel.aggregate(reviews, r["pool"], image_count=3)
            val.append({"key": saved["key"], "h0": r["h0"], "pool": r["pool"], "cmeans": means,
                        "truth": truth[saved["key"]], "archive": name,
                        "expected_gate": saved["predictions"]["gated_control"]})
    if len(train) != 136 or len(val) != 64 or len({r["key"] for r in val}) != 64:
        raise ValueError("declared cohort differs")
    return {"Training136": train, "VID11064": val}


def evaluate(rows):
    predictions, additions, changes = {}, {"original": Counter(), "guard": Counter()}, []
    for row in rows:
        prior = replay.prior_for(row["key"].split("_")[0])
        means, pool, h0 = row["cmeans"], row["pool"], row["h0"]
        means = means if isinstance(means, dict) else None
        base = replay.common.panel.select(h0, pool, means, threshold=4) if means is not None else h0
        kw = {"phase": h0["phase"][0], "veto_rate": .01, "add_rate": .7, "prune": ()}
        original, log0 = select_prior_gated(h0, pool, means, prior, **kw)
        original_again, _ = select_prior_gated(base, pool, None, prior, **kw)
        if original != original_again or ("expected_gate" in row and original != row["expected_gate"]):
            raise ValueError("baseline replay mismatch")
        # Panel already selected base; filtering now changes only prior-add eligibility.
        allowed = {"propositions": [p for p in pool["propositions"] if p["task"] != "ivt"
                   or (means is not None and means.get(p["id"]) is not None and means[p["id"]] >= 3)]}
        guarded, log1 = select_prior_gated(base, allowed, None, prior, **kw)
        if log0["vetoed"] != log1["vetoed"]:
            raise ValueError("guard changed veto path")
        predictions[row["key"]] = {"original": original, "guard": guarded}
        for arm, log in (("original", log0), ("guard", log1)):
            for ivt in log["prior_added"]:
                additions[arm]["correct" if ivt in row["truth"]["gt"]["ivt"] else "wrong"] += 1
        if original != guarded:
            changes.append({"key": row["key"], "original": original, "guard": guarded,
                            "removed_additions": sorted(set(log0["prior_added"]) - set(log1["prior_added"]))})
    metrics = replay.score(rows, predictions, ("original", "guard"))
    checks = {"wrong_additions_decrease": additions["guard"]["wrong"] < additions["original"]["wrong"],
              "correct_additions_not_decrease": additions["guard"]["correct"] >= additions["original"]["correct"],
              "mean_f1_not_decrease": metrics["guard"]["mean_f1"] >= metrics["original"]["mean_f1"],
              "errors_not_increase": metrics["guard"]["errors"] <= metrics["original"]["errors"]}
    return {"targets": len(rows), "metrics": metrics, "prior_additions": additions,
            "checks": checks, "passed": all(checks.values()), "changed_targets": changes}


if __name__ == "__main__":
    if OUTPUT.exists():
        raise ValueError("preserve previous audit")
    report = {"api_calls": 0, "candidate": "prior-add requires compact mean >=3; missing mean rejects add",
              "standard": "Wrong adds strictly decrease, correct adds do not decrease, mean F1 does not decrease, errors do not increase; required in both cohorts.",
              "cohorts": {name: evaluate(rows) for name, rows in rows_for().items()}}
    report["adopt"] = all(c["passed"] for c in report["cohorts"].values())
    replay.save(OUTPUT, report)
    print(json.dumps({"adopt": report["adopt"], "cohorts": {name: {k: c[k] for k in ("targets", "prior_additions", "checks", "passed")} for name, c in report["cohorts"].items()}}, indent=2))
