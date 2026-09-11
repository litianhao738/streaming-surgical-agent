"""Deduplicated experiment index and costs; recovered copies count once."""
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/preflight"
RUNS = {
    "dev8_v1": "joint_phase_feedback_dev8_20260910_v3",
    "first_fresh_inputs": "joint_phase_fresh16_inputs_20260910_v1",
    "first_confirmation_v1": "joint_phase_feedback_fresh16_20260910_v2",
    "independent_repair_dev16_v2": "joint_phase_independent_repair_dev16_20260910_recovered",
    "blind_review_dev16_v3": "joint_phase_blind_review_dev16_20260910_resumed",
    "alternative_repair_dev16_v4": "joint_phase_alternative_repair_dev16_20260910_v1",
    "second_fresh_inputs": "joint_phase_independent_fresh16_inputs_20260910_v3",
    "confirmation_primary_v2": "joint_phase_confirm16_independent_20260910_v1",
    "confirmation_secondary_v4": "joint_phase_confirm16_alternative_20260910_v1",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    result, charges, total_calls, targets = {}, defaultdict(Decimal), 0, set()
    for name, folder in RUNS.items():
        path = BASE / folder
        if not (path / "completion.json").exists():
            result[name] = {"status": "not_completed", "archive": str(path)}
            continue
        done, budget, plan = (read(path / f"{f}.json") for f in ("completion", "budget", "plan"))
        total_calls += len(budget["calls"])
        targets.update((s["video_id"], s["frame_id"]) for s in plan["selection"])
        costs = defaultdict(Decimal)
        for c in budget["calls"]:
            key = c["account"] + ":" + c["charge_kind"]
            costs[key] += Decimal(c["charge"])
            charges[key] += Decimal(c["charge"])
        entry = {"archive": str(path), "calls": len(budget["calls"]), "fatal_error": done["fatal_error"],
            "interrupted": done.get("recovered_interrupted_run", False), "costs": {k:str(v) for k,v in costs.items()},
            "statuses": dict(Counter(c["status"] for c in budget["calls"])), "cohort": plan.get("cohort", "upstream collection")}
        if (path / "metrics.json").exists():
            report = read(path / "metrics.json")
            entry.update(success=report["success"], checks=report["checks"], changes=report["changes"], f1={})
            for arm, m in report["metrics"].items():
                values = {t:s["micro_f1"]*100 for t,s in m["tasks"].items()}
                entry["f1"][arm] = {**values, "five_head_mean": sum(values.values())/5}
        result[name] = entry
    report = {"experiments": result, "total_actual_attempts": total_calls, "unique_targets": len(targets),
        "deduplicated_costs": {k:str(v) for k,v in charges.items()},
        "notes": ["One canonical archive per logical paid run; original/resumed/recovered copies are not charged twice.",
            "Cached parent calls are not recharged. Unknown reserved costs are not confirmed charges.",
            "All target videos are the same four Training videos; fresh targets do not establish unseen-video generalization."]}
    if all("f1" in result[k] for k in ("confirmation_primary_v2", "confirmation_secondary_v4")):
        report["followup_recommendation"] = {
            "preferred_for_expanded_validation": "confirmation_primary_v2:joint_r1",
            "highest_observed_confirmation_mean_f1": "confirmation_secondary_v4:joint_r2",
            "basis": "First round has fewer label errors and fewer calls; second round raises IVT/mean F1 but adds net label errors relative to first round.",
            "scope": "Observed same-video fresh16 confirmation only; earlier fresh16 cohort had failed and became development data.",
            "default_promoted": False,
        }
    output = ROOT / "artifacts/research/joint_phase_feedback_series_20260910.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "experiments"}, ensure_ascii=False))


if __name__ == "__main__":
    summarize()
