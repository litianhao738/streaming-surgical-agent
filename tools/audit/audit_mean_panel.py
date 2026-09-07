"""Offline replay, independent set counting, and native-cost reconciliation."""
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_mean_panel_trial as runner
from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.research.verification.mean_panel import run_arm
from surgical_agent.research.verification.prior_panel import TASKS


def audit(output):
    read = runner.read
    plan, rows = read(output / "plan.json"), read(output / "predictions.json")
    done = read(output / "prediction_completion.json")
    assert done["prediction_sha256"] == sha256_file(output / "predictions.json")
    assert [(r["video_id"], r["frame_id"]) for r in rows] == [
        (s["video_id"], s["frame_id"]) for s in plan["selection"]]
    h0s = {(r["video_id"], r["frame_id"]): r["h0"] for r in read(output / "h0_input.json")}
    mean_roots = [Path(p["path"]) for p in plan["predecessors"]
                  if Path(p["path"]).name.startswith("h0_five_mean_")] + [output]
    stops, rounds = Counter(), Counter()
    patch_schema = read(runner.legacy.CONTRACTS / "repair_response.schema.json")
    for row in rows:
        assert row["h0"] == h0s[row["video_id"], row["frame_id"]]
        records = {r["round"]: r for r in row["arm"]["history"]}
        assert len(records) <= 3
        selected = next(s for s in plan["selection"] if (s["video_id"], s["frame_id"]) ==
                        (row["video_id"], row["frame_id"]))
        # Grounding references come from preserved H0 request image identities.
        refs = []
        for record in records.values():
            for edit in (record.get("patch") or {}).get("edits", []):
                refs.extend(edit["evidence_refs"])
        # Verify references against actual submitted request, never infer from GT.
        requests = [f for root in mean_roots for f in (root / "calls" / selected["key"]).rglob("request.json")]
        for request in requests:
            wire = read(request)["wire"]
            text = wire["messages"][1]["content"][0]["text"]
            assert text.startswith('{"academic_context":')
            packet = json.loads(text)
            actual_refs = [im["ref"] for im in packet["images"]]
            assert set(refs) <= set(actual_refs)
            if "judge" in request.parent.name and packet["round"] == 1:
                assert "current_prediction" not in packet and "your_previous_response" not in packet
        replay = run_arm(row["h0"], lambda n, d, p, records=records: records[n]["responses"],
                         lambda n, s, i, records=records: records[n].get("patch"), patch_schema,
                         sorted(set(refs)), lambda *a: None)
        assert replay == row["arm"]
        stops[row["arm"]["stop_reason"]] += 1
        rounds.update(r["round"] for r in records.values() if r["status"] == "VALID")
    metrics = read(output / "metrics.json")
    independent, edits = {}, {}
    for stage in ("D1", "D2", "D3"):
        scored = read(output / "scored" / f"{stage}.json")
        independent[stage] = {}
        for arm in ("h0", "final"):
            independent[stage][arm] = {}
            for q in (*TASKS, "phase"):
                tp = fp = fn = exact = n = 0
                for r in scored:
                    if not r["mask"][q]:
                        continue
                    predicted, truth = set(r[arm][q]), set(r["gt"][q])
                    n += 1
                    tp += len(predicted & truth)
                    fp += len(predicted - truth)
                    fn += len(truth - predicted)
                    exact += predicted == truth
                values = {"valid_targets": n, "tp": tp, "fp": fp, "fn": fn,
                          "exact_matches": exact, "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
                          "exact_set_accuracy": exact / n if n else None}
                assert all(metrics[stage]["arms"][arm]["tasks"][q][k] == v for k, v in values.items())
                independent[stage][arm][q] = values
        if stage == "D3":
            for q in TASKS:
                counts = Counter({k: 0 for k in ("beneficial_add", "harmful_add", "beneficial_remove", "harmful_remove")})
                for r in scored:
                    if not r["mask"][q]:
                        continue
                    b, a, gt = set(r["h0"][q]), set(r["final"][q]), set(r["gt"][q])
                    for c in a - b:
                        counts["beneficial_add" if c in gt else "harmful_add"] += 1
                    for c in b - a:
                        counts["harmful_remove" if c in gt else "beneficial_remove"] += 1
                edits[q] = dict(counts)
    cost, unknown, calls, categories = Decimal(0), Decimal(0), 0, Counter()
    native_rows = read(output / "budget.json")["calls"]
    for name, r in native_rows.items():
        if r["state"] in {"RESERVED", "NOT_SENT"}:
            continue
        calls += 1
        categories["smoke" if name.startswith("smoke/") else "judge" if "judge" in name else "repair"] += 1
        if r["cost_usd"] is not None:
            cost += Decimal(r["cost_usd"])
        else:
            unknown += Decimal(r["reserve_usd"])
    assert calls <= plan["max_calls"]
    occupied = cost + unknown + Decimal(plan["carryover_liability_usd"])
    assert occupied <= Decimal(plan["total_authorized_usd"])
    phase_cost, phase_unknown, phase_calls, attempted = Decimal(0), Decimal(0), 0, set()
    phase_categories = Counter()
    for root in mean_roots:
        for name, r in read(root / "budget.json")["calls"].items():
            if r["state"] in {"RESERVED", "NOT_SENT"}:
                continue
            phase_calls += 1
            category = "smoke" if name.startswith("smoke/") else "judge" if "judge" in name else "repair"
            phase_categories[category] += 1
            if category != "smoke":
                attempted.add(name.split("/")[0])
            if r["cost_usd"] is None:
                phase_unknown += Decimal(r["reserve_usd"])
            else:
                phase_cost += Decimal(r["cost_usd"])
    assert phase_calls <= 142
    report = {"passed": True, "development_replay": True, "targets": len(rows),
              "valid_rounds": dict(rounds), "stop_reasons": dict(stops), "independent_metrics": independent,
              "final_label_edits": edits, "new_calls": calls, "call_categories": dict(categories),
              "new_settled_usd": str(cost), "new_unknown_reserved_usd": str(unknown),
              "cumulative_occupied_usd": str(occupied),
              "experiment_calls": phase_calls, "experiment_settled_usd": str(phase_cost),
              "experiment_unknown_reserved_usd": str(phase_unknown),
              "experiment_call_categories": dict(phase_categories), "attempted_targets": len(attempted),
              "frame_changes": metrics["D3"]["paired"]["h0_to_final"]["frames"],
              "prediction_sha256": done["prediction_sha256"]}
    with (output / "independent_audit.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "independent_metrics"}), flush=True)


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
