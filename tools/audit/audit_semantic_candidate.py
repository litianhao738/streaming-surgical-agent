"""Independent saved-artifact arithmetic, binding, billing and masked edit audit."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(root):
    summary, budget, plan = [read(root / name) for name in ("summary.json", "budget.json", "plan.json")]
    assert budget["stopped"] and len(budget["calls"]) <= plan["max_calls"]
    assert hashlib.sha256((root / "predictions.json").read_bytes()).hexdigest() == summary["prediction_sha256"]
    truth = read(root / "scored_predictions.json")
    compare_old = plan.get("compare_old", True)
    shared_pool = "ablation" not in plan and compare_old
    tables = {}
    arms = [
        ("h0", truth, "h0", summary["comparison"]["arms"]["h0"]),
        ("semantic_final", truth, "final", summary["comparison"]["arms"]["final"]),
    ]
    if compare_old:
        arms.append(("old_first", read(root / "old_scored_predictions.json"), "final",
                     summary["old_comparison"]["arms"]["final"]))
    for name, rows, field, expected in arms:
        tables[name] = {}
        for task, metrics in expected["tasks"].items():
            valid = [r for r in rows if r["mask"][task]]
            tp = fp = fn = exact = 0
            for r in valid:
                p, g = set(r[field][task]), set(r["gt"][task])
                tp += len(p & g)
                fp += len(p - g)
                fn += len(g - p)
                exact += p == g
            assert (tp, fp, fn, len(valid), exact) == tuple(metrics[k] for k in
                ("tp", "fp", "fn", "valid_targets", "exact_matches"))
            f1 = (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0) if valid else None
            assert metrics["micro_f1"] == f1
            assert metrics["exact_set_accuracy"] == (exact / len(valid) if valid else None)
            tables[name][task] = {"f1": f1, "exact": metrics["exact_set_accuracy"], "valid": len(valid)}
    bound_images, native = {}, {k: Decimal(0) for k in budget["limits"]}
    for folder in sorted((root / "calls").iterdir()):
        record, request = read(folder / "record.json"), read(folder / "request.json")
        images = [b["image_url"]["data_url_sha256"] for m in request["messages"]
                  if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "image_url"]
        assert len(images) == 3 and images == bound_images.setdefault(record["target"], images)
        if record["stage"] != "h0":
            packet = json.loads(request["messages"][0]["content"][0]["text"])
            assert next(iter(packet)) == "academic_context"
            assert not ({"gt", "ground_truth", "task_masks", "annotation"} & set(packet))
            if record["stage"].startswith(("old_", "semantic_")):
                assert "current_prediction" not in packet
        if record["stage"].startswith("semantic_") and record["status"] == "JSON_PARSED":
            wire = json.loads(read(folder / "response.json")["body"]["choices"][0]["message"]["content"])
            round_no = int(record["stage"].rsplit("_", 1)[1]) if record["stage"].rsplit("_", 1)[1].isdigit() else 1
            history = read(root / "targets" / record["target"] / f"round_{round_no}.json")
            if record["seat"] == "gemini" and isinstance(wire, dict) and set(wire) == {"rows"}:
                identifiers = [row.get("candidate_id") for row in wire["rows"]]
                wanted = {p["id"] for p in history["pool"]["propositions"]}
                if len(set(identifiers)) == len(identifiers) and set(identifiers) == wanted:
                    wire = {"judgments": {row["candidate_id"]: {k: v for k, v in row.items() if k != "candidate_id"}
                                          for row in wire["rows"]}}
            assert wire == history["raw"][record["seat"]]
        if record["charge_kind"] == "native":
            usage = read(folder / "response.json")["body"]["usage"]
            billed = (Decimal(str(usage["cost_in_usd_ticks"])) / Decimal(10**10)
                      if record["seat"] == "grok" else Decimal(str(usage["cost"])))
            assert billed == Decimal(record["charge"])
            native[record["account"]] += billed
    for account, amount in native.items():
        assert amount == Decimal(summary["native_costs"][account])
        assert Decimal(budget["occupied"][account]) == Decimal(budget["carried_occupied"][account]) + sum(
            Decimal(r["charge"]) for r in budget["calls"] if r["account"] == account)
    errors, blocks, rounds, opportunities = Counter(), Counter(), [], []
    seats = {"grok", "qwen", "gpt", "gemini", "deepseek"}
    for row in truth:
        key = f"{row['video_id']}_{row['frame_id']}"
        result = read(root / "targets" / key / "result.json")
        # Raw H0 may emit a different ordering of the same multi-label set.
        assert all(set(result[arm][task]) == set(row[arm][task])
                   for arm in ("h0", "final") for task in row[arm])
        assert len(result["history"]) <= 3
        for history in result["history"]:
            if history["status"] != "VALID":
                continue
            assert set(history["raw"]) == seats
            fully_valid = 0
            for pid in history["means"]:
                clean = history["normalized"]
                invalid = any(pid in clean[s]["errors"] for s in seats)
                values = [clean[s]["scores"][pid] for s in seats]
                conflict = not invalid and min(values) <= 2 and max(values) >= 4
                expected = 3 if invalid or conflict else sum(values) / 5
                assert history["means"][pid] == expected
                fully_valid += not invalid
                if invalid or conflict:
                    blocks["INVALID_EVIDENCE" if invalid else "EXPLICIT_CONFLICT"] += 1
                for s in seats:
                    if pid in clean[s]["errors"]:
                        errors[f"{s}:{clean[s]['errors'][pid]}"] += 1
            rounds.append({"target": key, "round": history["round"], "candidates": len(history["means"]),
                           "all_five_valid_items": fully_valid})
        if not result["history"]:
            continue
        if not shared_pool:
            continue
        first = result["history"][0]
        old = read(root / "targets" / key / "old_review.json")
        assert first["pool"] == old["pool"]
        for p in first["pool"]["propositions"]:
            task, c = p["task"], p["label_id"]
            if not row["mask"][task]:
                continue
            present = c in row["h0"][task]
            correct = c in row["gt"][task]
            opportunities.append({"target": key, "task": task, "id": c,
                "edit": "remove" if present else "add", "beneficial": present != correct,
                "old_accepted": (c in old["final"][task]) != present,
                "new_accepted": (c in result["snapshots"][0][task]) != present})
    admission = {}
    for good in (True, False):
        eligible = [r for r in opportunities if r["beneficial"] == good]
        admission["beneficial" if good else "harmful"] = {"opportunities": len(eligible),
            **{arm: {"accepted": sum(r[arm + "_accepted"] for r in eligible),
                      "rate": sum(r[arm + "_accepted"] for r in eligible) / len(eligible) if eligible else None}
               for arm in ("old", "new")}}
    report = {"passed": True, "same_causal_images": True, "same_first_pool": shared_pool,
              "independent_masked_counts_match": True, "native_billing_matches": True,
              "metrics": tables, "actual_rounds": rounds, "validation_errors": dict(errors),
              "blocked_candidates": dict(blocks), "first_round_edit_admission": admission if shared_pool else None,
              "opportunities": opportunities}
    (root / "independent_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "opportunities"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
