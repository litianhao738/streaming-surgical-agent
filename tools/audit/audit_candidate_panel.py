"""Independent arithmetic / label / billing audit of a saved candidate trial."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(root):
    predictions = read(root / "predictions.json")
    summary = read(root / "summary.json")
    truth = read(root / "scored_predictions.json")
    budget = read(root / "budget.json")
    assert hashlib.sha256((root / "predictions.json").read_bytes()).hexdigest() == summary["prediction_sha256"]
    rounds, images, billed = 0, {}, {k: Decimal(0) for k in budget["limits"]}
    for row in predictions:
        key = f"{row['video_id']}_{row['frame_id']}"
        result = read(root / "targets" / key / "result.json")
        assert result["h0"] == row["h0"] and result["final"] == row["final"]
        assert len(result["history"]) <= 3
        for history in result["history"]:
            if history["status"] != "VALID":
                continue
            assert set(history["raw"]) == {"grok", "qwen", "gpt", "gemini", "deepseek"}
            for candidate in history["pool"]["propositions"]:
                pid = candidate["id"]
                scores = [r["scores"][pid] for r in history["raw"].values()]
                assert all(type(s) is int and 1 <= s <= 5 for s in scores)
                assert sum(scores)/5 == history["means"][pid]
            rounds += 1
    for folder in sorted((root / "calls").iterdir()):
        record = read(folder / "record.json")
        request = read(folder / "request.json")
        target = record["target"]
        content = request["messages"][0]["content"]
        packet = json.loads(content[0]["text"])
        assert next(iter(packet)) == "academic_context"
        assert not ({"gt", "ground_truth", "task_masks", "annotation"} & set(packet))
        identity = [p["image_url"]["data_url_sha256"] for p in content[1:]]
        assert len(identity) == 3
        assert identity == images.setdefault(target, identity)
        usage = read(folder / "response.json").get("body", {}).get("usage", {}) if (folder / "response.json").exists() else {}
        usage = usage or {}
        if record["charge_kind"] == "native":
            cost = (Decimal(usage["cost_in_usd_ticks"])/Decimal(10**10)
                    if record["seat"] == "grok" else Decimal(str(usage["cost"])))
            assert cost == Decimal(record["charge"])
            billed[record["account"]] += cost
    for account, amount in billed.items():
        assert amount == Decimal(summary["native_costs"][account])
        occupied = Decimal(budget.get("carried_occupied", {}).get(account, "0")) + sum(
            Decimal(r["charge"]) for r in budget["calls"] if r["account"] == account)
        assert occupied == Decimal(budget["occupied"][account])
    for arm, field in [("h0", "h0_metrics"), ("final", "final_metrics")]:
        for task, expected in summary[field]["tasks"].items():
            valid = [r for r in truth if r["mask"][task]]
            tp = fp = fn = exact = 0
            for r in valid:
                pred, gt = set(r[arm][task]), set(r["gt"][task])
                tp += len(pred & gt)
                fp += len(pred-gt)
                fn += len(gt-pred)
                exact += pred == gt
            assert (tp, fp, fn, len(valid), exact) == tuple(expected[k] for k in
                       ("tp", "fp", "fn", "valid_targets", "exact_matches"))
            f1 = (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0) if valid else None
            assert f1 == expected["micro_f1"]
            assert (exact/len(valid) if valid else None) == expected["exact_set_accuracy"]
    result = {"passed": True, "valid_five_model_rounds": rounds,
              "post_calls": len(budget["calls"]), "same_causal_images_across_calls": True,
              "independent_masked_counts_match": True, "native_billing_matches_raw": True}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
