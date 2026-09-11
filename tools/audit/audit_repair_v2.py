"""Read-only independent metrics/raw-GT/billing audit for the repair-v2 trial."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def audit(root):
    summary, predictions, truth, budget = [read(root / name) for name in
        ("summary.json", "predictions.json", "scored_predictions.json", "budget.json")]
    assert hashlib.sha256((root / "predictions.json").read_bytes()).hexdigest() == summary["prediction_sha256"]
    assert budget["stopped"] and len(budget["calls"]) <= 20
    tables, edits = {}, {}
    for arm, rows in truth.items():
        tables[arm], edits[arm] = {}, {"correct_add": 0, "wrong_add": 0, "correct_delete": 0, "wrong_delete": 0}
        for task, metrics in summary["metrics"][arm]["tasks"].items():
            valid = [r for r in rows if r["mask"][task]]
            tp = fp = fn = exact = 0
            for r in valid:
                gt, pred, h0 = set(r["gt"][task]), set(r["final"][task]), set(r["h0"][task])
                tp += len(gt & pred)
                fp += len(pred - gt)
                fn += len(gt - pred)
                exact += gt == pred
                edits[arm]["correct_add"] += len((pred - h0) & gt)
                edits[arm]["wrong_add"] += len((pred - h0) - gt)
                edits[arm]["correct_delete"] += len((h0 - pred) - gt)
                edits[arm]["wrong_delete"] += len((h0 - pred) & gt)
            assert (tp, fp, fn, exact, len(valid)) == tuple(metrics[k] for k in
                   ("tp", "fp", "fn", "exact_matches", "valid_targets"))
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            assert f1 == metrics["micro_f1"] and exact / len(valid) == metrics["exact_set_accuracy"]
            tables[arm][task] = {"f1": f1, "exact": exact / len(valid), "tp": tp, "fp": fp, "fn": fn}
    raw_rows, causes = [], []
    for p, r in zip(predictions, truth["expanded_quorum"], strict=True):
        assert p["key"] == f"{r['video_id']}_{r['frame_id']}"
        path = Path("D:/cholec_dataset/Training") / r["video_id"] / f"{r['video_id'].lower()}.json"
        annotations = read(path)["annotations"][str(r["frame_id"])]
        for task in r["gt"]:
            field = "triplet" if task == "ivt" else task
            assert set(r["gt"][task]) == {a[field] for a in annotations}
        raw_rows.append({"key": p["key"], "source": str(path), "annotations": annotations})
        for prop in p["pool"]["propositions"]:
            pid = prop["id"]
            records = p["quorum_reviews"]
            valid = [s for s, v in records.items() if pid not in v["errors"]]
            values = [records[s]["scores"][pid] for s in valid]
            positive, negative = sum(v >= 4 for v in values), sum(v <= 2 for v in values)
            avg = sum(values) / len(values) if values else None
            blocked = len(valid) < 4 or (positive and negative) or (
                avg is not None and ((avg >= 4 and positive < 3) or (avg <= 2 and negative < 3)))
            assert p["quorum_means"][pid] == (3 if blocked else avg)
        for task in ("instrument", "verb", "target", "ivt"):
            pool_ids = {v["label_id"] for v in p["pool"]["propositions"] if v["task"] == task}
            for missing in set(r["gt"][task]) - set(r["final"][task]):
                pid = f"{task}_{missing}"
                diagnostic = p["quorum_reviews"]["gpt"].get("quorum", {}).get(pid)
                reason = ("NOT_IN_CANDIDATE_POOL" if missing not in pool_ids else
                          diagnostic["blocked_reason"] if diagnostic["blocked_reason"] else
                          "COMPONENT_DEPENDENCY_OR_OUTPUT_CAP" if p["quorum_means"][pid] >= 4 else
                          "LOW_VISUAL_SUPPORT")
                causes.append({"key": p["key"], "task": task, "label_id": missing, "reason": reason,
                               "review": diagnostic})
    images, native = {}, {k: Decimal(0) for k in budget["occupied"]}
    for folder in sorted((root / "calls").iterdir()):
        record, request = read(folder / "record.json"), read(folder / "request.json")
        blocks = request["messages"][0]["content"]
        identity = [b["image_url"]["data_url_sha256"] for b in blocks if b.get("type") == "image_url"]
        assert identity == images.setdefault(record["target"], identity) and len(identity) == 3
        packet = json.loads(blocks[0]["text"])
        assert not ({"gt", "current_prediction", "annotation", "task_masks"} & set(packet))
        if record["charge_kind"] == "native":
            usage = read(folder / "response.json")["body"]["usage"]
            amount = Decimal(str(usage["cost_in_usd_ticks"])) / Decimal(10**10) if record["seat"] == "grok" else Decimal(str(usage["cost"]))
            assert amount == Decimal(record["charge"])
            native[record["account"]] += amount
    for account, amount in native.items():
        assert amount == Decimal(summary["native_costs"][account])
        assert Decimal(budget["occupied"][account]) == Decimal(budget["carried_occupied"][account]) + sum(
            Decimal(r["charge"]) for r in budget["calls"] if r["account"] == account)
    output = {"passed": True, "raw_gt_matches_scored_gt": True, "independent_counts_match": True,
              "same_causal_images": True, "native_billing_matches": True, "tables": tables,
              "label_edits": edits, "miss_causes": dict(Counter(c["reason"] for c in causes)),
              "miss_details": causes}
    (root / "raw_gt_audit.json").write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
    (root / "independent_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return {k: v for k, v in output.items() if k != "miss_details"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
