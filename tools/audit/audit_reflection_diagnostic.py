"""Independent saved reflection metrics, diff binding, reference isolation and bills."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(root):
    plan, summary, rows, budget = [read(root / f) for f in
        ("plan.json", "summary.json", "scored_predictions.json", "budget.json")]
    assert budget["stopped"] and len(budget["calls"]) <= plan["max_calls"]
    assert hashlib.sha256((root / "predictions.json").read_bytes()).hexdigest() == summary["prediction_sha256"]
    gallery = plan.get("gallery", [])
    excluded = {s["video_id"] for s in plan["selection"]}
    for ref in gallery:
        assert ref["video_id"] not in excluded
        annotation = Path(ref["annotation_source"])
        raw = read(annotation)
        assert raw["video"]["split"].lower() == "training"
        assert hashlib.sha256(annotation.read_bytes()).hexdigest() == ref["annotation_sha256"]
        assert hashlib.sha256(Path(ref["image_path"]).read_bytes()).hexdigest() == ref["image_sha256"]
        assert any(a["triplet"] == ref["triplet_id"] and a["tool_bbox"] == ref["tool_bbox"]
                   for a in raw["annotations"][str(ref["frame_id"])])
    counts = {}
    for arm in ("h0", "final"):
        counts[arm] = {}
        for task, expected in summary["comparison"]["arms"][arm]["tasks"].items():
            valid = [r for r in rows if r["mask"][task]]
            tp = fp = fn = exact = 0
            for r in valid:
                predicted, gt = set(r[arm][task]), set(r["gt"][task])
                tp += len(predicted & gt)
                fp += len(predicted - gt)
                fn += len(gt - predicted)
                exact += predicted == gt
            assert (tp, fp, fn, exact, len(valid)) == tuple(expected[k] for k in
                ("tp", "fp", "fn", "exact_matches", "valid_targets"))
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            assert f1 == expected["micro_f1"] and exact / len(valid) == expected["exact_set_accuracy"]
            counts[arm][task] = {"f1": f1, "exact": exact / len(valid), "tp": tp, "fp": fp, "fn": fn}
    native = Decimal(0)
    for folder in (root / "calls").iterdir():
        record, request = read(folder / "record.json"), read(folder / "request.json")
        content = request["messages"][0]["content"]
        images = [c for c in content if c.get("type") == "image_url"]
        packet = json.loads(content[0]["text"])
        assert len(images) == len(gallery) + 3 and "gt" not in packet and "query_gt" not in packet
        if gallery:
            assert [r["image_index"] for r in packet["query_images"]] == list(range(len(gallery), len(gallery) + 3))
            assert [r["seconds_relative_to_target"] for r in packet["query_images"]] == [-2, -1, 0]
        result = read(root / "targets" / f"{record['target']}.json")
        if result["status"] == "VALID_PATCH":
            changes = result["raw"]["changes"]
            expected = {f"{q}_{c}": "ADD" if c in result["final"][q] else "REMOVE"
                        for q in ("instrument", "verb", "target", "ivt")
                        for c in set(result["final"][q]) ^ set(result["h0"][q])}
            assert len(changes) == len(expected)
            assert {c["candidate_id"]: c["operation"] for c in changes} == expected
            assert all(len(images) - 1 in c["image_indices"] for c in changes)
            assert all(c["scope"] == "WHOLE_FRAME" for c in changes if c["operation"] == "REMOVE")
        if record["charge_kind"] == "native":
            actual = Decimal(str(read(folder / "response.json")["body"]["usage"]["cost"]))
            assert actual == Decimal(record["charge"])
            native += actual
    assert native == Decimal(summary["native_usd"])
    result = {"passed": True, "independent_masked_counts": counts, "native_usd": str(native),
              "bound_actual_changes": True, "references_exclude_query_videos": True,
              "reference_count": len(gallery), "changes": summary["comparison"]["paired"]["h0_to_final"]}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
