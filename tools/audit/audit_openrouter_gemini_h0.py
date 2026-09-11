"""Compare the saved Gemini H0 wire to the historical Qwen H0, plus masked scores."""
import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from audit_semantic_candidate import audit, read


def check(root, reference):
    common = audit(root)
    plan, budget = read(root / "plan.json"), read(root / "budget.json")
    reference_plan = read(reference / "plan.json")
    expected_keys = [s["key"] for s in plan["selection"]]
    assert expected_keys == [s["key"] for s in reference_plan["selection"]]
    old_calls = {}
    for folder in (reference / "calls").iterdir():
        record = read(folder / "record.json")
        if record["stage"] == "h0":
            old_calls[record["target"]] = read(folder / "request.json")
    h0_targets, stages, native_by_stage = [], Counter(), {}
    for folder in sorted((root / "calls").iterdir()):
        record, request = read(folder / "record.json"), read(folder / "request.json")
        stages[record["stage"]] += 1
        if record["charge_kind"] == "native":
            key = f"{record['stage']}:{record['account']}"
            native_by_stage[key] = native_by_stage.get(key, Decimal(0)) + Decimal(record["charge"])
        if record["seat"] == "base":
            assert request["model"] == plan["h0"] == "google/gemini-3.8-flash"
            assert request["provider"] == {"only": ["google-ai-studio"], "allow_fallbacks": False,
                                           "require_parameters": True}
            assert record["account"] == "openrouter_usd"
            response = read(folder / "response.json")["body"]
            assert response["model"] == request["model"] and response["provider"] == "Google AI Studio"
        if record["stage"] != "h0":
            continue
        key = record["target"]
        h0_targets.append(key)
        assert request == read(root / "h0_preflight" / f"{key}.json")
        old_request = old_calls[key]
        assert request.pop("model") != old_request.pop("model")
        assert request["provider"].pop("only") == ["google-ai-studio"]
        assert old_request["provider"].pop("only") == ["alibaba"]
        assert request == old_request  # Prompt, schema, images, detail and parameters identical.
    assert h0_targets == expected_keys
    new_truth, old_truth = read(root / "scored_predictions.json"), read(reference / "scored_predictions.json")
    assert [(r["video_id"], r["frame_id"], r["mask"], r["gt"]) for r in new_truth] == [
        (r["video_id"], r["frame_id"], r["mask"], r["gt"]) for r in old_truth]
    old_metrics = {}
    for task in ("instrument", "verb", "target", "ivt", "phase"):
        valid = [r for r in old_truth if r["mask"][task]]
        tp = fp = fn = exact = 0
        for row in valid:
            p, g = set(row["h0"][task]), set(row["gt"][task])
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
            exact += p == g
        old_metrics[task] = {"f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                             "exact": exact / len(valid) if valid else None, "valid": len(valid)}
    summary = read(root / "summary.json")
    result = {"passed": True, "historical_qwen": old_metrics, **common,
              "h0_wire_diff_only_model_and_route": True, "same_gt_masks": True,
              "call_stages": dict(stages), "native_by_stage": {k: str(v) for k, v in native_by_stage.items()},
              "repair_changes": summary["comparison"]["paired"],
              "native_costs": summary["native_costs"],
              "estimated_aliyun_cny": str(sum(Decimal(r["charge"]) for r in budget["calls"]
                                             if r["charge_kind"] == "conservative_estimate"))}
    (root / "h0_switch_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {k: v for k, v in result.items() if k not in ("actual_rounds", "validation_errors", "repair_changes")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/preflight/semantic_complete_gt_20260908_v1"))
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.reference)))
