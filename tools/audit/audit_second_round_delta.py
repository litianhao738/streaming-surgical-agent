"""Independent arithmetic and saved-wire audit for two cached-R1 continuations."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    plan, budget, summary = (read(root / f"{name}.json") for name in ("plan", "budget", "summary"))
    assert budget["stopped"] and len(budget["calls"]) <= plan["max_calls"]
    assert all(sha(Path(p)) == h for p, h in plan["shared_input_sha256"].items())
    tables, truths = {}, {}
    for name in ("round1", "original_round2", "delta_round2"):
        assert sha(root / f"{name}_predictions.json") == summary["prediction_sha256"][name]
        rows = read(root / f"{name}_scored.json")
        truths[name] = rows
        tables[name] = {}
        for arm, data in summary["comparisons"][name]["arms"].items():
            tables[name][arm] = {}
            for task, metrics in data["tasks"].items():
                valid = [r for r in rows if r["mask"][task]]
                tp = fp = fn = exact = 0
                for row in valid:
                    prediction = (row["h0"] if row["h1"] is None else row["h1"]) if arm == "h1_policy" else row[arm]
                    p, g = set(prediction[task]), set(row["gt"][task])
                    tp += len(p & g)
                    fp += len(p - g)
                    fn += len(g - p)
                    exact += p == g
                assert (tp, fp, fn, exact, len(valid)) == tuple(metrics[k] for k in
                    ("tp", "fp", "fn", "exact_matches", "valid_targets"))
                f1 = (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0) if valid else None
                accuracy = exact / len(valid) if valid else None
                assert f1 == metrics["micro_f1"] and accuracy == metrics["exact_set_accuracy"]
                tables[name][arm][task] = {"f1": f1, "exact": accuracy, "valid": len(valid)}
    source = Path(plan["source"])
    source_truth = read(source / "scored_predictions.json")
    identity = lambda rows: [(r["video_id"], r["frame_id"], r["h0"], r["gt"], r["mask"]) for r in rows]
    assert all(identity(rows) == identity(source_truth) for rows in truths.values())
    calls, native, counts = Counter(), {k: Decimal(0) for k in budget["limits"]}, []
    for folder in sorted((root / "calls").iterdir()):
        record, request = read(folder / "record.json"), read(folder / "request.json")
        assert record["stage"] != "h0"
        calls[record["stage"]] += 1
        packet = json.loads(request["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        assert not {"gt", "ground_truth", "task_masks", "query_gt"} & set(packet)
        source_request = read(source / "h0_preflight" / f"{record['target']}.json")
        images = lambda body: [b["image_url"]["data_url_sha256"] for m in body["messages"]
            if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "image_url"]
        assert images(request) == images(source_request)
        if "review" in record["stage"]:
            assert "current_prediction" not in packet and "issues" not in packet
        if record["stage"] == "llm_delta_2":
            assert set(packet["response_schema"]["properties"]) == {"changes"}
            assert packet["full_ontology"]
        if record["charge_kind"] == "native":
            response = read(folder / "response.json")["body"]
            charge = Decimal(str(response["usage"]["cost_in_usd_ticks"])) / Decimal(10**10) if record["seat"] == "grok" else Decimal(str(response["usage"]["cost"]))
            assert charge == Decimal(record["charge"])
            native[record["account"]] += charge
        if record["seat"] == "base" and record["status"] == "JSON_PARSED":
            response = read(folder / "response.json")["body"]
            assert request["provider"]["only"] == ["google-ai-studio"]
            assert response["provider"] == "Google AI Studio" and response["model"] == "google/gemini-3.8-flash"
    for account, amount in native.items():
        assert amount == Decimal(summary["native_costs"][account])
        assert Decimal(budget["occupied"][account]) == Decimal(budget["carried_occupied"][account]) + sum(
            Decimal(r["charge"]) for r in budget["calls"] if r["account"] == account)
    for s in plan["selection"]:
        folder = root / "targets" / s["key"]
        classic = read(folder / "original_result.json")
        cached = read(source / "targets" / s["key"] / "result.json")
        assert classic["history"][0]["raw"] == cached["history"][0]["raw"]
        assert classic["snapshots"][0] == cached["final"]
        assert len(classic["history"]) <= 2
        delta = read(folder / "delta_result.json")
        if "compiled" not in delta:
            assert delta["final"] == cached["final"]
            counts.append({"target": s["key"], "original_actual_rounds": len(classic["history"]),
                           "delta_status": delta["status"], "proposed_edits": None, "accepted_edits": 0,
                           "targeted": 0, "full_pool": None})
            continue
        compiled = delta["compiled"]
        expected = {f"{q}_{c}": "ADD" if c in compiled["tentative"][q] else "REMOVE"
                    for q in ("instrument", "verb", "target", "ivt")
                    for c in set(compiled["current"][q]) ^ set(compiled["tentative"][q])}
        assert {e["candidate_id"]: e["operation"] for e in compiled["changes"]} == expected
        actual = {f"{q}_{c}": "ADD" if c in delta["final"][q] else "REMOVE"
                  for q in ("instrument", "verb", "target", "ivt")
                  for c in set(compiled["current"][q]) ^ set(delta["final"][q])}
        assert all(pid in expected and expected[pid] == op for pid, op in actual.items())
        assert all(2 in e["image_indices"] for e in compiled["changes"])
        assert compiled["current"]["phase"] == delta["final"]["phase"]
        if delta["status"] == "REVIEWED":
            for pid, mean in delta["means"].items():
                opinions = list(delta["normalized"].values())
                ratings = [o["scores"][pid] for o in opinions]
                blocked = any(pid in o["errors"] for o in opinions) or (min(ratings) <= 2 and max(ratings) >= 4)
                assert mean == (3 if blocked else sum(ratings)/5)
            assert all(delta["means"][pid] >= 4 if op == "ADD" else delta["means"][pid] <= 2 for pid, op in actual.items())
        counts.append({"target": s["key"], "original_actual_rounds": len(classic["history"]),
                       "delta_status": delta["status"], "proposed_edits": len(expected), "accepted_edits": len(actual),
                       **delta["review_item_counts"]})
    result = {"passed": True, "new_h0_calls": 0, "same_h0_gt_masks_and_images": True, "metrics": tables,
              "call_stages": dict(calls), "targets": counts, "native_costs": summary["native_costs"],
              "estimated_aliyun_cny": summary["estimated_aliyun_cny"],
              "changes": {name: report["paired"] for name, report in summary["comparisons"].items()}}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {k: v for k, v in result.items() if k != "changes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
