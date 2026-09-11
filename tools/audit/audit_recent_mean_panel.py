"""Independent saved-output audit for the recent five-family mean trial."""
import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

TASKS = ("instrument", "verb", "target", "ivt", "phase")
SEATS = ("grok", "qwen", "gpt", "gemini", "deepseek")


def pooled(rows, arm):
    result = {}
    for task in TASKS:
        tp = fp = fn = exact = valid = 0
        for row in rows:
            if not row["mask"][task]:
                continue
            valid += 1
            truth = set(row["gt"][task])
            pred = set(row[arm][task]) if row[arm] is not None else set()
            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)
            exact += row[arm] is not None and pred == truth
        result[task] = {"valid": valid, "tp": tp, "fp": fp, "fn": fn,
                        "f1": 2 * tp / (2 * tp + fp + fn) if valid and 2 * tp + fp + fn else 0 if valid else None,
                        "accuracy": exact / valid if valid else None}
    return result


def images(body):
    return [part["image_url"]["data_url_sha256"] for msg in body["messages"]
            for part in msg["content"] if isinstance(part, dict) and part.get("type") == "image_url"]


def audit(output):
    completion = read(output / "completion.json")
    plan, budget, states = read(output / "plan.json"), read(output / "budget.json"), read(output / "inference_states.json")
    assert budget["stopped"]
    source_drift = {}
    for path, digest in plan["source_sha256"].items():
        if sha(ROOT / path) != digest:
            source_drift[path] = {"frozen": digest, "current": sha(ROOT / path)}
        assert sha(output / "frozen_source" / path) == digest, path
    selection = {s["key"]: s for s in plan["selection"]}
    assert len(selection) == len(plan["selection"]) > 0
    assert completion["targets"] == len(selection)
    for s in selection.values():
        assert all(s["gt_availability_only"].values())
        for im in s["images"]:
            assert sha(im["path"]) == im["sha256"]
    calls = budget["calls"]
    assert len(calls) == completion["post_calls"] <= plan["max_calls"]
    reference = {k: images(read(output / "h0_preflight" / f"{k}.json")) for k in selection}
    cache_path = output / "cache_hits.json"
    cache_hits = read(cache_path) if cache_path.exists() else []
    for hit in cache_hits:
        folder = Path(hit["source"])
        assert sha(folder / "request.json") == hit["request_sha256"]
        assert sha(folder / "response.json") == hit["response_sha256"]
        assert images(read(folder / "request.json")) == reference[hit["target"]]
    for row in calls:
        folder = output / "calls" / f"{row['index']:03d}_{row['target']}_{row['stage']}_{row['seat']}"
        body = read(folder / "request.json")
        assert images(body) == reference[row["target"]]
        assert body["model"] == (plan["h0"] if row["stage"] == "h0" else plan["proposer"] if row["seat"] == "base"
                                  else plan["models"][row["seat"]])
        if row["seat"] != "base":
            packet = json.loads(body["messages"][0]["content"][0]["text"])
            assert next(iter(packet)) == "academic_context"
            assert not {"gt", "ground_truth", "current_prediction", "h0"} & set(packet)
    rounds = {}
    previous_by_arm = {}
    for n in range(1, completion["round_snapshots_written"] + 1):
        snapshot = read(output / f"round_{n}_predictions.json")
        saved = read(output / "scores" / f"round_{n}.json")
        assert sha(output / f"round_{n}_predictions.json") == saved["prediction_sha256"]
        arms = {}
        for arm in ("primary", "shadow"):
            truth = read(output / "scores" / f"round_{n}_{arm}_truth.json")
            expected = saved["reports"][arm]
            metrics = pooled(truth, "final")
            for t, m in metrics.items():
                official = expected["arms"]["final"]["tasks"][t]
                assert (m["tp"], m["fp"], m["fn"], m["valid"]) == tuple(official[k] for k in ("tp", "fp", "fn", "valid_targets"))
                assert m["f1"] == official["micro_f1"] and m["accuracy"] == official["exact_set_accuracy"]
            edits, per_target = {t: Counter() for t in TASKS}, []
            previous = previous_by_arm.get(arm, {})
            for row in truth:
                key = f"{row['video_id']}_{row['frame_id']}"
                before = previous.get(key, row["h0"])
                after = row["final"]
                target_edits = {}
                if before is not None and after is not None:
                    for t in TASKS:
                        if not row["mask"][t]:
                            continue
                        gt, left, right = set(row["gt"][t]), set(before[t]), set(after[t])
                        added, removed = right - left, left - right
                        beneficial, harmful = len(added & gt) + len(removed - gt), len(added - gt) + len(removed & gt)
                        edits[t].update(beneficial=beneficial, harmful=harmful, added=len(added), removed=len(removed))
                        if added or removed:
                            target_edits[t] = {"added": sorted(added), "removed": sorted(removed),
                                               "beneficial": beneficial, "harmful": harmful}
                per_target.append({"key": key, "edits_from_previous_round": target_edits})
            previous_by_arm[arm] = {f"{r['video_id']}_{r['frame_id']}": r["final"] for r in truth}
            arms[arm] = {"metrics": metrics, "changes_from_h0": expected["paired"]["h0_to_final"],
                         "label_edits_from_previous_round": edits, "targets": per_target}
        errors, conflicts, coverage = Counter(), 0, {t: {"gt": 0, "pool_hits": 0} for t in TASKS[:4]}
        rating_quality = {t: Counter() for t in TASKS[:4]}
        truth_by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in truth}
        for key, state in states.items():
            records = [r for r in state["history"] if r["round"] <= n]
            if not records:
                continue
            record = records[-1]
            if record["round"] == n:
                for pid, diagnostic in record["diagnostics"].items():
                    vals = diagnostic["scores"]
                    independent_mean = None if any(v is None for v in vals) else sum(vals) / 5
                    assert record["means"][pid] == independent_mean
                    errors.update(f"{seat}:{error}" for seat, error in diagnostic["invalid"].items())
                    conflicts += diagnostic["explicit_conflict"]
            row = truth_by_key[key]
            for p in record["pool"]["propositions"]:
                task, pid = p["task"], p["id"]
                if not row["mask"][task]:
                    continue
                mean = record["means"][pid]
                correct = p["label_id"] in row["gt"][task]
                rating_quality[task]["pool_true" if correct else "pool_false"] += 1
                if mean is None:
                    rating_quality[task]["invalid_true" if correct else "invalid_false"] += 1
                elif mean >= 4:
                    rating_quality[task]["high_score_true" if correct else "high_score_false"] += 1
                elif mean <= 2:
                    rating_quality[task]["low_score_true" if correct else "low_score_false"] += 1
                else:
                    rating_quality[task]["uncertain_true" if correct else "uncertain_false"] += 1
            for t in TASKS[:4]:
                if row["mask"][t]:
                    wanted = set(row["gt"][t])
                    available = {p["label_id"] for p in record["pool"]["propositions"] if p["task"] == t}
                    coverage[t]["gt"] += len(wanted)
                    coverage[t]["pool_hits"] += len(wanted & available)
        stage_rows = [r for r in calls if r["stage"].endswith(f"_{n}") or n == 1 and r["stage"] == "h0"]
        rounds[str(n)] = {"arms": arms, "actual_target_reviews": sum(r["reviewed_this_round"] for r in snapshot),
                          "candidate_coverage": coverage, "invalid_items_by_seat": errors, "conflicting_items": conflicts,
                          "rating_quality_primary_threshold": rating_quality,
                          "costs": {k: str(sum(Decimal(r["charge"]) for r in stage_rows if r["account"] == k)) for k in budget["limits"]},
                          "post_calls": len(stage_rows)}
    h0_truth = read(output / "scores/round_1_primary_truth.json")
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    _, reread_truth = score_saved(adapter, h0_truth)
    assert reread_truth == h0_truth, "fresh GT/mask differs from saved scoring input"
    summary = {"verified": True, "plan_sha256": sha(output / "plan.json"), "h0_metrics": pooled(h0_truth, "h0"),
               "rounds": rounds, "completion": completion, "models": plan["models"],
               "successful_calls_reused": len(cache_hits),
               "current_source_differences": source_drift,
               "call_statuses": dict(Counter(r["status"] for r in calls)),
               "http_statuses": dict(Counter(str(r.get("http_status")) for r in calls)),
               "audited_statuses": {k: "EMPTY_POOL_UNVERIFIED" if s["status"] == "MODEL_PASS"
                                    and not s["pool"]["propositions"] else s["status"] for k, s in states.items()},
               "model_pass_is_not_gt_accuracy": True, "shadow_is_not_separate_closed_loop": True}
    save(output / "independent_audit.json", summary)
    print(json.dumps({"verified": True, "calls": len(calls), "h0": summary["h0_metrics"],
                      "rounds": {n: {a: d["metrics"] for a, d in r["arms"].items()} for n, r in rounds.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
