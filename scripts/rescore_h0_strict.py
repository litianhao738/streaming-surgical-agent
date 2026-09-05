"""Independently rescore saved visible SSE JSON against dataset source labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("instrument", "verb", "target", "ivt", "phase")
BOUNDS = {"instrument": 6, "verb": 9, "target": 14, "ivt": 99, "phase": 6}
MAX_ITEMS = {"instrument": 3, "verb": 4, "target": 5, "ivt": 8}
RUNS = ("h0_prompt_refinement_qwen0902_20260906", "h0_schema_confirmation_qwen0902_20260906")
ARMS = ("baseline", "schema_only")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def labels_from_visible_stream(http):
    if http["status_code"] != 200:
        raise ValueError("non-success HTTP response")
    content, endings, models = [], set(), set()
    for line in http["body"].splitlines():
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        event = json.loads(line[5:])
        if event.get("model"):
            models.add(event["model"])
        for choice in event.get("choices", []):
            if choice.get("index") != 0:
                raise ValueError("unexpected completion choice")
            delta = choice.get("delta", {})
            if delta.get("content") is not None:
                if not isinstance(delta["content"], str):
                    raise ValueError("non-text visible content")
                content.append(delta["content"])
            if choice.get("finish_reason"):
                endings.add(choice["finish_reason"])
    if endings != {"stop"} or models != {"qwen/qwen3.8-max-0902"}:
        raise ValueError("incomplete or unexpected model response")
    payload = json.loads("".join(content), object_pairs_hook=unique_object)
    if set(payload) != {*TASKS, "schema_version"} or payload["schema_version"] != "joint_perception_final_only_v1":
        raise ValueError("wrong or incomplete five-head response")
    labels = {}
    for task in TASKS:
        key = "selected_id" if task == "phase" else "selected_ids"
        if not isinstance(payload[task], dict) or set(payload[task]) != {key}:
            raise ValueError("wrong head fields")
        values = [payload[task][key]] if task == "phase" else payload[task][key]
        if (not isinstance(values, list) or len(values) > MAX_ITEMS.get(task, 1)
                or any(type(v) is not int or not 0 <= v <= BOUNDS[task] for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("invalid label IDs or duplicate predictions")
        labels[task] = values
    return labels


def source_truth(dataset, samples):
    raw = {v: read_json(dataset / "Training" / v / f"{v.lower()}.json")["annotations"]
           for v in {s["video_id"] for s in samples} if v != "VID31"}
    actions = read_json(dataset / "Training/VID31/vid31_frame_ivt_repaired.json")["frames"]
    phases = read_json(dataset / "Training/VID31/vid31_phase_repaired.json")["phase_by_track20_image_frame_id"]
    result = {}
    for sample in samples:
        video, frame = sample["video_id"], sample["frame_id"]
        mask, gt = {}, {}
        if video == "VID31":
            action, phase = actions.get(str(frame)), phases.get(str(frame))
            if action and action.get("negative_sentinel_row_count") != 0:
                raise ValueError("ambiguous partial sidecar labels need review")
            for task in TASKS:
                if task == "phase":
                    values = [phase["phase_id"]] if phase else None
                else:
                    key = "triplet_ids" if task == "ivt" else f"{task}_ids"
                    values = action.get(key) if action else None
                mask[task] = isinstance(values, list) and all(type(v) is int and 0 <= v <= BOUNDS[task] for v in values)
                gt[task] = sorted(set(values)) if mask[task] else None
            if action and phase and action["cholect80_phase_id"] != phase["phase_id"]:
                raise ValueError("phase sidecars disagree")
        else:
            instances = raw[video].get(str(frame), [])
            for task in TASKS:
                field = "triplet" if task == "ivt" else task
                values = [i.get(field) for i in instances]
                mask[task] = bool(instances) and all(type(v) is int and 0 <= v <= BOUNDS[task] for v in values)
                gt[task] = sorted(set(values)) if mask[task] else None
            if mask["phase"] and len(gt["phase"]) != 1:
                raise ValueError("conflicting instance phases")
        result[(video, frame)] = {"mask": mask, "gt": gt}
    return result


def score(rows):
    keys = {(r["video_id"], r["frame_id"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate target in cohort")
    metrics = {}
    for task in TASKS:
        valid = [r for r in rows if r["mask"][task]]
        tp = fp = fn = exact = 0
        for row in valid:
            truth = set(row["gt"][task])
            predicted = set(row["prediction"][task]) if row["valid_response"] else set()
            tp += len(truth & predicted)
            fp += len(predicted - truth)
            fn += len(truth - predicted)
            exact += row["valid_response"] and predicted == truth
        metrics[task] = {
            "tp": tp, "fp": fp, "fn": fn, "valid_gt": len(valid),
            "missing_gt": len(rows) - len(valid), "exact_correct": exact,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0.0 if valid else None),
            "set_exact_accuracy": exact / len(valid) if valid else None,
        }
    joint = [r for r in rows if all(r["mask"].values())]
    correct = sum(r["valid_response"] and all(set(r["prediction"][t]) == set(r["gt"][t]) for t in TASKS) for r in joint)
    return {"tasks": metrics, "joint_all_five_exact": {"correct": correct, "valid_gt": len(joint),
            "accuracy": correct / len(joint) if joint else None},
            "valid_responses": sum(r["valid_response"] for r in rows), "targets": len(rows)}


def run(dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    tracked, plans = {}, {}
    for name in RUNS:
        root = ROOT / "artifacts/preflight" / name
        plans[name] = read_json(root / "plan.json")
        for name_, sha in plans[name]["annotation_sources_sha256"].items():
            if digest(Path(name_)) != sha:
                raise ValueError("source annotation changed since inference")
            tracked[name_] = sha
        for path in (root / "plan.json", root / "summary.json", root / "predictions.json"):
            tracked[str(path)] = digest(path)
    samples = [s for p in plans.values() for s in p["samples"]]
    truth = source_truth(dataset, samples)
    rows, prediction_mismatches, gt_mismatches = [], [], []
    for name, plan in plans.items():
        root = ROOT / "artifacts/preflight" / name
        saved = {r["key"]: r for r in read_json(root / "predictions.json")}
        old = read_json(root / "summary.json")
        for item in plan["call_order"]:
            if item["arm"] not in ARMS:
                continue
            path = root / "calls" / item["key"] / "http_response.json"
            tracked[str(path)] = digest(path)
            gt = truth[(item["video_id"], item["frame_id"])]
            try:
                labels = labels_from_visible_stream(read_json(path))
                error = None
            except (ValueError, KeyError, TypeError) as exc:
                labels, error = None, str(exc)
            row = {**item, "run": name, **gt, "prediction": labels, "valid_response": error is None, "error": error}
            rows.append(row)
            if labels != saved[item["key"]].get("selected_ids"):
                prediction_mismatches.append(item["key"])
            previous_gt = next(r for r in old["groups"][item["arm"]]["evaluation"]["frames"]
                               if (r["video_id"], r["frame_id"]) == (item["video_id"], item["frame_id"]))
            if gt["mask"] != previous_gt["mask"] or gt["gt"] != previous_gt["gt"]:
                gt_mismatches.append(item["key"])
    results = {}
    for cohort, included in (("previous16", [RUNS[0]]), ("new24", [RUNS[1]]), ("pooled40", RUNS)):
        results[cohort] = {arm: score([r for r in rows if r["run"] in included and r["arm"] == arm]) for arm in ARMS}
    after = {name: digest(Path(name)) for name in tracked}
    if after != tracked:
        raise ValueError("source changed during read-only rescore")
    result = {"protocol": "Standard per-head micro-F1 and exact-set accuracy. Whole IVT IDs must match; null IDs retained; task-local missing GT masked; invalid responses retained as failures; joint exact uses only fully labeled targets.",
              "new_api_calls": 0, "raw_reconstructed_responses": len(rows),
              "prediction_mismatches_vs_saved": prediction_mismatches, "gt_mismatches_vs_saved": gt_mismatches,
              "source_sha256_before_and_after": tracked, "results": results, "frames": rows}
    (output / "strict_rescore.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "source_sha256_before_and_after")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/h0_strict_rescore_20260906")
    args = parser.parse_args()
    run(args.dataset, args.output)
