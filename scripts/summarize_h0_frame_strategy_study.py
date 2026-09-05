"""Offline paired analysis, including shared successful targets and wire audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_h0_frame_strategy_study import OFFSETS, TASKS, VIDEOS
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file


def metrics(frames, task):
    tp = fp = fn = exact = valid = 0
    for row in frames:
        if not row["mask"][task]:
            continue
        valid += 1
        truth = set(row["gt"][task])
        prediction = set(row["h0"][task]) if row["status"] == "OK" else set()
        tp += len(prediction & truth)
        fp += len(prediction - truth)
        fn += len(truth - prediction)
        exact += row["status"] == "OK" and prediction == truth
    return {"valid": valid, "exact": exact, "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0}


def run(output):
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] not in {"COMPLETE_WITH_PRIOR_LOST_RESPONSE", "STOPPED"}:
        raise ValueError("wait until predictions and offline scoring finish")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    if sha256_file(output / "predictions.json") != summary["predictions_sha256_before_scoring"]:
        raise ValueError("predictions changed after scoring")
    frames = {group: summary["groups"][group]["evaluation"]["frames"] for group in OFFSETS}
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    pilot_targets = {(item["video_id"], item["frame_id"]) for item in plan["call_order"][:128]}
    frames = {group: [row for row in values if (row["video_id"], row["frame_id"]) in pilot_targets]
              for group, values in frames.items()}
    primary = {group: {task: metrics(values, task) for task in TASKS} for group, values in frames.items()}
    per_video = {video: {group: {task: metrics([row for row in values if row["video_id"] == video], task)
                               for task in TASKS} for group, values in frames.items()} for video in VIDEOS}
    shared = set.intersection(*({(row["video_id"], row["frame_id"]) for row in values if row["status"] == "OK"}
                                for values in frames.values()))
    paired = {group: {task: metrics([row for row in values if (row["video_id"], row["frame_id"]) in shared], task)
                      for task in TASKS} for group, values in frames.items()}
    differences = {}
    index = {group: {(row["video_id"], row["frame_id"]): row for row in values} for group, values in frames.items()}
    for group in ("A", "C", "D"):
        differences[group + "_vs_B"] = {}
        for task in TASKS:
            wins = losses = ties = changed = 0
            for key in shared:
                left, right = index["B"][key], index[group][key]
                if not left["mask"][task] or not right["mask"][task]:
                    continue
                truth = set(left["gt"][task])
                assert truth == set(right["gt"][task])
                compared_sets = [set(row["h0"][task]) for row in (left, right)]
                f1 = [2 * len(pred & truth) / (len(pred) + len(truth)) if pred or truth else 1 for pred in compared_sets]
                wins += f1[1] > f1[0]
                losses += f1[1] < f1[0]
                ties += f1[1] == f1[0]
                changed += compared_sets[0] != compared_sets[1]
            differences[group + "_vs_B"][task] = {"frame_f1_wins": wins, "losses": losses, "ties": ties,
                                                     "changed_sets": changed}
    image_audit = json.loads((output / "input_image_audit.json").read_text(encoding="utf-8"))
    wire_errors, image_count, wire_count = [], 0, 0
    for item in plan["call_order"]:
        path = output / "calls" / item["key"] / "wire_request.json"
        if not path.exists():
            continue
        wire_count += 1
        wire = json.loads(path.read_text(encoding="utf-8"))
        image_parts = [part["image_url"] for message in wire["messages"] if isinstance(message["content"], list)
                       for part in message["content"] if part.get("type") == "image_url"]
        expected = plan["requests"][item["key"]]["images"]
        if len(image_parts) != len(expected):
            wire_errors.append({"key": item["key"], "reason": "image_count"})
            continue
        for part, source in zip(image_parts, expected, strict=True):
            image_count += 1
            if part["data_url_sha256"] != image_audit["expected_images"][source["identifier"]]["data_url_sha256"]:
                wire_errors.append({"key": item["key"], "reason": "image_hash"})
        if [part.get("detail") for part in image_parts] != ["low"] * (len(expected) - 1) + ["high"]:
            wire_errors.append({"key": item["key"], "reason": "image_detail"})
        if wire["model"] != plan["model"]:
            wire_errors.append({"key": item["key"], "reason": "model"})
    analysis = {"pilot_targets": len(pilot_targets), "pilot_metrics": primary, "pilot_per_video": per_video,
                "full_plan_targets": 80,
                "attempted_predictions": sum(row["status"] != "NOT_ATTEMPTED" for row in predictions),
                "successful_predictions": summary["successful_predictions"],
                "not_attempted_predictions": sum(row["status"] == "NOT_ATTEMPTED" for row in predictions),
                "confirmed_cost_usd": summary["usage"]["reported_cost_usd"], "cost_is_lower_bound": True,
                "shared_successful_targets": len(shared), "paired_metrics": paired, "paired_changes": differences,
                "failures": [{k: row.get(k) for k in ("key", "status", "error", "http_status", "billing_reconciled")}
                             for row in predictions if row["status"] not in {"OK", "NOT_ATTEMPTED"}],
                "offline_recovered_successes": sum(bool(row.get("recovered_from_saved_response")) for row in predictions),
                "wire_audit": {"status": "PASS" if not wire_errors else "FAIL", "requests": wire_count,
                               "image_references": image_count, "errors": wire_errors},
                "summary_sha256": sha256_file(output / "summary.json")}
    atomic_write_json(output / "paired_analysis.json", analysis)
    lines = ["# H0 frame strategy Training pilot — 2026-09-05", "",
             "Model: `qwen/qwen3.8-max-0902` through OpenRouter. Pure joint five-head H0; no Tracker, repair or memory.",
             "Original plan: 80 time-spread targets from four Training videos, four paired image strategies, no retries.",
             "Account balance could not support the full plan. This report covers its first 32 targets (8 per video, 128 requests); later 48 targets were not run.",
             "The smaller pilot covers earlier video portions and does not retain full-video temporal coverage. The original 80-target manifest and raw summary remain preserved.",
             "A: current image; B: [-2,-1,0] seconds; C: [-5,-2,0]; D: [-5,-4,-3,-2,-1,0].",
             "Only the final target is scored. History images use low detail; target uses high detail. Temperature 0.", "",
             f"Successful predictions: {summary['successful_predictions']}/128 in this pilot; shared successful targets: {len(shared)}/32.",
             f"Confirmed cost: ${summary['usage']['reported_cost_usd']:.6f}, plus unknown cost for the prior lost response (VID103_5351_B).",
             "The $3 unknown-cost reserve is a budget reserve, not a measured charge. The lost request was not repeated.",
             "Known spend is a lower bound: it excludes the lost original response and unpriced gateway rejections; those are never relabeled as successful predictions.",
             "Primary pilot metrics retain all 32 targets with valid GT; failed calls receive empty predictions. Secondary metrics use shared successful targets.",
             "All 320 requests were frozen before scoring; selection used time bins and validity masks, not label identities.",
             "Interrupted continuations triggered partial offline scoring. Remaining frozen requests were unchanged; final scoring followed completion of the affordable pilot.", "",
             "## Primary micro-F1 (all 32 pilot targets per arm, masked by task)", "",
             "| Task | A single | B 3/2s | C 3/5s | D 6/5s |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        lines.append("| " + task + " | " + " | ".join(f"{100 * primary[g][task]['f1']:.2f}%" for g in OFFSETS) + " |")
    lines += ["", "## Secondary micro-F1 (shared successful targets)", "",
              "| Task | A single | B 3/2s | C 3/5s | D 6/5s |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        lines.append("| " + task + " | " + " | ".join(f"{100 * paired[g][task]['f1']:.2f}%" for g in OFFSETS) + " |")
    lines += ["", "## Per-video IVT micro-F1 (primary)", "",
              "| Video | A | B | C | D |", "|---|---:|---:|---:|---:|"]
    for video in VIDEOS:
        lines.append("| " + video + " | " + " | ".join(f"{100 * per_video[video][g]['ivt']['f1']:.2f}%" for g in OFFSETS) + " |")
    lines += ["", "## Limits", "",
              "This is a small development pilot on Training videos, not held-out test performance or official mAP. IVT includes null classes 94–99.",
              "Targets are spread over time, not a continuous streaming run; the study does not validate throughput, latency or cross-window state at 1 Hz.",
              "A four-video result cannot establish that six images improve every action, target or phase. No default pipeline change follows automatically.",
              f"Wire image audit: {analysis['wire_audit']['status']}; {wire_count} requests, {image_count} image references.",
              "Original frozen scripts, sampling plan and prior artifacts were preserved. Continuations used eight concurrent calls, reduced to four for the balance limit, with no retries.", ""]
    report = ROOT / "docs/H0_FRAME_STRATEGY_TRAINING32_QWEN_2026-09-05.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report), **analysis}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
