"""Freeze score aggregation from 72 old targets; evaluate only after new inference closes."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_default_improvement_trial import SUCCESS, passing
from scripts.run_prior_panel_trial import now, read, save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification import weighted_review as weighted
from surgical_agent.research.verification.phase_extension import apply_phase_choices

OUTPUT = ROOT / "artifacts/preflight/weighted_review_confirmation_20260909_v1"
QUERY = ROOT / "artifacts/preflight/default_improvement_confirm_20260909_v1"
SOURCES = ("new_training_verb_guard_20260909_v1", "verb_prompt_20260909_v1", "transactional_trial_20260909_v2")


def prepare():
    if OUTPUT.exists() or (QUERY / "scored_truth.json").exists() or (QUERY / "metrics.json").exists():
        raise ValueError("freeze before looking at query scores")
    examples, hashes, identities = [], {}, set()
    for name in SOURCES:
        root = ROOT / "artifacts/preflight" / name
        truth_path = root / "scored_truth.json"
        truth = read(truth_path)
        hashes[str(truth_path)] = sha(truth_path)
        pools = {r["key"]: r["pool"] for r in read(root / "initial_state.json")} if name == SOURCES[1] else None
        if pools is not None:
            hashes[str(root / "initial_state.json")] = sha(root / "initial_state.json")
        for row in truth:
            key = f"{row['video_id']}_{row['frame_id']}"
            if key in identities:
                raise ValueError("duplicate training target")
            identities.add(key)
            path = root / "targets" / key / ("control.json" if pools is not None else "result.json")
            record = read(path)
            graph = record if pools is not None else record["graph"]
            hashes[str(path)] = sha(path)
            if not graph or not graph.get("reviews"):
                continue
            pool = pools[key] if pools is not None else graph["pool"]
            _, diagnostics = panel.aggregate(graph["reviews"], pool, image_count=3)
            for item in pool["propositions"]:
                task, pid = item["task"], item["id"]
                scores = diagnostics[pid]["scores"]
                if row["mask"][task] and all(s is not None for s in scores):
                    examples.append({"video_id": row["video_id"], "key": key, "task": task,
                        "scores": scores, "present": int(item["label_id"] in row["gt"][task])})
    query_plan = read(QUERY / "plan.json")
    if identities & {s["key"] for s in query_plan["selection"]}:
        raise ValueError("new target overlaps fitting data")
    models = {video: weighted.fit(examples, video) for video in sorted({s["video_id"] for s in query_plan["selection"]})}
    save(OUTPUT / "models.json", models)
    save(OUTPUT / "training_examples.json", examples)
    code = [Path(__file__).resolve(), ROOT / "src/surgical_agent/research/verification/weighted_review.py",
            ROOT / "tests/unit/test_weighted_review.py"]
    plan = {"created_utc": now(), "version": weighted.VERSION, "query_plan_sha256": sha(QUERY / "plan.json"),
        "query_scores_absent_at_freeze": True, "training_targets": len(identities), "training_examples": len(examples),
        "positive_threshold": weighted.POSITIVE, "negative_threshold": weighted.NEGATIVE, "success_rule": SUCCESS,
        "models_sha256": sha(OUTPUT / "models.json"), "examples_sha256": sha(OUTPUT / "training_examples.json"),
        "sources": hashes, "code": {p.relative_to(ROOT).as_posix(): sha(p) for p in code},
        "protocol": "Independent prospective offline side study; not a new arm of the running ROI experiment. "
        "Fit four monotone L2-regularized logistic score aggregators per excluded query video, using only old "
        "original-panel Training targets. Five valid ratings required; no H0 membership, class IDs, graph frequency "
        "or query labels as features. Frozen thresholds 0.65/0.35 with explicit visual support/refutation guards. "
        "Original Python component/IVT dependency rules and shared Phase kept. Scores are not validated confidence. "
        "No API calls. Current source trial GT only read after closed inference. No post-score tuning. "
        "Whole-query-video exclusion applies to fitted score models; historical feature-generating pipelines "
        "were not independently rerun as full-system OOF."}
    save(OUTPUT / "plan.json", plan)
    for p in code:
        target = OUTPUT / "frozen_source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    print(json.dumps({"prepared": True, "training_targets": len(identities), "examples": len(examples),
                      "plan_sha256": sha(OUTPUT / "plan.json"), "api_calls": 0}))


def score():
    plan = read(OUTPUT / "plan.json")
    if sha(QUERY / "plan.json") != plan["query_plan_sha256"]:
        raise ValueError("changed query plan")
    for group in ("sources", "code"):
        for path, digest in plan[group].items():
            assert sha(ROOT / path) == digest
    assert sha(OUTPUT / "models.json") == plan["models_sha256"]
    assert sha(OUTPUT / "training_examples.json") == plan["examples_sha256"]
    done, source = read(QUERY / "completion.json"), read(QUERY / "metrics.json")
    if done["fatal_error"] or not source["raw_replayed"]:
        raise ValueError("closed and replayed source required")
    for path, digest in done["hashes"].items():
        assert sha(QUERY / path) == digest
    truths = {(r["video_id"], r["frame_id"]): r for r in read(QUERY / "scored_truth.json")}
    rows, models, details = read(QUERY / "predictions.json"), read(OUTPUT / "models.json"), []
    for row in rows:
        record = read(QUERY / "targets" / row["key"] / "result.json")
        graph = record["arms"].get("control", {})
        prediction, state = row["control"], None
        if graph.get("reviews"):
            state = weighted.select(row["h0"], graph["pool"], graph["reviews"], models[row["video_id"]], row["video_id"])
            prediction = apply_phase_choices(state["prediction"], record["phase"]["raw"], 3)[0]
        row["weighted"] = prediction
        details.append({"key": row["key"], "state": state})
    metrics = {v: compute_repair_comparison([{**truths[r['video_id'], r['frame_id']], "h0": r['h0'],
        "h1": None, "final": r[v]} for r in rows])["arms"]["final"] for v in ("h0", "control", "weighted")}
    deltas = [{"key": r["key"], **frame_delta(r["control"], r["weighted"], truths[r['video_id'], r['frame_id']]["gt"],
        truths[r['video_id'], r['frame_id']]["mask"])} for r in rows]
    changes = summarize_deltas(deltas)
    checks = passing(metrics, changes, "weighted")
    save(OUTPUT / "predictions.json", rows)
    save(OUTPUT / "details.json", details)
    save(OUTPUT / "frame_deltas.json", deltas)
    report = {"metrics": metrics, "changes": changes, "checks": checks, "success": all(checks.values()), "api_calls": 0,
              "source_closed_hash": sha(QUERY / "completion.json"), "source_metrics_sha256": sha(QUERY / "metrics.json")}
    save(OUTPUT / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    {"prepare": prepare, "score": score}[parser.parse_args().command]()
