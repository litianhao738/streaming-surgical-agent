"""Frozen eight-target quorum4/5 x direct-second-review mechanism experiment.

No H0/proposal/GT inference. Reuse graph R1; fresh reviewer measurements are
shared between deterministic policies. Archived feedback R2 is a comparator,
never the starting point of another paid round. No retries or third round.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls, sources
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_feedback_continuation import metadata_preflight, same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_prior_feedback_continuation import audit_records, audit_wires
from scripts.score_prior_feedback_continuation import validate as validate_source
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification import flexible_quorum as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import digest

PROFILE = "quorum_direct_recheck_v1"
POLICIES = {"q5": 5, "q4": 4}
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
DEFAULT_SOURCE = ROOT / "artifacts/preflight/prior_graph_second_round_20260908_v1"
STAGE = "direct_review_2"


def pool_for(initial):
    old = initial["graph_r1"]
    pool = make_pool(initial["h0"], old["proposal"], make_pool(initial["h0"]))
    same(pool, old["pool"], "canonical graph R1 pool")
    return pool


def initial_policies(initial):
    pool = pool_for(initial)
    reviews, formats = normalize_five(initial["graph_r1"]["raw"], pool, 3)
    same(reviews, initial["graph_r1"]["reviews"], "source normalized judgments")
    same(formats, initial["graph_r1"]["format_diagnostics"], "source normalization diagnostics")
    policies = {}
    for policy, minimum in POLICIES.items():
        means, diagnostics = panel.aggregate(reviews, pool, minimum_valid=minimum, image_count=3)
        prediction = panel.select(initial["h0"], pool, means, threshold=4)
        if minimum == 5:
            same(means, initial["graph_r1"]["means"], "quorum5 original means")
            same(prediction, initial["graph_r1"]["prediction"], "quorum5 original prediction")
        policies[policy] = {"minimum_valid": minimum, "prediction": prediction,
                            "means": means, "diagnostics": diagnostics,
                            "queue": panel.recheck_queue(prediction, pool, means, diagnostics)}
    return {"key": initial["key"], "policies": policies}


def archived_feedback_replay(initial, record):
    """One-step admission ablation, not an adaptive quorum4 feedback trajectory."""
    before = initial["graph_r1"]["prediction"]
    if (not record["reviewed"] or record["review_calls_observed"] != 5
            or record["status"] in {"SELECTION_FAILED", "INCOMPLETE_PANEL"}):
        return deepcopy(before)
    reviews, _ = normalize_five(record["raw"], record["pool"], 3)
    means, _ = panel.aggregate(reviews, record["pool"], minimum_valid=4, image_count=3)
    return panel.select(before, record["pool"], means, threshold=4)


def empty_record(initial, state):
    record = {"pool": pool_for(initial), "raw": None, "reviews": None,
              "format_diagnostics": None, "panel_seconds": 0.0,
              "review_calls_observed": 0, "request_fingerprints": {},
              "reviewed": False, "attempted": False, "policies": {}}
    for policy, data in state["policies"].items():
        record["policies"][policy] = {"minimum_valid": POLICIES[policy],
            "before": deepcopy(data["prediction"]), "prediction": deepcopy(data["prediction"]),
            "eligible": bool(data["queue"]), "queue": deepcopy(data["queue"]),
            "queue_after": deepcopy(data["queue"]), "means": deepcopy(data["means"]),
            "diagnostics": deepcopy(data["diagnostics"]),
            "status": "NOT_ATTEMPTED" if data["queue"] else "NO_RECHECK_NEEDED"}
    return record


def run_target(calls, base, selected, initial, state):
    record = empty_record(initial, state)
    eligible = [p for p, r in record["policies"].items() if r["eligible"]]
    if not eligible:
        return record
    if calls.stopped:
        for policy in eligible:
            record["policies"][policy]["status"] = "BUDGET_STOPPED"
        return record
    bodies = {seat: review_wire(seat, base, selected, record["pool"]) for seat in SEATS}
    record["request_fingerprints"] = {seat: fingerprint(body) for seat, body in bodies.items()}
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], STAGE, s, bodies[s]), SEATS), strict=True))
    record["panel_seconds"] = perf_counter() - start
    count = sum(r["target"] == selected["key"] and r["stage"] == STAGE for r in calls.rows)
    reviews, formats = normalize_five(raw, record["pool"], len(base.images))
    record.update(raw=raw, reviews=reviews, format_diagnostics=formats,
                  review_calls_observed=count, reviewed=count == 5, attempted=count > 0)
    for policy in eligible:
        result = record["policies"][policy]
        if count != 5:
            result["status"] = "INCOMPLETE_PANEL"
            continue
        means, diagnostics = panel.aggregate(reviews, record["pool"], minimum_valid=POLICIES[policy], image_count=3)
        try:
            prediction = panel.select(result["before"], record["pool"], means, threshold=4)
            queue = panel.recheck_queue(prediction, record["pool"], means, diagnostics)
            result.update(prediction=prediction, queue_after=queue, means=means, diagnostics=diagnostics,
                          status="REVIEWED_PENDING" if queue else "REVIEWED_NO_PENDING")
        except (ApiSchemaError, ValueError, TypeError, KeyError):
            result["status"] = "SELECTION_FAILED"
    return record


def verify_plan(plan):
    same(plan["profile"], PROFILE, "profile")
    for name, value in (("policies", POLICIES), ("models", MODELS), ("h0", PROPOSER),
                        ("limits", LIMITS), ("rates", RATES_V2), ("threshold", 4), ("round_cap", 2)):
        same(plan[name], value, name)
    same(plan["max_calls"], len(plan["eligible_targets"]) * 5, "call cap")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen source changed: " + name)
    source = Path(plan["source_root"])
    for name, expected in plan["source_snapshot_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("frozen archive changed: " + name)
    for selected in plan["selection"]:
        for im in selected["images"]:
            if sha(im["path"]) != im["sha256"]:
                raise ValueError("causal image changed")
    validate_source(source)


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    source = source.resolve()
    old_plan, old_rows, initials, ledger = validate_source(source)
    records, _ = audit_records(source, old_plan, old_rows, initials, ledger)
    audit_wires(source, old_plan, initials, records, ledger, adapter)
    states = [initial_policies(initial) for initial in initials]
    preflight = {}
    for selected, initial, state in zip(old_plan["selection"], initials, states, strict=True):
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "causal image binding")
        if any(value["queue"] for value in state["policies"].values()):
            preflight[initial["key"]] = {seat: redact_images(review_wire(seat, base, selected, pool_for(initial))) for seat in SEATS}
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(),
        ROOT / "scripts/run_prior_candidate_trial.py", ROOT / "scripts/score_prior_candidate_trial.py",
        ROOT / "scripts/run_prior_feedback_continuation.py", ROOT / "scripts/score_prior_feedback_continuation.py",
        ROOT / "scripts/score_quorum_recheck_trial.py", ROOT / "src/surgical_agent/research/verification/flexible_quorum.py"})
    archives = [source / f"{name}.json" for name in ("plan", "completion", "predictions", "budget", "initial_state")]
    archives += list((source / "calls").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "source_snapshot_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(archives)},
        "selection": old_plan["selection"], "policies": POLICIES, "models": MODELS, "h0": PROPOSER,
        "threshold": 4, "round_cap": 2, "max_calls": len(preflight) * 5, "eligible_targets": list(preflight),
        "limits": LIMITS, "rates": RATES_V2,
        "review_preflight_fingerprints": {key: {seat: digest(body) for seat, body in wires.items()} for key, wires in preflight.items()},
        "quorum_policy": "Average only semantically valid observed scores; minimum4 vs minimum5. Invalid/missing never becomes neutral/negative evidence. Selection thresholds/component dependencies unchanged.",
        "recheck_policy": "Each policy starts its own R1. Selected meanNone/<4 or any valid low<=2/high>=4 conflict triggers one fresh full-pool panel. Review union, share raw, apply only to eligible policy. No unselected-uncertain-only trigger; no queue is not visual proof.",
        "limitations": ["Eight previously scored development targets, not independent validation.",
            "Same images/pool/reviewer prompt are measured again; no new visual evidence and no claim of independent errors.",
            "Graph/candidates/proposal logic unchanged. Archived feedback quorum4 is a one-step replay from original graphR1, not a regenerated trajectory.",
            "No retries, new H0, proposal, training or third round. API/model changes versus archived rounds may affect repeated-review comparison."]}
    save(output / "initial_state.json", {"targets": initials})
    save(output / "policy_state.json", {"targets": states})
    plan.update({f"{name}_sha256": sha(output / f"{name}.json") for name in ("initial_state", "policy_state")})
    save(output / "plan.json", plan)
    save(output / "preflight_metadata.json", metadata)
    for key, wires in preflight.items():
        for seat, body in wires.items():
            save(output / "review_preflight" / key / f"{seat}.json", body)
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": len(initials),
        "eligible_targets": len(preflight), "max_calls": plan["max_calls"], "limits": LIMITS,
        "policy_eligible": {p: sum(bool(s["policies"][p]["queue"]) for s in states) for p in POLICIES}}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use paid experiment; no overwrite or replay")
    verify_plan(plan)
    for name in ("initial_state", "policy_state"):
        same(sha(output / f"{name}.json"), plan[f"{name}_sha256"], name)
    initials = read(output / "initial_state.json")["targets"]
    states = read(output / "policy_state.json")["targets"]
    same(states, [initial_policies(initial) for initial in initials], "frozen per-policy state")
    bases = {s["key"]: build_gemini_base(adapter, s) for s in plan["selection"]}
    for selected, initial in zip(plan["selection"], initials, strict=True):
        base = bases[initial["key"]]
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "causal image binding")
        if initial["key"] in plan["eligible_targets"]:
            for seat in SEATS:
                body = review_wire(seat, base, selected, pool_for(initial))
                same(digest(redact_images(body)), plan["review_preflight_fingerprints"][initial["key"]][seat], "frozen review request")
    source = Path(plan["source_root"])
    old_rows = {r["key"]: r for r in read(source / "predictions.json")["targets"]}
    rows = []
    for initial, state in zip(initials, states, strict=True):
        row = {k: deepcopy(initial[k]) for k in ("key", "video_id", "frame_id", "h0")}
        row.update(graph_r1=deepcopy(initial["graph_r1"]["prediction"]),
            graph_feedback_r2=deepcopy(old_rows[row["key"]]["arms"]["evidence_feedback"]["prediction"]),
            quorum4_r1=deepcopy(state["policies"]["q4"]["prediction"]),
            archived_feedback_quorum4=archived_feedback_replay(initial, read(source / "targets" / row["key"] / "evidence_feedback.json")),
            direct_recheck_5=deepcopy(state["policies"]["q5"]["prediction"]),
            direct_recheck_4=deepcopy(state["policies"]["q4"]["prediction"]))
        rows.append(row)
        save(output / "targets" / row["key"] / "direct.json", empty_record(initial, state))
    save(output / "predictions.json", {"targets": rows})
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
        providers=PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"))
    started, fatal = perf_counter(), None
    try:
        for selected, initial, state, row in zip(plan["selection"], initials, states, rows, strict=True):
            record = run_target(calls, bases[row["key"]], selected, initial, state)
            save(output / "targets" / row["key"] / "direct.json", record)
            for policy, field in (("q5", "direct_recheck_5"), ("q4", "direct_recheck_4")):
                row[field] = deepcopy(record["policies"][policy]["prediction"])
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": row["key"], "statuses": {p: r["status"] for p, r in record["policies"].items()},
                              "post_calls": len(calls.rows)}), flush=True)
        verify_plan(plan)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": rows})
        artifacts = [*(output / "calls").rglob("*.json"), *(output / "targets").rglob("*.json")]
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
            **{f"{name}_sha256": sha(output / f"{name}.json") for name in ("plan", "predictions", "budget", "initial_state", "policy_state")},
            "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(artifacts)},
            "post_calls": len(calls.rows), "call_statuses": dict(Counter(r["status"] for r in calls.rows)),
            "occupied": {k: str(v) for k, v in calls.occupied.items()}, "inference_seconds": perf_counter() - started,
            "new_h0_calls": 0, "new_proposal_calls": 0, "query_gt_scoring_not_run_during_inference": True})
    print(json.dumps(read(output / "completion.json") | {"inference_artifact_sha256": "stored"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.source, adapter)
    else:
        execute(args.output, adapter)
