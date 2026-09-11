"""Continue the archived eight-target graph arm once, without regenerating H0.

Run observation feedback first, then an equal-opportunity issues-only control.
No new pool means no repeated panel. Same complete review requests share their
original answers, including failures. GT is consumed only by the separate scorer.
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

import requests

from scripts.check_candidate_panel_providers import key_for, redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_evidence_feedback_trial import proposal_wire as feedback_proposal_wire
from scripts.run_graph_review_trial import TimedCalls, sources
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import (
    MODELS,
    PROVIDERS,
    RATES_V2,
    ROUTES,
    review_wire,
)
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_prior_candidate_trial import validate as validate_original
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import digest
from surgical_agent.research.verification.review_feedback import build_review_feedback

PROFILE = "prior_graph_feedback_continuation_v1"
ARMS = ("evidence_feedback", "issues_only")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
DEFAULT_SOURCE = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1"


def same(actual, expected, context):
    if digest(actual) != digest(expected):
        raise ValueError(context + " differs from frozen archive")


def initial_row_replay(row):
    """Validate the imported first round without loading any GT label values."""
    for arm in ("control_r1", "graph_r1"):
        record = row[arm]
        pool = make_pool(row["h0"], record["proposal"], make_pool(row["h0"]))
        same(pool, record["pool"], "initial pool")
        reviews, formats = normalize_five(record["raw"], pool, 3)
        same(reviews, record["reviews"], "initial normalized reviews")
        same(formats, record["format_diagnostics"], "initial formats")
        means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
        same(means, record["means"], "initial means")
        same(diagnostics, record["diagnostics"], "initial diagnostics")
        prediction = panel.select(row["h0"], pool, means, threshold=4)
        issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
        same(prediction, record["prediction"], "initial prediction")
        same(issues, record["issues"], "initial issues")
        same(record["status"], "UNRESOLVED" if issues else "MODEL_PASS", "initial status")


def proposal_wire(base, selected, initial, arm):
    if arm not in ARMS:
        raise ValueError("unknown continuation arm")
    old = initial["graph_r1"]
    feedback = (build_review_feedback(old["pool"], old["reviews"], old["issues"], image_count=len(base.images))
                if arm == "evidence_feedback" else None)
    body = feedback_proposal_wire(base, selected, old["prediction"], old["pool"], old["issues"], feedback)
    hints = initial["hints"]["packet"]
    if hints is not None:
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        packet["candidate_relation_hints"] = deepcopy(hints)
        body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def empty_record(initial, arm):
    old = initial["graph_r1"]
    return {"round": 2, "arm": arm, "before": deepcopy(old["prediction"]),
            "pool_before": deepcopy(old["pool"]), "input_issues": deepcopy(old["issues"]),
            "review_evidence_feedback": (build_review_feedback(old["pool"], old["reviews"], old["issues"], image_count=3)
                                         if arm == "evidence_feedback" else None),
            "candidate_relation_hints": deepcopy(initial["hints"]["packet"]),
            "proposal": None, "pool": deepcopy(old["pool"]), "raw": None, "reviews": None,
            "format_diagnostics": None, "means": None, "diagnostics": None,
            "issues": deepcopy(old["issues"]), "prediction": deepcopy(old["prediction"]),
            "status": "NOT_ATTEMPTED", "round2_attempted": False, "reviewed": False,
            "shared_from": None, "review_calls_observed": 0, "new_review_calls": 0,
            "review_request_fingerprints": {}, "proposal_request_fingerprint": None,
            "proposal_seconds": 0.0, "panel_seconds": None, "original_panel_seconds": None,
            "total_seconds": 0.0}


def run_target(calls, base, selected, initial, arm, reusable=None):
    record = empty_record(initial, arm)
    start = perf_counter()
    if initial["graph_r1"]["status"] == "MODEL_PASS":
        record["status"] = "INHERITED_MODEL_PASS"
        return record, None
    if calls.stopped:
        record["status"] = "BUDGET_STOPPED"
        return record, None
    body = proposal_wire(base, selected, initial, arm)
    record["proposal_request_fingerprint"] = fingerprint(body)
    begin = perf_counter()
    proposal = calls.call(selected["key"], f"{arm}_proposal_2", "base", body)
    record["proposal_seconds"] = perf_counter() - begin
    record["round2_attempted"] = any(r["target"] == selected["key"] and r["stage"] == f"{arm}_proposal_2"
                                     for r in calls.rows)
    record["proposal"] = proposal
    try:
        if proposal is None:
            raise ValueError("no proposal response")
        pool = make_pool(record["before"], proposal, record["pool_before"])
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        record["status"] = "PROPOSAL_FAILED" if record["round2_attempted"] else "BUDGET_STOPPED"
        record["total_seconds"] = perf_counter() - start
        return record, None
    record["pool"] = pool
    if pool == record["pool_before"]:
        record["status"] = "NO_NEW_CANDIDATES"
        record["total_seconds"] = perf_counter() - start
        return record, None
    bodies = {seat: review_wire(seat, base, selected, pool) for seat in SEATS}
    record["review_request_fingerprints"] = {seat: fingerprint(body) for seat, body in bodies.items()}
    if reusable is not None and reusable["target"] == selected["key"] and bodies == reusable["bodies"]:
        raw = deepcopy(reusable["raw"])
        count = reusable["review_count"]
        record.update(shared_from=reusable["arm"], original_panel_seconds=reusable["panel_seconds"])
    else:
        begin = perf_counter()
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], f"{arm}_review_2", s, bodies[s]), SEATS), strict=True))
        record["panel_seconds"] = perf_counter() - begin
        count = sum(r["target"] == selected["key"] and r["stage"] == f"{arm}_review_2" for r in calls.rows)
        record["new_review_calls"] = count
    reviews, formats = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(reviews, pool, image_count=len(base.images))
    record.update(raw=raw, reviews=reviews, format_diagnostics=formats, means=means,
                  diagnostics=diagnostics, review_calls_observed=count, reviewed=count == 5)
    if count != 5:
        record["status"] = "INCOMPLETE_PANEL"
    else:
        try:
            prediction = panel.select(record["before"], pool, means, threshold=4)
            issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
            record.update(prediction=prediction, issues=issues, status="UNRESOLVED" if issues else "MODEL_PASS")
        except (ApiSchemaError, TypeError, ValueError, KeyError):
            record["status"] = "SELECTION_FAILED"
    record["total_seconds"] = perf_counter() - start
    cache = {"arm": arm, "target": selected["key"], "bodies": bodies, "raw": deepcopy(raw), "review_count": count,
             "panel_seconds": record["panel_seconds"]}
    return record, cache


def verify_plan(plan):
    if plan["profile"] != PROFILE or plan["arms"] != list(ARMS) or plan["threshold"] != 4 or plan["round_cap"] != 2:
        raise ValueError("continuation policy changed")
    same(plan["models"], MODELS, "frozen reviewer models")
    same(plan["h0"], PROPOSER, "frozen proposer model")
    same(plan["limits"], LIMITS, "frozen spending limits")
    same(plan["max_calls"], len(plan["eligible_targets"]) * 12, "call cap")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen runtime source changed: " + name)
    source = Path(plan["source_root"])
    for name, expected in plan["source_snapshot_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("original first-round archive changed: " + name)
    for row in plan["selection"]:
        for im in row["images"]:
            if sha(im["path"]) != im["sha256"]:
                raise ValueError("causal source image changed")


def metadata_preflight():
    metadata = {"checked_utc": now(), "openrouter_routes": {}, "direct_routes": {}}
    for seat, route in ROUTES.items():
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if endpoint["status"] != 0 or not {"response_format", "reasoning"} <= set(endpoint["supported_parameters"]):
            raise ValueError("frozen model route unavailable")
        if any(Decimal(endpoint["pricing"][f]) > Decimal(rate)
               for f, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("current route price exceeds reserved envelope")
        metadata["openrouter_routes"][seat] = data
    secret = key_for("gpt")
    response = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + secret.reveal()}, timeout=25)
    response.raise_for_status()
    key_data = response.json()["data"]
    metadata["openrouter_key"] = {k: key_data.get(k) for k in ("limit", "limit_remaining", "usage", "is_free_tier")}
    if key_data.get("limit_remaining") is not None and Decimal(str(key_data["limit_remaining"])) < Decimal(LIMITS["openrouter_usd"]):
        raise ValueError("key budget insufficient")
    response = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + secret.reveal()}, timeout=25)
    metadata["credits_http_status"] = response.status_code
    if response.ok:
        credit_data = response.json()["data"]
        balance = Decimal(str(credit_data["total_credits"])) - Decimal(str(credit_data["total_usage"]))
        metadata["openrouter_balance_usd"] = str(balance)
        if balance < Decimal(LIMITS["openrouter_usd"]):
            raise ValueError("account budget insufficient")
    # Local credentials are loaded for presence/shape only; no extra paid probes.
    for seat in ("grok", "qwen"):
        key_for(seat)
        metadata["direct_routes"][seat] = {"credential_loaded": True, "model": MODELS[seat],
            "live_balance_verified": False, "rates": list(RATES_V2[seat]),
            "rate_note": "Existing frozen conservative envelope; provider usage or unknown reserve retained."}
    return metadata


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("use a new output directory")
    source = source.resolve()
    original_plan, original_rows, _ = validate_original(source)
    if original_plan["h0"] != PROPOSER or original_plan["models"] != MODELS or len(original_rows) != 8:
        raise ValueError("expected exact eight-target graph experiment")
    initials, preflight = [], {}
    for selected, old in zip(original_plan["selection"], original_rows, strict=True):
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "cached H0 image/protocol binding")
        row = {k: deepcopy(old[k]) for k in ("key", "video_id", "frame_id", "h0")}
        row.update(control_r1=read(source / "targets" / old["key"] / "control.json"),
                   graph_r1=read(source / "targets" / old["key"] / "prior_graph.json"),
                   hints=read(source / "targets" / old["key"] / "hints.json"))
        initial_row_replay(row)
        initials.append(row)
        if row["graph_r1"]["status"] == "UNRESOLVED":
            wires = {a: proposal_wire(base, selected, row, a) for a in ARMS}
            stripped = deepcopy(wires["evidence_feedback"])
            packet = json.loads(stripped["messages"][0]["content"][0]["text"])
            packet.pop("review_evidence_feedback")
            stripped["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
            same(stripped, wires["issues_only"], "only feedback field differs between proposer arms")
            preflight[row["key"]] = {a: redact_images(wires[a]) for a in ARMS}
    unresolved_count = sum(r["graph_r1"]["status"] == "UNRESOLVED" for r in initials)
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_prior_candidate_trial.py",
                           ROOT / "scripts/run_prior_candidate_trial.py", ROOT / "scripts/score_prior_feedback_continuation.py"})
    source_files = [source / name for name in ("plan.json", "completion.json", "predictions.json", "budget.json", "training_counts.json")]
    source_files += list((source / "targets").rglob("*.json")) + list((source / "priors").rglob("*.json"))
    source_files += list((source / "calls").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source),
            "source_snapshot_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(source_files)},
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
            "selection": original_plan["selection"], "arms": list(ARMS), "models": MODELS,
            "h0": PROPOSER, "threshold": 4, "round_cap": 2, "max_calls": unresolved_count * 12,
            "limits": LIMITS, "rates": RATES_V2, "cached_h0_calls": 0,
            "proposal_preflight_fingerprints": {key: {a: digest(body) for a, body in wires.items()} for key, wires in preflight.items()},
            "eligible_targets": [r["key"] for r in initials if r["graph_r1"]["status"] == "UNRESOLVED"],
            "execution_order": "all evidence_feedback targets first, then all issues_only targets, per user request",
            "order_limitation": "No counterbalancing; provider drift/response variation cannot be separated from arm effects.",
            "stopping_rule": "Inherited MODEL_PASS stops; unresolved gets one proposal; unchanged pool stops without a review; changed pool gets at most one complete panel.",
            "shared_review_policy": "same target, all five complete request bodies equal: share original raw results including failures, never mix per-item scores",
            "graph_policy": "Keep the originally frozen leave-query-video-out hint packet in both second-round proposer arms; no new retrieval or GT lookup.",
            "gt_policy": "Previously scored development targets intentionally reused. No new GT labels in inference; score only after both paid arms close. This is not an independent confirmation or Testing result.",
            "success_rule": "Compare graphR1 with each R2 and issuesR2 with feedbackR2. Report net label errors, each head P/R/F1/exact accuracy, harms and runtime; no threshold or prompt adjustment using these targets."}
    save(output / "initial_state.json", {"targets": initials})
    plan["initial_state_sha256"] = sha(output / "initial_state.json")
    save(output / "plan.json", plan)
    save(output / "preflight_metadata.json", metadata)
    for key, wires in preflight.items():
        for arm, wire in wires.items():
            save(output / "proposal_preflight" / key / f"{arm}.json", wire)
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": len(initials), "eligible_targets": unresolved_count,
                      "max_calls": plan["max_calls"], "limits": LIMITS}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("paid replay prohibited; original output is immutable")
    verify_plan(plan)
    if sha(output / "initial_state.json") != plan["initial_state_sha256"]:
        raise ValueError("cached initial state changed")
    initials = read(output / "initial_state.json")["targets"]
    bases = {s["key"]: build_gemini_base(adapter, s) for s in plan["selection"]}
    for row, selected in zip(initials, plan["selection"], strict=True):
        initial_row_replay(row)
        same(canonical_request_metadata(bases[row["key"]]).to_mapping(), selected["request_metadata"], "causal image binding")
        if row["graph_r1"]["status"] == "UNRESOLVED":
            for arm in ARMS:
                wire = proposal_wire(bases[row["key"]], selected, row, arm)
                same(digest(redact_images(wire)), plan["proposal_preflight_fingerprints"][row["key"]][arm], "frozen proposal request")
    with (output / "execution.lock").open("x") as f:
        f.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                       providers=PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"))
    rows = [{k: deepcopy(initial[k]) for k in ("key", "video_id", "frame_id", "h0")} for initial in initials]
    for initial, row in zip(initials, rows, strict=True):
        row.update(control_r1=deepcopy(initial["control_r1"]["prediction"]),
                   graph_r1=deepcopy(initial["graph_r1"]["prediction"]), arms={})
        for arm in ARMS:
            record_path = f"targets/{row['key']}/{arm}.json"
            record = empty_record(initial, arm)
            save(output / record_path, record)
            row["arms"][arm] = {k: record[k] for k in ("prediction", "status", "round2_attempted", "reviewed")}
            row["arms"][arm]["record"] = record_path
    save(output / "predictions.json", {"targets": rows})
    caches, fatal = {}, None
    started = perf_counter()
    try:
        for arm in ARMS:
            arm_start = perf_counter()
            for row, initial, selected in zip(rows, initials, plan["selection"], strict=True):
                record, cache = run_target(calls, bases[row["key"]], selected, initial, arm, caches.get(row["key"]))
                if cache is not None and arm == "evidence_feedback":
                    caches[row["key"]] = cache
                save(output / row["arms"][arm]["record"], record)
                row["arms"][arm].update({k: record[k] for k in ("prediction", "status", "round2_attempted", "reviewed")})
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"arm": arm, "target": row["key"], "status": record["status"],
                                  "pool_size": len(record["pool"]["propositions"]), "post_calls": len(calls.rows)}), flush=True)
            save(output / f"{arm}_finished.json", {"finished_utc": now(), "arm_seconds": perf_counter() - arm_start,
                 "statuses": dict(Counter(r["arms"][arm]["status"] for r in rows)), "post_calls_so_far": len(calls.rows)})
        verify_plan(plan)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": rows})
        artifacts = [*(output / "targets").rglob("*.json"), *(output / "calls").rglob("*.json")]
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
             **{f"{name}_sha256": sha(output / f"{name}.json") for name in ("plan", "predictions", "budget", "initial_state")},
             "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(artifacts)},
             "all_targets_recorded": len(rows) == len(plan["selection"]), "post_calls": len(calls.rows),
             "call_statuses": dict(Counter(r["status"] for r in calls.rows)),
             "occupied": {k: str(v) for k, v in calls.occupied.items()}, "inference_seconds": perf_counter() - started,
             "new_h0_calls": 0, "query_gt_scoring_not_run_during_inference": True})
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
