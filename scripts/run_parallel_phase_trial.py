"""Fixed-H0 graph repair and blind Phase review run concurrently, then merge."""
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

from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls, sources
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_phase_extension_trial import phase_wire
from scripts.run_prior_candidate_trial import TASKS as ORIGINAL_TASKS
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_feedback_continuation import metadata_preflight, same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_phase_extension_trial import audit as audit_source
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.parallel_phase import run_parallel_repair
from surgical_agent.research.verification.phase_extension import phase_choice_error

PROFILE = "fixed_h0_parallel_graph_and_phase_v1"
SOURCE = ROOT / "artifacts/preflight/phase_extension_eight_20260909_v1"
GRAPH_SOURCE = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1"
BRANCHES = ("graph", "phase")
STAGES = {"proposal": "graph_proposal", "review": "graph_review", "phase": "phase_review"}
BRANCH_MAX_CALLS = {"graph": 48, "phase": 40}
BRANCH_LIMITS = {b: {"openrouter_usd": "1", "xai_usd": "1", "aliyun_cny": "1"} for b in BRANCHES}
TOTAL_LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}


def run_graph(calls, base, selected, h0, prior):
    """Exactly the original one-round graph protocol, starting from original H0."""
    # save() sorts archive keys. Restore original insertion order inside the
    # serialized prompt, so even its text bytes match the initial live run.
    h0 = {task: deepcopy(h0[task]) for task in ORIGINAL_TASKS}
    record = {"prediction": deepcopy(h0), "status": "PROPOSAL_FAILED", "hints": None, "proposal": None,
        "pool": make_pool(h0), "raw_reviews": None, "reviews": None, "format_diagnostics": None,
        "means": None, "diagnostics": None, "issues": [], "error": None,
        "request_fingerprints": {"proposal": None, "reviews": {}},
        "timing": dict.fromkeys(("retrieval_seconds", "proposal_seconds", "compile_seconds",
                                "review_prepare_seconds", "review_seconds", "decision_seconds"), 0.0)}
    start = perf_counter()
    hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
    record["hints"] = hints
    record["timing"]["retrieval_seconds"] = perf_counter() - start
    body = proposal_wire(base, selected, h0, record["pool"], hints["packet"])
    record["request_fingerprints"]["proposal"] = fingerprint(body)
    start = perf_counter()
    raw = calls.call(selected["key"], STAGES["proposal"], "base", body)
    record["timing"]["proposal_seconds"] = perf_counter() - start
    record["proposal"] = raw
    start = perf_counter()
    try:
        if raw is None:
            raise ValueError("missing proposal")
        record["pool"] = make_pool(h0, raw, record["pool"])
    except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        record["timing"]["compile_seconds"] = perf_counter() - start
        return record
    record["timing"]["compile_seconds"] = perf_counter() - start
    if not record["pool"]["propositions"]:
        record["status"] = "EMPTY_POOL_UNVERIFIED"
        return record
    start = perf_counter()
    bodies = {s: review_wire(s, base, selected, record["pool"]) for s in SEATS}
    record["request_fingerprints"]["reviews"] = {s: fingerprint(b) for s, b in bodies.items()}
    record["timing"]["review_prepare_seconds"] = perf_counter() - start
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw_reviews = dict(zip(SEATS, workers.map(
            lambda s: calls.call(selected["key"], STAGES["review"], s, bodies[s]), SEATS), strict=True))
    record["timing"]["review_seconds"] = perf_counter() - start
    start = perf_counter()
    normalized, formatting = normalize_five(raw_reviews, record["pool"], len(base.images))
    means, diagnostics = panel.aggregate(normalized, record["pool"], image_count=len(base.images))
    record.update(raw_reviews=raw_reviews, reviews=normalized, format_diagnostics=formatting,
                  means=means, diagnostics=diagnostics)
    try:
        record["prediction"] = panel.select(h0, record["pool"], means, threshold=4)
        record["issues"] = panel.unresolved(record["prediction"], record["pool"], means, diagnostics, threshold=4)
        record["status"] = "UNRESOLVED" if record["issues"] else "MODEL_PASS"
    except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
        record.update(prediction=deepcopy(h0), status="SELECTION_FAILED", error=type(exc).__name__ + ": " + str(exc))
    count = sum(r["target"] == selected["key"] and r["stage"] == STAGES["review"] for r in calls.rows)
    if count != 5:
        record["status"] = "INCOMPLETE_PANEL"
    record["timing"]["decision_seconds"] = perf_counter() - start
    same(record["prediction"]["phase"], h0["phase"], "graph branch must retain original Phase")
    return record


def collect_phase(calls, selected):
    """Collect independent Phase choices without any current prediction argument."""
    start = perf_counter()
    bodies = {s: phase_wire(s, selected) for s in SEATS}
    hashes = {s: fingerprint(b) for s, b in bodies.items()}
    prepare_seconds = perf_counter() - start
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw = dict(zip(SEATS, workers.map(
            lambda s: calls.call(selected["key"], STAGES["phase"], s, bodies[s]), SEATS), strict=True))
    api_seconds = perf_counter() - start
    start = perf_counter()
    errors = {s: phase_choice_error(raw[s], len(selected["images"])) for s in SEATS}
    return {"raw_reviews": raw, "errors": errors, "request_fingerprints": hashes,
        "status": "INVALID_PANEL" if any(errors.values()) else "REVIEWED",
        "timing": {"prepare_seconds": prepare_seconds, "api_seconds": api_seconds,
                   "validation_seconds": perf_counter() - start}}


def verify_plan(plan):
    for name, value in (("profile", PROFILE), ("models", MODELS), ("proposer", PROPOSER),
        ("providers", PROVIDERS), ("rates", RATES_V2), ("branch_limits", BRANCH_LIMITS),
        ("limits", TOTAL_LIMITS), ("branch_max_calls", BRANCH_MAX_CALLS), ("max_calls", 88),
        ("automatic_retries", 0), ("stages", STAGES)):
        same(plan[name], value, name)
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "runtime source " + name)
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "source archive " + name)
    for name, value in plan["prior_paths_sha256"].items():
        same(sha(name), value, "original LOVO prior")
    for selected in plan["selection"]:
        for im in selected["images"]:
            same(sha(im["path"]), im["sha256"], "original image")


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    old_plan, previous, _, _, _, _ = audit_source(source, adapter)
    originals = {r["key"]: r for r in read(GRAPH_SOURCE / "predictions.json")["targets"]}
    graph_plan = read(GRAPH_SOURCE / "plan.json")
    initials, prior_hashes, preflight = [], {}, {}
    for selected, row in zip(old_plan["selection"], previous, strict=True):
        key = row["key"]
        ordered_h0 = {task: deepcopy(row["h0"][task]) for task in ORIGINAL_TASKS}
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        same(originals[key]["h0"], row["h0"], "cached H0")
        same(originals[key]["arms"]["prior_graph"]["prediction"], row["graph_r1"], "original graph prediction")
        prior_path = GRAPH_SOURCE / "priors" / f"{row['video_id']}.json"
        same(sha(prior_path), graph_plan["prior_sha256"][row["video_id"]], "original prior hash")
        prior = read(prior_path)
        if row["video_id"] in prior["fit_videos"] or any(
            adapter.entries[v].split is not DatasetSplit.TRAINING for v in prior["fit_videos"]):
            raise ValueError("LOVO Training prior required")
        prior_hashes[str(prior_path.resolve())] = sha(prior_path)
        initials.append({k: deepcopy(row[k]) for k in ("key", "video_id", "frame_id", "h0")}
                        | {"previous_graph": deepcopy(row["graph_r1"]), "previous_final": deepcopy(row["phase_short"])})
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "original H0 image protocol")
        hints = retrieve_candidate_hints(ordered_h0, prior, video_id=row["video_id"])
        same(hints, read(GRAPH_SOURCE / "targets" / key / "hints.json"), "original graph hints")
        wire = proposal_wire(base, selected, ordered_h0, make_pool(ordered_h0), hints["packet"])
        same(fingerprint(wire), read(GRAPH_SOURCE / "targets" / key / "prior_graph.json")["proposal_fingerprint"], "same proposer wire")
        phase_selected = old_plan["phase_inputs"][key]["phase_short"]
        phase_hashes = {s: fingerprint(phase_wire(s, phase_selected)) for s in SEATS}
        same(phase_hashes, old_plan["wire_fingerprints"][key]["phase_short"], "same Phase wire")
        preflight[key] = {"proposal": fingerprint(wire), "phase": phase_hashes}
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_parallel_phase_trial.py",
        ROOT / "scripts/run_phase_extension_trial.py", ROOT / "scripts/score_phase_extension_trial.py",
        ROOT / "scripts/run_five_head_repair_trial.py", ROOT / "scripts/score_five_head_repair_trial.py",
        ROOT / "scripts/run_visual_repair_trial.py", ROOT / "scripts/score_visual_repair_trial.py",
        ROOT / "scripts/run_prior_candidate_trial.py", ROOT / "scripts/score_prior_candidate_trial.py",
        ROOT / "scripts/run_prior_feedback_continuation.py", ROOT / "scripts/score_prior_feedback_continuation.py"})
    archive = [source / f"{n}.json" for n in ("plan", "initial_state", "completion", "predictions", "budget")]
    archive += list((source / "calls").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source.resolve()),
        "original_graph_root": str(GRAPH_SOURCE.resolve()), "selection": old_plan["selection"],
        "phase_inputs": {k: v["phase_short"] for k, v in old_plan["phase_inputs"].items()},
        "models": MODELS, "proposer": PROPOSER, "providers": PROVIDERS, "rates": RATES_V2,
        "limits": TOTAL_LIMITS, "branch_limits": BRANCH_LIMITS, "branch_max_calls": BRANCH_MAX_CALLS,
        "max_calls": 88, "automatic_retries": 0, "stages": STAGES, "wire_fingerprints": preflight,
        "prior_paths_sha256": prior_hashes,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "archive_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(archive)},
        "policy": "Cached original H0. Original graph retrieval/proposal/five-seat review and blind Phase five-seat review overlap. Independent branch budgets prevent a Phase provider error stopping graph calls. No review cache reuse, retries, GT or Phase feedback to graph. Merge changes only Phase on the newly computed graph result. Five-valid mean4 original four-head rule and five-valid majority3 Phase rule unchanged.",
        "limitations": ["Eight previously inspected Training targets, not independent validation.",
            "Fresh graph/Phase API responses may vary despite identical requests; old cached predictions are historical references.",
            "H0 is reused, so actual measured latency and charges cover post-H0 repair only.",
            "Per-target branches overlap; target processing remains sequential. No Tracker or Gate.",
            "Serial-equivalent timing is arithmetic from this concurrent run, not a randomized serial control."]}
    save(output / "initial_state.json", {"targets": initials})
    plan["initial_state_sha256"] = sha(output / "initial_state.json")
    save(output / "plan.json", plan)
    save(output / "preflight_metadata.json", metadata)
    for path in dependencies:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"prepared": str(output), "targets": len(initials), "max_calls": 88, "limits": TOTAL_LIMITS}), flush=True)


def combined_budget(output, calls):
    snapshot = {"stopped": all(c.stopped for c in calls.values()), "limits": TOTAL_LIMITS,
        "branch_limits": BRANCH_LIMITS,
        "calls": [{"branch": b, **r} for b in BRANCHES for r in calls[b].rows],
        "branch_budget_sha256": {b: sha(output / "branches" / b / "budget.json") for b in BRANCHES}}
    save(output / "budget.json", snapshot)
    return snapshot


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no paid overwrite")
    verify_plan(plan)
    same(sha(output / "initial_state.json"), plan["initial_state_sha256"], "initial state")
    initials = read(output / "initial_state.json")["targets"]
    rows = [{**deepcopy(i), "graph_only": deepcopy(i["h0"]), "final": deepcopy(i["h0"])} for i in initials]
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = {b: TimedCalls(output / "branches" / b,
        limits={k: Decimal(v) for k, v in BRANCH_LIMITS[b].items()}, rates=RATES_V2,
        providers=PROVIDERS, max_calls=BRANCH_MAX_CALLS[b], reasoning_seats=("grok", "gemini")) for b in BRANCHES}
    for c in calls.values():
        c.persist()
    save(output / "predictions.json", {"targets": rows})
    start, fatal = perf_counter(), None
    try:
        for selected, row in zip(plan["selection"], rows, strict=True):
            input_start = perf_counter()
            base = build_gemini_base(adapter, selected)
            same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "same images")
            prior = read(Path(plan["original_graph_root"]) / "priors" / f"{selected['video_id']}.json")
            phase_selected = plan["phase_inputs"][row["key"]]
            input_seconds = perf_counter() - input_start
            result = run_parallel_repair(row["h0"],
                lambda h0, base=base, selected=selected, prior=prior: run_graph(calls["graph"], base, selected, h0, prior),
                lambda phase_selected=phase_selected: collect_phase(calls["phase"], phase_selected), image_count=len(base.images))
            result["input_seconds"] = input_seconds
            same(result["graph"]["request_fingerprints"]["proposal"], plan["wire_fingerprints"][row["key"]]["proposal"], "frozen proposer")
            same(result["phase"]["request_fingerprints"], plan["wire_fingerprints"][row["key"]]["phase"], "frozen Phase")
            save(output / "targets" / row["key"] / "pipeline.json", result)
            row.update(graph_only=result["graph"]["prediction"], final=result["final"])
            save(output / "predictions.json", {"targets": rows})
            ledger = combined_budget(output, calls)
            print(json.dumps({"target": row["key"], "graph_status": result["graph"]["status"],
                "phase_status": result["phase"]["status"], "phase_decision": result["phase_decision"],
                "parallel_seconds": result["timing"]["parallel_seconds"], "calls": len(ledger["calls"])}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        for c in calls.values():
            c.stopped = True
            c.persist()
        ledger = combined_budget(output, calls)
        save(output / "predictions.json", {"targets": rows})
        inference_seconds = perf_counter() - start
        validation_start = perf_counter()
        try:
            verify_plan(plan)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            artifacts = [*(output / "branches").rglob("*.json"), *(output / "targets").rglob("*.json")]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                **{f"{n}_sha256": sha(output / f"{n}.json") for n in ("plan", "initial_state", "predictions", "budget")},
                "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(artifacts)},
                "post_calls": len(ledger["calls"]), "statuses": dict(Counter(c["status"] for c in ledger["calls"])),
                "inference_seconds": inference_seconds, "closure_validation_seconds": perf_counter() - validation_start,
                "new_h0_calls": 0, "query_gt_not_loaded_during_inference": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.source, adapter)
    else:
        execute(args.output, adapter)
