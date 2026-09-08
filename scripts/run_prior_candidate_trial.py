"""Isolated paired trial: Training-prior graph hints BEFORE visual proposal.

No default pipeline changes. Shared fresh H0; distinct baseline/hinted proposals;
unchanged five-seat review and repair. Single-use paid run, separate GT scoring.
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

from scripts.check_candidate_panel_providers import redact_images
from scripts.prepare_final_only_gate_training import historical_targets, select_pilot
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls, check_sources, sources
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_panel_trial import now, read, save, source_paths, truth_row
from scripts.run_recent_mean_panel_trial import (
    MODELS,
    PROVIDERS,
    RATES_V2,
    ROUTES,
    review_wire,
)
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import (
    MAX_CHARS,
    retrieve_candidate_hints,
)
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import fit_prior, video_counts

PROFILE = "lovo_prior_guided_visual_proposer_v1"
ARMS = ("control", "prior_graph")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VIDEOS = ("VID103", "VID23", "VID31", "VID96")
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}


def proposal_wire(base, selected, h0, pool, hints=None):
    body = gemini_proposal(base, selected, h0, pool, [])
    if hints is not None:
        if len(json.dumps(hints, ensure_ascii=False)) > MAX_CHARS:
            raise ValueError("prior hint text exceeds frozen bound")
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        packet["candidate_relation_hints"] = deepcopy(hints)
        body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def verify(plan, output):
    check_sources(plan)
    for path, value in plan["knowledge_source_sha256"].items():
        if sha(path) != value:
            raise ValueError("Training knowledge source changed")
    for video, value in plan["prior_sha256"].items():
        if sha(output / "priors" / f"{video}.json") != value:
            raise ValueError("leave-video-out prior changed")


def prepare(output, adapter, dataset_root):
    if output.exists():
        raise ValueError("new output directory required")
    excluded, prior_plans = historical_targets(ROOT / "artifacts/preflight")
    pilot = ROOT / "artifacts/training/gate/final_only_preparation_20260908_v2/pilot_selection.json"
    if pilot.exists():
        prior_plans[str(pilot)] = sha(pilot)
        for row in read(pilot)["selection"]:
            excluded.setdefault(row["video_id"], set()).add(row["frame_id"])
    selected, inventory = select_pilot(adapter, VIDEOS, per_video=2, excluded=excluded)
    if len(selected) != 8:
        raise ValueError("all eight time/mask-only targets are required")
    for i, row in enumerate(selected):
        if len(row["causal_frame_ids"]) != 3:
            raise ValueError("three real causal images required")
        row["key"] = f"{row['video_id']}_{row['frame_id']}"
        row["arm_order"] = list(ARMS if i % 2 == 0 else reversed(ARMS))
        row["request_metadata"] = canonical_request_metadata(build_gemini_base(adapter, row)).to_mapping()
    # GT is consumed ONLY here to build Training statistics. The inference
    # request for a video receives a table fitted without that entire video.
    counts, knowledge_sources = {}, {}
    for video, entry in sorted(adapter.entries.items()):
        if entry.split is not DatasetSplit.TRAINING:
            continue
        rows = []
        seen_provenance = set()
        for resolved in adapter.iter_video(video):
            rows.append(truth_row(resolved))
            provenance_key = tuple(getattr(resolved.provenance, field) for field in
                                   ("annotation_source", "phase_source", "frame_action_source", "manifest_source"))
            if provenance_key not in seen_provenance:
                seen_provenance.add(provenance_key)
                for path in source_paths(resolved, dataset_root):
                    if str(path) not in knowledge_sources:
                        knowledge_sources[str(path)] = sha(path)
        counts[video] = video_counts(rows)
    priors = {video: fit_prior(counts, video) for video in VIDEOS}
    for video, prior in priors.items():
        if video in prior["fit_videos"] or any(adapter.entries[v].split is not DatasetSplit.TRAINING
                                             for v in prior["fit_videos"]):
            raise ValueError("knowledge source leakage")

    def endpoint(item):
        seat, route = item
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        result = response.json()
        chosen = next(e for e in result["data"]["endpoints"] if e["tag"] == route)
        if chosen["status"] != 0 or not {"response_format", "reasoning"} <= set(chosen["supported_parameters"]):
            raise ValueError("fixed route unavailable")
        if any(Decimal(chosen["pricing"][k]) > Decimal(rate)
               for k, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("current price exceeds reserve")
        return seat, result

    with ThreadPoolExecutor(max_workers=4) as workers:
        metadata = dict(workers.map(endpoint, ROUTES.items()))
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_prior_candidate_trial.py",
                           ROOT / "scripts/prepare_final_only_gate_training.py",
                           ROOT / "src/surgical_agent/research/signals/resources/ivt_components_v1.csv"})
    if any(not p.is_file() for p in dependencies):
        raise ValueError("missing source before freeze")
    for video, prior in priors.items():
        save(output / "priors" / f"{video}.json", prior)
    plan = {"profile": PROFILE, "created_utc": now(), "selection": selected,
            "selection_rule": "two chronological quantiles per complete-GT Training video, nearest mask-valid target beyond 75 raw frames from every prior/planned target; no GT labels or scores used for selection",
            "selection_inventory": inventory, "excluded_selection_sources": prior_plans,
            "arms": list(ARMS), "h0": PROPOSER, "models": MODELS, "threshold": 4, "round_cap": 1,
            "max_calls": 104, "limits": LIMITS, "rates": RATES_V2,
            "treatment": "At most two complete IVT hypotheses from leave-video-out Training priors appended ONLY to the existing visual proposer. Same response schema and maximum four newly proposed IVTs; original five reviewers see no prior text or source membership.",
            "prior_policy": "Existing video-balanced fit_prior, >=100 valid frames, >=3 valid videos, >=5 positive frames and >=2 positive videos; one phase-informed slot and one global slot, same instrument and one-component relation neighbors preferred; phase/global fallbacks never make absence claims.",
            "knowledge_scope": "Training frame-label cooccurrence on whole-IVT nodes plus existing ontology, no reference examples, no instance correspondence, no medical appearance knowledge, no GT from query video in its prior",
            "gt_policy": "Offline Training-prior fitting is authorized; query-video GT never enters its inference. All paid outputs close before independent score. Testing/Validation excluded from fitting, selection and tuning. This is development LOVO retrieval, not independent full-system OOF/Gate evaluation.",
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dependencies},
            "knowledge_source_sha256": knowledge_sources,
            "prior_sha256": {v: sha(output / "priors" / f"{v}.json") for v in priors},
            "success_rule": "Candidate IVT coverage increase alone is insufficient; seek IVT F1 or exact-set gain vs control without decreased Target metrics or increased total label errors; report all heads, harmful edits and failures.",
            "latency_target": "No extra deployed calls; unchanged candidate upper bound. Paired five-seat wall-time median increase <=10%, provider variation reported."}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    save(output / "training_counts.json", counts)
    for row in selected:
        save(output / "h0_preflight" / f"{row['key']}.json", redact_images(gemini_h0_wire(build_gemini_base(adapter, row))))
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": [r["key"] for r in selected],
                      "max_calls": 104, "limits": LIMITS,
                      "ivt_source_videos_per_query": {v: priors[v]["tasks"]["ivt"]["global"][0]["valid_videos"] for v in VIDEOS}}), flush=True)


def review(calls, base, selected, h0, pool, arm, reusable):
    bodies = {seat: review_wire(seat, base, selected, pool) for seat in SEATS}
    record = {"pool": pool, "request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()}}
    shared = next((r for r in reusable if r["bodies"] == bodies), None)
    if shared is None:
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], f"{arm}_review", s, bodies[s]), SEATS), strict=True))
        elapsed = perf_counter() - start
        n = sum(r["target"] == selected["key"] and r["stage"] == f"{arm}_review" for r in calls.rows)
        record.update(panel_seconds=elapsed, call_count=n)
        reusable.append({"arm": arm, "bodies": bodies, "raw": deepcopy(raw), "count": n, "seconds": elapsed})
    else:
        raw, n = deepcopy(shared["raw"]), shared["count"]
        record.update(shared_from=shared["arm"], panel_seconds=None, call_count=0,
                      original_panel_seconds=shared["seconds"])
    normalized, formatting = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(normalized, pool, image_count=len(base.images))
    try:
        prediction = panel.select(h0, pool, means, threshold=4)
        issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
        status = "UNRESOLVED" if issues else "MODEL_PASS"
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        prediction, issues, status = deepcopy(h0), [], "SELECTION_FAILED"
    if n != 5:
        status = "INCOMPLETE_PANEL"
    record.update(prediction=prediction, status=status, raw=raw, reviews=normalized,
                  format_diagnostics=formatting, means=means, diagnostics=diagnostics, issues=issues)
    return record


def execute(output, adapter):
    plan = read(output / "plan.json")
    if plan["profile"] != PROFILE or (output / "execution.lock").exists():
        raise ValueError("wrong profile or paid rerun prohibited")
    verify(plan, output)
    bases = {r["key"]: build_gemini_base(adapter, r) for r in plan["selection"]}
    for row in plan["selection"]:
        if canonical_request_metadata(bases[row["key"]]).to_mapping() != row["request_metadata"]:
            raise ValueError("H0 wire changed after preflight")
    with (output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                       rates=plan["rates"], providers=PROVIDERS, max_calls=plan["max_calls"],
                       reasoning_seats=("grok", "gemini"))
    targets = [{"key": r["key"], "video_id": r["video_id"], "frame_id": r["frame_id"], "h0": None,
                "arms": {a: {"prediction": None, "status": "NOT_ATTEMPTED"} for a in ARMS}}
               for r in plan["selection"]]
    fatal = None
    try:
        for selected, state in zip(plan["selection"], targets, strict=True):
            if calls.stopped:
                break
            key, base = selected["key"], bases[selected["key"]]
            raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
            try:
                validate_final_only(raw)
            except (ApiSchemaError, TypeError, ValueError):
                for arm in state["arms"].values():
                    arm["status"] = "H0_FAILED"
                calls.stopped = True
                break
            h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
            state["h0"] = h0
            initial = make_pool(h0)
            for arm in state["arms"].values():
                arm.update(prediction=deepcopy(h0), pool=deepcopy(initial))
            start = perf_counter()
            hint = retrieve_candidate_hints(h0, read(output / "priors" / f"{selected['video_id']}.json"), video_id=selected["video_id"])
            state["retrieval_seconds"] = perf_counter() - start
            save(output / "targets" / key / "hints.json", hint)
            reused_proposals, reused_reviews = [], []
            for arm in selected["arm_order"]:
                if calls.stopped:
                    break
                wire = proposal_wire(base, selected, h0, initial, hint["packet"] if arm == "prior_graph" else None)
                reused = next((r for r in reused_proposals if r["wire"] == wire), None)
                if reused:
                    proposal = deepcopy(reused["proposal"])
                else:
                    proposal = calls.call(key, f"{arm}_proposal", "base", wire)
                    reused_proposals.append({"arm": arm, "wire": wire, "proposal": deepcopy(proposal)})
                record = {"proposal": proposal, "proposal_fingerprint": fingerprint(wire),
                          "proposal_shared_from": reused["arm"] if reused else None,
                          "prediction": deepcopy(h0), "pool": deepcopy(initial), "status": "PROPOSAL_FAILED"}
                try:
                    if proposal is None:
                        raise ValueError("missing proposal")
                    pool = make_pool(h0, proposal, initial)
                except (ApiSchemaError, TypeError, ValueError, KeyError):
                    pass
                else:
                    record["pool"] = pool
                    if pool["propositions"]:
                        record.update(review(calls, base, selected, h0, pool, arm, reused_reviews))
                    else:
                        record["status"] = "EMPTY_POOL_UNVERIFIED"
                save(output / "targets" / key / f"{arm}.json", record)
                state["arms"][arm] = {k: v for k, v in record.items() if k not in
                                      {"raw", "reviews", "format_diagnostics", "means", "diagnostics", "issues"}}
                save(output / "predictions.json", {"targets": targets})
                print(json.dumps({"target": key, "arm": arm, "status": record["status"],
                                  "candidate_count": len(record["pool"]["propositions"]),
                                  "panel_seconds": record.get("panel_seconds"), "post_calls": len(calls.rows)}), flush=True)
        verify(plan, output)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": targets})
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
             "predictions_sha256": sha(output / "predictions.json"), "plan_sha256": sha(output / "plan.json"),
             "budget_sha256": sha(output / "budget.json"), "post_calls": len(calls.rows),
             "call_statuses": dict(Counter(r["status"] for r in calls.rows)),
             "occupied": {k: str(v) for k, v in calls.occupied.items()},
             "all_targets_have_h0": all(r["h0"] is not None for r in targets),
             "query_gt_scoring_not_run_during_inference": True})
    print(json.dumps(read(output / "completion.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    dataset = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, dataset, args.dataset_root)
    else:
        execute(args.output, dataset)
