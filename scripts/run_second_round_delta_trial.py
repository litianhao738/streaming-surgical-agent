"""Two continuations of cached Gemini H0/R1: original R2 vs sparse LLM repair/R2.

No new H0 calls. Original first-round requests, predictions and reviews remain
immutable. The LLM arm returns edits only, then five judges review their scope.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from functools import partial
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import PROVIDERS, RATES, Calls, sha
from scripts.run_complete_gt_semantic_trial import normalize_review_wire, review_body_v3
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification import candidate_coordinator as candidates
from surgical_agent.research.verification import semantic_coordinator as semantic
from surgical_agent.research.verification.delta_repair import (
    apply_reviewed_delta,
    compile_delta,
    delta_schema,
    issue_evidence,
)

SOURCE = ROOT / "artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1"
ALLOWANCE = {"openrouter_usd": Decimal("1.20"), "xai_usd": Decimal("0.50"), "aliyun_cny": Decimal("1.00")}


def delta_body(base, selected, current, history):
    body = gemini_proposal(base, selected, current, history["pool"], [])
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet["task"] = "Propose a minimal evidence-grounded repair of the CURRENT-frame label sets."
    packet["instructions"] = (
        "Return changes only; do not repeat the full prediction, unchanged labels, ontology names or a report. "
        "The current answer and prior reviewer opinions may be wrong. Reinspect all visible tools, including "
        "frame-edge tools and their actual action-object relationships. Check the target frame; use history for motion. "
        "Old candidate membership and agreement are not visual evidence. You may introduce an ID from the full "
        "ontology even if it was absent from the old candidate pool. This is a hypothesis for a separate review, "
        "not a correctness claim. Do not change Phase. Only propose actual ADD/REMOVE differences. "
        "Each changed candidate_id is task_ID (e.g. instrument_0); operation, rating, finding, scope, "
        "image_indices and observation are required. ADD needs MATCH and rating4/5 with current-frame evidence; "
        "REMOVE needs REFUTED and rating1/2, WHOLE_FRAME evidence excluding this label across all tools. "
        "Occlusion, uncertainty, or one local negative cannot justify a frame-level deletion. "
        "Cite the current image index for each change, not just history. "
        "For an added IVT, explicitly ADD any of its instrument/verb/target labels missing from the current answer. "
        "Removing an IVT does not automatically remove components; another relation may still require them. "
        "Do not remove a component needed by any retained/new IVT. Leave all unrelated predictions unchanged. "
        "Return an empty changes array if no edit has visual support; at most24 changes. "
        "Keep each observation to one short factual sentence where possible (hard limit1000 characters).")
    packet["issues"] = issue_evidence(current, history)
    packet["response_schema"] = delta_schema(len(base.images))
    packet["current_image_index"] = len(base.images) - 1
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(reasoning={"effort": "medium"}, max_tokens=8192, response_format={"type": "json_object"})
    return body


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    source_plan, budget = read(source / "plan.json"), read(source / "budget.json")
    if not budget["stopped"] or source_plan["h0"] != PROPOSER or source_plan["round_cap"] != 1:
        raise ValueError("requires finished Gemini H0 / one-round source")
    selected = source_plan["selection"]
    inputs = [source / "plan.json", source / "budget.json", source / "predictions.json"]
    for s in selected:
        base = build_gemini_base(adapter, s)
        if canonical_request_metadata(base).to_mapping() != s["request_metadata"]:
            raise ValueError("cached H0 image/request identity changed")
        result = read(source / "targets" / s["key"] / "result.json")
        if len(result["history"]) != 1 or result["history"][0]["status"] != "VALID":
            raise ValueError("requires a validated first-round review")
        inputs.append(source / "targets" / s["key"] / "result.json")
    url = f"https://openrouter.ai/api/v1/models/{PROPOSER}/endpoints"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    endpoint = next(e for e in metadata["data"]["endpoints"] if e["tag"] == "google-ai-studio")
    if endpoint["status"] != 0 or any(Decimal(endpoint["pricing"][p]) > Decimal(RATES["base"][i])
                                    for i, p in enumerate(("prompt", "completion"))):
        raise ValueError("provider unavailable or over price envelope")
    paths = [ROOT / path for path in source_plan["source_sha256"]]
    paths += [Path(__file__), ROOT / "src/surgical_agent/research/verification/delta_repair.py",
              ROOT / "tools/audit/audit_second_round_delta.py"]
    paths = sorted(set(paths))
    plan = {"created_utc": now(), "source": str(source), "selection": selected,
            "max_calls": 48, "additional_rounds_per_arm": 1, "new_h0_calls": 0,
            "arms": ["cached_r1", "original_flow_r2", "llm_sparse_tentative", "llm_sparse_reviewed"],
            "shared_input_sha256": {str(p): sha(p) for p in inputs},
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "carried_occupied": budget["occupied"],
            "limits": {k: str(Decimal(budget["occupied"][k]) + v) for k, v in ALLOWANCE.items()},
            "incremental_allowances": {k: str(v) for k, v in ALLOWANCE.items()},
            "gt_policy": "reuse exact four Training development targets; labels scored after inference stops; no Testing",
            "h0_model": PROPOSER, "original_refill_reasoning": "low", "delta_reasoning": "medium",
            "comparison_limit": "original vs sparse arms differ in repair role, scope and reasoning; not isolated token-format effect",
            "stopping": "one extra round; original stops on no new candidates, delta stops on empty/invalid patch; no automatic retry"}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    for path in paths:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"targets": [s["key"] for s in selected], "max_calls": 48,
                      "new_h0_calls": 0, "allowances": plan["incremental_allowances"]}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "budget.json").exists():
        raise ValueError("no paid overwrite or automatic replay")
    def verify_inputs():
        for key, value in plan["source_sha256"].items():
            if sha(ROOT / key) != value:
                raise ValueError("source changed after prepare")
        for key, value in plan["shared_input_sha256"].items():
            if sha(Path(key)) != value:
                raise ValueError("cached first round changed")
    verify_inputs()
    source = Path(plan["source"])
    calls = Calls(output, plan["carried_occupied"], limits={k: Decimal(v) for k, v in plan["limits"].items()},
                  providers={**PROVIDERS, PROPOSER: "Google AI Studio"}, max_calls=48)
    rows = {"round1": [], "original_round2": [], "delta_round2": []}
    for s in plan["selection"]:
        key, base = s["key"], build_gemini_base(adapter, s)
        original = read(source / "targets" / key / "result.json")
        first, h0, current = original["history"][0], original["h0"], original["final"]
        folder = output / "targets" / key
        def panel(pool, stage, base=base, s=s, key=key):
            def one(seat):
                body = review_body_v3(seat, base, s, pool)
                return normalize_review_wire(seat, calls.call(key, stage, seat, body), pool)
            with ThreadPoolExecutor(max_workers=5) as workers:
                raw = dict(zip(candidates.SEATS, workers.map(one, candidates.SEATS), strict=True))
            return raw
        def propose(n, state, pool, issues, first=first, key=key, base=base, s=s):
            if n == 0:
                return first["proposal"]
            return calls.call(key, "original_refill_1", "base", gemini_proposal(base, s, state, pool, issues))
        def review(n, state, pool, first=first, panel=panel):
            del state
            if n == 1:
                if pool != first["pool"]:
                    raise ValueError("replayed first-round candidate pool differs")
                return first["raw"]
            return panel(pool, "original_review_2")
        classic = candidates.run(h0, propose, review,
            lambda stage, value, folder=folder: save(folder / "original" / f"{stage}.json", value),
            aggregate_review=partial(semantic.aggregate, image_count=len(base.images)), max_rounds=2)
        if classic["snapshots"][0] != candidates.labels(current):
            raise ValueError("first-round replay changed accepted state")
        save(folder / "original_result.json", classic)
        raw_delta = calls.call(key, "llm_delta_2", "base", delta_body(base, s, current, first))
        tentative, final, compiled = None, current, None
        delta = {"raw": raw_delta, "input_state": current, "status": "INVALID_DELTA", "new_review_calls": 0}
        try:
            compiled = compile_delta(current, raw_delta, len(base.images))
            tentative = compiled["tentative"]
            delta["compiled"] = compiled
        except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
            delta["validation_error"] = type(exc).__name__
        if compiled is not None:
            if compiled["changes"]:
                raw = panel(compiled["review_pool"], "delta_review_2")
                means, normalized = semantic.aggregate(raw, compiled["review_pool"], image_count=len(base.images))
                final = apply_reviewed_delta(compiled, means)
                delta.update(status="REVIEWED", review_raw=raw, means=means, normalized=normalized,
                             new_review_calls=5)
            else:
                delta["status"] = "NO_CHANGES_PROPOSED"
            counterfactual = candidates.make_pool(current, previous={"propositions":
                first["pool"]["propositions"] + compiled["full_pool"]["propositions"]})
            delta["review_item_counts"] = {"targeted": len(compiled["review_pool"]["propositions"]),
                                            "full_pool": len(counterfactual["propositions"])}
        delta.update(final=final, tentative=tentative)
        save(folder / "delta_result.json", delta)
        identity = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0}
        rows["round1"].append({**identity, "h1": None, "final": current, "status": "CACHED_FIRST_ROUND"})
        rows["original_round2"].append({**identity, "h1": None, "final": classic["final"], "status": classic["stop_reason"]})
        rows["delta_round2"].append({**identity, "h1": tentative, "final": final, "status": delta["status"]})
        for name, values in rows.items():
            save(output / f"{name}_predictions.json", values)
        print(json.dumps({"target": key, "original_stop": classic["stop_reason"],
                          "original_rounds": len(classic["history"]), "delta_status": delta["status"],
                          "delta_changes": len(compiled["changes"]) if compiled else None,
                          "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    verify_inputs()
    hashes = {name: sha(output / f"{name}_predictions.json") for name in rows}
    reports = {}
    for name, values in rows.items():
        report, truth = score_saved(adapter, values)
        reports[name] = report
        save(output / f"{name}_comparison.json", report)
        save(output / f"{name}_scored.json", truth)
        if sha(output / f"{name}_predictions.json") != hashes[name]:
            raise ValueError("scoring changed predictions")
    summary = {"post_calls": len(calls.rows), "new_h0_calls": 0, "prediction_sha256": hashes,
               "comparisons": reports,
               "native_costs": {k: str(sum(Decimal(r["charge"]) for r in calls.rows
                                          if r["account"] == k and r["charge_kind"] == "native")) for k in ALLOWANCE},
               "estimated_aliyun_cny": str(sum(Decimal(r["charge"]) for r in calls.rows
                                               if r["charge_kind"] == "conservative_estimate")),
               "unknown_cost_calls": [r["index"] for r in calls.rows if r["charge_kind"] == "unknown_reserved"]}
    save(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.source, adapter)
    else:
        execute(args.output, adapter)
