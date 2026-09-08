"""Isolated one-round none/flat/graph comparison; existing entry points unchanged.

Shared fresh H0 and proposal, exact same five parallel reviewer seats. Retrieval
is local and cannot edit the pool, images, response contract or acceptance rule.
Paid execution is single-use; GT scoring is a separate command after closure.
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
from itertools import permutations
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_presence_review_trial import _mask_only
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import (
    MODELS,
    PROVIDERS,
    RATES_V2,
    ROUTES,
    review_wire,
)
from scripts.run_repair_revision_trial import (
    RevisionCalls,
    frozen_sources,
    normalize_five,
)
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.graph_review import knowledge_manifest, retrieve
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool

PROFILE = "isolated_local_graph_review_v1"
ARMS = ("none", "flat", "graph")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}


def add_reference(body, packet):
    """None preserves the original complete wire; no new output requirements."""
    result = deepcopy(body)
    if packet is None:
        return result
    encoded = json.dumps(packet, ensure_ascii=False)
    if len(encoded) > 2500:
        raise ValueError("reference packet exceeds the frozen character limit")
    text = json.loads(result["messages"][0]["content"][0]["text"])
    if "reference_knowledge" in text:
        raise ValueError("refuse to overwrite an existing reference block")
    text["reference_knowledge"] = packet
    result["messages"][0]["content"][0]["text"] = json.dumps(text, ensure_ascii=False)
    return result


class TimedCalls(RevisionCalls):
    def call(self, target, stage, seat, body):
        start = perf_counter()
        try:
            return super().call(target, stage, seat, body)
        finally:
            elapsed = perf_counter() - start
            with self.lock:
                rows = [r for r in self.rows if (r["target"], r["stage"], r["seat"]) == (target, stage, seat)]
                if len(rows) == 1:
                    row = rows[0]
                    row["elapsed_seconds"] = elapsed
                    folder = self.output / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
                    save(folder / "record.json", row)
                    self.persist()


def sources():
    return sorted({*frozen_sources(), Path(__file__).resolve(),
                   ROOT / "scripts/run_evidence_feedback_trial.py",
                   ROOT / "scripts/score_graph_review_trial.py",
                   *(ROOT / "src/surgical_agent/research/retrieval").rglob("*.py"),
                   *(ROOT / "src/surgical_agent/research/retrieval").rglob("*.json")})


def check_sources(plan):
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen source changed: " + name)
    for row in plan["selection"]:
        for image in row["images"]:
            if sha(image["path"]) != image["sha256"]:
                raise ValueError("frozen causal image changed")


def prepare(output, selection_path, adapter):
    if output.exists():
        raise ValueError("prepare requires a new output directory")
    selection = read(selection_path)
    if selection.get("no_gt_label_values_used_for_selection") is not True:
        raise ValueError("selection must document time/mask-only sampling")
    rows = deepcopy(selection["selection"])
    if not 1 <= len(rows) <= 6 or len({(r["video_id"], r["frame_id"]) for r in rows}) != len(rows):
        raise ValueError("one to six unique Training targets required")
    orders = list(permutations(ARMS))
    masks = {}
    for video in dict.fromkeys(r["video_id"] for r in rows):
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("Training targets required")
        frame_ids = {r["frame_id"] for r in rows if r["video_id"] == video}
        masks.update({(video, r.inference.target_frame_id): _mask_only(r)
                      for r in adapter.iter_video(video, frame_ids=frame_ids)})
    for index, row in enumerate(rows):
        if adapter.entries[row["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Testing/Validation cannot select this experiment")
        row["key"] = f"{row['video_id']}_{row['frame_id']}"
        if masks.get((row["video_id"], row["frame_id"])) != row["gt_availability_only"]:
            raise ValueError("actual GT availability differs from selection manifest")
        sample = next(s for s in adapter.iter_inference_video(row["video_id"]) if s.target_frame_id == row["frame_id"])
        if (len(row["images"]) != len(sample.media_refs)
                or any(Path(im["path"]).resolve() != Path(path).resolve()
                       for im, path in zip(row["images"], sample.media_refs, strict=True))):
            raise ValueError("declared image paths differ from canonical causal images")
        row["arm_order"] = list(orders[index])
        base = build_gemini_base(adapter, row)
        row["request_metadata"] = canonical_request_metadata(base).to_mapping()

    def endpoint(item):
        seat, route = item
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        chosen = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if chosen["status"] != 0 or not {"response_format", "reasoning"} <= set(chosen["supported_parameters"]):
            raise ValueError("fixed provider unavailable: " + seat)
        if any(Decimal(chosen["pricing"][k]) > Decimal(rate)
               for k, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("provider price exceeds conservative reserve: " + seat)
        return seat, data

    with ThreadPoolExecutor(max_workers=4) as workers:
        metadata = dict(workers.map(endpoint, ROUTES.items()))
    dependencies = sources()
    if any(not p.is_file() for p in dependencies):
        raise ValueError("all dependencies must exist before freezing")
    plan = {"profile": PROFILE, "created_utc": now(), "selection": rows, "arms": list(ARMS),
            "selection_manifest": selection, "selection_sha256": sha(selection_path),
            "models": MODELS, "h0": PROPOSER, "threshold": 4, "round_cap": 1,
            "max_calls": len(rows) * 17, "limits": LIMITS, "rates": RATES_V2,
            "retrieval_policy": {"max_reference_chars": 2500, "max_items": 4, "max_items_per_candidate": 2},
            "knowledge_scope": "Graph-organized existing project definitions and ontology; no new medical knowledge, GT statistics or reference images.",
            "treatment": "Only append reference_knowledge before the five reviewers; shared H0/proposal/pool/images, unchanged response schema and mean threshold 4.",
            "shared_review_policy": "Exact complete-body equality shares original replies, including failures; no independent latency/effect claim for reused arms.",
            "gt_policy": "No GT values consumed by inference. Score all planned targets with per-task masks after paid ledger closure and immutable prediction hash.",
            "semantic_rule": "Graph must improve Target or IVT micro-F1 or exact-set accuracy versus none, with neither head metric decreasing and no increased total label errors; report flat control and all head regressions. Small development evidence only.",
            "latency_rule": "No extra POST or review rounds per deployed arm. Report paired panel wall times and added token counts; median graph/none wall ratio <=1.10 is the engineering target, not a statistical guarantee.",
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dependencies}}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    save(output / "knowledge_manifest.json", knowledge_manifest())
    for row in rows:
        save(output / "h0_preflight" / f"{row['key']}.json", redact_images(gemini_h0_wire(build_gemini_base(adapter, row))))
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": [r["key"] for r in rows],
                      "max_calls": plan["max_calls"], "limits": LIMITS}), flush=True)


def run_panel(calls, base, selected, h0, pool, arm, reuse):
    start = perf_counter()
    retrieved = retrieve(pool, mode=arm)
    retrieval_seconds = perf_counter() - start
    packet = retrieved["reference_knowledge"]
    bodies = {s: add_reference(review_wire(s, base, selected, pool), packet) for s in SEATS}
    record = {"prediction": deepcopy(h0), "status": "NOT_ATTEMPTED", "retrieval": retrieved,
              "retrieval_seconds": retrieval_seconds, "reference_chars": 0 if packet is None else
              len(json.dumps(packet, ensure_ascii=False)),
              "request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()}}
    shared = next((item for item in reuse if all(item["bodies"][s] == bodies[s] for s in SEATS)), None)
    if shared is not None:
        raw = deepcopy(shared["raw"])
        record.update(shared_from=shared["arm"], panel_seconds=None, call_count=0,
                      original_panel_seconds=shared["panel_seconds"])
        dispatched = shared["dispatched"]
    else:
        start = perf_counter()
        def one(seat):
            return calls.call(selected["key"], f"{arm}_review", seat, bodies[seat])
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
        elapsed = perf_counter() - start
        dispatched = sum(r["target"] == selected["key"] and r["stage"] == f"{arm}_review" for r in calls.rows)
        record.update(panel_seconds=elapsed, call_count=dispatched)
        reuse.append({"arm": arm, "bodies": bodies, "raw": deepcopy(raw), "dispatched": dispatched, "panel_seconds": elapsed})
    reviews, formatting = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(reviews, pool, image_count=len(base.images))
    try:
        final = panel.select(h0, pool, means, threshold=4)
        issues = panel.unresolved(final, pool, means, diagnostics, threshold=4)
        status = "UNRESOLVED" if issues else "MODEL_PASS"
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        final, issues, status = deepcopy(h0), [], "SELECTION_FAILED"
    if dispatched != 5:
        status = "INCOMPLETE_PANEL"
    record.update(prediction=final, status=status, raw=raw, reviews=reviews,
                  format_diagnostics=formatting, means=means, diagnostics=diagnostics, issues=issues)
    return record


def execute(output, adapter):
    plan = read(output / "plan.json")
    if plan["profile"] != PROFILE or (output / "execution.lock").exists():
        raise ValueError("wrong profile or paid replay prohibited; use a fresh experiment")
    check_sources(plan)
    bases = {r["key"]: build_gemini_base(adapter, r) for r in plan["selection"]}
    for row in plan["selection"]:
        if canonical_request_metadata(bases[row["key"]]).to_mapping() != row["request_metadata"]:
            raise ValueError("H0 preflight request changed")
    with (output / "execution.lock").open("x") as marker:
        marker.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                       rates=plan["rates"], providers=PROVIDERS, max_calls=plan["max_calls"],
                       reasoning_seats=("grok", "gemini"))
    targets = [{"key": r["key"], "video_id": r["video_id"], "frame_id": r["frame_id"], "h0": None,
                "arms": {a: {"prediction": None, "status": "NOT_ATTEMPTED"} for a in ARMS}}
               for r in plan["selection"]]
    fatal_error = None
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
            state["arms"] = {a: {"prediction": deepcopy(h0), "status": "NOT_ATTEMPTED"} for a in ARMS}
            initial_pool = make_pool(h0)
            proposal = calls.call(key, "shared_proposal", "base", gemini_proposal(base, selected, h0, initial_pool, []))
            try:
                if proposal is None:
                    raise ValueError("proposal unavailable")
                pool = make_pool(h0, proposal, initial_pool)
            except (ApiSchemaError, TypeError, ValueError, KeyError):
                for arm in state["arms"].values():
                    arm["status"] = "PROPOSAL_FAILED"
                save(output / "predictions.json", {"targets": targets})
                continue
            save(output / "targets" / key / "shared.json", {"h0": h0, "proposal": proposal, "pool": pool})
            if not pool["propositions"]:
                for arm in state["arms"].values():
                    arm["status"] = "EMPTY_POOL_UNVERIFIED"
                continue
            reusable = []
            for arm in selected["arm_order"]:
                if calls.stopped:
                    break
                record = run_panel(calls, base, selected, h0, pool, arm, reusable)
                save(output / "targets" / key / f"{arm}.json", record)
                state["arms"][arm] = {k: v for k, v in record.items() if k not in
                                      {"raw", "reviews", "format_diagnostics", "means", "diagnostics", "retrieval", "issues"}}
                save(output / "predictions.json", {"targets": targets})
                print(json.dumps({"target": key, "arm": arm, "status": record["status"],
                                  "panel_seconds": record["panel_seconds"], "reference_chars": record["reference_chars"],
                                  "post_calls": len(calls.rows)}), flush=True)
        check_sources(plan)
    except Exception as exc:
        fatal_error = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": targets})
        save(output / "completion.json", {"closed_utc": now(), "predictions_sha256": sha(output / "predictions.json"),
             "plan_sha256": sha(output / "plan.json"), "budget_sha256": sha(output / "budget.json"),
             "post_calls": len(calls.rows), "call_statuses": dict(Counter(r["status"] for r in calls.rows)),
             "occupied": {k: str(v) for k, v in calls.occupied.items()}, "fatal_error": fatal_error,
             "all_targets_have_h0": all(r["h0"] is not None for r in targets),
             "all_panels_attempted": all(r["arms"][a]["status"] in {"MODEL_PASS", "UNRESOLVED"} for r in targets for a in ARMS),
             "gt_values_not_read_during_inference": True})
    print(json.dumps(read(output / "completion.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        if args.selection is None:
            parser.error("prepare requires --selection")
        prepare(args.output, args.selection, adapter)
    else:
        execute(args.output, adapter)


if __name__ == "__main__":
    main()
