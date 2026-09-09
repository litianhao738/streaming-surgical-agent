"""Same-eight graph R1 -> unscored single-VLM edits -> five-seat review.

Independent version; no H0 regeneration, old-source edits, retries or third
round. Separate prepare/execute/score commands keep inference closed before GT.
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
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from scripts.score_prior_feedback_continuation import audit_records, audit_wires
from scripts.score_prior_feedback_continuation import validate as validate_source
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import digest
from surgical_agent.research.verification.repair_feedback_v2 import (
    build_repair_feedback,
)
from surgical_agent.research.verification.visual_repair import (
    apply_reviewed_repair,
    compile_repair,
    repair_schema,
)

PROFILE = "visual_repair_unscored_hypotheses_v1"
SOURCE = ROOT / "artifacts/preflight/prior_graph_second_round_20260908_v1"
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
REPAIR_STAGE, REVIEW_STAGE = "visual_repair_2", "repair_review_2"


def repair_wire(base, selected, initial):
    old = initial["graph_r1"]
    feedback = build_repair_feedback(old["pool"], old["raw"], old["reviews"], old["issues"], image_count=len(base.images))
    # No aggregate scores or self-certification: observations remain fallible.
    for candidate in feedback["candidates"]:
        for observation in candidate["valid_observations"]:
            observation.pop("rating")
    body = gemini_proposal(base, selected, old["prediction"], old["pool"], [])
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet.update(task="Reinspect the CURRENT frame and propose specific visual repairs for independent verification.",
        instructions=(
            "Inspect all visible tools, including frame edges, and each tool's action and directly acted-on tissue. "
            "Use the historical images to distinguish motion; report current-frame relations only. "
            "The current answer, old candidates and reviewer observations may all be wrong. "
            "Return changes (ADD/REMOVE) and recheck candidate IDs. A replacement is an explicit REMOVE plus ADD. "
            "You may propose any ID in the full ontology, including a missing instrument or a relation outside the old pool. "
            "Give a short visual reason and query image indices for each proposed change; cite the target image. "
            "These are hypotheses awaiting review, so do not supply a self-rating, majority score or final pass claim. "
            "A plausible alternative may be proposed without meeting the final admission threshold. "
            "Do not invent changes merely to fill a quota. When uncertain whether an existing label is wrong, request recheck. "
            "A local negative cannot establish absence across the whole frame; another tool may support the label. "
            "Explicitly request REMOVE if you believe an old relation is wrong; omission never means deletion. "
            "Do not change Phase. Removing an IVT does not automatically remove its component labels. "
            "Required components of a new IVT will be added as separate hypotheses and independently checked. "
            "Do not repeat an ADD already selected or a REMOVE already absent. At most16 changes and16 recheck IDs. "
            "Return empty lists if no plausible repair or recheck is supported. No prose outside the JSON object. "
            "Each observation is at most1000 characters. Check actual tool-action-object relations, not anatomical visibility."),
        response_schema=repair_schema(len(base.images)),
        review_feedback=feedback, candidate_relation_hints=deepcopy(initial["hints"]["packet"]),
        current_image_index=len(base.images) - 1)
    packet.pop("issues", None)
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(reasoning={"effort": "medium"}, max_tokens=8192, response_format={"type": "json_object"})
    return body


def empty_record(initial):
    before = initial["graph_r1"]["prediction"]
    return {"before": deepcopy(before), "temporary": deepcopy(before), "prediction": deepcopy(before),
            "raw_repair": None, "compiled": None, "raw_reviews": None, "reviews": None,
            "format_diagnostics": None, "means": None, "diagnostics": None,
            "status": "NOT_ATTEMPTED", "compile_error": None, "repair_seconds": 0,
            "review_seconds": 0, "review_calls_observed": 0, "request_fingerprints": {}}


def run_target(calls, base, selected, initial, persist=lambda record: None):
    record = empty_record(initial)
    if calls.stopped:
        record["status"] = "BUDGET_STOPPED"
        return record
    body = repair_wire(base, selected, initial)
    record["request_fingerprints"]["base"] = fingerprint(body)
    started = perf_counter()
    raw = calls.call(selected["key"], REPAIR_STAGE, "base", body)
    record.update(raw_repair=raw, repair_seconds=perf_counter() - started)
    try:
        compiled = compile_repair(record["before"], initial["graph_r1"]["pool"], raw,
                                  initial["graph_r1"]["issues"], image_count=len(base.images))
    except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
        record.update(status="REPAIR_FAILED", compile_error=type(exc).__name__ + ": " + str(exc))
        return record
    record.update(compiled=compiled, temporary=compiled["temporary"], status="PROPOSED")
    persist(record)
    # An unchanged pool or empty changes does not stop a pending recheck.
    if not compiled["review_pool"]["propositions"]:
        record["status"] = "NO_REPAIR_OR_RECHECK"
        return record
    bodies = {seat: review_wire(seat, base, selected, compiled["review_pool"]) for seat in SEATS}
    record["request_fingerprints"].update({s: fingerprint(b) for s, b in bodies.items()})
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw_reviews = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], REVIEW_STAGE, s, bodies[s]), SEATS), strict=True))
    count = sum(c["target"] == selected["key"] and c["stage"] == REVIEW_STAGE for c in calls.rows)
    reviews, formatting = normalize_five(raw_reviews, compiled["review_pool"], len(base.images))
    means, diagnostics = panel.aggregate(reviews, compiled["review_pool"], image_count=len(base.images))
    record.update(raw_reviews=raw_reviews, reviews=reviews, format_diagnostics=formatting, means=means,
                  diagnostics=diagnostics, review_seconds=perf_counter() - started, review_calls_observed=count)
    if count != 5:
        record["status"] = "INCOMPLETE_PANEL"
    else:
        try:
            record.update(prediction=apply_reviewed_repair(compiled, means), status="REVIEWED")
        except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
            record.update(status="SELECTION_FAILED", compile_error=type(exc).__name__ + ": " + str(exc))
    return record


def verify_plan(plan):
    for name, value in (("profile", PROFILE), ("models", MODELS), ("repair_model", PROPOSER),
                        ("rates", RATES_V2), ("limits", LIMITS), ("max_calls", 48), ("round_cap", 2)):
        same(plan[name], value, name)
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "runtime source " + name)
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "source archive " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "image")


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    old_plan, old_rows, initials, ledger = validate_source(source)
    records, _ = audit_records(source, old_plan, old_rows, initials, ledger)
    audit_wires(source, old_plan, initials, records, ledger, adapter)
    preflight = {}
    for selected, initial in zip(old_plan["selection"], initials, strict=True):
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "causal image binding")
        preflight[initial["key"]] = redact_images(repair_wire(base, selected, initial))
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_visual_repair_trial.py",
        ROOT / "scripts/run_prior_candidate_trial.py", ROOT / "scripts/run_prior_feedback_continuation.py",
        ROOT / "scripts/score_prior_candidate_trial.py", ROOT / "scripts/score_prior_feedback_continuation.py"})
    archive = [source / f"{n}.json" for n in ("plan", "initial_state", "completion", "predictions", "budget")]
    archive += list((source / "calls").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source.resolve()),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "archive_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(archive)},
        "selection": old_plan["selection"], "models": MODELS, "repair_model": PROPOSER,
        "rates": RATES_V2, "limits": LIMITS, "max_calls": 48, "round_cap": 2,
        "repair_preflight_fingerprints": {key: digest(body) for key, body in preflight.items()},
        "policy": "All eight fixed Training targets, including prior MODEL_PASS. One unscored visual repair plus at most one five-seat targeted review. Existing unresolved IDs always rechecked even with no new candidates. Original five-valid mean >=4 ADD / <=2 REMOVE and component protection. Phase frozen.",
        "limitations": ["Previously inspected development data, not independent confirmation.",
            "Combined feedback-diagnostics, proposal-role and review-queue change; not a one-factor efficacy ablation.",
            "Same original causal images. No new external visual evidence, tracker or gate.",
            "Repair may introduce full-ontology labels; unapproved temporary outputs are scored separately, not production answers."]}
    save(output / "initial_state.json", {"targets": initials})
    plan["initial_state_sha256"] = sha(output / "initial_state.json")
    save(output / "plan.json", plan)
    save(output / "preflight_metadata.json", metadata)
    for key, body in preflight.items():
        save(output / "repair_preflight" / f"{key}.json", body)
    for path in dependencies:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"prepared": str(output), "targets": len(initials), "max_calls": 48, "limits": LIMITS}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no overwrite or retry")
    verify_plan(plan)
    same(sha(output / "initial_state.json"), plan["initial_state_sha256"], "initial state")
    initials = read(output / "initial_state.json")["targets"]
    bases = {}
    for selected, initial in zip(plan["selection"], initials, strict=True):
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "causal image binding")
        same(digest(redact_images(repair_wire(base, selected, initial))), plan["repair_preflight_fingerprints"][initial["key"]], "repair request")
        bases[initial["key"]] = base
    rows = [{**{k: deepcopy(i[k]) for k in ("key", "video_id", "frame_id", "h0")},
             "graph_r1": deepcopy(i["graph_r1"]["prediction"]),
             "temporary": deepcopy(i["graph_r1"]["prediction"]), "final": deepcopy(i["graph_r1"]["prediction"])} for i in initials]
    save(output / "predictions.json", {"targets": rows})
    for initial in initials:
        save(output / "targets" / initial["key"] / "repair.json", empty_record(initial))
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                       providers=PROVIDERS, max_calls=48, reasoning_seats=("grok", "gemini"))
    start, fatal = perf_counter(), None
    try:
        for selected, initial, row in zip(plan["selection"], initials, rows, strict=True):
            destination = output / "targets" / row["key"] / "repair.json"
            record = run_target(calls, bases[row["key"]], selected, initial, lambda r, dest=destination: save(dest, r))
            save(destination, record)
            row.update(temporary=record["temporary"], final=record["prediction"])
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": row["key"], "status": record["status"],
                "changes": len(record["compiled"]["directions"]) if record["compiled"] else None,
                "calls": len(calls.rows)}), flush=True)
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
            **{f"{name}_sha256": sha(output / f"{name}.json") for name in ("plan", "initial_state", "predictions", "budget")},
            "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(artifacts)},
            "post_calls": len(calls.rows), "statuses": dict(Counter(r["status"] for r in calls.rows)),
            "inference_seconds": perf_counter() - start, "new_h0_calls": 0,
            "query_gt_not_loaded_during_inference": True})


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
