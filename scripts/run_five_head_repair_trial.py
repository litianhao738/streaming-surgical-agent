"""Same-eight, fixed-H0 five-head visual repair with two shared-proposal arms."""
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
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from scripts.score_visual_repair_trial import audit as audit_source
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.five_head_repair import (
    aggregate_five_heads,
    compile_joint,
    consistency_issues,
    joint_schema,
    normalize_five_heads,
    phase_hints,
    select_five_heads,
)
from surgical_agent.research.verification.prior_panel import digest
from surgical_agent.research.verification.repair_feedback_v2 import (
    build_repair_feedback,
)

PROFILE = "five_head_surgreflect_adaptation_v1"
SOURCE = ROOT / "artifacts/preflight/visual_repair_eight_20260909_v1"
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
REPAIR_STAGE, REVIEW_STAGE = "joint_repair_2", "five_head_review_2"
PHASE_SEMANTICS = (
    "Phase is the single current surgical workflow stage, not a visible object or an IVT. "
    "Rate all seven phase alternatives against the current whole scene and causal history. "
    "Do not infer a stage solely from one tool, anatomical visibility or a common IVT. "
    "A short window may not reveal the stage; use UNCLEAR/rating 3 then. "
    "Use scope WHOLE_FRAME for a supported or refuted stage claim, with the current image index. "
    "Do not mark several incompatible stages as definitely correct."
)


def repair_wire(base, selected, initial):
    old = initial["graph_r1"]
    feedback = build_repair_feedback(old["pool"], old["raw"], old["reviews"], old["issues"], image_count=len(base.images))
    for candidate in feedback["candidates"]:
        for observation in candidate["valid_observations"]:
            observation.pop("rating")
    body = gemini_proposal(base, selected, old["prediction"], old["pool"], [])
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet.update(task="Visually re-evaluate and revise all FIVE heads of the CURRENT frame.",
        instructions=(
            "Inspect every visible tool, its action and the directly acted-on tissue; use the older images for motion only. "
            "The prior prediction and feedback may be wrong. Resolve the supplied consistency issues by looking again. "
            "Return the COMPLETE revised instrument, verb, target, ivt AND phase label sets, not ADD/REMOVE lists. "
            "All five heads may change. Omitted old labels are removed. Preserve supported labels and revise visual errors. "
            "Use any ID in the full ontology; you are not restricted to the previous candidate pool. "
            "Do not invent changes to fill a quota. Target means directly acted-on tissue, not everything visible. "
            "Form IVTs only when the instrument-action-object relation is visually supported. "
            "Independent heads may include evidence not captured by a selected IVT, but each selected IVT needs its components. "
            "A local absence does not disprove a label across the whole frame; inspect the other tools too. "
            "Phase is exactly one ID and must be re-evaluated, including when the four interaction heads stay unchanged. "
            "Phase-IVT statistics are fallible hints from other Training videos, not a lookup answer or visual proof. "
            "Do not force a phase from a possibly mistaken IVT. If the stage cannot be distinguished, retain the old phase. "
            "Provide one short English observation for each head (at most1000 characters), tied to the current image. "
            "Do not output self-ratings or a pass decision. Return only a JSON object matching response_schema."),
        response_schema=joint_schema(), review_feedback=feedback,
        candidate_relation_hints=deepcopy(initial["hints"]["packet"]),
        phase_compatibility_hints=phase_hints(old["prediction"], initial["phase_prior"], video_id=selected["video_id"]),
        consistency_issues=consistency_issues(old["prediction"]),
        phase_semantics=PHASE_SEMANTICS, current_image_index=len(base.images) - 1)
    packet.pop("issues", None)
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(reasoning={"effort": "medium"}, max_tokens=8192, response_format={"type": "json_object"})
    return body


def five_review_wire(seat, base, selected, pool):
    body = review_wire(seat, base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet["phase_semantics"] = PHASE_SEMANTICS
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def empty_record(initial):
    before = deepcopy(initial["graph_r1"]["prediction"])
    return {"before": before, "llm_raw": deepcopy(before), "paper_style": deepcopy(before),
        "panel_five": deepcopy(before), "panel_four_shadow": deepcopy(before),
        "raw_repair": None, "compiled": None, "raw_reviews": None, "reviews": None,
        "format_diagnostics": None, "means": None, "diagnostics": None, "phase_decision": None,
        "status": "NOT_ATTEMPTED", "compile_error": None, "repair_seconds": 0,
        "review_seconds": 0, "review_calls_observed": 0, "request_fingerprints": {}}


def run_target(calls, base, selected, initial, persist=lambda record: None):
    record = empty_record(initial)
    if calls.stopped:
        record["status"] = "BUDGET_STOPPED"
        return record
    body = repair_wire(base, selected, initial)
    record["request_fingerprints"]["base"] = fingerprint(body)
    start = perf_counter()
    raw = calls.call(selected["key"], REPAIR_STAGE, "base", body)
    record.update(raw_repair=raw, repair_seconds=perf_counter() - start)
    try:
        compiled = compile_joint(record["before"], initial["graph_r1"]["pool"], raw)
    except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
        record.update(status="REPAIR_FAILED", compile_error=type(exc).__name__ + ": " + str(exc))
        return record
    record.update(compiled=compiled, llm_raw=compiled["raw_prediction"], paper_style=compiled["paper_style"], status="PROPOSED")
    persist(record)
    pool = compiled["pool"]
    bodies = {seat: five_review_wire(seat, base, selected, pool) for seat in SEATS}
    record["request_fingerprints"].update({s: fingerprint(b) for s, b in bodies.items()})
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw_reviews = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], REVIEW_STAGE, s, bodies[s]), SEATS), strict=True))
    count = sum(c["target"] == selected["key"] and c["stage"] == REVIEW_STAGE for c in calls.rows)
    reviews, formatting = normalize_five_heads(raw_reviews, pool, image_count=len(base.images))
    means, diagnostics = aggregate_five_heads(reviews, pool, image_count=len(base.images))
    record.update(raw_reviews=raw_reviews, reviews=reviews, format_diagnostics=formatting, means=means,
        diagnostics=diagnostics, review_seconds=perf_counter() - start, review_calls_observed=count)
    if count != 5:
        record["status"] = "INCOMPLETE_PANEL"
        return record
    try:
        final, decision = select_five_heads(record["before"], pool, means)
    except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
        record.update(status="SELECTION_FAILED", compile_error=type(exc).__name__ + ": " + str(exc))
        return record
    shadow = deepcopy(final)
    shadow["phase"] = deepcopy(record["before"]["phase"])
    record.update(panel_five=final, panel_four_shadow=shadow, phase_decision=decision, status="REVIEWED")
    return record


def verify_plan(plan):
    for name, value in (("profile", PROFILE), ("models", MODELS), ("repair_model", PROPOSER),
                        ("rates", RATES_V2), ("limits", LIMITS), ("max_calls", 48), ("round_cap", 2)):
        same(plan[name], value, name)
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "runtime source " + name)
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "source archive " + name)
    for name, value in plan["prior_files_sha256"].items():
        same(sha(name), value, "prior file")
    for selected in plan["selection"]:
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "image")


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    old_plan, previous_rows, inherited, _, _, _ = audit_source(source, adapter)
    graph_root = Path(read(Path(old_plan["source_root"]) / "plan.json")["source_root"])
    initials, priors, preflight = [], {}, {}
    for selected, original, previous in zip(old_plan["selection"], inherited, previous_rows, strict=True):
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        prior_file = graph_root / "priors" / f"{selected['video_id']}.json"
        prior = read(prior_file)
        if any(adapter.entries[v].split is not DatasetSplit.TRAINING for v in prior["fit_videos"]):
            raise ValueError("prior must use other Training videos only")
        initial = {**deepcopy(original), "previous_four": deepcopy(previous["final"]), "phase_prior": prior}
        initials.append(initial)
        priors[str(prior_file.resolve())] = sha(prior_file)
        base = build_gemini_base(adapter, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "causal image binding")
        preflight[initial["key"]] = redact_images(repair_wire(base, selected, initial))
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_five_head_repair_trial.py",
        ROOT / "scripts/run_visual_repair_trial.py", ROOT / "scripts/score_visual_repair_trial.py",
        ROOT / "scripts/run_prior_feedback_continuation.py", ROOT / "scripts/score_prior_feedback_continuation.py"})
    archive = [source / f"{n}.json" for n in ("plan", "initial_state", "completion", "predictions", "budget")]
    archive += list((source / "calls").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source.resolve()),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "archive_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(archive)},
        "prior_files_sha256": priors, "selection": old_plan["selection"], "models": MODELS, "repair_model": PROPOSER,
        "rates": RATES_V2, "limits": LIMITS, "max_calls": 48, "round_cap": 2,
        "repair_preflight_fingerprints": {key: digest(body) for key, body in preflight.items()},
        "policy": "All eight cached Training targets. Shared complete five-head VLM revision. paper_style accepts legal revised heads with IVT component union and all-four-empty guard; panel_five independently reviews the expanded four-head pool plus all seven phases. Four-head five-valid mean >=4 ADD / <=2 REMOVE and component protection; atomic Phase switch only to unique supported (>=4) best beating a valid old-phase score. No retries or third round.",
        "parent_commit": "3028fce4a8eb7ba181314b593c0336ad69b9b67b",
        "limitations": ["Previously inspected development data, not independent confirmation.",
            "Author-inspired adaptation to full ontology, not reproduction of the parent's MCQ experiments.",
            "Joint rewrite and five-head review differ from the old targeted edit protocol; not a one-factor Phase ablation.",
            "Same cached H0 and causal images; no tracker/gate or new external visual evidence.",
            "Phase-IVT prior is a soft hint with query video excluded; it never overwrites Phase deterministically."]}
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
    arms = ("llm_raw", "paper_style", "panel_five", "panel_four_shadow")
    rows = [{**{k: deepcopy(i[k]) for k in ("key", "video_id", "frame_id", "h0", "previous_four")},
        "graph_r1": deepcopy(i["graph_r1"]["prediction"]),
        **{arm: deepcopy(i["graph_r1"]["prediction"]) for arm in arms}} for i in initials]
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
            row.update({arm: record[arm] for arm in arms})
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": row["key"], "status": record["status"],
                "phase_decision": record["phase_decision"], "calls": len(calls.rows)}), flush=True)
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
