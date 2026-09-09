"""Isolated Phase extension of frozen graph R1; short versus sparse causal history."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import ACADEMIC, redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls, sources
from scripts.run_prior_feedback_continuation import metadata_preflight, same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, review_wire
from scripts.score_five_head_repair_trial import audit as audit_source
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    choose_history,
    phase_choice_error,
    phase_pool,
)

PROFILE = "phase_isolated_single_choice_short_vs_context_v1"
SOURCE = ROOT / "artifacts/preflight/five_head_repair_eight_20260909_v1"
ARMS = ("phase_short", "phase_context")
OFFSETS = {"phase_short": [2, 1, 0], "phase_context": [30, 10, 0]}
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
MAX_CALLS = 80
PHASE_GUIDE = [
    {"id": 0, "name": "Preparation", "meaning": "Initial preparation and exposure of the gallbladder region."},
    {"id": 1, "name": "CalotTriangleDissection", "meaning": "Dissecting around the gallbladder neck and Calot triangle to expose the cystic duct and artery."},
    {"id": 2, "name": "ClippingCutting", "meaning": "The workflow of clipping and dividing the cystic duct and artery."},
    {"id": 3, "name": "GallbladderDissection", "meaning": "Separating the gallbladder from its liver bed."},
    {"id": 4, "name": "GallbladderPackaging", "meaning": "Placing the detached gallbladder into a specimen retrieval bag."},
    {"id": 5, "name": "CleaningCoagulation", "meaning": "Cleaning or irrigating/suctioning the operative field and coagulating for hemostasis."},
    {"id": 6, "name": "GallbladderExtraction", "meaning": "Withdrawing the gallbladder or specimen bag through the extraction port; not ordinary traction on an attached gallbladder."},
]
for _phase in PHASE_GUIDE:
    _phase["name"] = _TASK_NAMES["phase"][_phase["id"]]


def phase_schema(image_count):
    return {"type": "object", "properties": {
        "phase_id": {"type": ["integer", "null"], "enum": [*range(7), None]},
        "image_indices": {"type": "array", "items": {"type": "integer", "enum": list(range(image_count))}},
        "observation": {"type": "string"}},
        "required": ["phase_id", "image_indices", "observation"], "additionalProperties": False}


def phase_wire(seat, selected):
    """No old prediction, reviewer feedback, GT or phase-IVT prior is exposed."""
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=Path(im["path"]).read_bytes())
                                  for im in selected["images"]],
                           payload={"image_details": ["low"] * (len(selected["images"]) - 1) + ["high"]})
    body = review_wire(seat, base, selected, phase_pool())
    schema = phase_schema(len(base.images))
    packet = {"academic_context": ACADEMIC,
        "task": "Identify the ONE current surgical workflow phase at the final target image.",
        "instructions": (
            "Inspect the whole target scene, operative objective and anatomical location; use the earlier images only as causal context. "
            "Choose exactly one phase ID, or null if the supplied images do not distinguish the current workflow stage. "
            "A phase spans a workflow interval: its namesake action need not be visible at every second. "
            "A visible tool, clip or bag alone is not enough to establish a phase. Use actual ongoing activity, location and progression. "
            "Earlier activity is not automatically the target phase. Do not impose a rigid phase order or infer a transition from absence of a tool. "
            "Distinguish operating at the gallbladder neck from separating the gallbladder at the liver bed, and that dissection from field cleanup. "
            "Use the supplied dataset ID mapping. Report one short English observation explaining the discriminating evidence or uncertainty (1..1000 characters). "
            "Cite real input image indices; a non-null choice must cite the final target image. "
            "Return only the JSON object specified by response_schema. No self-score, alternatives, other annotation heads or decision on another model's answer."),
        "target_frame_id": selected["frame_id"],
        "images": [{"index": i, "frame_id": f, "seconds_relative_to_target": (f - selected["frame_id"]) / 25}
                   for i, f in enumerate(selected["causal_frame_ids"])],
        "phase_definitions": PHASE_GUIDE, "current_image_index": len(base.images) - 1,
        "response_schema": schema}
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body["max_tokens"] = 4096
    if body["response_format"]["type"] == "json_schema":
        body["response_format"]["json_schema"] = {"name": "single_phase_choice_v1", "strict": True, "schema": schema}
    return body


def build_inputs(adapter, selection):
    videos = {v: {s.target_frame_id: s for s in adapter.iter_inference_video(v)}
              for v in dict.fromkeys(s["video_id"] for s in selection)}
    out = {}
    for original in selection:
        video, target = original["video_id"], original["frame_id"]
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        samples = videos[video]
        out[original["key"]] = {}
        for arm in ARMS:
            ids = choose_history(sorted(samples), target, OFFSETS[arm])
            if arm == "phase_short":
                same(ids, original["causal_frame_ids"], "original H0 short history")
            out[original["key"]][arm] = {"key": original["key"], "video_id": video, "frame_id": target,
                "causal_frame_ids": ids, "images": [{"frame_id": f, "path": str(samples[f].media_refs[-1]),
                    "sha256": sha(samples[f].media_refs[-1])} for f in ids]}
    return out


def run_panel(calls, selected, current, stage):
    if stage not in ARMS:
        raise ValueError("unknown experiment arm")
    bodies = {s: phase_wire(s, selected) for s in SEATS}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        raw = dict(zip(SEATS, pool.map(lambda s: calls.call(selected["key"], stage, s, bodies[s]), SEATS), strict=True))
    image_count = len(selected["images"])
    errors = {s: phase_choice_error(raw[s], image_count) for s in SEATS}
    prediction, decision = apply_phase_choices(current, raw, image_count)
    return {"raw_reviews": raw, "errors": errors, "prediction": prediction, "decision": decision,
        "status": "INVALID_PANEL" if any(errors.values()) else "REVIEWED",
        "request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()}, "seconds": perf_counter() - started}


def verify_plan(plan):
    for name, value in (("profile", PROFILE), ("arms", ARMS), ("offsets_seconds", OFFSETS),
                        ("models", MODELS), ("providers", PROVIDERS), ("rates", RATES_V2),
                        ("limits", LIMITS), ("max_calls", MAX_CALLS), ("automatic_retries", 0)):
        same(plan[name], value, name)
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "runtime source " + name)
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "source archive " + name)
    for group in plan["phase_inputs"].values():
        for selected in group.values():
            for im in selected["images"]:
                same(sha(im["path"]), im["sha256"], "image")


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    old_plan, old_rows, _, _, _, _ = audit_source(source, adapter)
    inputs = build_inputs(adapter, old_plan["selection"])
    initials = [{k: deepcopy(r[k]) for k in ("key", "video_id", "frame_id", "h0", "graph_r1")}
                | {"previous_five": deepcopy(r["panel_five"])} for r in old_rows]
    preflight = {key: {arm: {s: redact_images(phase_wire(s, item)) for s in SEATS}
                      for arm, item in group.items()} for key, group in inputs.items()}
    metadata = metadata_preflight()
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/score_phase_extension_trial.py",
        ROOT / "scripts/run_five_head_repair_trial.py", ROOT / "scripts/score_five_head_repair_trial.py",
        ROOT / "scripts/run_visual_repair_trial.py", ROOT / "scripts/score_visual_repair_trial.py",
        ROOT / "scripts/run_prior_feedback_continuation.py", ROOT / "scripts/score_prior_feedback_continuation.py"})
    archive = [source / f"{n}.json" for n in ("plan", "initial_state", "completion", "predictions", "budget")]
    archive += list((source / "calls").rglob("*.json")) + list((source / "targets").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "source_root": str(source.resolve()),
        "selection": old_plan["selection"], "phase_inputs": inputs, "arms": ARMS,
        "offsets_seconds": OFFSETS, "models": MODELS, "providers": PROVIDERS, "rates": RATES_V2,
        "limits": LIMITS, "max_calls": MAX_CALLS, "automatic_retries": 0,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "archive_sha256": {p.relative_to(source).as_posix(): sha(p) for p in sorted(archive)},
        "wire_fingerprints": {key: {arm: {s: fingerprint(phase_wire(s, item)) for s in SEATS}
                            for arm, item in group.items()} for key, group in inputs.items()},
        "policy": "Original graph R1 four heads retained exactly. Phase-only blind single choice or abstention from five families. All five structurally valid; >=3 agreeing votes selects Phase, otherwise KEEP. No H0, candidate generation, IVT prior, GT, old prediction or reviewer feedback in model input. Both history arms share prompt and rule. No retries, no label-specific exceptions.",
        "limitations": ["Previously inspected eight Training development targets; not independent validation.",
            "Four-head invariance is guaranteed by separation, not improved visual recognition.",
            "All five model calls per target are incremental; temporal arm has at most three images.",
            "Sparse history stays inside contiguous available 25-frame segment; no future or padded frames.",
            "Phase rubric is a project adaptation of dataset names, not an official boundary annotation manual."],
        "phase_rubric_sources": ["https://arxiv.org/html/1602.03012", "https://arxiv.org/html/2312.07352v2"]}
    save(output / "initial_state.json", {"targets": initials})
    plan["initial_state_sha256"] = sha(output / "initial_state.json")
    save(output / "plan.json", plan)
    save(output / "preflight_metadata.json", metadata)
    save(output / "wire_preflight.json", preflight)
    for path in dependencies:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"prepared": str(output), "targets": len(initials), "max_calls": MAX_CALLS, "limits": LIMITS}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment")
    verify_plan(plan)
    same(build_inputs(adapter, plan["selection"]), plan["phase_inputs"], "causal media resolution")
    same(sha(output / "initial_state.json"), plan["initial_state_sha256"], "initial state")
    for key, group in plan["phase_inputs"].items():
        for arm, item in group.items():
            same({s: fingerprint(phase_wire(s, item)) for s in SEATS}, plan["wire_fingerprints"][key][arm], "frozen wire")
    initials = read(output / "initial_state.json")["targets"]
    rows = [{**deepcopy(i), **{arm: deepcopy(i["graph_r1"]) for arm in ARMS}} for i in initials]
    save(output / "predictions.json", {"targets": rows})
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = TimedCalls(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                       providers=PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
    start, fatal = perf_counter(), None
    try:
        for index, row in enumerate(rows):
            # Counterbalance order; neither arm can use the other's decision.
            for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS))):
                record = run_panel(calls, plan["phase_inputs"][row["key"]][arm], row["graph_r1"], arm)
                save(output / "targets" / row["key"] / f"{arm}.json", record)
                row[arm] = record["prediction"]
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "arm": arm, "status": record["status"],
                    "decision": record["decision"], "calls": len(calls.rows)}), flush=True)
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
            **{f"{n}_sha256": sha(output / f"{n}.json") for n in ("plan", "initial_state", "predictions", "budget")},
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
