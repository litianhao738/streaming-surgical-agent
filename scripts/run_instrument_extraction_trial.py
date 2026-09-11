"""Instrument-anchored extraction probe, paired against a closed archive.

Isolated experiment. It does not change any default, does not re-run H0, does
not touch the source archive, and never feeds extraction into the published
admission chain. The question is narrow: given the same targets, the same three
images and the same cached H0 as a finished run, how precise and how complete
are the relations a reviewer extracts, compared with the relations the rating
pipeline admitted?

Commands
  prepare    freeze targets, images, wires and budget into a new directory
  preflight  zero API: dump redacted wires and drive the whole path on mocks
  execute    single-use paid run, five seats per target
  score      offline scoring after inference is closed, against archived arms
  smoke      standalone deterministic run, no archive, no dataset, no network
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_glm_parallel_repair as roster
from scripts.check_candidate_panel_providers import redact_images
from scripts.instrument_extraction_wire import CONTRACT, extraction_wire
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_prior_candidate_trial import TASKS as ALL_TASKS
from scripts.run_prior_feedback_continuation import same
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.instrument_extraction import (
    VERSION,
    combine,
    describe,
    merge_with_h0,
    response_error,
)

PROFILE = "instrument_anchored_extraction_probe_v1"
STAGE = "extraction"
SOURCE = ROOT / "artifacts/preflight/joint_phase_confirm16_alternative_20260910_v1"
LIMITS = {"openrouter_usd": "1", "xai_usd": "0", "aliyun_cny": "1"}
COMPARATORS = ("h0", "control", "joint_r1", "joint_r2")
ARMS = ("extract_majority", "extract_union_h0", "extract_replace_h0")
CLAUDE_LIGHT = "anthropic/claude-haiku-4.5"
# Legacy seat keys stay fixed so archives and the published validators keep
# working; `models` and `reviewer_families` in the plan are authoritative, as in
# the GLM roster. Claude answers through a strict schema because the shared
# transport parses content with json.loads and cannot strip Markdown fences.
ROSTERS = {
    "lightweight": {},
    "claude_for_deepseek": {"deepseek": {
        "model": CLAUDE_LIGHT,
        "provider": {"only": ["anthropic"], "order": ["anthropic"],
                     "allow_fallbacks": False, "require_parameters": True},
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_schema", "json_schema": {}},
        "_schema_dialect": "anthropic"}},
}
ROSTER_RATES = {"claude_for_deepseek": {"deepseek": ("0.000001", "0.000005")}}
ROSTER_PROVIDERS = {"claude_for_deepseek": {CLAUDE_LIGHT: "Anthropic"}}
ROSTER_FAMILIES = {"claude_for_deepseek": {"deepseek": "Claude"}}
RUNTIME_FILES = [Path(__file__).resolve(),
                 ROOT / "scripts/instrument_extraction_wire.py",
                 ROOT / "src/surgical_agent/research/verification/instrument_extraction.py"]


def effective(name):
    """Models, rates, providers and families one roster actually binds."""
    overrides = ROSTERS[name]
    return {"models": {**roster.MODELS, **{s: o["model"] for s, o in overrides.items()}},
            "rates": {**roster.RATES, **ROSTER_RATES.get(name, {})},
            "providers": {**roster.PROVIDERS, **ROSTER_PROVIDERS.get(name, {})},
            "families": {**roster.FAMILIES, **ROSTER_FAMILIES.get(name, {})}}


def base_for(selected):
    """Build the transport base from the frozen image paths, as the Phase wire does."""
    images = [SimpleNamespace(mime_type="image/png", content=Path(i["path"]).read_bytes())
              for i in selected["images"]]
    return SimpleNamespace(images=images, payload={
        "image_details": ["low"] * (len(images) - 1) + ["high"]})


def roster_for(h0):
    """Extraction is anchored on the instruments H0 already reports."""
    return sorted({v for v in (h0.get("instrument") or []) if type(v) is int})


def run_target(calls, selected, h0, *, quorum=3, base_builder=base_for, overrides=None):
    instruments = roster_for(h0)
    record = {"key": selected["key"], "instruments": instruments, "raw": None,
              "extraction": None, "arms": {}, "seconds": 0.0, "status": "NO_INSTRUMENT"}
    if not instruments:
        # H0 reported no tool; there is nothing to anchor on and no call is made.
        record["arms"] = {arm: deepcopy(h0) for arm in ARMS}
        return record
    base = base_builder(selected)
    overrides = overrides or {}
    bodies = {seat: extraction_wire(seat, base, selected, instruments,
                                    override=overrides.get(seat)) for seat in SEATS}
    record["request_fingerprints"] = {s: fingerprint(b) for s, b in bodies.items()}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=len(SEATS)) as workers:
        raw = dict(zip(SEATS, workers.map(
            lambda s: calls.call(selected["key"], STAGE, s, bodies[s]), SEATS), strict=True))
    record["seconds"] = perf_counter() - started
    record["raw"] = raw
    extraction = combine(raw, instruments, len(base.images), quorum=quorum)
    record["extraction"] = extraction
    record["status"] = extraction["status"]
    record["arms"] = {
        "extract_majority": {**{t: list(extraction["prediction"][t]) for t in ALL_TASKS[:4]},
                             "phase": list(h0["phase"])},
        "extract_union_h0": merge_with_h0(h0, extraction, mode="union"),
        "extract_replace_h0": merge_with_h0(h0, extraction, mode="replace"),
    }
    if extraction["status"] != "EXTRACTED":
        record["arms"]["extract_majority"] = deepcopy(h0)
    return record


def prepare(output, source, roster_name="lightweight"):
    if roster_name not in ROSTERS:
        raise ValueError("unknown reviewer roster " + roster_name)
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    overrides, bound = ROSTERS[roster_name], effective(roster_name)
    plan_source = read(source / "plan.json")
    truth_keys = {f"{r['video_id']}_{r['frame_id']}" for r in read(source / "scored_truth.json")}
    selection, initials = [], []
    for selected in plan_source["selection"]:
        key = selected["key"]
        if key not in truth_keys:
            raise ValueError("source target lacks scored ground truth: " + key)
        result = read(source / "targets" / key / "result.json")
        h0 = {t: list(result["h0"][t]) for t in ALL_TASKS}
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "frozen source image " + key)
        selection.append(deepcopy(selected))
        initials.append({"key": key, "video_id": selected["video_id"],
                         "frame_id": selected["frame_id"], "h0": h0,
                         "instruments": roster_for(h0),
                         "comparators": {a: deepcopy(result["predictions"][a])
                                         for a in COMPARATORS if a in result["predictions"]}})
    wires = {}
    for selected, initial in zip(selection, initials, strict=True):
        if not initial["instruments"]:
            continue
        base = base_for(selected)
        wires[selected["key"]] = {
            s: fingerprint(extraction_wire(s, base, selected, initial["instruments"],
                                           override=overrides.get(s)))
            for s in SEATS}
    plan = {"profile": PROFILE, "contract": CONTRACT, "contract_version": VERSION,
            "created_utc": now(), "source_archive": str(source.resolve()),
            "source_plan_sha256": sha(source / "plan.json"),
            "source_truth_sha256": sha(source / "scored_truth.json"),
            "selection": selection, "roster": roster_name, "models": bound["models"],
            "providers": bound["providers"], "rates": bound["rates"],
            "reviewer_families": bound["families"],
            "seats": list(SEATS), "quorum": 3, "limits": LIMITS,
            "max_calls": len(wires) * len(SEATS), "automatic_retries": 0,
            "stage": STAGE, "arms": list(ARMS), "comparators": list(COMPARATORS),
            "wire_fingerprints": wires,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_FILES},
            "policy": (
                "Cached H0, frozen images and the source archive's own arms are reused without "
                "re-running them. Each seat answers one instrument-anchored extraction request; "
                "Python assembles ontology-legal triplets and admits a relation on a three-seat "
                "majority. Extraction is scored beside the archived arms and is never fed into "
                "the published admission chain. No retries, no default change, no GT in any "
                "request."),
            "limitations": [
                "Sixteen already-analysed Training targets from four videos; not independent validation.",
                "Extraction is anchored on H0 instruments, so an instrument H0 missed cannot be recovered.",
                "Alternatives are collected but excluded from the majority unless explicitly enabled.",
                "Comparator arms come from the source archive and are not re-run, so provider variation between runs is not controlled.",
                "Phase is outside this contract and is carried over from H0 unchanged."]}
    save(output / "initial_state.json", {"targets": initials})
    plan["initial_state_sha256"] = sha(output / "initial_state.json")
    save(output / "plan.json", plan)
    for path in RUNTIME_FILES:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": len(initials),
                      "targets_with_instruments": len(wires),
                      "max_calls": plan["max_calls"], "limits": LIMITS}), flush=True)


def verify_plan(plan):
    same(plan["profile"], PROFILE, "profile")
    same(plan["contract_version"], VERSION, "contract version")
    same(plan["models"], effective(plan.get("roster", "lightweight"))["models"], "roster models")
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "runtime source " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "frozen image")


class MockCalls:
    """Deterministic local answers; never opens a socket."""

    def __init__(self, answer):
        self.answer, self.rows, self.bodies = answer, [], []

    def call(self, target, stage, seat, body):
        self.bodies.append((target, stage, seat, body))
        self.rows.append({"target": target, "stage": stage, "seat": seat, "status": "MOCK"})
        return deepcopy(self.answer(target, seat))


def mock_answer(instruments, image_count=3):
    def build(_target, seat):
        rows = []
        for n, i in enumerate(sorted(instruments)):
            # One seat disagrees on the last tool, exercising the quorum path.
            shift = 1 if seat == SEATS[-1] and n == len(instruments) - 1 else 0
            rows.append({"instrument_id": i, "present": True, "verb_id": (2 + shift) % 10,
                         "target_id": 0, "alternative_verb_id": None,
                         "alternative_target_id": None,
                         "image_indices": [image_count - 1],
                         "observation": "Deterministic offline extraction row."})
        return {"instruments": rows}
    return build


def preflight(output):
    plan = read(output / "plan.json")
    verify_plan(plan)
    initials = {r["key"]: r for r in read(output / "initial_state.json")["targets"]}
    dumps, checks = {}, []
    overrides = ROSTERS[plan.get("roster", "lightweight")]
    for selected in plan["selection"]:
        key = selected["key"]
        instruments = initials[key]["instruments"]
        if not instruments:
            continue
        base = base_for(selected)
        bodies = {s: extraction_wire(s, base, selected, instruments,
                                     override=overrides.get(s)) for s in SEATS}
        same({s: fingerprint(b) for s, b in bodies.items()},
             plan["wire_fingerprints"][key], "frozen wire " + key)
        dumps[key] = {s: redact_images(b) for s, b in bodies.items()}
        for body in bodies.values():
            packet = json.loads(body["messages"][0]["content"][0]["text"])
            if {"gt", "ground_truth", "current_prediction", "candidate_pool"} & packet.keys():
                raise ValueError("extraction request must not carry answers or GT")
        record = run_target(MockCalls(mock_answer(instruments)), selected,
                            initials[key]["h0"], overrides=overrides)
        checks.append({"key": key, "status": record["status"],
                       "valid_seats": record["extraction"]["valid_seats"],
                       "majority_ivt": record["arms"]["extract_majority"]["ivt"],
                       "rejected_triplets": len(record["extraction"]["rejected_triplets"])})
    save(output / "wire_preflight.json", dumps)
    save(output / "offline_preflight.json", {"checked_utc": now(), "api_calls": 0,
                                             "targets": checks})
    print(json.dumps({"preflight": str(output), "targets": len(checks), "api_calls": 0,
                      "all_valid": all(c["valid_seats"] == len(SEATS) for c in checks)}), flush=True)


def execute(output):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no paid overwrite")
    verify_plan(plan)
    same(sha(output / "initial_state.json"), plan["initial_state_sha256"], "initial state")
    initials = {r["key"]: r for r in read(output / "initial_state.json")["targets"]}
    rows = [{"key": r["key"], "video_id": r["video_id"], "frame_id": r["frame_id"],
             "h0": deepcopy(r["h0"]), "comparators": deepcopy(r["comparators"]),
             "arms": {a: deepcopy(r["h0"]) for a in ARMS}, "status": "NOT_ATTEMPTED"}
            for r in read(output / "initial_state.json")["targets"]]
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    fatal, start = None, perf_counter()
    with roster.lightweight_protocol():
        calls = roster.GLMCalls(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                                rates=plan["rates"], providers=plan["providers"],
                                max_calls=plan["max_calls"], reasoning_seats=("gemini",))
        try:
            for selected, row in zip(plan["selection"], rows, strict=True):
                if calls.stopped:
                    break
                record = run_target(calls, selected, initials[row["key"]]["h0"],
                                    quorum=plan["quorum"],
                                    overrides=ROSTERS[plan.get("roster", "lightweight")])
                if record.get("request_fingerprints"):
                    same(record["request_fingerprints"], plan["wire_fingerprints"][row["key"]],
                         "frozen wire at execution")
                save(output / "targets" / row["key"] / "extraction.json", record)
                row.update(arms=record["arms"], status=record["status"])
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "status": record["status"],
                                  "instruments": record["instruments"],
                                  "calls": len(calls.rows)}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            save(output / "predictions.json", {"targets": rows})
            save(output / "completion.json", {
                "closed_utc": now(), "fatal_error": fatal,
                "plan_sha256": sha(output / "plan.json"),
                "predictions_sha256": sha(output / "predictions.json"),
                "post_calls": len(calls.rows),
                "statuses": dict(Counter(r["status"] for r in calls.rows)),
                "inference_seconds": perf_counter() - start,
                "new_h0_calls": 0, "gt_not_loaded_during_inference": True})


def score(output):
    plan = read(output / "plan.json")
    if not (output / "completion.json").exists():
        raise ValueError("inference must be closed before ground truth is read")
    truth = {f"{r['video_id']}_{r['frame_id']}": r
             for r in read(Path(plan["source_archive"]) / "scored_truth.json")}
    rows = read(output / "predictions.json")["targets"]
    versions = {a: [] for a in (*plan["comparators"], *ARMS)}
    for row in rows:
        item = truth[row["key"]]
        for name, collected in versions.items():
            prediction = row["arms"].get(name) or row["comparators"].get(name)
            if prediction is not None:
                collected.append((prediction, item))
    metrics = {}
    for name, pairs in versions.items():
        if not pairs:
            continue
        entry = {}
        for task in ALL_TASKS:
            tp = fp = fn = valid = 0
            for prediction, item in pairs:
                if not item["mask"].get(task):
                    continue
                got, gt = set(prediction[task]), set(item["gt"][task] or [])
                tp, fp, fn = tp + len(got & gt), fp + len(got - gt), fn + len(gt - got)
                valid += 1
            entry[task] = {"tp": tp, "fp": fp, "fn": fn, "valid_targets": valid,
                           "micro_f1": round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None,
                           "micro_precision": round(100 * tp / (tp + fp), 2) if tp + fp else None,
                           "micro_recall": round(100 * tp / (tp + fn), 2) if tp + fn else None}
        entry["total_label_errors"] = sum(entry[t]["fp"] + entry[t]["fn"] for t in ALL_TASKS)
        entry["mean_f1"] = round(sum(entry[t]["micro_f1"] or 0 for t in ALL_TASKS) / len(ALL_TASKS), 2)
        metrics[name] = entry
    rejected = Counter()
    for row in rows:
        path = output / "targets" / row["key"] / "extraction.json"
        if path.exists():
            for item in read(path)["extraction"]["rejected_triplets"]:
                rejected[describe_safe(item)] += 1
    save(output / "metrics.json", {"scored_utc": now(), "metrics": metrics,
                                   "illegal_triplets": dict(rejected.most_common(20))})
    print(f"{'arm':<20}" + "".join(f"{t:>12}" for t in ALL_TASKS) + f"{'meanF1':>9}{'errors':>8}")
    for name, entry in metrics.items():
        print(f"{name:<20}" + "".join(f"{entry[t]['micro_f1']:>12}" for t in ALL_TASKS)
              + f"{entry['mean_f1']:>9}{entry['total_label_errors']:>8}")


def describe_safe(item):
    ivt = item.get("ivt")
    if isinstance(ivt, int):
        return describe(ivt)
    return f"i{item['instrument']}/v{item['verb']}/t{item['target']}"


def smoke():
    """Whole path on fixed answers: no archive, no dataset, no network."""
    h0 = {"instrument": [0, 2], "verb": [2], "target": [0], "ivt": [], "phase": [1]}
    instruments = roster_for(h0)
    calls = MockCalls(mock_answer(instruments))
    selected = {"key": "SMOKE_100", "video_id": "SMOKE", "frame_id": 100,
                "causal_frame_ids": [50, 75, 100], "images": []}
    synthetic = SimpleNamespace(
        images=[SimpleNamespace(mime_type="image/png", content=b"offline")] * 3,
        payload={"image_details": ["low", "low", "high"]})
    record = run_target(calls, selected, h0, base_builder=lambda _s: synthetic)
    answer = mock_answer(instruments)
    invalid = {s: response_error(answer("SMOKE_100", s), instruments, 3) for s in SEATS}
    invalid["_a_malformed_row_is_rejected"] = response_error(
        {"instruments": [{"instrument_id": instruments[0], "present": True, "verb_id": 2,
                          "target_id": 0, "alternative_verb_id": None,
                          "alternative_target_id": None, "image_indices": [0],
                          "observation": "history only"}]}, instruments, 3)
    summary = {"profile": PROFILE, "api_calls": 0, "mock_calls": len(calls.rows),
               "instruments": instruments, "status": record["status"],
               "valid_seats": record["extraction"]["valid_seats"],
               "votes": record["extraction"]["votes"],
               "arms": record["arms"], "seat_validation": invalid,
               "rejected_triplets": record["extraction"]["rejected_triplets"],
               "limitations": ["Fixed local answers and a synthetic target; wiring only.",
                               "No images, dataset, credentials or ground truth involved."]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score", "smoke"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--roster", choices=tuple(ROSTERS), default="lightweight")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "prepare":
        prepare(args.output, args.source, args.roster)
    elif args.command == "preflight":
        preflight(args.output)
    elif args.command == "execute":
        execute(args.output)
    else:
        score(args.output)
