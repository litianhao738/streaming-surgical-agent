"""Training-only conditional repeat versus two real earlier frames.

The completed first names review is the shared fallback for both supplementary
arms. Default is offline preflight. No H0/candidate regeneration or retry occurs.
"""

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_verifier_variant_trial as shared
from scripts.run_factored_verifier_trial import (
    checked_saved_names_wire,
    verify_names_completion,
)
from surgical_agent.api.contracts import ApiImageInput, canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.api.errors import ApiError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.context_builder import encode_rgb_png
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
    validate_presence_review,
)
from surgical_agent.research.verification.temporal_memory import (
    assert_memory_extension,
    extend_history_request,
    trigger_propositions,
)

VARIANTS = ("initial_names1000", "repeat1000", "memory1000")
SUPPLEMENTAL = VARIANTS[1:]
PROTOCOL = ROOT / "docs/VERIFIER_TEMPORAL_EVIDENCE_PROTOCOL_2026-09-07.md"


def check_temporal_envelope(request):
    """Same budget envelope as shared; only its image-count limit is now eight."""
    from PIL import Image

    text_bytes = len(request.payload["system_text"].encode()) + len(request.payload["input_text"].encode())
    if text_bytes > 40000 or not 1 <= len(request.images) <= 8:
        raise ValueError("request exceeds reserved text/image-count envelope")
    if sum(len(image.content) for image in request.images) > 12 * 1024 * 1024:
        raise ValueError("request exceeds reserved image-byte envelope")
    for image in request.images:
        with Image.open(io.BytesIO(image.content)) as decoded:
            if max(decoded.size) > 2048 or decoded.width * decoded.height > 2048 * 2048:
                raise ValueError("request exceeds reserved image-pixel envelope")
    shared.wire_body(request)


class TemporalBudgetedCalls(shared.BudgetedCalls):
    """Retain the original ledger/transport logic with an explicit eight-image check."""

    def call(self, key, stage, request):
        if self.stopped or self.used >= self.cap:
            return None
        check_temporal_envelope(request)
        reservation_id = hashlib.sha256(canonical_json_bytes({
            "run_id": self.run_id, "key": key, "stage": stage,
            "request_hash": canonical_request_metadata(request).request_hash})).hexdigest()
        try:
            self.ledger.reserve(reservation_id, run_id=self.run_id, key=key,
                                variant=stage, phase=self.phase)
        except shared.GoalBudgetStop:
            self.stopped = "SHARED_GOAL_BUDGET_STOP"
            return None
        payload = shared.DirectCalls.call(self, key, stage, request)
        record = self.records[-1]
        evidence = self.output / "calls" / key / stage / "result.json"
        if record["cost_usd"] is None:
            self.ledger.unknown(reservation_id, evidence_path=evidence,
                                reason=record.get("error", "MISSING_NATIVE_COST"))
        else:
            self.ledger.settle(reservation_id, record["cost_usd"], evidence_path=evidence,
                               provider_request_id=record.get("provider_request_id"))
        if record.get("error") == "model_or_provider_identity_mismatch":
            self.stopped = "MODEL_OR_PROVIDER_IDENTITY_MISMATCH"
        if self.stopped:
            self.ledger.halt(self.stopped)
        if self.ledger.snapshot()["halt_reasons"]:
            self.stopped = self.stopped or "SHARED_GOAL_HALTED"
        return payload


def completed_training(source):
    plan, completion = shared.read(source / "plan.json"), shared.read(source / "summary.json")
    if plan.get("source_split") != "Training" or plan.get("model") != shared.MODEL:
        raise ValueError("only the frozen Training model is supported; no Validation/Testing")
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != plan["plan_sha256"]
            or completion.get("plan_sha256") != plan["plan_sha256"]
            or completion.get("predictions_sha256") != shared.sha(source / "predictions.json")):
        raise ValueError("source completion/plan/prediction hash mismatch")
    return plan, shared.read(source / "predictions.json")


def check_history(selected, feasible, hashes):
    """Walk the real 25-original-frame grid backwards and stop at its first gap."""
    key, target = selected["key"], selected["frame_id"]
    if (feasible["key"] != key or feasible["video_id"] != selected["video_id"]
            or feasible["target_frame_id"] != target or feasible["source_split"] != "Training"
            or selected["causal_frame_ids"] != [target - 50, target - 25, target]):
        raise ValueError("frozen temporal target identity mismatch")
    folder = Path(selected["images"][-1]["path"]).resolve().parent
    found, missing = [], None
    for frame in (target - 75, target - 100):
        path = folder / f"{frame:06d}.png"
        if not path.is_file():
            missing = str(path)
            break
        found.append((frame, path))
    found.reverse()
    if [f for f, _ in found] != [e["frame_id"] for e in feasible["extra_frames"]]:
        raise ValueError("available contiguous history differs from frozen feasibility")
    for (frame, path), extra in zip(found, feasible["extra_frames"], strict=True):
        digest = shared.sha(path)
        if (path != Path(extra["path"]).resolve() or digest != extra["source_sha256"]
                or extra["relative_seconds"] != (frame - target) / 25 or extra["detail"] != "low"):
            raise ValueError("extra history path/hash/time/detail mismatch")
        hashes[str(path)] = digest
    return found, missing


def load_extra_images(video, found):
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(2)
    result = []
    for frame, path in found:
        with Image.open(path) as source:
            array = np.array(source.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        result.append(ApiImageInput(f"cholectrack20:{video}:frame:{frame}",
                                    "image/png", encode_rgb_png(tensor)))
    return tuple(result)


def evaluate(request, row, review):
    body = json.loads(request.payload["input_text"])
    context = {"h0": row["h0"], "h1": row["h1"],
               "allowed_evidence_refs": body["allowed_evidence_refs"],
               "full_frame_ref": body["full_frame_ref"],
               "schema_version": PRESENCE_REVIEW_1000_VERSION}
    validate_presence_review(review, **context)
    return evaluate_presence_review(review=review, **context)


def checked_initial_result(row, names_request, previous):
    if previous.get("status") != "OK" or previous.get("review") is None:
        raise ValueError("changed candidates require a valid first names review")
    result = evaluate(names_request, row, previous["review"])
    if any(result[key] != previous[key] for key in result):
        raise ValueError("first names decisions differ from unchanged A/B semantics")
    return deepcopy(previous)


def prepare(args):
    if not 1 <= args.max_calls <= 18 or not 1 <= args.timeout_seconds <= 600:
        raise ValueError("this bounded Training trial permits at most 18 calls")
    source, batch, output = args.source.resolve(), args.names_batch.resolve(), args.output.resolve()
    if any(output == p or p in output.parents for p in (source, batch)):
        raise ValueError("new output must be separate from both frozen source runs")
    if (output / "execution.lock").exists():
        raise ValueError("already dispatched; never repeat this run")
    source_plan, old_rows = completed_training(source)
    names_plan, old_names = completed_training(batch)
    keys = [s["key"] for s in source_plan["selection"]]
    if (len(keys) != 16 or len(set(keys)) != 16 or names_plan["keys"] != keys
            or Path(names_plan["source"]).resolve() != source):
        raise ValueError("retain the same complete sixteen-target Training collection")
    old_by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in old_rows}
    names_by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in old_names}
    if (len(old_rows) != 16 or len(old_names) != 16
            or set(old_by_key) != set(keys) or set(names_by_key) != set(keys)):
        raise ValueError("source rows omit or duplicate frozen targets")
    feasibility = shared.read(args.feasibility)
    feasible_by_key = {r["key"]: r for r in feasibility["targets"]}
    if feasibility["source_split"] != "Training" or list(feasible_by_key) != keys:
        raise ValueError("feasibility is not the same fixed Training selection")
    for path, digest in feasibility["source_artifact_sha256"].items():
        if shared.sha(path) != digest:
            raise ValueError("feasibility source binding changed")
    hashes = {str(p): shared.sha(p) for p in (args.feasibility.resolve(), args.pricing_snapshot.resolve())}
    hashes.update({str(p / n): shared.sha(p / n) for p in (source, batch)
                   for n in ("plan.json", "summary.json", "predictions.json")})
    endpoint, reservation = shared.checked_pricing(shared.read(args.pricing_snapshot))
    requests, names_requests, rows, selection, targets, binding_selection, absent = {}, {}, [], [], [], [], []
    trigger_index = 0
    for selected in source_plan["selection"]:
        key, feasible = selected["key"], feasible_by_key[selected["key"]]
        old, first = old_by_key[key], names_by_key[key]
        if selected["source_split"] != "Training" or any(first[k] != old[k] for k in ("h0", "h1")):
            raise ValueError("first names H0/H1 or split differs from collection")
        found, missing = check_history(selected, feasible, hashes)
        if missing:
            absent.append(missing)
        _, restored = shared.restore_target(source, selected, old, hashes)
        row = {"video_id": old["video_id"], "frame_id": old["frame_id"],
               "h0": old["h0"], "h1": old["h1"], "variants": {}}
        initial = deepcopy(first["variants"]["names1000"])
        triggers, names = [], None
        if restored is not None:
            raw, _evidence, full_ref = restored
            names = shared.variant_request(raw, "names1000")
            saved_path = batch / "requests" / key / "names1000.json"
            wire_hash = checked_saved_names_wire(shared.read(saved_path), names)
            hashes[str(saved_path)] = shared.sha(saved_path)
            initial = checked_initial_result(row, names, initial)
            triggers = trigger_propositions(names, initial["review"], h0=row["h0"], h1=row["h1"])
            names_requests[key] = names
            binding_selection.append({"key": key, "source_names_wire_sha256": wire_hash,
                "source_names_request_metadata": canonical_request_metadata(names).to_mapping()})
        elif initial.get("review") is not None or any(initial[a] != old["h0"] for a in ("final_a", "final_b")):
            raise ValueError("unchanged/unavailable candidate must preserve original H0")
        trigger = bool(triggers)
        if trigger != feasible["trigger"] or triggers != feasible["trigger_propositions"]:
            raise ValueError("GT-free trigger differs from the frozen feasibility")
        dispatch = trigger and bool(found)
        reason = "NOT_DISPATCHED_FALLBACK" if dispatch else (
            "NO_EXTRA_HISTORY" if trigger else "NOT_TRIGGERED" if names else "NO_CHANGED_CANDIDATE")
        row["variants"][VARIANTS[0]] = {**deepcopy(initial), "review_source": "initial_names1000",
                                        "supplemental_status": "BASELINE_REUSED_NO_NEW_CALL"}
        for variant in SUPPLEMENTAL:
            row["variants"][variant] = {**deepcopy(initial), "review_source": "initial_names1000",
                "supplemental_status": reason, "supplemental_review": None}
        rows.append(row)
        targets.append({"key": key, "trigger": trigger, "trigger_propositions": triggers,
            "dispatch_pair": dispatch, "extra_frame_ids": [f for f, _ in found],
            "fallback_reason_without_dispatch": reason})
        if not dispatch:
            continue
        images = load_extra_images(old["video_id"], found)
        memory, memory_evidence, memory_full_ref = extend_history_request(names, images, [f for f, _ in found])
        assert_memory_extension(names, memory)
        variants = SUPPLEMENTAL if trigger_index % 2 == 0 else tuple(reversed(SUPPLEMENTAL))
        trigger_index += 1
        for variant in variants:
            request = names if variant == "repeat1000" else memory
            check_temporal_envelope(request)
            requests[(key, variant)] = request
            body = json.loads(request.payload["input_text"])
            selection.append({"key": key, "variant": variant,
                "evidence_manifest": body["evidence_manifest"], "full_frame_ref": body["full_frame_ref"],
                "request_metadata": canonical_request_metadata(request).to_mapping(),
                "wire_sha256": hashlib.sha256(canonical_json_bytes(shared.wire_body(request))).hexdigest(),
                "source_names_wire_sha256": wire_hash})
        if memory_full_ref != full_ref or memory_evidence != json.loads(memory.payload["input_text"])["evidence_manifest"]:
            raise ValueError("memory evidence return does not match its actual input")
    if len(selection) > args.max_calls or len(selection) != feasibility["max_calls"]:
        raise ValueError("fixed paired dispatch count does not fit the declared cap/feasibility")
    binding_plan = {"source_names_batch": str(batch), "source_names_batch_plan_sha256": names_plan["plan_sha256"],
                    "keys": keys, "selection": binding_selection}
    binding = verify_names_completion(binding_plan, names_requests, rows)
    if any(c["baseline_local_status"] != "OK" for c in binding["reference_calls"]):
        raise ValueError("all changed candidates require successful bound first names responses")
    hashes.update(binding["source_artifact_sha256"])
    sources = list(dict.fromkeys(shared._source_files() + [Path(__file__), PROTOCOL,
        ROOT / "scripts/run_factored_verifier_trial.py",
        ROOT / "src/surgical_agent/research/verification/factored_presence.py",
        ROOT / "src/surgical_agent/research/verification/temporal_memory.py"]))
    plan = {"schema_version": "temporal_memory_trial_plan_v1", "source": str(source),
        "source_names_batch": str(batch), "source_split": "Training", "confirmation": False,
        "model": shared.MODEL, "variants": list(VARIANTS), "keys": keys, "target_count": 16,
        "targets": targets, "selection": selection, "review_call_count": len(selection),
        "trigger_targets": sum(t["trigger"] for t in targets), "dispatch_targets": trigger_index,
        "max_provider_calls": args.max_calls, "retries": 0, "maximum_images": 8,
        "timeout_seconds": args.timeout_seconds, "goal_id": args.goal_id,
        "goal_budget_usd": str(shared.GOAL_CAP_USD), "development_budget_usd": str(shared.DEVELOPMENT_CAP_USD),
        "budget_ledger": str(args.budget_ledger.resolve()), "reservation": reservation,
        "pricing": endpoint["pricing"], "source_artifact_sha256": hashes,
        "source_sha256": {str(p.relative_to(ROOT)): shared.sha(p) for p in sources},
        "frozen_absent_paths": absent, "gt_values_used_in_requests": False,
        "gt_values_used_for_trigger": False, "scheme_development_used_training_results": True,
        "fallback_policy": "per arm: invalid/missing supplementary response reuses full first names review; valid replaces it whole",
        "interpretation": "conditional supplementary review; main contrast memory versus exact-wire repeat, not strict factored fallback"}
    if shared.save_plan(output, plan, sources):
        atomic_write_json(output / "frozen_predictions.json", rows)
        atomic_write_json(output / "names_completion_binding.json", binding)
        for (key, variant), request in requests.items():
            atomic_write_json(output / "requests" / key / f"{variant}.json", {
                "metadata": canonical_request_metadata(request).to_mapping(), "payload": thaw_json(request.payload),
                "wire": shared.safe_wire(shared.wire_body(request))})
    print(json.dumps({"status": "PREFLIGHT_PASSED", "targets": 16, "trigger_targets": plan["trigger_targets"],
                      "new_calls": len(selection), "provider_calls": 0, "maximum_images": 8}), flush=True)
    return plan, requests, rows


def assert_frozen(plan):
    shared.assert_frozen(plan)
    if any(Path(p).exists() for p in plan["frozen_absent_paths"]):
        raise ValueError("previously missing history became available after freezing")


def apply_supplement(row, request, review, *, dispatched):
    """Every failed supplementary arm reuses its whole valid first review."""
    previous = deepcopy(row["variants"]["initial_names1000"])
    error = None
    if review is not None:
        try:
            result = evaluate(request, row, review)
        except (ApiError, TypeError, ValueError, OverflowError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            return {**result, "review": review, "status": "OK", "review_source": "supplemental",
                    "supplemental_status": "VALID_REPLACED", "supplemental_review": review}
    return {**previous, "review_source": "initial_names1000", "supplemental_review": review,
        "supplemental_status": "INVALID_REVIEW_FALLBACK" if review is not None else (
            "REQUEST_FAILURE_FALLBACK" if dispatched else "NOT_DISPATCHED_FALLBACK"),
        "supplemental_error": error}


def run(args):
    plan, requests, rows = prepare(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("--execute requires an external key file")
    assert_frozen(plan)
    secret, ledger, _unused_six_image_calls = shared.start_execution(args, plan)
    calls = TemporalBudgetedCalls(args.output, args.max_calls, secret, ledger,
                                  phase="development", timeout=args.timeout_seconds)
    by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in rows}
    atomic_write_json(args.output / "predictions.json", rows)
    for selected in plan["selection"]:
        if calls.stopped:
            break
        key, variant = selected["key"], selected["variant"]
        before = calls.used
        review = calls.call(key, variant, requests[(key, variant)])
        by_key[key]["variants"][variant] = apply_supplement(by_key[key], requests[(key, variant)],
                                                          review, dispatched=calls.used > before)
        if calls.used > before:
            by_key[key]["variants"][variant]["supplemental_call_status"] = calls.records[-1]["status"]
            by_key[key]["variants"][variant]["supplemental_call_error"] = calls.records[-1].get("error")
        atomic_write_json(args.output / "predictions.json", rows)
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = shared.sha(args.output / "predictions.json")
    assert_frozen(plan)
    scored, comparisons = shared.score_saved_rows(args, rows, VARIANTS)
    atomic_write_json(args.output / "scored_predictions.json", scored)
    between = {arm: compute_repair_comparison([{**r,
        "h0": r["variants"]["repeat1000"][arm], "h1": r["variants"]["memory1000"][arm],
        "final": r["variants"]["memory1000"][arm]} for r in scored]) for arm in ("final_a", "final_b")}
    report = {"schema_version": "temporal_memory_trial_result_v1", "model": shared.MODEL,
        "source_split": "Training", "targets": len(rows), "variants": list(VARIANTS),
        "trigger_targets": plan["trigger_targets"], "dispatch_targets": plan["dispatch_targets"],
        "stop_reason": calls.stopped, "plan_sha256": plan["plan_sha256"],
        "predictions_sha256": predictions_hash, "comparisons": comparisons,
        "memory_vs_repeat": between, "goal_budget": ledger.snapshot(),
        "supplemental_status_counts": {v: dict(Counter(r["variants"][v]["supplemental_status"] for r in rows))
                                       for v in VARIANTS},
        "initial_names_new_calls": 0, "fallback_policy": plan["fallback_policy"],
        "interpretation": plan["interpretation"], **shared.accounting(calls.records)}
    if shared.sha(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", report)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps({k: v for k, v in report.items() if k not in ("comparisons", "memory_vs_repeat")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "names-batch", "feasibility", "output", "dataset-root", "pricing-snapshot", "budget-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
