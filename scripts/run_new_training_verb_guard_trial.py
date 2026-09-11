"""Fresh Training comparison of original Graph R1 and a stricter new-Verb rule.

Prepare and execute never read query GT. Pre-fitted leave-video-out priors are
inputs; scoring follows immutable closure and raw-response replay. Shared H0,
proposal and five reviewers produce both arms. No independent Phase API calls.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
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
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_parallel_phase_trial import STAGES as GRAPH_STAGES
from scripts.run_parallel_phase_trial import run_graph
from scripts.run_prior_feedback_continuation import same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import digest, labels
from surgical_agent.research.verification.verb_guard import VERSION as GUARD_VERSION
from surgical_agent.research.verification.verb_guard import select_with_verb_guard

PROFILE = "new_training_graph_r1_vs_new_verb_guard_v1"
TARGET_COUNT = 24
MAX_CALLS = TARGET_COUNT * 7
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VERSIONS = ("h0", "original", "verb_guard")
H0_STAGE = "h0"
ALLOWED_CALLS = {(H0_STAGE, "base"), (GRAPH_STAGES["proposal"], "base"),
                 *((GRAPH_STAGES["review"], seat) for seat in SEATS)}
FORBIDDEN_INPUT_KEYS = {"gt", "ground_truth", "ground_truth_labels", "annotation_labels", "label_values"}
SUCCESS_RULE = (
    "Versus the shared-answer original arm: Verb micro-Precision strictly increases, Verb micro-F1 and "
    "IVT micro-F1 do not decrease, and Verb micro-Precision is at least shared H0. At least one actual "
    "beneficial label change must occur on a successful guard selection. No changes or only error fallback "
    "cannot establish success. This frozen reporting rule never controls inference or threshold search."
)


class InferenceOnlyAdapter:
    """Expose only media-only iteration; training-label readers are unavailable."""
    def __init__(self, adapter):
        self.entries = adapter.entries
        self._iterate = adapter.iter_inference_video

    def iter_inference_video(self, video):
        return self._iterate(video)

    def iter_video(self, *args, **kwargs):
        raise RuntimeError("query GT access is forbidden in prepare/execute")


def reject_query_gt(value):
    if isinstance(value, dict):
        if FORBIDDEN_INPUT_KEYS & value.keys():
            raise ValueError("selection input contains query GT label fields")
        for item in value.values():
            reject_query_gt(item)
    elif isinstance(value, list):
        for item in value:
            reject_query_gt(item)


def selection_rows(manifest, adapter):
    reject_query_gt(manifest)
    if manifest.get("no_gt_label_values_used_for_selection") is not True:
        raise ValueError("selection must declare mask/time-only selection without GT values")
    raw = manifest.get("selection")
    if not isinstance(raw, list) or len(raw) != TARGET_COUNT:
        raise ValueError("exactly 24 preselected targets required")
    rows, seen = [], set()
    for item in raw:
        key, video, frame = item["key"], item["video_id"], item["frame_id"]
        if (type(frame) is not int or key != f"{video}_{frame}" or key in seen
                or adapter.entries[video].split is not DatasetSplit.TRAINING):
            raise ValueError("unique canonical Training identities required")
        if item.get("source_split", item.get("split", "Training")) != "Training":
            raise ValueError("Testing/Validation cannot enter this experiment")
        seen.add(key)
        ids, images = item["causal_frame_ids"], item["images"]
        same(ids, [frame - 50, frame - 25, frame], "three actual causal frame identities")
        if len(images) != 3:
            raise ValueError("three original image files required")
        for frame_id, image in zip(ids, images, strict=True):
            same(image["frame_id"], frame_id, "selection image identity")
            same(sha(image["path"]), image["sha256"], "selection image bytes")
        rows.append({"key": key, "video_id": video, "frame_id": frame, "source_split": "Training",
                     "causal_frame_ids": deepcopy(ids), "images": deepcopy(images)})
    return rows


def validate_prior(prior, video, adapter):
    claimed = prior.get("table_sha256")
    if (prior.get("excluded_video") != video or not prior.get("fit_videos")
            or video in prior["fit_videos"] or len(set(prior["fit_videos"])) != len(prior["fit_videos"])
            or any(adapter.entries[v].split is not DatasetSplit.TRAINING for v in prior["fit_videos"])
            or digest({k: v for k, v in prior.items() if k != "table_sha256"}) != claimed):
        raise ValueError("valid pre-fitted whole-query-video-excluded Training prior required")


def validate_budget(budget):
    limits, rates = budget.get("limits"), budget.get("rates")
    if not isinstance(limits, dict) or set(limits) != {"openrouter_usd", "xai_usd", "aliyun_cny"}:
        raise ValueError("three explicit account budget caps required")
    if not isinstance(rates, dict) or set(rates) != set(RATES_V2):
        raise ValueError("rate bounds for the fixed base and five reviewers required")
    values = [*limits.values()]
    for row in rates.values():
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("input/output token rates required")
        values.extend(row)
    if any(not Decimal(str(v)).is_finite() or Decimal(str(v)) <= 0 for v in values):
        raise ValueError("finite positive budget limits and token rates required")
    return {k: str(v) for k, v in limits.items()}, {k: [str(x) for x in v] for k, v in rates.items()}


class BoundCalls(TimedCalls):
    """Capture every data-dependent request before dispatch, without retries."""
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                         rates=plan["rates"], providers=PROVIDERS, max_calls=MAX_CALLS,
                         reasoning_seats=("grok", "gemini"))
        self.known = {s["key"] for s in plan["selection"]}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or (stage, seat) not in ALLOWED_CALLS:
            raise ValueError("undeclared target, stage or reviewer; Phase calls are prohibited")
        identity = (target, stage, seat)
        with self.lock:
            if identity in self.attempted:
                raise ValueError("duplicate stage request; automatic retries prohibited")
            self.attempted.add(identity)
            save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json",
                 {"target": target, "stage": stage, "seat": seat,
                  "request_fingerprint": fingerprint(body), "request": redact_images(body)})
        return super().call(target, stage, seat, body)


def run_target(calls, base, selected, prior):
    record = {"h0": None, "original": None, "verb_guard": None, "status": "H0_FAILED",
              "h0_raw": None, "graph": None, "guard": None, "guard_error": None,
              "guard_status": "H0_FAILED"}
    raw = calls.call(selected["key"], H0_STAGE, "base", gemini_h0_wire(base))
    record["h0_raw"] = raw
    try:
        validate_final_only(raw)
    except (ApiSchemaError, TypeError, ValueError):
        return record
    h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
    record.update(h0=deepcopy(h0), original=deepcopy(h0), verb_guard=deepcopy(h0))
    graph = run_graph(calls, base, selected, h0, prior)
    record.update(graph=graph, original=graph["prediction"], status=graph["status"], guard_status="GRAPH_FALLBACK")
    if graph["means"] is not None:
        try:
            guard = select_with_verb_guard(h0, graph["pool"], graph["means"], graph["diagnostics"])
            record.update(guard=guard, verb_guard=guard["prediction"], guard_status="SELECTION_COMPLETE")
        except (ApiSchemaError, TypeError, ValueError, KeyError) as exc:
            record["guard_error"] = {"status": "SELECTION_FAILED", "exception_type": type(exc).__name__}
            record["guard_status"] = "SELECTION_FAILED"
    for arm in ("original", "verb_guard"):
        same(record[arm]["phase"], h0["phase"], "shared H0 Phase must remain unchanged")
    return record


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "timing" and not k.endswith("_seconds")}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify_plan(output):
    plan = read(output / "plan.json")
    for name, expected in (("profile", PROFILE), ("target_count", TARGET_COUNT), ("max_calls", MAX_CALLS),
                           ("models", MODELS), ("proposer", PROPOSER), ("providers", PROVIDERS),
                           ("guard_version", GUARD_VERSION), ("automatic_retries", 0), ("phase_api_calls", 0)):
        same(plan[name], expected, "frozen " + name)
    validate_budget(plan)
    same(plan["success_rule"], SUCCESS_RULE, "predeclared success rule")
    reject_query_gt(plan["selection"])
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "frozen source " + name)
    for name, value in plan["input_sha256"].items():
        same(sha(output / name), value, "frozen input " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "same causal image")
    return plan


def prepare(output, selection_path, prior_root, budget_path, adapter):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    view = InferenceOnlyAdapter(adapter)
    manifest, budget = read(selection_path), read(budget_path)
    metadata_path = selection_path.parent / "model_metadata.json"
    metadata = read(metadata_path)
    same(metadata["paid_calls"], 0, "read-only official endpoint metadata")
    history_path = Path(manifest["history_inventory"])
    same(sha(history_path), manifest["history_inventory_sha256"], "original history identity inventory")
    same(sha(ROOT / "scripts/prepare_new_training_verb_selection.py"),
         manifest["selection_source_sha256"], "selection implementation")
    selected = selection_rows(manifest, view)
    limits, rates = validate_budget(budget)
    priors, h0_wires, h0_hashes = {}, {}, {}
    for video in sorted({r["video_id"] for r in selected}):
        same(sha(prior_root / f"{video}.json"), manifest["prior_sha256"][video], "original frozen LOVO input")
        prior = read(prior_root / f"{video}.json")
        validate_prior(prior, video, view)
        priors[video] = prior
    for item in selected:
        base = build_gemini_base(view, item)
        item["request_metadata"] = canonical_request_metadata(base).to_mapping()
        wire = gemini_h0_wire(base)
        h0_wires[item["key"]] = redact_images(wire)
        h0_hashes[item["key"]] = fingerprint(wire)
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/run_parallel_phase_trial.py",
        ROOT / "scripts/prepare_new_training_verb_selection.py",
        ROOT / "src/surgical_agent/research/verification/verb_guard.py",
        ROOT / "scripts/run_phase_extension_trial.py", ROOT / "scripts/score_five_head_repair_trial.py",
        ROOT / "scripts/score_graph_review_trial.py", ROOT / "scripts/score_prior_feedback_continuation.py",
        ROOT / "configs/perception/joint_openrouter_h0.yaml",
        ROOT / "src/surgical_agent/research/signals/resources/ivt_components_v1.csv",
        ROOT / "src/surgical_agent/perception/prompts/perception_schema_gate_owned_compact.json"})
    save(output / "selection_manifest.json", manifest)
    save(output / "budget_configuration.json", budget)
    save(output / "h0_preflight.json", h0_wires)
    shutil.copyfile(metadata_path, output / "model_metadata.json")
    shutil.copyfile(history_path, output / "history_inventory.json")
    for video, prior in priors.items():
        save(output / "priors" / f"{video}.json", prior)
    inputs = [output / name for name in ("selection_manifest.json", "budget_configuration.json", "h0_preflight.json",
                                        "model_metadata.json", "history_inventory.json")]
    inputs += list((output / "priors").glob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "target_count": TARGET_COUNT,
            "selection": selected, "selection_source": str(selection_path.resolve()),
            "selection_source_sha256": sha(selection_path), "prior_source_root": str(prior_root.resolve()),
            "models": MODELS, "proposer": PROPOSER, "providers": PROVIDERS, "limits": limits, "rates": rates,
            "guard_version": GUARD_VERSION, "max_calls": MAX_CALLS, "automatic_retries": 0, "phase_api_calls": 0,
            "success_rule": SUCCESS_RULE,
            "h0_fingerprints": h0_hashes,
            "input_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(inputs)},
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
            "protocol": "Fresh shared final-only Gemini H0; original pre-fitted LOVO graph hints, one proposal, five original visual reviewers. Original mean4 repair and new-Verb guard share every answer. Only novel Verb labels additionally require all five valid integer scores >=4; blocked Verb mean becomes None before rerunning the unchanged selector from H0, including IVT component checks. Existing labels and other means are untouched. Phase stays equal to shared H0. No retries, reduced quorum, replacement frames or Phase API calls.",
            "request_freeze_scope": "H0 wire fixed before execution. Proposal and reviewer content depend on fresh model answers; frozen source defines their builders and each actual wire is captured before dispatch. No claim that unknown future model-dependent wires were precomputed.",
            "gt_policy": "Selection uses supplied time/mask-only manifest. Prepare/execute have a media-only adapter, no GT fitting or iter_video access. Existing priors exclude the whole query video and use Training videos only. All planned targets remain in closed scoring; missing task GT is masked out, and failed predictions are reported.",
            "limitations": ["New targets evaluate a predeclared local rule, not a full held-out test benchmark.",
                            "Unanimous model support is not calibrated certainty or independent visual truth.",
                            "No Tracker, Gate, phase repair or additional candidate round in this isolated test."]}
    save(output / "plan.json", plan)
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    verify_plan(output)
    print(json.dumps({"prepared": str(output), "targets": TARGET_COUNT, "max_calls": MAX_CALLS,
                      "limits": limits, "phase_calls": 0, "paid_calls": 0}), flush=True)


def execute(output, adapter):
    plan = verify_plan(output)
    if (output / "execution.lock").exists():
        raise ValueError("single-use execution; paid rerun prohibited")
    view = InferenceOnlyAdapter(adapter)
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = BoundCalls(output, plan)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
             "h0": None, "original": None, "verb_guard": None, "status": "NOT_ATTEMPTED",
             "guard_status": "NOT_ATTEMPTED"} for s in plan["selection"]]
    started, fatal = perf_counter(), None
    try:
        for selected, row in zip(plan["selection"], rows, strict=True):
            if calls.stopped:
                row["status"] = "BUDGET_OR_PROVIDER_STOPPED"
                row["guard_status"] = "BUDGET_OR_PROVIDER_STOPPED"
                continue
            base = build_gemini_base(view, selected)
            same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "frozen H0 image input")
            same(fingerprint(gemini_h0_wire(base)), plan["h0_fingerprints"][row["key"]], "frozen H0 wire")
            prior = read(output / "priors" / f"{row['video_id']}.json")
            validate_prior(prior, row["video_id"], view)
            record = run_target(calls, base, selected, prior)
            save(output / "targets" / row["key"] / "result.json", record)
            row.update({name: record[name] for name in (*VERSIONS, "status", "guard_status")})
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": row["key"], "status": row["status"], "calls": len(calls.rows),
                              "occupied": {k: str(v) for k, v in calls.occupied.items()}}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": rows})
        verify_plan(output)
        files = [output / name for name in ("plan.json", "budget.json", "predictions.json", "execution.lock")]
        files += [p for folder in ("calls", "targets", "request_intents") for p in (output / folder).rglob("*.json")]
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal, "planned_targets": TARGET_COUNT,
            "post_calls": len(calls.rows), "statuses": dict(Counter(r["status"] for r in calls.rows)),
            "target_statuses": dict(Counter(r["status"] for r in rows)), "elapsed_seconds": perf_counter() - started,
            "guard_statuses": dict(Counter(r["guard_status"] for r in rows)),
            "gt_values_not_read_during_inference": True,
            "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output, adapter):
    plan = verify_plan(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] is not None or not ledger["stopped"] or any(r["status"] == "DISPATCHED" for r in ledger["calls"]):
        raise ValueError("closed nonfatal inference required before GT scoring")
    expected_files = {p.relative_to(output).as_posix() for folder in ("calls", "targets", "request_intents")
                      for p in (output / folder).rglob("*.json")}
    if not expected_files <= set(done["inference_artifact_sha256"]):
        raise ValueError("closure omits an inference artifact")
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(output / name), value, "closed inference evidence")
    calls = ledger["calls"]
    if len(calls) > MAX_CALLS or done["post_calls"] != len(calls):
        raise ValueError("invalid paid call count")
    if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate request or undeclared retry")
    known = {r["key"] for r in plan["selection"]}
    if any(c["target"] not in known or (c["stage"], c["seat"]) not in ALLOWED_CALLS for c in calls):
        raise ValueError("undeclared request identity")
    for call in calls:
        expected_model = PROPOSER if call["seat"] == "base" else MODELS[call["seat"]]
        same(call["model"], expected_model, "logged model identity")
        if call["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
            folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
            response = read(folder / "response.json")["body"]
            same(response["model"], expected_model, "returned model identity")
            if call["seat"] in PROVIDERS:
                same(response.get("provider"), PROVIDERS[call["seat"]], "returned provider identity")
    for account, value in ledger["occupied"].items():
        charges = [Decimal(c["charge"]) for c in calls if c["account"] == account]
        if any(not charge.is_finite() or charge < 0 for charge in charges):
            raise ValueError("invalid charge")
        if sum(charges) != Decimal(value) or Decimal(value) > Decimal(plan["limits"][account]):
            raise ValueError("ledger sum or cap mismatch")
    rows = read(output / "predictions.json")["targets"]
    same([(r["key"], r["video_id"], r["frame_id"]) for r in rows],
         [(r["key"], r["video_id"], r["frame_id"]) for r in plan["selection"]], "all 24 planned identities and order")
    view = InferenceOnlyAdapter(adapter)
    for row, selected in zip(rows, plan["selection"], strict=True):
        target_calls = [c for c in calls if c["target"] == row["key"]]
        path = output / "targets" / row["key"] / "result.json"
        if not path.exists():
            if target_calls or any(row[v] is not None for v in VERSIONS) or row["status"] != "BUDGET_OR_PROVIDER_STOPPED":
                raise ValueError("missing attempted target evidence")
            continue
        replay = ReplayCalls(output, target_calls)
        rebuilt = run_target(replay, build_gemini_base(view, selected), selected,
                             read(output / "priors" / f"{row['video_id']}.json"))
        same(without_timing(rebuilt), without_timing(read(path)), "raw response replay " + row["key"])
        same(row["guard_status"], rebuilt["guard_status"], "saved guard status")
        if len(replay.rows) != len(target_calls):
            raise ValueError("unconsumed API response")
        for version in VERSIONS:
            same(row[version], rebuilt[version], "saved " + version)
            if row[version] is not None:
                labels(row[version])
        for call in target_calls:
            intent = read(output / "request_intents" / f"{call['target']}_{call['stage']}_{call['seat']}.json")
            folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
            same(intent["request"], read(folder / "request.json"), "pre-dispatch request intent")
            same(digest(intent["request"]), intent["request_fingerprint"], "intent fingerprint")
    return plan, rows, ledger, done


def diagnostic_summary(rows, truths, records, metrics):
    coverage = {t: Counter(valid_targets=0, gt_positive=0, pool_true=0, pool_false=0,
                           missing_from_pool=0, targets_without_pool=0) for t in TASKS[:4]}
    blocked = {t: Counter(valid_targets=0, true_additions_blocked=0, false_additions_blocked=0,
                         unexpected_removed_existing_labels=0) for t in ("verb", "ivt")}
    rating_counts = {s: Counter() for s in SEATS}
    rejection_reasons = {s: Counter() for s in SEATS}
    novel_verb_reasons, candidate_quorum = Counter(), Counter()
    beneficial_successful_guard_changes = 0
    guard_selection_failures = 0
    for row in rows:
        gt = truths[row["video_id"], row["frame_id"]]
        record = records.get(row["key"], {})
        if record.get("guard_error") is not None:
            guard_selection_failures += 1
        graph = record.get("graph") or {}
        pool = graph.get("pool", {}).get("propositions", [])
        for task, values in coverage.items():
            if not gt["mask"][task]:
                continue
            expected = set(gt["gt"][task])
            available = {p["label_id"] for p in pool if p["task"] == task}
            values.update(valid_targets=1, gt_positive=len(expected), pool_true=len(available & expected),
                          pool_false=len(available - expected), missing_from_pool=len(expected - available),
                          targets_without_pool=int(not pool))
        successful_guard = record.get("guard") is not None and record.get("guard_error") is None
        if successful_guard and row["h0"] is not None and row["original"] is not None and row["verb_guard"] is not None:
            for task, counts in blocked.items():
                if not gt["mask"][task]:
                    continue
                removed = set(row["original"][task]) - set(row["verb_guard"][task])
                fresh = removed - set(row["h0"][task])
                expected = set(gt["gt"][task])
                counts.update(valid_targets=1, true_additions_blocked=len(fresh & expected),
                              false_additions_blocked=len(fresh - expected),
                              unexpected_removed_existing_labels=len(removed & set(row["h0"][task])))
        if successful_guard:
            delta = frame_delta(row["original"], row["verb_guard"], gt["gt"], gt["mask"])
            beneficial_successful_guard_changes += delta["fixed_label_errors"]
            for choice in record["guard"]["new_verb_decisions"]:
                novel_verb_reasons[choice["reason"]] += 1
        if graph.get("reviews") is not None:
            for seat in SEATS:
                accepted = len(graph["reviews"][seat]["judgments"])
                rating_counts[seat].update(panels=1, expected_candidate_ratings=len(pool),
                                          valid_candidate_ratings=accepted, invalid_candidate_ratings=len(pool) - accepted)
                for reasons in graph["format_diagnostics"][seat]["errors"].values():
                    rejection_reasons[seat].update(reasons)
            for mean in graph["means"].values():
                candidate_quorum["five_valid" if mean is not None else "incomplete_or_invalid"] += 1
    original, guard, h0 = (metrics[v]["tasks"] for v in ("original", "verb_guard", "h0"))
    values = (guard["verb"]["micro_precision"], original["verb"]["micro_precision"],
              guard["verb"]["micro_f1"], original["verb"]["micro_f1"], guard["ivt"]["micro_f1"],
              original["ivt"]["micro_f1"], h0["verb"]["micro_precision"])
    vp, op, vf, of, vi, oi, hp = values
    evaluable = all(value is not None for value in values)
    checks = {"Verb_precision_strictly_above_original": evaluable and vp > op,
              "Verb_F1_not_below_original": evaluable and vf >= of,
              "IVT_F1_not_below_original": evaluable and vi >= oi,
              "Verb_precision_at_least_H0": evaluable and vp >= hp,
              "no_guard_selection_failures": guard_selection_failures == 0,
              "beneficial_actual_change_on_successful_guard": beneficial_successful_guard_changes > 0}
    return {"shared_candidate_coverage": {t: dict(v) for t, v in coverage.items()},
            "blocked_actual_new_labels": {t: dict(v) for t, v in blocked.items()},
            "review_validity": {s: {**dict(rating_counts[s]), "reasons": dict(rejection_reasons[s])} for s in SEATS},
            "candidate_quorum": dict(candidate_quorum), "new_verb_guard_reasons": dict(novel_verb_reasons),
            "guard_selection_failures": guard_selection_failures,
            "predeclared_success": {"rule": SUCCESS_RULE, "checks": checks, "confirmed": all(checks.values()),
                                     "beneficial_successful_guard_label_changes": beneficial_successful_guard_changes,
                                     "scope": "Same new Training targets only; no threshold search or population-level claim."}}


def score(output, adapter):
    plan, rows, ledger, done = audit(output, adapter)
    # The first query GT read happens only after every inference replay/check.
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["verb_guard"]} for r in rows])
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    records = {r["key"]: read(output / "targets" / r["key"] / "result.json") for r in rows
               if (output / "targets" / r["key"] / "result.json").exists()}
    metrics = {version: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[version]} for r in rows])["arms"]["final"] for version in VERSIONS}
    details, comparisons = {}, {}
    for before, after in (("h0", "original"), ("h0", "verb_guard"), ("original", "verb_guard")):
        name = before + "_to_" + after
        details[name] = [{"key": row["key"], **frame_delta(row[before], row[after],
            truths[row["video_id"], row["frame_id"]]["gt"], truths[row["video_id"], row["frame_id"]]["mask"])} for row in rows]
        comparisons[name] = summarize_deltas(details[name])
    costs = {account: {"native": str(sum(Decimal(c["charge"]) for c in ledger["calls"]
                 if c["account"] == account and c["charge_kind"] == "native")),
        "estimated": str(sum(Decimal(c["charge"]) for c in ledger["calls"]
                 if c["account"] == account and c["charge_kind"] == "conservative_estimate")),
        "unknown_reserved": str(sum(Decimal(c["charge"]) for c in ledger["calls"]
                 if c["account"] == account and c["charge_kind"] == "unknown_reserved"))} for account in plan["limits"]}
    report = {"profile": PROFILE, "planned_targets": TARGET_COUNT, "metrics": metrics,
              "comparisons": comparisons, "costs_by_account": costs, "post_calls": len(ledger["calls"]),
              "guard_extra_api_calls": 0, "elapsed_seconds": done["elapsed_seconds"],
              "transport_statuses": done["statuses"], "target_statuses": done["target_statuses"],
              "guard_statuses": done["guard_statuses"],
              "audit": {"raw_outputs_replayed": True, "gt_loaded_after_closed_inference": True,
                        "phase_equals_shared_h0": True, "all_planned_targets_retained": True},
              **diagnostic_summary(rows, truths, records, metrics),
              "limitations": plan["limitations"]}
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", details)
    save(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "audit", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        if any(value is None for value in (args.selection, args.prior_root, args.budget)):
            parser.error("prepare requires --selection, --prior-root and --budget")
        prepare(args.output, args.selection, args.prior_root, args.budget, adapter)
    elif args.command == "execute":
        execute(args.output, adapter)
    elif args.command == "audit":
        _, rows, ledger, _ = audit(args.output, adapter)
        print(json.dumps({"verified": True, "targets": len(rows), "calls": len(ledger["calls"])}))
    else:
        score(args.output, adapter)


if __name__ == "__main__":
    main()
