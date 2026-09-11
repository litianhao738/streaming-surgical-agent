"""Fresh shared-H0 comparison with one bounded visual recheck of unclear IVTs."""
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

from scripts import run_new_training_verb_guard_trial as base_trial
from scripts.check_candidate_panel_providers import ACADEMIC, redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_feedback_continuation import same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS
from scripts.run_semantic_candidate_trial import BOUNDARIES, PROPOSER
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.disputed_relation import (
    VERSION,
    apply_groups,
    eligible_groups,
    response_schema,
)

PROFILE = "new_training_unclear_relation_single_recheck_v1"
STAGE = "relation_recheck"
TARGET_COUNT = 24
MAX_CALLS = TARGET_COUNT * 8
VERSIONS = (*base_trial.VERSIONS, "joint_recheck")
TASKS = base_trial.TASKS
ALLOWED_CALLS = base_trial.ALLOWED_CALLS | {(STAGE, "base")}
SUCCESS_RULE = (
    "On all new planned Training targets, versus original shared-answer repair: joint-recheck Verb "
    "Precision strictly improves, Verb F1 and IVT F1 do not decrease, Verb Precision is at least shared "
    "H0; at least one actual beneficial label change occurs and there are no selector exceptions. "
    "No threshold adjustment after GT. Recheck must actually run and accept at least one group; "
    "otherwise its recovery benefit is unestablished. All head counts, harms and API failures reported."
)
PROTOCOL = (
    "Fresh shared H0, original LOVO graph proposal and five reviews. Original mean4 and strict "
    "new-Verb arms share answers. One existing Gemini base-model visual call per eligible frame, "
    "all eligible IVTs packed together. Eligibility: originally admitted new IVT blocked only by "
    "valid UNCLEAR novel-Verb votes; all IVT/I/V/T votes valid, each mean>=4, none <=2. No H0 answers, "
    "votes, scores, reviewer names, graph priors or GT sent to rechecker. SUPPORT atomically restores "
    "only that IVT and needed originally supported components onto strict output. REFUTE/UNCLEAR "
    "does not delete frame-level labels. Failures retain strict output. No new candidates, no "
    "retries, no repeated review, no Phase repair, Tracker or Gate. Scores are never rewritten."
)


def recheck_wire(base, groups):
    if not 1 <= len(groups) <= 4:
        raise ValueError("one to four original candidate relations required")
    body = gemini_h0_wire(base)
    images = [deepcopy(part) for message in body["messages"]
              if isinstance(message.get("content"), list) for part in message["content"]
              if part.get("type") == "image_url"]
    if len(images) != 3:
        raise ValueError("exactly three original causal images required")
    schema = response_schema(groups, len(images))
    packet = {
        "academic_context": ACADEMIC,
        "task": "Independently check whether each supplied instrument-action-target relation is present in CURRENT image 2.",
        "image_order": ["t-2 seconds", "t-1 second", "current target"],
        "proposition_semantics": BOUNDARIES,
        "candidate_relations": deepcopy(groups),
        "instructions": (
            "Candidate membership is not evidence. Check the actual tool tip, the directly acted-on "
            "tissue and the action of that SAME tool on that SAME tissue. Mere contact, three unrelated "
            "components or phase plausibility does not establish the relation. Use real motion from "
            "the ordered images where visible; do not invent continuous motion between them. The "
            "relation must occur now, not only in history. Return SUPPORT only when observed tool, "
            "action and direct target jointly support the whole relation. REFUTE needs visible "
            "contradictory evidence after checking other possible tools; occlusion, uncertain tissue "
            "identity or unclear action requires UNCLEAR. A rejected relation does not establish "
            "absence of the action across the whole frame. For each group provide one short English "
            "observation of the tool/action/object and its location, or the specific uncertainty. "
            "SUPPORT and REFUTE must cite current image 2. Do not output scores, new candidates, "
            "Phase, or a rewritten prediction. Observation maximum 1000 characters. Return only JSON."
        ),
        "response_schema": schema,
    }
    body["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": json.dumps(packet, ensure_ascii=False)}, *images]}]
    body["max_tokens"] = 4096
    body["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "unclear_relation_recheck_v1", "strict": True, "schema": schema}}
    return body


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                         rates=plan["rates"], providers=PROVIDERS, max_calls=MAX_CALLS,
                         reasoning_seats=("grok", "gemini"))
        self.known = {s["key"] for s in plan["selection"]}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or (stage, seat) not in ALLOWED_CALLS:
            raise ValueError("undeclared target/stage/seat")
        with self.lock:
            identity = (target, stage, seat)
            if identity in self.attempted:
                raise ValueError("duplicate call; no retry or second recheck")
            self.attempted.add(identity)
            save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json", {
                "target": target, "stage": stage, "seat": seat,
                "request_fingerprint": fingerprint(body), "request": redact_images(body)})
        return super().call(target, stage, seat, body)


def run_target(calls, base, selected, prior):
    result = base_trial.run_target(calls, base, selected, prior)
    result.update(joint_recheck=deepcopy(result["verb_guard"]), eligibility=None, recheck_raw=None,
                  recheck=None, recheck_status="UPSTREAM_UNAVAILABLE", recheck_error=None,
                  recheck_fingerprint=None, recheck_seconds=0.0)
    if result["guard"] is None or result["guard_error"] is not None:
        return result
    try:
        eligible = eligible_groups(result["h0"], result["graph"], result["guard"], image_count=3)
        result["eligibility"] = eligible
        if not eligible["groups"]:
            result["recheck_status"] = "NO_ELIGIBLE_GROUPS"
            return result
        wire = recheck_wire(base, eligible["groups"])
        result["recheck_fingerprint"] = fingerprint(wire)
        start = perf_counter()
        raw = calls.call(selected["key"], STAGE, "base", wire)
        result.update(recheck_raw=raw, recheck_seconds=perf_counter() - start)
        repaired = apply_groups(result["verb_guard"], result["original"], eligible["groups"], raw)
        result.update(recheck=repaired, recheck_status=repaired["status"], joint_recheck=repaired["prediction"])
    except (ValueError, KeyError, TypeError) as exc:
        result.update(recheck_status="SELECTION_FAILED", recheck_error=type(exc).__name__)
    same(result["joint_recheck"]["phase"], result["h0"]["phase"], "Phase remains shared H0")
    return result


def verify_plan(output):
    plan = read(output / "plan.json")
    for name, value in (("profile", PROFILE), ("target_count", TARGET_COUNT), ("max_calls", MAX_CALLS),
                        ("models", MODELS), ("proposer", PROPOSER), ("providers", PROVIDERS),
                        ("recheck_version", VERSION), ("success_rule", SUCCESS_RULE), ("protocol", PROTOCOL),
                        ("automatic_retries", 0), ("phase_api_calls", 0)):
        same(plan[name], value, "frozen " + name)
    base_trial.validate_budget(plan)
    base_trial.reject_query_gt(plan["selection"])
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "frozen source " + name)
    for name, value in plan["input_sha256"].items():
        same(sha(output / name), value, "frozen input " + name)
    for selected in plan["selection"]:
        for image in selected["images"]:
            same(sha(image["path"]), image["sha256"], "frozen causal image")
    return plan


def prepare(output, selection, prior_root, budget, adapter):
    # Reuse only the established media/prior/input freezer, before any dispatch.
    base_trial.prepare(output, selection, prior_root, budget, adapter)
    plan = read(output / "plan.json")
    plan.update(profile=PROFILE, recheck_version=VERSION, max_calls=MAX_CALLS,
                versions=list(VERSIONS), success_rule=SUCCESS_RULE, protocol=PROTOCOL,
                recheck_model=PROPOSER, max_rechecks_per_target=1,
                limitations=["New targets from previously used Training videos; no unseen-video or Testing claim.",
                             "Rechecker reobserves the same images and may share errors with proposer/reviewers.",
                             "No Phase review, Tracker, Gate or candidate expansion in this isolated test."])
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_new_training_verb_guard_trial.py",
                 ROOT / "scripts/prepare_disputed_relation_inputs.py",
                 ROOT / "src/surgical_agent/research/verification/disputed_relation.py"):
        rel = path.relative_to(ROOT).as_posix()
        plan["source_sha256"][rel] = sha(path)
        dest = output / "frozen_source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    save(output / "plan.json", plan)
    verify_plan(output)
    print(json.dumps({"prepared": PROFILE, "targets": TARGET_COUNT, "max_calls": MAX_CALLS,
                      "limits": plan["limits"], "paid_calls": 0}), flush=True)


def execute(output, adapter):
    plan = verify_plan(output)
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; paid rerun prohibited")
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    view = base_trial.InferenceOnlyAdapter(adapter)
    calls = BoundCalls(output, plan)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
             **dict.fromkeys(VERSIONS), "status": "NOT_ATTEMPTED", "guard_status": "NOT_ATTEMPTED",
             "recheck_status": "NOT_ATTEMPTED"} for s in plan["selection"]]
    started, fatal = perf_counter(), None
    try:
        for selected, row in zip(plan["selection"], rows, strict=True):
            if calls.stopped:
                row.update(status="BUDGET_OR_PROVIDER_STOPPED", guard_status="BUDGET_OR_PROVIDER_STOPPED",
                           recheck_status="BUDGET_OR_PROVIDER_STOPPED")
                continue
            base = build_gemini_base(view, selected)
            same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "frozen H0 input")
            same(fingerprint(gemini_h0_wire(base)), plan["h0_fingerprints"][row["key"]], "frozen H0 wire")
            prior = read(output / "priors" / f"{row['video_id']}.json")
            base_trial.validate_prior(prior, row["video_id"], view)
            result = run_target(calls, base, selected, prior)
            save(output / "targets" / row["key"] / "result.json", result)
            row.update({k: result[k] for k in (*VERSIONS, "status", "guard_status", "recheck_status")})
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": row["key"], "status": row["status"], "recheck": row["recheck_status"],
                              "calls": len(calls.rows)}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": rows})
        try:
            verify_plan(output)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            files = [output / n for n in ("plan.json", "budget.json", "predictions.json", "execution.lock")]
            files += [p for folder in ("calls", "targets", "request_intents") for p in (output / folder).rglob("*.json")]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                "planned_targets": TARGET_COUNT, "post_calls": len(calls.rows),
                "statuses": dict(Counter(r["status"] for r in calls.rows)),
                "target_statuses": dict(Counter(r["status"] for r in rows)),
                "guard_statuses": dict(Counter(r["guard_status"] for r in rows)),
                "recheck_statuses": dict(Counter(r["recheck_status"] for r in rows)),
                "elapsed_seconds": perf_counter() - started, "gt_values_not_read_during_inference": True,
                "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output, adapter):
    plan, done, ledger = verify_plan(output), read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"] or any(c["status"] == "DISPATCHED" for c in ledger["calls"]):
        raise ValueError("closed nonfatal inference required")
    expected = {p.relative_to(output).as_posix() for folder in ("calls", "targets", "request_intents")
                for p in (output / folder).rglob("*.json")}
    if not expected <= set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen inference artifact")
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(output / name), value, "closed inference artifact")
    calls = ledger["calls"]
    if len(calls) > MAX_CALLS or len(calls) != done["post_calls"]:
        raise ValueError("invalid call count")
    if len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate call")
    known = {r["key"] for r in plan["selection"]}
    for call in calls:
        if call["target"] not in known or (call["stage"], call["seat"]) not in ALLOWED_CALLS:
            raise ValueError("undeclared call")
        model = PROPOSER if call["seat"] == "base" else MODELS[call["seat"]]
        same(call["model"], model, "logged model")
        if call["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
            folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
            response = read(folder / "response.json")["body"]
            same(response["model"], model, "returned model")
            if call["seat"] in PROVIDERS:
                same(response.get("provider"), PROVIDERS[call["seat"]], "returned provider")
    for account, value in ledger["occupied"].items():
        charges = [Decimal(c["charge"]) for c in calls if c["account"] == account]
        if any(not c.is_finite() or c < 0 for c in charges):
            raise ValueError("invalid charge")
        same(sum(charges), Decimal(value), "ledger sum")
        if Decimal(value) > Decimal(plan["limits"][account]):
            raise ValueError("budget exceeded")
    rows = read(output / "predictions.json")["targets"]
    same([(r["key"], r["video_id"], r["frame_id"]) for r in rows],
         [(s["key"], s["video_id"], s["frame_id"]) for s in plan["selection"]], "all planned identities")
    view = base_trial.InferenceOnlyAdapter(adapter)
    for row, selected in zip(rows, plan["selection"], strict=True):
        target_calls = [c for c in calls if c["target"] == row["key"]]
        path = output / "targets" / row["key"] / "result.json"
        if not path.exists():
            if target_calls or any(row[v] is not None for v in VERSIONS) or row["status"] != "BUDGET_OR_PROVIDER_STOPPED":
                raise ValueError("missing attempted target")
            continue
        replay = ReplayCalls(output, target_calls)
        rebuilt = run_target(replay, build_gemini_base(view, selected), selected,
                             read(output / "priors" / f"{row['video_id']}.json"))
        same(base_trial.without_timing(rebuilt), base_trial.without_timing(read(path)), "raw-response replay")
        same(len(replay.rows), len(target_calls), "all responses consumed")
        for key in (*VERSIONS, "status", "guard_status", "recheck_status"):
            same(row[key], rebuilt[key], "saved " + key)
        for call in target_calls:
            intent = read(output / "request_intents" / f"{call['target']}_{call['stage']}_{call['seat']}.json")
            folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
            same(intent["request"], read(folder / "request.json"), "captured before dispatch")
            same(fingerprint_from_redacted(intent["request"]), intent["request_fingerprint"], "request hash")
    return plan, rows, ledger, done


def fingerprint_from_redacted(body):
    from surgical_agent.research.verification.prior_panel import digest
    return digest(body)


def score(output, adapter):
    plan, rows, ledger, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["joint_recheck"]} for r in rows])
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    records = {r["key"]: read(output / "targets" / r["key"] / "result.json") for r in rows
               if (output / "targets" / r["key"] / "result.json").exists()}
    metrics = {v: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[v]} for r in rows])["arms"]["final"] for v in VERSIONS}
    pairs = (("h0", "original"), ("h0", "verb_guard"), ("h0", "joint_recheck"),
             ("original", "verb_guard"), ("original", "joint_recheck"), ("verb_guard", "joint_recheck"))
    details = {a + "_to_" + b: [{"key": r["key"], **frame_delta(r[a], r[b],
        truths[r["video_id"], r["frame_id"]]["gt"], truths[r["video_id"], r["frame_id"]]["mask"])}
        for r in rows] for a, b in pairs}
    costs = {account: {kind: str(sum(Decimal(c["charge"]) for c in ledger["calls"]
                if c["account"] == account and c["charge_kind"] == value))
        for kind, value in (("native", "native"), ("estimated", "conservative_estimate"),
                            ("unknown_reserved", "unknown_reserved"))} for account in plan["limits"]}
    diagnostics = base_trial.diagnostic_summary(rows, truths, records, metrics)
    diagnostics["strict_guard_success"] = diagnostics.pop("predeclared_success")
    original, joint, h0 = (metrics[v]["tasks"] for v in ("original", "joint_recheck", "h0"))
    values = [x[t][k] for x, t, k in ((joint, "verb", "micro_precision"), (original, "verb", "micro_precision"),
        (joint, "verb", "micro_f1"), (original, "verb", "micro_f1"), (joint, "ivt", "micro_f1"),
        (original, "ivt", "micro_f1"), (h0, "verb", "micro_precision"))]
    vp, op, vf, of, vi, oi, hp = values
    ok = all(v is not None for v in values)
    accepted = sum(len((r.get("recheck") or {}).get("accepted_groups", [])) for r in records.values())
    checks = {"Verb_precision_above_original": ok and vp > op, "Verb_F1_not_below_original": ok and vf >= of,
        "IVT_F1_not_below_original": ok and vi >= oi, "Verb_precision_at_least_H0": ok and vp >= hp,
        "no_selection_exceptions": not any(r.get("guard_error") or r.get("recheck_error") for r in records.values()),
        "actual_group_recovery": accepted > 0,
        "beneficial_change": sum(r["fixed_label_errors"] for r in details["original_to_joint_recheck"]) > 0}
    recheck_calls = [c for c in ledger["calls"] if c["stage"] == STAGE]
    report = {"profile": PROFILE, "planned_targets": TARGET_COUNT, "metrics": metrics,
        "comparisons": {k: summarize_deltas(v) for k, v in details.items()}, "costs_by_account": costs,
        "post_calls": len(ledger["calls"]), "additional_recheck_calls": len(recheck_calls),
        "additional_recheck_charge_usd": str(sum(Decimal(c["charge"]) for c in recheck_calls)),
        "elapsed_seconds": done["elapsed_seconds"], "transport_statuses": done["statuses"],
        "target_statuses": done["target_statuses"], "guard_statuses": done["guard_statuses"],
        "recheck_statuses": done["recheck_statuses"], "recheck_groups_accepted": accepted,
        "eligible_groups": sum(len((r.get("eligibility") or {}).get("groups", [])) for r in records.values()),
        "predeclared_success": {"rule": SUCCESS_RULE, "checks": checks, "confirmed": all(checks.values())},
        "audit": {"raw_outputs_replayed": True, "gt_loaded_after_closed_inference": True,
                  "all_planned_targets_retained": True, "phase_equals_shared_h0": True},
        **diagnostics, "limitations": plan["limitations"]}
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
        if any(x is None for x in (args.selection, args.prior_root, args.budget)):
            parser.error("prepare needs selection, prior-root and budget")
        prepare(args.output, args.selection, args.prior_root, args.budget, adapter)
    elif args.command == "execute":
        execute(args.output, adapter)
    elif args.command == "score":
        score(args.output, adapter)
    else:
        _, rows, ledger, _ = audit(args.output, adapter)
        print(json.dumps({"verified": True, "targets": len(rows), "calls": len(ledger["calls"])}))


if __name__ == "__main__":
    main()
