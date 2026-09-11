"""Paired prompt-only proposal trial with cached H0 and one unchanged panel.

Prepare/execute cannot read query GT. Score is offline after immutable closure.
Previously scored Training targets are a development replay, not confirmation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts import run_disputed_relation_trial as previous_trial
from scripts import run_new_training_verb_guard_trial as infrastructure
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire, review
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, ROUTES
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_disputed_relation_trial import exact_same as same
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.contact_first_candidate import (
    FIELD,
    TEMPLATE,
    VERSION,
    append_contact_audit,
)
from surgical_agent.research.verification.semantic_coordinator import item_error

PROFILE = "cached_h0_contact_first_proposal_paired_v1"
SOURCE = ROOT / "artifacts/preflight/disputed_relation_20260909_v1"
ARMS = ("control", "contact_first")
VERSIONS = ("h0", "previous_original", *ARMS)
TASKS = ("instrument", "verb", "target", "ivt", "phase")
TARGET_COUNT = 24
MAX_CALLS = TARGET_COUNT * 12
LIMITS = {"openrouter_usd": "8", "xai_usd": "3", "aliyun_cny": "3"}
ALLOWED_CALLS = {(a + "_proposal", "base") for a in ARMS} | {
    (a + "_review", s) for a in ARMS for s in SEATS}
SUCCESS_RULE = (
    "Versus contemporaneous control on all 24 fixed Training targets, contact-first IVT micro-F1 "
    "must strictly increase, Verb Precision and F1 and Target Precision and F1 must not decrease, "
    "and total label errors must not increase. At least one beneficial actual edit and no selector "
    "exception. Candidate coverage alone is not success. No GT-based threshold/prompt changes. "
    "This is a previously inspected development cohort, not independent generalization evidence."
)
PROTOCOL = (
    "Only append private_contact_audit to the existing graph-hinted proposal JSON input. "
    "Keep all original proposal fields, including H0, graph hints, ontology, schema and images. "
    "This is not blinded generation. Same cached H0 and query-video-excluded priors; two fresh "
    "proposals, original five-seat review and mean4 local repair; alternate arm order by target. "
    "Share the exact full panel request/replies (including failures) only when all five bodies "
    "are equal within a target; never reuse a subset of votes on a changed pool. No extra round, "
    "strict Verb guard, disputed recheck, Phase repair, Tracker or Gate. One deployed arm still "
    "uses at most one proposal plus five reviewers. Same output caps; no added audit text."
)


def audit_previous(source, adapter):
    original = previous_trial.same
    try:
        previous_trial.same = same
        return previous_trial.audit(source, adapter)
    finally:
        previous_trial.same = original


def wire_for(base, selected, h0, pool, hints, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    h0 = {t: deepcopy(h0[t]) for t in TASKS}
    body = proposal_wire(base, selected, h0, pool, hints)
    return append_contact_audit(body) if arm == "contact_first" else body


def ordered_hints(output, selected, initial):
    """Rebuild insertion order lost by sorted JSON archive serialization."""
    prior = read(output / f"prior_{selected['video_id']}.json")
    restored = retrieve_candidate_hints(initial["h0"], prior, video_id=selected["video_id"])
    same(restored, initial["hints"], "same frozen graph hints")
    return restored["packet"]


def run_arm(calls, base, selected, h0, hints, arm, reusable):
    h0 = {t: deepcopy(h0[t]) for t in TASKS}
    pool = make_pool(h0)
    body = wire_for(base, selected, h0, pool, hints, arm)
    started = perf_counter()
    raw = calls.call(selected["key"], arm + "_proposal", "base", body)
    record = {"prediction": deepcopy(h0), "status": "PROPOSAL_FAILED", "pool": pool,
              "proposal": raw, "proposal_fingerprint": fingerprint(body),
              "proposal_seconds": perf_counter() - started, "error": None}
    try:
        if raw is None:
            raise ValueError("missing proposal")
        pool = make_pool(h0, raw, pool)
    except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
        record["error"] = type(exc).__name__
        return record
    record["pool"] = pool
    if pool["propositions"]:
        record.update(review(calls, base, selected, h0, pool, arm, reusable))
    else:
        record["status"] = "EMPTY_POOL_UNVERIFIED"
    same(record["prediction"]["phase"], h0["phase"], "unchanged H0 Phase")
    return record


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
                         rates=plan["rates"], providers=PROVIDERS, max_calls=MAX_CALLS,
                         reasoning_seats=("grok", "gemini"))
        self.known = {s["key"] for s in plan["selection"]}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or (stage, seat) not in ALLOWED_CALLS:
            raise ValueError("undeclared call, H0 call or extra round")
        with self.lock:
            identity = (target, stage, seat)
            if identity in self.attempted:
                raise ValueError("no duplicate attempt or retry")
            self.attempted.add(identity)
            save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json", {
                "target": target, "stage": stage, "seat": seat,
                "request_fingerprint": fingerprint(body), "request": redact_images(body)})
        return super().call(target, stage, seat, body)


def metadata_preflight():
    def one(item):
        seat, route = item
        model = PROPOSER if seat == "base" else MODELS[seat]
        url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if endpoint["status"] != 0 or not {"response_format", "reasoning"} <= set(endpoint["supported_parameters"]):
            raise ValueError("fixed route unavailable")
        if any(Decimal(endpoint["pricing"][k]) > Decimal(rate) for k, rate in
               zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("current rate exceeds frozen reserve")
        return seat, {"url": url, "response": data}
    with ThreadPoolExecutor(max_workers=4) as workers:
        return {"queried_utc": now(), "paid_calls": 0, "endpoints": dict(workers.map(one, ROUTES.items()))}


def verify_plan(output):
    plan = read(output / "plan.json")
    expected = {"profile": PROFILE, "template_version": VERSION, "models": MODELS,
                "providers": PROVIDERS, "proposer": PROPOSER, "limits": LIMITS, "rates": RATES_V2,
                "arms": list(ARMS), "max_calls": MAX_CALLS, "target_count": TARGET_COUNT,
                "success_rule": SUCCESS_RULE, "protocol": PROTOCOL, "automatic_retries": 0}
    for name, value in expected.items():
        same(plan[name], value, name)
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "frozen source")
    for name, value in plan["input_sha256"].items():
        same(sha(output / name), value, "frozen input")
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "cached source archive")
    for selected in plan["selection"]:
        for im in selected["images"]:
            same(sha(im["path"]), im["sha256"], "original causal image")
    infrastructure.validate_budget(plan)
    infrastructure.reject_query_gt(plan["selection"])
    return plan


def prepare(output, source, adapter):
    if output.exists():
        raise ValueError("new experiment directory required")
    old_plan, old_rows, _, old_done = audit_previous(source, adapter)
    same(len(old_rows), TARGET_COUNT, "all 24 targets")
    view = infrastructure.InferenceOnlyAdapter(adapter)
    selection = deepcopy(old_plan["selection"])
    initials, preflight = [], {}
    for index, (selected, row) in enumerate(zip(selection, old_rows, strict=True)):
        if view.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training targets only")
        selected["arm_order"] = list(ARMS if index % 2 == 0 else reversed(ARMS))
        base = build_gemini_base(view, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "frozen input")
        same(fingerprint(gemini_h0_wire(base)), old_plan["h0_fingerprints"][selected["key"]], "H0 wire")
        prior = read(source / "priors" / f"{selected['video_id']}.json")
        infrastructure.validate_prior(prior, selected["video_id"], view)
        hints = retrieve_candidate_hints(row["h0"], prior, video_id=selected["video_id"])
        initial = {"key": row["key"], "h0": deepcopy(row["h0"]),
                   "previous_original": deepcopy(row["original"]), "hints": hints}
        initials.append(initial)
        bodies = {arm: wire_for(base, selected, row["h0"], make_pool(row["h0"]), hints["packet"], arm)
                  for arm in ARMS}
        old_result = read(source / "targets" / selected["key"] / "result.json")
        same(fingerprint(bodies["control"]), old_result["graph"]["request_fingerprints"]["proposal"],
             "control matches old proposal exactly")
        reverted = deepcopy(bodies["contact_first"])
        packet = json.loads(reverted["messages"][0]["content"][0]["text"])
        del packet[FIELD]
        reverted["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
        same(reverted, bodies["control"], "single input-field difference")
        preflight[selected["key"]] = {arm: {"fingerprint": fingerprint(body), "request": redact_images(body)}
                                      for arm, body in bodies.items()}
    metadata = metadata_preflight()
    save(output / "initial_state.json", {"targets": initials})
    save(output / "proposal_preflight.json", preflight)
    save(output / "model_metadata.json", metadata)
    for video in dict.fromkeys(s["video_id"] for s in selection):
        shutil.copyfile(source / "priors" / f"{video}.json", output / f"prior_{video}.json")
    dependencies = {ROOT / n for n in old_plan["source_sha256"]} | {
        Path(__file__).resolve(), TEMPLATE, ROOT / "src/surgical_agent/research/verification/contact_first_candidate.py",
        ROOT / "scripts/score_disputed_relation_trial.py", ROOT / "tests/unit/test_contact_first_candidate.py"}
    inputs = [output / n for n in ("initial_state.json", "proposal_preflight.json", "model_metadata.json")]
    inputs += list(output.glob("prior_*.json"))
    archive = {**old_done["inference_artifact_sha256"], "completion.json": sha(source / "completion.json")}
    plan = {"profile": PROFILE, "template_version": VERSION, "created_utc": now(), "source_root": str(source.resolve()),
        "selection": selection, "target_count": TARGET_COUNT, "models": MODELS, "providers": PROVIDERS,
        "proposer": PROPOSER, "limits": LIMITS, "rates": RATES_V2, "arms": list(ARMS), "max_calls": MAX_CALLS,
        "success_rule": SUCCESS_RULE, "protocol": PROTOCOL, "automatic_retries": 0,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(dependencies)},
        "input_sha256": {p.relative_to(output).as_posix(): sha(p) for p in inputs}, "archive_sha256": archive,
        "gt_policy": "Previously scored Training development replay. No query GT access in prepare/execute, no refitting priors. GT scoring only after new inference closure and raw-response replay.",
        "comparison": "Fresh paired control is primary; previously saved original is a secondary drift reference, not another replicate.",
        "request_freeze_scope": "Both proposal requests fixed now; reviewer bodies depend on unknown proposal replies and are captured before dispatch using frozen builders.",
        "phase_calls": 0, "new_h0_calls": 0, "rounds_per_arm": 1}
    save(output / "plan.json", plan)
    for path in dependencies:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    verify_plan(output)
    # Verify the actual persisted/reloaded form before allowing any paid run.
    for selected, initial in zip(selection, read(output / "initial_state.json")["targets"], strict=True):
        base = build_gemini_base(view, selected)
        hints = ordered_hints(output, selected, initial)
        for arm in ARMS:
            actual = wire_for(base, selected, initial["h0"], make_pool(initial["h0"]), hints, arm)
            same(fingerprint(actual), preflight[selected["key"]][arm]["fingerprint"], "roundtrip proposal wire")
    print(json.dumps({"prepared": PROFILE, "targets": TARGET_COUNT, "max_calls": MAX_CALLS,
                      "limits": LIMITS, "paid_calls": 0, "plan_sha256": sha(output / "plan.json")}), flush=True)


def execute(output, adapter):
    plan = verify_plan(output)
    if (output / "execution.lock").exists():
        raise ValueError("single-use paid run; no rerun")
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    initials = read(output / "initial_state.json")["targets"]
    preflight = read(output / "proposal_preflight.json")
    view = infrastructure.InferenceOnlyAdapter(adapter)
    calls = BoundCalls(output, plan)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
             "h0": i["h0"], "previous_original": i["previous_original"],
             **{arm: deepcopy(i["h0"]) for arm in ARMS},
             "statuses": dict.fromkeys(ARMS, "NOT_ATTEMPTED")} for s, i in zip(plan["selection"], initials, strict=True)]
    start, fatal = perf_counter(), None
    try:
        for selected, initial, row in zip(plan["selection"], initials, rows, strict=True):
            base = build_gemini_base(view, selected)
            same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "input metadata")
            hints = ordered_hints(output, selected, initial)
            reuse = []
            for arm in selected["arm_order"]:
                if calls.stopped:
                    row["statuses"][arm] = "BUDGET_OR_PROVIDER_STOPPED"
                    continue
                body = wire_for(base, selected, initial["h0"], make_pool(initial["h0"]), hints, arm)
                same(fingerprint(body), preflight[row["key"]][arm]["fingerprint"], "frozen proposal wire")
                record = run_arm(calls, base, selected, initial["h0"], hints, arm, reuse)
                save(output / "targets" / row["key"] / f"{arm}.json", record)
                row[arm], row["statuses"][arm] = record["prediction"], record["status"]
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "arm": arm, "status": record["status"],
                    "pool": len(record["pool"]["propositions"]), "shared": record.get("shared_from"),
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
            fatal = fatal or "FROZEN_INPUT_CHANGED"
            raise
        finally:
            files = [output / n for n in ("plan.json", "budget.json", "predictions.json", "execution.lock")]
            files += [p for folder in ("calls", "targets", "request_intents") for p in (output / folder).rglob("*.json")]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                "elapsed_seconds": perf_counter() - start, "post_calls": len(calls.rows),
                "transport_statuses": dict(Counter(c["status"] for c in calls.rows)),
                "arm_statuses": {a: dict(Counter(r["statuses"][a] for r in rows)) for a in ARMS},
                "query_gt_not_read_during_inference": True,
                "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output, adapter):
    plan = verify_plan(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or ledger["stopped"] is not True:
        raise ValueError("closed nonfatal inference required")
    files = {p.relative_to(output).as_posix() for folder in ("calls", "targets", "request_intents")
             for p in (output / folder).rglob("*.json")}
    if files | {"plan.json", "budget.json", "predictions.json", "execution.lock"} != set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen/missing artifact")
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(output / name), value, "closed artifact")
    calls = ledger["calls"]
    same(len(calls), done["post_calls"], "call count")
    if len(calls) > MAX_CALLS or len({(c["target"], c["stage"], c["seat"]) for c in calls}) != len(calls):
        raise ValueError("duplicate/excess calls")
    known = {s["key"] for s in plan["selection"]}
    for c in calls:
        if c["target"] not in known or (c["stage"], c["seat"]) not in ALLOWED_CALLS or c["status"] == "DISPATCHED":
            raise ValueError("undeclared/outstanding call")
        model = PROPOSER if c["seat"] == "base" else MODELS[c["seat"]]
        same(c["model"], model, "model")
        folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        intent = read(output / "request_intents" / f"{c['target']}_{c['stage']}_{c['seat']}.json")
        same(intent["request"], read(folder / "request.json"), "captured before dispatch")
        same(previous_trial.fingerprint_from_redacted(intent["request"]), intent["request_fingerprint"], "request hash")
        if c["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
            response = read(folder / "response.json")["body"]
            same(response["model"], model, "returned model")
            if c["seat"] in PROVIDERS:
                same(response.get("provider"), PROVIDERS[c["seat"]], "returned provider")
    for account, value in ledger["occupied"].items():
        charges = [Decimal(c["charge"]) for c in calls if c["account"] == account]
        if any(not c.is_finite() or c < 0 for c in charges):
            raise ValueError("invalid charges")
        same(sum(charges), Decimal(value), "exact ledger arithmetic")
        if Decimal(value) > Decimal(plan["limits"][account]):
            raise ValueError("budget exceeded")
    rows = read(output / "predictions.json")["targets"]
    initials = read(output / "initial_state.json")["targets"]
    same([(r["key"], r["video_id"], r["frame_id"]) for r in rows],
         [(s["key"], s["video_id"], s["frame_id"]) for s in plan["selection"]], "all planned rows")
    view, records = infrastructure.InferenceOnlyAdapter(adapter), {}
    for selected, initial, row in zip(plan["selection"], initials, rows, strict=True):
        same(row["h0"], initial["h0"], "cached H0")
        same(row["previous_original"], initial["previous_original"], "historical reference")
        base = build_gemini_base(view, selected)
        hints = ordered_hints(output, selected, initial)
        target_calls = [c for c in calls if c["target"] == row["key"]]
        replay, reuse = ReplayCalls(output, target_calls), []
        records[row["key"]] = {}
        for arm in selected["arm_order"]:
            path = output / "targets" / row["key"] / f"{arm}.json"
            if not path.exists():
                if row["statuses"][arm] != "BUDGET_OR_PROVIDER_STOPPED":
                    raise ValueError("missing target record")
                same(row[arm], initial["h0"], "unattempted fallback")
                continue
            stored = read(path)
            rebuilt = run_arm(replay, base, selected, initial["h0"], hints, arm, reuse)
            same(infrastructure.without_timing(rebuilt), infrastructure.without_timing(stored), "raw response replay")
            same(row[arm], rebuilt["prediction"], "scored prediction")
            same(row["statuses"][arm], rebuilt["status"], "arm status")
            records[row["key"]][arm] = stored
        same(len(replay.rows), len(target_calls), "all responses consumed")
    return plan, rows, records, ledger, done


def summarize(plan, rows, records, truth, ledger, done):
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics, manual = {}, {}
    for version in VERSIONS:
        inputs = [{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None,
                   "final": r[version]} for r in rows]
        metrics[version] = compute_repair_comparison(inputs)["arms"]["final"]
        manual[version] = {}
        for task in TASKS:
            valid = [r for r in rows if truths[r["video_id"], r["frame_id"]]["mask"][task]]
            sets = [(set(r[version][task]), set(truths[r["video_id"], r["frame_id"]]["gt"][task])) for r in valid]
            counts = {"tp": sum(len(p & g) for p, g in sets), "fp": sum(len(p - g) for p, g in sets),
                      "fn": sum(len(g - p) for p, g in sets), "exact_matches": sum(p == g for p, g in sets),
                      "valid_targets": len(valid)}
            for key, value in counts.items():
                same(metrics[version]["tasks"][task][key], value, "independent counts")
            manual[version][task] = counts
    pairs = (("h0", "control"), ("h0", "contact_first"), ("control", "contact_first"),
             ("previous_original", "control"), ("previous_original", "contact_first"))
    deltas = {a + "_to_" + b: [{"key": r["key"], **frame_delta(r[a], r[b],
        truths[r["video_id"], r["frame_id"]]["gt"], truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows]
        for a, b in pairs}
    comparisons = {k: summarize_deltas(v) for k, v in deltas.items()}
    coverage = {a: {t: Counter() for t in TASKS[:-1]} for a in ARMS}
    validity = {a: {s: Counter() for s in SEATS} for a in ARMS}
    pool_details = []
    for row in rows:
        gt = truths[row["video_id"], row["frame_id"]]
        for arm in ARMS:
            record = records[row["key"]].get(arm, {})
            pool = record.get("pool", make_pool(row["h0"]))["propositions"]
            for task in TASKS[:-1]:
                if not gt["mask"][task]:
                    continue
                available = {p["label_id"] for p in pool if p["task"] == task}
                expected = set(gt["gt"][task])
                coverage[arm][task].update({"valid_targets": 1, "gt_positive": len(expected),
                    "pool_true": len(available & expected), "pool_false": len(available - expected),
                    "missing_from_pool": len(expected - available)})
                pool_details.append({"key": row["key"], "arm": arm, "task": task,
                    "true_in_pool": sorted(available & expected), "false_in_pool": sorted(available - expected),
                    "missing_from_pool": sorted(expected - available)})
            if record.get("reviews") is not None:
                for seat in SEATS:
                    for prop in pool:
                        error = item_error(record["reviews"][seat]["judgments"].get(prop["id"]), prop["task"], 3)
                        validity[arm][seat][error or "VALID"] += 1
    costs = {account: {kind: str(sum(Decimal(c["charge"]) for c in ledger["calls"]
                if c["account"] == account and c["charge_kind"] == code))
        for kind, code in (("native", "native"), ("estimated", "conservative_estimate"), ("unknown_reserved", "unknown_reserved"))}
        for account in plan["limits"]}
    stages = defaultdict(list)
    for c in ledger["calls"]:
        stages[c["stage"]].append(c["elapsed_seconds"])
    timing = {k: {"calls": len(v), "request_seconds_sum": sum(v), "request_seconds_mean": mean(v)} for k, v in stages.items()}
    for arm in ARMS:
        seconds = [x[arm]["panel_seconds"] for x in records.values() if x.get(arm, {}).get("panel_seconds") is not None]
        timing[arm + "_panel_wall"] = {"fresh_panels": len(seconds),
            "mean_seconds": mean(seconds) if seconds else None, "median_seconds": median(seconds) if seconds else None}
    control, contact = (metrics[a]["tasks"] for a in ARMS)
    checks = {"IVT_F1_increases": contact["ivt"]["micro_f1"] > control["ivt"]["micro_f1"],
        **{f"{t}_{m}_does_not_decrease": contact[t][m] >= control[t][m]
           for t in ("verb", "target") for m in ("micro_precision", "micro_f1")},
        "total_errors_do_not_increase": comparisons["control_to_contact_first"]["net_errors_removed"] >= 0,
        "beneficial_actual_change": comparisons["control_to_contact_first"]["fixed_label_errors"] > 0,
        "no_selector_failure": all(r["statuses"][a] != "SELECTION_FAILED" for r in rows for a in ARMS)}
    report = {"profile": PROFILE, "targets": len(rows), "metrics": metrics, "comparisons": comparisons,
        "candidate_coverage": coverage, "review_validity": validity, "costs_by_account": costs,
        "post_calls": len(ledger["calls"]), "elapsed_seconds": done["elapsed_seconds"], "timing": timing,
        "transport_statuses": done["transport_statuses"], "arm_statuses": done["arm_statuses"],
        "shared_panels": sum(bool(r.get(a, {}).get("shared_from")) for r in records.values() for a in ARMS),
        "predeclared_success": {"rule": SUCCESS_RULE, "checks": checks, "confirmed": all(checks.values())},
        "audit": {"all_targets_retained": True, "raw_replay_passed": True, "independent_counts_passed": True,
                  "GT_loaded_after_closed_inference": True, "new_h0_calls": 0, "phase_fixed": True},
        "limitations": ["Previously scored Training development cohort, not new confirmation.",
                        "Changing pool can change valid reviewer judgments; reviewers are not independent truth.",
                        "Same-pool sharing is paired reuse, not two independent timing/model replicates.",
                        "Private observation instructions do not prove the model followed them."]}
    return report, deltas, pool_details, manual


def score(output, adapter):
    plan, rows, records, ledger, done = audit(output, adapter)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["contact_first"]} for r in rows])
    report, deltas, pools, manual = summarize(plan, rows, records, truth, ledger, done)
    save(output / "scored_truth.json", truth)
    save(output / "metrics.json", report)
    save(output / "frame_deltas.json", deltas)
    save(output / "candidate_coverage_details.json", pools)
    save(output / "independent_counts.json", manual)
    print(json.dumps({k: report[k] for k in ("targets", "post_calls", "costs_by_account", "predeclared_success")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    dataset = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.source, dataset)
    elif args.command == "execute":
        execute(args.output, dataset)
    else:
        score(args.output, dataset)
