"""Frozen paired development/confirmation of the selected graph+Phase default."""
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
from scripts.prepare_new_training_verb_selection import VIDEOS
from scripts.prepare_new_training_verb_selection import prepare as select_new
from scripts.run_candidate_panel_trial import sha
from scripts.run_contact_first_candidate_trial import metadata_preflight
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls, sources
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_new_training_verb_guard_trial import (
    InferenceOnlyAdapter,
    validate_prior,
    without_timing,
)
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_phase_extension_trial import phase_wire
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_disputed_relation_trial import exact_same as same
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification import roi_evidence as roi
from surgical_agent.research.verification import transactional_review as tx
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.phase_extension import apply_phase_choices

PROFILE = "default_graph_phase_improvement_v1"
GRAPH = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1"
PHASE = ROOT / "artifacts/preflight/phase_extension_eight_20260909_v1"
TRACKER = ROOT / "artifacts/training/tracker_clip_v2_oof5_20260906"
INPUTS = ROOT / "artifacts/preflight/default_improvement_inputs_20260909_v1"
TASKS = ("instrument", "verb", "target", "ivt", "phase")
LIMITS = {"development": {"openrouter_usd": "3", "xai_usd": "2", "aliyun_cny": "2"},
          "confirmation": {"openrouter_usd": "6", "xai_usd": "4", "aliyun_cny": "4"}}
SUCCESS = ("Compared with contemporaneous control: equal-head mean F1 strictly increases; IVT F1 and "
           "equal-head mean Precision do not decrease; summed FP+FN decreases; at least one real beneficial "
           "change. Report every head and failure, with shared Phase. Development selects at most one arm "
           "by mean F1, then IVT F1, then fixed arm order. Freeze it before new-target confirmation. "
           "Confirmation must pass the same rule; no post-score threshold or prompt tuning. "
           "Performance conditional on no panel transport failures is reported separately, not substituted.")


def select_inputs(dataset_root):
    inventory = INPUTS.with_name(INPUTS.name + "_history.json")
    if INPUTS.exists() or inventory.exists():
        raise ValueError("preserve frozen input selection")
    parent = ROOT / "artifacts/preflight/verb_addition_confirmation_20260909_inventory/historical_inventory.json"
    excluded = {v: set(fs) for v, fs in read(parent)["excluded_targets_by_video"].items()}
    hashes = {str(parent): sha(parent)}

    def visit(value, video=None):
        if isinstance(value, dict):
            video = value.get("video_id", video)
            if video in VIDEOS:
                for key in ("frame_id", "target_frame_id"):
                    if type(value.get(key)) is int:
                        excluded.setdefault(video, set()).add(value[key])
                excluded.setdefault(video, set()).update(x for x in value.get("causal_frame_ids", []) if type(x) is int)
            for child in value.values():
                visit(child, video)
        elif isinstance(value, list):
            for child in value:
                visit(child, video)

    for path in sorted((ROOT / "artifacts/preflight").rglob("plan.json")):
        if "frozen_source" in path.parts or "clean_checkout" in path.parts:
            continue
        visit(read(path))
        hashes[str(path.relative_to(ROOT))] = sha(path)
    save(inventory, {"source_sha256": hashes, "excluded_targets_by_video": {v: sorted(fs) for v, fs in excluded.items()}})
    select_new(inventory, INPUTS, dataset_root)


def tracker_rows(selection):
    index_path = TRACKER / "oof/index.json"
    index = read(index_path)
    sources_hash = {str(index_path): sha(index_path)}
    result = {}
    for video in dict.fromkeys(s["video_id"] for s in selection):
        rel = index["video_to_artifact"][video]
        path = TRACKER / "oof" / rel
        same(sha(path), index["artifacts"][rel], "OOF prediction hash")
        data = read(path)
        manifest_path = (TRACKER / "full/training_manifest.json" if video == "VID31"
                         else path.parent / "training_manifest.json")
        manifest = read(manifest_path)
        if (video in manifest["training_video_ids"] or manifest["bbox_policy"] != "clip_to_frame_v2"
                or not data["causal"] or data["inference_mode"] != "online_forward_only"):
            raise ValueError("causal corrected held-out tool predictions required")
        same(data["checkpoint_sha256"], manifest["checkpoint_sha256"], "OOF checkpoint")
        if video != "VID31" and video not in manifest["excluded_video_ids"]:
            raise ValueError("query video must be held out")
        sources_hash.update({str(path): sha(path), str(manifest_path): sha(manifest_path)})
        frames = {r["frame_id"]: r["tracks"] for r in data["videos"][video]["frames"]}
        for row in selection:
            if row["video_id"] == video:
                result[row["key"]] = {"tracks": deepcopy(frames.get(row["frame_id"], [])),
                    "frame_available": row["frame_id"] in frames, "source": str(path),
                    "vid31_box_gt_unavailable": video == "VID31"}
    return result, sources_hash


def prepare(output, cohort, candidate, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    if cohort == "development":
        selection = deepcopy(read(GRAPH / "plan.json")["selection"])
        rows = read(GRAPH / "predictions.json")["targets"]
        initial = {r["key"]: {"h0": r["h0"], "phase_raw": read(PHASE / "targets" / r["key"] / "phase_short.json")["raw_reviews"]} for r in rows}
        arms = [*roi.ARMS, "verifier_current"]
        references = {str(p): sha(p) for p in [GRAPH / "plan.json", GRAPH / "predictions.json",
                      PHASE / "plan.json", PHASE / "predictions.json", ROOT / "BEST_PHASE_RESULT.json"]}
        for s in selection:
            p = PHASE / "targets" / s["key"] / "phase_short.json"
            references[str(p)] = sha(p)
    else:
        if candidate not in (*roi.ARMS[1:], "verifier_current", "selector_only"):
            raise ValueError("freeze exactly one candidate before confirmation")
        selection = deepcopy(read(INPUTS / "selection.json")["selection"])
        initial = {s["key"]: {"h0": None, "phase_raw": None} for s in selection}
        arms = ["control"] if candidate == "selector_only" else ["control", candidate]
        references = {str(INPUTS / "selection.json"): sha(INPUTS / "selection.json")}
    tracks, tracker_hashes = tracker_rows(selection)
    references.update(tracker_hashes)
    view = InferenceOnlyAdapter(adapter)
    for i, s in enumerate(selection):
        if adapter.entries[s["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        proposal_arms = [a for a in arms if a != "verifier_current"]
        s["arm_order"] = proposal_arms[i % len(proposal_arms):] + proposal_arms[:i % len(proposal_arms)]
        base = build_gemini_base(view, s)
        metadata = canonical_request_metadata(base).to_mapping()
        if "request_metadata" in s:
            same(s["request_metadata"], metadata, "original H0 media")
        s["request_metadata"] = metadata
        initial[s["key"]]["views"] = roi.create_views(s, tracks[s["key"]]["tracks"], output / "views" / s["key"])
        initial[s["key"]]["tracker"] = tracks[s["key"]]
        s["h0_fingerprint"] = fingerprint(gemini_h0_wire(base))
    for video in dict.fromkeys(s["video_id"] for s in selection):
        p = GRAPH / "priors" / f"{video}.json"
        prior = read(p)
        validate_prior(prior, video, view)
        save(output / "priors" / f"{video}.json", prior)
        references[str(p)] = sha(p)
    save(output / "initial_state.json", initial)
    save(output / "model_metadata.json", metadata_preflight())
    dependencies = {*sources(), *(ROOT / "scripts").glob("*.py"), *(ROOT / "src/surgical_agent").rglob("*.py"),
                    ROOT / "DEFAULT_PIPELINE_VERSION.json", ROOT / "tests/unit/test_roi_evidence.py"}
    dependencies = sorted(p for p in dependencies if p.is_file())
    for p in dependencies:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    inputs = [p for folder in ("views", "priors") for p in (output / folder).rglob("*") if p.is_file()]
    inputs += [output / "initial_state.json", output / "model_metadata.json"]
    plan = {"profile": PROFILE, "created_utc": now(), "cohort": cohort, "candidate": candidate,
        "selection": selection, "arms": arms, "versions": ["h0", *arms, "selector_only"],
        "limits": LIMITS[cohort], "rates": RATES_V2, "models": MODELS, "providers": PROVIDERS,
        "proposer": PROPOSER, "max_calls": len(selection) * (len(proposal_arms) + 5 * len(arms) + (6 if cohort == "confirmation" else 0)),
        "success_rule": SUCCESS, "automatic_retries": 0,
        "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        "inputs": {p.relative_to(output).as_posix(): sha(p) for p in inputs}, "references": references,
        "protocol": "Same frozen H0 and original full-frame review, pool limits and mean4 acceptance. "
        "Current/temporal auxiliary predicted-box crops alter proposer visual input only. Exact identical "
        "proposal/review requests share results INCLUDING failures within a target. Verifier-current uses the "
        "CONTROL pool and target full-scene+ROI gallery, always after other arms (fixed order limitation). "
        "No new review rounds within an arm. Selector-only reuses control panel with transactional selection. "
        "One shared independent short Phase panel (archived for development; freshly called alongside graph for confirmation). "
        "All arms merged with same Phase. Development cohort was previously inspected; confirmation uses new "
        "time/mask-selected targets in same Training videos. No GT before inference closure. "
        "VID31 predictions use full detector not trained on VID31 instance annotations; box GT unavailable. "
        "ROIs introduce predicted-Tracker dependence, relevant to future Tracker ablations."}
    save(output / "plan.json", plan)
    verify(output)
    print(json.dumps({"prepared": str(output), "cohort": cohort, "targets": len(selection),
        "max_calls": plan["max_calls"], "limits": plan["limits"], "plan_sha256": sha(output / "plan.json"),
        "regions_per_target": {k: len(v["views"]["roi_current"]) for k, v in initial.items()}}), flush=True)


def verify(output):
    plan = read(output / "plan.json")
    for k, v in {"profile": PROFILE, "success_rule": SUCCESS, "models": MODELS, "providers": PROVIDERS, "rates": RATES_V2}.items():
        same(plan[k], v, "frozen " + k)
    for group, root in (("sources", ROOT), ("inputs", output), ("references", Path())):
        for name, digest in plan[group].items():
            same(sha(root / name), digest, group + " " + name)
    for s in plan["selection"]:
        for im in s["images"]:
            same(sha(im["path"]), im["sha256"], "original image")
    return plan


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in plan["limits"].items()},
            rates=RATES_V2, providers=PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"))
        self.known = {s["key"] for s in plan["selection"]}
        self.allowed = {(a + "_proposal", "base") for a in plan["arms"] if a != "verifier_current"} | {
            (a + "_review", s) for a in plan["arms"] for s in SEATS}
        if plan["cohort"] == "confirmation":
            self.allowed |= {("h0", "base")} | {("phase", s) for s in SEATS}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or (stage, seat) not in self.allowed:
            raise ValueError("undeclared request")
        with self.lock:
            key = target, stage, seat
            if key in self.attempted:
                raise ValueError("retry forbidden")
            self.attempted.add(key)
            save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json",
                 {"fingerprint": fingerprint(body), "request": redact_images(body)})
        return super().call(target, stage, seat, body)


def collect_phase(calls, selected):
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw = dict(zip(SEATS, workers.map(lambda seat: calls.call(selected["key"], "phase", seat,
            phase_wire(seat, selected)), SEATS), strict=True))
    return {"raw": raw, "phase_seconds": perf_counter() - start}


def review_pool(calls, base, selected, h0, pool, arm, reusable, views=()):
    bodies = {s: roi.augment_review(review_wire(s, base, selected, pool), views) for s in SEATS}
    shared = next((r for r in reusable if r["bodies"] == bodies), None)
    start = perf_counter()
    if shared:
        raw = deepcopy(shared["raw"])
        count = shared["count"]
    else:
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], arm + "_review", s, bodies[s]), SEATS), strict=True))
        count = sum(r["target"] == selected["key"] and r["stage"] == arm + "_review" for r in calls.rows)
        reusable.append({"arm": arm, "bodies": bodies, "raw": deepcopy(raw), "count": count})
    normalized, formatting = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(normalized, pool, image_count=len(base.images))
    prediction = panel.select(h0, pool, means, threshold=4)
    issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
    return {"prediction": prediction, "raw": raw, "reviews": normalized, "format_diagnostics": formatting,
        "means": means, "diagnostics": diagnostics, "issues": issues,
        "status": "INCOMPLETE_PANEL" if count != 5 else "UNRESOLVED" if issues else "MODEL_PASS",
        "request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()},
        "shared_from": shared["arm"] if shared else None, "panel_seconds": perf_counter() - start,
        "call_count": 0 if shared else count}


def run_case(calls, base, selected, initial, prior, plan):
    start = perf_counter()
    record = {"h0": deepcopy(initial["h0"]), "h0_raw": None, "arms": {}, "predictions": {}, "status": "READY"}
    if record["h0"] is None:
        raw = calls.call(selected["key"], "h0", "base", gemini_h0_wire(base))
        record["h0_raw"] = raw
        try:
            validate_final_only(raw)
            record["h0"] = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
        except (ApiSchemaError, ValueError, TypeError, KeyError):
            record.update(status="H0_FAILED", predictions=dict.fromkeys(plan["versions"]))
            return record
    h0 = {t: deepcopy(record["h0"][t]) for t in TASKS}
    hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
    record["hints"] = hints
    proposals, reusable = [], []
    with ThreadPoolExecutor(max_workers=1) as phase_worker:
        future = phase_worker.submit(collect_phase, calls, selected) if initial["phase_raw"] is None else None
        for arm in selected["arm_order"]:
            pool = make_pool(h0)
            body = roi.augment_proposal(proposal_wire(base, selected, h0, pool, hints["packet"]), initial["views"][arm])
            shared = next((p for p in proposals if p["body"] == body), None)
            begin = perf_counter()
            raw = deepcopy(shared["raw"]) if shared else calls.call(selected["key"], arm + "_proposal", "base", body)
            proposals.append({"body": body, "raw": deepcopy(raw), "arm": arm})
            item = {"prediction": deepcopy(h0), "pool": pool, "proposal": raw, "status": "PROPOSAL_FAILED",
                "proposal_fingerprint": fingerprint(body), "proposal_shared_from": shared["arm"] if shared else None,
                "proposal_seconds": perf_counter() - begin}
            record["arms"][arm] = item
            try:
                if raw is None:
                    raise ValueError("proposal unavailable")
                item["pool"] = make_pool(h0, raw, pool)
            except (ApiSchemaError, ValueError, TypeError, KeyError):
                continue
            if item["pool"]["propositions"]:
                item.update(review_pool(calls, base, selected, h0, item["pool"], arm, reusable))
            else:
                item["status"] = "EMPTY_POOL_UNVERIFIED"
        if "verifier_current" in plan["arms"]:
            item = deepcopy(record["arms"]["control"])
            item["proposal_shared_from"] = "control"
            item["proposal_seconds"] = 0
            if item["pool"]["propositions"] and "reviews" in item:
                item.update(review_pool(calls, base, selected, h0, item["pool"], "verifier_current", reusable,
                                        initial["views"]["roi_current"]))
            record["arms"]["verifier_current"] = item
        phase = future.result() if future else {"raw": initial["phase_raw"], "phase_seconds": 0}
    record["phase"] = phase
    for arm, item in record["arms"].items():
        final, decision = apply_phase_choices(item["prediction"], phase["raw"], len(base.images))
        record["predictions"][arm] = final
        item["phase_decision"] = decision
    control = record["arms"]["control"]
    if "reviews" in control:
        evidence = tx.assess_legacy(control["reviews"], control["pool"], len(base.images))
        record["selector"] = tx.apply(h0, control["pool"], evidence)
        prediction = record["selector"]["prediction"]
    else:
        prediction = control["prediction"]
    record["predictions"]["selector_only"] = apply_phase_choices(prediction, phase["raw"], len(base.images))[0]
    record["predictions"]["h0"] = h0
    record["total_seconds"] = perf_counter() - start
    return record


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as f:
        f.write(sha(output / "plan.json"))
    initial = read(output / "initial_state.json")
    calls = BoundCalls(output, plan)
    calls.persist()
    view, rows = InferenceOnlyAdapter(adapter), []
    started, fatal = perf_counter(), None
    try:
        for selected in plan["selection"]:
            base = build_gemini_base(view, selected)
            same(fingerprint(gemini_h0_wire(base)), selected["h0_fingerprint"], "fixed H0")
            record = run_case(calls, base, selected, initial[selected["key"]],
                              read(output / "priors" / f"{selected['video_id']}.json"), plan)
            save(output / "targets" / selected["key"] / "result.json", record)
            rows.append({k: selected[k] for k in ("key", "video_id", "frame_id")} | record["predictions"])
            save(output / "predictions.json", rows)
            print(json.dumps({"target": selected["key"], "completed": len(rows), "post_calls": len(calls.rows),
                "statuses": {a: r["status"] for a, r in record["arms"].items()},
                "seconds": record.get("total_seconds"), "budget_stopped": calls.stopped}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", rows)
        files = [output / n for n in ("plan.json", "initial_state.json", "budget.json", "predictions.json", "execution.lock")]
        files += [p for folder in ("targets", "calls", "request_intents") for p in (output / folder).rglob("*.json")]
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
            "elapsed_seconds": perf_counter() - started, "post_calls": len(calls.rows),
            "hashes": {p.relative_to(output).as_posix(): sha(p) for p in files}})


def passing(metrics, changes, arm):
    before, after = metrics["control"]["tasks"], metrics[arm]["tasks"]
    avg = lambda values, name: sum(values[t][name] for t in TASKS) / len(TASKS)
    return {"mean_f1_improved": avg(after, "micro_f1") > avg(before, "micro_f1"),
            "mean_precision_not_decreased": avg(after, "micro_precision") >= avg(before, "micro_precision"),
            "ivt_f1_not_decreased": after["ivt"]["micro_f1"] >= before["ivt"]["micro_f1"],
            "net_errors_reduced": changes["net_errors_removed"] > 0,
            "beneficial_changes": changes["fixed_label_errors"] > 0}


def score(output, adapter):
    plan = verify(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal inference required")
    for name, digest in done["hashes"].items():
        same(sha(output / name), digest, "closed inference " + name)
    same(len(ledger["calls"]), done["post_calls"], "POST count")
    if len(ledger["calls"]) > plan["max_calls"]:
        raise ValueError("call cap exceeded")
    for account, amount in ledger["occupied"].items():
        charges = [Decimal(c["charge"]) for c in ledger["calls"] if c["account"] == account]
        if any(not c.is_finite() or c < 0 for c in charges):
            raise ValueError("invalid charges")
        same(sum(charges), Decimal(amount), "ledger arithmetic")
        if Decimal(amount) > Decimal(plan["limits"][account]):
            raise ValueError("account budget exceeded")
    rows, initial = read(output / "predictions.json"), read(output / "initial_state.json")
    same(len(rows), len(plan["selection"]), "all targets")
    replay, records = ReplayCalls(output, ledger["calls"]), {}
    for selected, row in zip(plan["selection"], rows, strict=True):
        same(row["key"], selected["key"], "row order")
        record = run_case(replay, build_gemini_base(InferenceOnlyAdapter(adapter), selected), selected,
            initial[selected["key"]], read(output / "priors" / f"{selected['video_id']}.json"), plan)
        saved = read(output / "targets" / row["key"] / "result.json")
        same(without_timing(record), without_timing(saved), "raw request/response replay")
        same({v: row[v] for v in plan["versions"]}, record["predictions"], "final predictions")
        records[row["key"]] = saved
    same(len(replay.rows), len(ledger["calls"]), "all calls replayed")
    _, truth = score_saved(adapter, [{**r, "h1": None, "final": r["control"]} for r in rows])
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics = {v: compute_repair_comparison([{**truths[r['video_id'], r['frame_id']], "h0": r["h0"],
        "h1": None, "final": r[v]} for r in rows])["arms"]["final"] for v in plan["versions"]}
    details = {a + "_to_" + b: [{"key": r["key"], **frame_delta(r[a], r[b],
        truths[r['video_id'], r['frame_id']]["gt"], truths[r['video_id'], r['frame_id']]["mask"])} for r in rows]
        for a, b in [("h0", v) for v in plan["versions"] if v != "h0"] +
                    [("control", v) for v in plan["versions"] if v not in ("h0", "control")]}
    changes = {k: summarize_deltas(v) for k, v in details.items()}
    checks = {v: passing(metrics, changes["control_to_" + v], v) for v in plan["versions"] if v not in ("h0", "control")}
    eligible = [v for v, c in checks.items() if all(c.values())]
    rank = lambda v: (sum(metrics[v]["tasks"][t]["micro_f1"] for t in TASKS), metrics[v]["tasks"]["ivt"]["micro_f1"], -plan["versions"].index(v))
    winner = max(eligible, key=rank) if eligible else None
    coverage = {a: Counter() for a in plan["arms"]}
    for row in rows:
        gt = truths[row['video_id'], row['frame_id']]
        for a in plan["arms"]:
            if not gt["mask"]["ivt"] or a not in records[row["key"]]["arms"]:
                continue
            pool = {p["label_id"] for p in records[row["key"]]["arms"][a]["pool"]["propositions"] if p["task"] == "ivt"}
            gold = set(gt["gt"]["ivt"])
            coverage[a].update(gt=len(gold), covered=len(pool & gold), missing=len(gold - pool), wrong_candidates=len(pool - gold))
    costs = {a: {kind: str(sum(Decimal(c['charge']) for c in ledger['calls'] if c['account'] == a and c['charge_kind'] == kind))
                 for kind in ('native', 'conservative_estimate', 'unknown_reserved')} for a in plan["limits"]}
    sensitivity = {}
    for variant in checks:
        source_arm = "control" if variant == "selector_only" else variant
        usable = [r for r in rows if all(
            all(isinstance(records[r["key"]]["arms"].get(a, {}).get("raw", {}).get(s), dict) for s in SEATS)
            for a in ("control", source_arm))]
        sensitivity[variant] = {"targets": len(usable), "excluded_keys": [r["key"] for r in rows if r not in usable],
            "metrics": {v: compute_repair_comparison([{**truths[r['video_id'], r['frame_id']], "h0": r["h0"],
                "h1": None, "final": r[v]} for r in usable])["arms"]["final"]
                for v in ("h0", "control", variant)} if usable else {}}
    costs_by_stage = {stage: {a: {kind: str(sum(Decimal(c['charge']) for c in ledger['calls']
        if c['stage'] == stage and c['account'] == a and c['charge_kind'] == kind))
        for kind in ('native', 'conservative_estimate', 'unknown_reserved')} for a in plan['limits']}
        for stage in sorted({c['stage'] for c in ledger['calls']})}
    report = {"metrics": metrics, "comparisons": changes, "checks": checks, "development_candidate": winner,
        "confirmation_candidate": plan["candidate"], "confirmation_pass": plan["cohort"] == "confirmation" and all(checks[plan['candidate']].values()),
        "candidate_coverage": {a: dict(c) for a, c in coverage.items()}, "costs": costs,
        "costs_by_stage": costs_by_stage, "transport_sensitivity": sensitivity,
        "post_calls": len(ledger["calls"]), "transport": dict(Counter(c["status"] for c in ledger["calls"])),
        "elapsed_seconds": done["elapsed_seconds"], "raw_replayed": True}
    save(output / "metrics.json", report)
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", details)
    print(json.dumps({"candidate": winner, "confirmation_pass": report["confirmation_pass"], "coverage": report["candidate_coverage"],
        "f1": {a: {t: m['micro_f1'] for t, m in v['tasks'].items()} for a, v in metrics.items()}, "costs": costs}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "prepare", "execute", "score"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohort", choices=("development", "confirmation"), default="development")
    parser.add_argument("--candidate", choices=("roi_current", "roi_temporal", "verifier_current", "selector_only"))
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    if args.command == "select":
        select_inputs(args.dataset_root)
    else:
        if args.output is None:
            parser.error("--output required")
        adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
        if args.command == "prepare":
            prepare(args.output, args.cohort, args.candidate, adapter)
        else:
            {"execute": execute, "score": score}[args.command](args.output, adapter)
