"""Fresh paired proposal wording trial using the unchanged compact v1.1 reviewers.

One shared H0 and independent Phase panel per target; two proposal arms. GT is
available only to the separate scorer after closure and exact response replay.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from threading import RLock
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_compact_parallel_repair import (
    PROMPT_FILES,
    PROMPT_PROFILE,
    phase_wire,
    review_wire,
)
from scripts.run_compact_parallel_repair import VERSION as DEFAULT_VERSION
from scripts.run_contact_first_candidate_trial import metadata_preflight
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_new_training_verb_guard_trial import (
    InferenceOnlyAdapter,
    reject_query_gt,
    selection_rows,
    validate_prior,
    without_timing,
)
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_five_head_repair_trial import ReplayCalls
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.action_cue_candidate import (
    TEMPLATE,
    VERSION,
    replace_action_cues,
    token_audit,
)
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    phase_choice_error,
)

PROFILE = "fresh_training_action_cue_proposal_compact_v1"
ARMS = ("control", "action_cue")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
TARGET_COUNT = 24
TARGET_WORKERS = 2
SELECTION_GAP_FRAMES = 75
MAX_CALLS = TARGET_COUNT * 18
LIMITS = {"openrouter_usd": "4", "xai_usd": "3", "aliyun_cny": "2"}
ALLOWED_CALLS = {("h0", "base"), *(("shared_phase", s) for s in SEATS),
                 *((a + "_proposal", "base") for a in ARMS),
                 *((a + "_review", s) for a in ARMS for s in SEATS)}
SUCCESS_RULE = (
    "Versus fresh control: Verb candidate GT coverage and final Verb micro-F1 strictly increase; "
    "Verb Precision, IVT F1, Instrument F1 and Target F1 do not decrease. No selector exception; "
    "all 24 targets attempted and all 24 paired native proposal prompt-token counts measurable "
    "and non-increasing. Shared Phase is not an improvement claim. No GT-based prompt, threshold "
    "or sample changes. No automatic promotion. New frames from previously used Training videos "
    "are not independent-video generalization."
)
PROTOCOL = (
    "Replace only proposal instructions with concrete action cues. One shared fresh Gemini H0, "
    "same query-video-excluded Training prior and images; unchanged compact v1.1 five reviewers, "
    "mean4 additions / mean2 deletions and original component protection. Alternate proposal-arm "
    "order by target. Share an entire five-seat panel including failures only if all five request "
    "bodies match within the same target; never share a subset for different pools. One independent "
    "compact Phase panel shared across arms. No Phase or reviewer answers in proposals. Two target "
    "workers; Phase overlaps the two sequential graph arms. No extra rounds or API retries. "
    "Same JSON schemas, caps, models, routes, reasoning and image details. Local UTF8 and two "
    "tokenizers are preflight proxies; only native usage can establish actual proposal input tokens. "
    "Changed candidate pools can change downstream reviewer input/output tokens."
)


def same(actual, expected, message):
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(message)


def empty_prediction():
    return {t: [] for t in TASKS}


def proposal_pair(base, selected, h0, hints):
    h0 = {t: deepcopy(h0[t]) for t in TASKS}
    control = proposal_wire(base, selected, h0, make_pool(h0), hints)
    treatment = replace_action_cues(control)
    audit = token_audit(control, treatment)
    return {"control": control, "action_cue": treatment}, audit


def review_panel(calls, base, selected, h0, pool, arm, reusable):
    bodies = {s: review_wire(s, base, selected, pool) for s in SEATS}
    record = {"request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()}}
    shared = next((r for r in reusable if r["bodies"] == bodies), None)
    if shared is None:
        begin = perf_counter()
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(
                lambda s: calls.call(selected["key"], arm + "_review", s, bodies[s]), SEATS), strict=True))
        seconds = perf_counter() - begin
        count = sum(r["target"] == selected["key"] and r["stage"] == arm + "_review" for r in calls.rows)
        reusable.append({"arm": arm, "bodies": bodies, "raw": deepcopy(raw), "count": count, "seconds": seconds})
        record.update(panel_seconds=seconds, call_count=count)
    else:
        raw, count = deepcopy(shared["raw"]), shared["count"]
        record.update(shared_from=shared["arm"], panel_seconds=None, call_count=0,
                      original_panel_seconds=shared["seconds"])
    normalized, formatting = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(normalized, pool, image_count=len(base.images))
    try:
        prediction = panel.select(h0, pool, means, threshold=4)
        issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
        status = "UNRESOLVED" if issues else "MODEL_PASS"
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        prediction, issues, status = deepcopy(h0), [], "SELECTION_FAILED"
    if count != 5:
        status = "INCOMPLETE_PANEL"
    record.update(prediction=prediction, status=status, raw=raw, reviews=normalized,
                  format_diagnostics=formatting, means=means, diagnostics=diagnostics, issues=issues)
    return record


def run_arm(calls, base, selected, h0, body, arm, reusable):
    pool = make_pool(h0)
    begin = perf_counter()
    raw = calls.call(selected["key"], arm + "_proposal", "base", body)
    record = {"prediction": deepcopy(h0), "status": "PROPOSAL_FAILED", "pool": pool,
              "proposal": raw, "proposal_fingerprint": fingerprint(body), "error": None,
              "proposal_seconds": perf_counter() - begin}
    try:
        if raw is None:
            raise ValueError("missing proposal")
        pool = make_pool(h0, raw, pool)
    except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
        record["error"] = type(exc).__name__
        return record
    record["pool"] = pool
    if pool["propositions"]:
        record.update(review_panel(calls, base, selected, h0, pool, arm, reusable))
    else:
        record["status"] = "EMPTY_POOL_UNVERIFIED"
    same(record["prediction"]["phase"], h0["phase"], "graph changed Phase")
    return record


def shared_phase(calls, selected):
    bodies = {s: phase_wire(s, selected) for s in SEATS}
    begin = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw = dict(zip(SEATS, workers.map(
            lambda s: calls.call(selected["key"], "shared_phase", s, bodies[s]), SEATS), strict=True))
    errors = {s: phase_choice_error(raw[s], len(selected["images"])) for s in SEATS}
    return {"raw_reviews": raw, "errors": errors, "phase_seconds": perf_counter() - begin,
            "request_fingerprints": {s: fingerprint(b) for s, b in bodies.items()},
            "status": "INVALID_PANEL" if any(errors.values()) else "REVIEWED"}


def run_target(calls, base, selected, prior, persist=None):
    persist = persist or (lambda name, value: None)
    row = {k: selected[k] for k in ("key", "video_id", "frame_id")}
    row.update(h0=None, **{a: empty_prediction() for a in ARMS},
               statuses=dict.fromkeys(ARMS, "H0_FAILED"))
    begin = perf_counter()
    raw = calls.call(selected["key"], "h0", "base", gemini_h0_wire(base))
    shared = {"h0_raw": raw, "h0_seconds": perf_counter() - begin, "status": "H0_FAILED"}
    records = {}
    try:
        validate_final_only(raw)
    except (ApiSchemaError, TypeError, ValueError):
        persist("shared", shared)
        return row, records, shared
    h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
    row["h0"] = deepcopy(h0)
    hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
    bodies, proxy = proposal_pair(base, selected, h0, hints["packet"])
    shared.update(status="H0_VALID", hints=hints, proposal_token_proxy=proxy,
                  proposal_fingerprints={a: fingerprint(b) for a, b in bodies.items()})
    persist("shared", shared)
    reusable = []
    with ThreadPoolExecutor(max_workers=1) as workers:
        phase_future = workers.submit(shared_phase, calls, selected)
        for arm in selected["arm_order"]:
            record = run_arm(calls, base, selected, h0, bodies[arm], arm, reusable)
            records[arm] = record
            row[arm], row["statuses"][arm] = record["prediction"], record["status"]
            persist(arm, record)
        shared["phase"] = phase_future.result()
    for arm in ARMS:
        row[arm], decision = apply_phase_choices(row[arm], shared["phase"]["raw_reviews"], len(base.images))
        shared.setdefault("phase_decisions", {})[arm] = decision
    same(row["control"]["phase"], row["action_cue"]["phase"], "shared Phase differs")
    persist("shared", shared)
    return row, records, shared


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                         providers=PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.known = {s["key"] for s in plan["selection"]}
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or (stage, seat) not in ALLOWED_CALLS:
            raise ValueError("undeclared target/stage/seat")
        same(body["model"], PROPOSER if seat == "base" else MODELS[seat], "changed model")
        reject_query_gt(body)
        with self.lock:
            identity = (target, stage, seat)
            if identity in self.attempted:
                raise ValueError("automatic API retry forbidden")
            self.attempted.add(identity)
            save(self.output / "request_intents" / f"{target}_{stage}_{seat}.json",
                 {"target": target, "stage": stage, "seat": seat, "request_fingerprint": fingerprint(body),
                  "request": redact_images(body)})
        return super().call(target, stage, seat, body)

    def persist(self):
        # Retry only the local atomic ledger write; never repeat an HTTP call.
        for attempt in range(6):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 5:
                    raise
                sleep(0.05 * (attempt + 1))


def runtime_paths():
    paths = {Path(__file__).resolve(), TEMPLATE, *PROMPT_FILES,
             ROOT / "scripts/score_action_cue_candidate_trial.py",
             ROOT / "tests/unit/test_action_cue_trial.py",
             ROOT / "tests/unit/test_action_cue_candidate.py",
             ROOT / "tests/unit/test_action_cue_candidate_scoring.py",
             ROOT / "tools/audit/preflight_action_cue_replay.py",
             ROOT / "configs/perception/joint_openrouter_h0.yaml"}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == ".py" and ".venv" not in str(path):
                paths.add(path)
    paths.update(p for p in (ROOT / "src/surgical_agent/perception/prompts").glob("*") if p.is_file())
    return sorted(paths)


def verify_plan(output):
    plan = read(output / "plan.json")
    expected = {"profile": PROFILE, "template_version": VERSION, "default_version": DEFAULT_VERSION,
                "verifier_prompt_profile": PROMPT_PROFILE, "models": MODELS, "proposer": PROPOSER,
                "providers": PROVIDERS, "limits": LIMITS, "rates": RATES_V2, "arms": ARMS,
                "target_count": TARGET_COUNT, "max_calls": MAX_CALLS, "target_workers": TARGET_WORKERS,
                "selection_gap_frames": SELECTION_GAP_FRAMES,
                "success_rule": SUCCESS_RULE, "protocol": PROTOCOL, "automatic_retries": 0}
    for field, value in expected.items():
        same(plan[field], value, "changed frozen " + field)
    for section, root in (("source_sha256", ROOT), ("input_sha256", output)):
        for name, digest in plan[section].items():
            same(sha(root / name), digest, "changed " + section + " " + name)
    for selected in plan["selection"]:
        for im in selected["images"]:
            same(sha(im["path"]), im["sha256"], "changed image")
    return plan


def prepare(output, selection_path, prior_root, adapter):
    if output.exists():
        raise ValueError("a new single-use experiment directory is required")
    view = InferenceOnlyAdapter(adapter)
    manifest = read(selection_path)
    history_path = Path(manifest["history_inventory"])
    same(sha(history_path), manifest["history_inventory_sha256"], "historical exclusion inventory")
    same(manifest["gap_raw_frames"], SELECTION_GAP_FRAMES, "pre-inference history spacing")
    selection_source = ROOT / "scripts/prepare_new_training_verb_selection.py"
    same(sha(selection_source), manifest["selection_source_sha256"], "deterministic selection implementation")
    selection_override = selection_path.parent.parent / "select_gap75.py"
    spacing_evidence = selection_path.parent.parent / "spacing_feasibility.json"
    same(sha(selection_override), manifest["parameter_override_script_sha256"], "declared spacing adaptation")
    same(sha(spacing_evidence), manifest["spacing_feasibility_sha256"], "spacing availability evidence")
    selection = selection_rows(manifest, view)
    historical = read(history_path)["excluded_targets_by_video"]
    masks = {s["key"]: s["gt_availability_only"] for s in manifest["selection"]}
    seen = {}
    for selected in selection:
        video, frame = selected["video_id"], selected["frame_id"]
        mask = masks[selected["key"]]
        if (set(mask) != set(TASKS) or not all(mask.values())
                or any(abs(frame - old) <= SELECTION_GAP_FRAMES
                       for old in historical.get(video, []) + seen.get(video, []))
                or set(selected["causal_frame_ids"]) & set(historical.get(video, []))):
            raise ValueError("full-mask and history/cohort-separated new targets required")
        seen.setdefault(video, []).append(frame)
    same(Counter(s["video_id"] for s in selection), dict.fromkeys(("VID103", "VID23", "VID31", "VID96"), 6),
         "four videos with six new targets each")
    priors = {}
    for video in sorted({s["video_id"] for s in selection}):
        same(sha(prior_root / f"{video}.json"), manifest["prior_sha256"][video], "original prior bytes")
        prior = read(prior_root / f"{video}.json")
        validate_prior(prior, video, view)
        priors[video] = prior
    preflight = {}
    for index, selected in enumerate(selection):
        selected["arm_order"] = list(ARMS if index % 2 == 0 else reversed(ARMS))
        base = build_gemini_base(view, selected)
        selected["request_metadata"] = canonical_request_metadata(base).to_mapping()
        # H0 is unknown until inference. Empty dummy state audits fixed wording,
        # schemas and images; actual H0-dependent proposal pairs are checked again.
        dummy = {**empty_prediction(), "phase": [0]}
        hints = retrieve_candidate_hints(dummy, priors[selected["video_id"]], video_id=selected["video_id"])
        pair, proxy = proposal_pair(base, selected, dummy, hints["packet"])
        preflight[selected["key"]] = {
            "h0": fingerprint(gemini_h0_wire(base)),
            "phase": {s: fingerprint(phase_wire(s, selected)) for s in SEATS},
            "dummy_proposal_token_proxy": proxy,
            "dummy_proposal_fingerprints": {a: fingerprint(b) for a, b in pair.items()}}
    metadata = metadata_preflight()
    save(output / "selection_manifest.json", manifest)
    shutil.copyfile(history_path, output / "historical_inventory.json")
    shutil.copyfile(selection_override, output / "selection_override.py")
    shutil.copyfile(selection_source, output / "selection_source.py")
    shutil.copyfile(spacing_evidence, output / "spacing_feasibility.json")
    save(output / "preflight.json", preflight)
    save(output / "model_metadata.json", metadata)
    for video, prior in priors.items():
        save(output / "priors" / f"{video}.json", prior)
    inputs = [output / n for n in ("selection_manifest.json", "historical_inventory.json", "selection_override.py",
                                   "selection_source.py", "spacing_feasibility.json", "preflight.json", "model_metadata.json")]
    inputs.extend((output / "priors").glob("*.json"))
    paths = runtime_paths()
    plan = {"profile": PROFILE, "template_version": VERSION, "default_version": DEFAULT_VERSION,
            "verifier_prompt_profile": PROMPT_PROFILE, "created_utc": now(), "selection": selection,
            "models": MODELS, "proposer": PROPOSER, "providers": PROVIDERS, "limits": LIMITS,
            "rates": RATES_V2, "arms": ARMS, "target_count": TARGET_COUNT, "max_calls": MAX_CALLS,
            "selection_gap_frames": SELECTION_GAP_FRAMES,
            "target_workers": TARGET_WORKERS, "success_rule": SUCCESS_RULE, "protocol": PROTOCOL,
            "automatic_retries": 0, "query_gt_policy": "mask/time selection only; GT scoring after closure and replay",
            "selection_source": str(selection_path.resolve()), "selection_source_sha256": sha(selection_path),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "input_sha256": {p.relative_to(output).as_posix(): sha(p) for p in inputs}}
    save(output / "plan.json", plan)
    for path in paths:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    verify_plan(output)
    print(json.dumps({"prepared": PROFILE, "targets": TARGET_COUNT, "max_calls": MAX_CALLS,
                      "limits": LIMITS, "paid_calls": 0, "plan_sha256": sha(output / "plan.json")}), flush=True)


def execute(output, adapter):
    plan = verify_plan(output)
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    view, calls = InferenceOnlyAdapter(adapter), BoundCalls(output, plan)
    calls.persist()
    preflight = read(output / "preflight.json")
    rows = [{**{k: s[k] for k in ("key", "video_id", "frame_id")}, "h0": None,
             **{a: empty_prediction() for a in ARMS}, "statuses": dict.fromkeys(ARMS, "NOT_ATTEMPTED")}
            for s in plan["selection"]]
    output_lock = RLock()
    begin, fatal = perf_counter(), None

    def one(index, selected):
        base = build_gemini_base(view, selected)
        same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "input metadata")
        same(fingerprint(gemini_h0_wire(base)), preflight[selected["key"]]["h0"], "H0 preflight")
        same({s: fingerprint(phase_wire(s, selected)) for s in SEATS},
             preflight[selected["key"]]["phase"], "Phase preflight")
        prior = read(output / "priors" / f"{selected['video_id']}.json")
        row, records, _shared = run_target(calls, base, selected, prior,
            persist=lambda name, value: save(output / "targets" / selected["key"] / f"{name}.json", value))
        with output_lock:
            rows[index] = row
            save(output / "predictions.json", {"targets": rows})
        print(json.dumps({"target": row["key"], "statuses": row["statuses"],
                          "pool_sizes": {a: len(r["pool"]["propositions"]) for a, r in records.items()},
                          "shared_panel": any("shared_from" in r for r in records.values()),
                          "post_calls": len(calls.rows)}), flush=True)

    try:
        with ThreadPoolExecutor(max_workers=TARGET_WORKERS) as workers:
            futures = [workers.submit(one, i, s) for i, s in enumerate(plan["selection"])]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    with calls.lock:
                        calls.stopped = True
                    for pending in futures:
                        pending.cancel()
                    raise
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
                "elapsed_seconds": perf_counter() - begin, "post_calls": len(calls.rows),
                "transport_statuses": dict(Counter(c["status"] for c in calls.rows)),
                "arm_statuses": {a: dict(Counter(r["statuses"][a] for r in rows)) for a in ARMS},
                "query_gt_not_read_during_inference": True,
                "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output, adapter):
    plan = verify_plan(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if not done.get("closed_utc") or done["fatal_error"] or ledger["stopped"] is not True:
        raise ValueError("closed nonfatal inference required")
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(output / name), value, "closed artifact " + name)
    files = {p.relative_to(output).as_posix() for folder in ("calls", "targets", "request_intents")
             for p in (output / folder).rglob("*.json")}
    same(sorted(files | {"plan.json", "budget.json", "predictions.json", "execution.lock"}),
         sorted(done["inference_artifact_sha256"]), "unfrozen or missing artifacts")
    call_rows = ledger["calls"]
    same(len(call_rows), done["post_calls"], "call count")
    if (len(call_rows) > MAX_CALLS or any(c["status"] == "DISPATCHED" for c in call_rows)
            or len({(c["target"], c["stage"], c["seat"]) for c in call_rows}) != len(call_rows)):
        raise ValueError("invalid/outstanding/duplicate call")
    known = {s["key"] for s in plan["selection"]}
    for c in call_rows:
        if c["target"] not in known or (c["stage"], c["seat"]) not in ALLOWED_CALLS:
            raise ValueError("undeclared call")
        same(c["model"], PROPOSER if c["seat"] == "base" else MODELS[c["seat"]], "call model")
        folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        intent = read(output / "request_intents" / f"{c['target']}_{c['stage']}_{c['seat']}.json")
        same(intent["request"], read(folder / "request.json"), "pre-dispatch request")
    for account, occupied in ledger["occupied"].items():
        charges = [Decimal(c["charge"]) for c in call_rows if c["account"] == account]
        if any(not v.is_finite() or v < 0 for v in charges):
            raise ValueError("invalid charge")
        if sum(charges) != Decimal(occupied) or Decimal(occupied) > Decimal(LIMITS[account]):
            raise ValueError("ledger arithmetic or cap")
    rows = read(output / "predictions.json")["targets"]
    same([(r["key"], r["video_id"], r["frame_id"]) for r in rows],
         [(s["key"], s["video_id"], s["frame_id"]) for s in plan["selection"]], "planned target identities")
    view, records = InferenceOnlyAdapter(adapter), {}
    for selected, row in zip(plan["selection"], rows, strict=True):
        base = build_gemini_base(view, selected)
        prior = read(output / "priors" / f"{selected['video_id']}.json")
        target_calls = [c for c in call_rows if c["target"] == selected["key"]]
        replay = ReplayCalls(output, target_calls)
        rebuilt_row, rebuilt_records, rebuilt_shared = run_target(replay, base, selected, prior)
        same(rebuilt_row, row, "raw-response final replay")
        same(without_timing(rebuilt_shared), without_timing(read(output / "targets" / row["key"] / "shared.json")),
             "shared H0/Phase replay")
        records[row["key"]] = {}
        for arm, rebuilt in rebuilt_records.items():
            stored = read(output / "targets" / row["key"] / f"{arm}.json")
            same(without_timing(rebuilt), without_timing(stored), "graph replay")
            records[row["key"]][arm] = stored
        same(len(replay.rows), len(target_calls), "all responses consumed")
    return plan, rows, records, ledger, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "audit", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--priors", type=Path, default=ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1/priors")
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        if args.selection is None:
            raise ValueError("selection manifest required")
        prepare(args.output, args.selection, args.priors, adapter)
    elif args.command == "execute":
        execute(args.output, adapter)
    elif args.command == "audit":
        _, rows, _, ledger, _ = audit(args.output, adapter)
        print(json.dumps({"replay_verified": True, "targets": len(rows), "post_calls": len(ledger["calls"])}))
    else:
        from scripts.score_action_cue_candidate_trial import score
        score(args.output, adapter)


if __name__ == "__main__":
    main()
