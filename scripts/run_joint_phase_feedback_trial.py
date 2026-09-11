"""Isolated, frozen current-roster joint Phase review and second-round repair.

No defaults are changed. Cached upstream inputs are shared by all arms. The
first repair call recommends Phase only; four-head proposal remains unchanged.
The second repair may expand hypotheses, but only the panel selects outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import check_candidate_panel_providers as credential_loader
from scripts import run_glm_parallel_repair as roster
from scripts import run_split_review_trial as common
from scripts.run_candidate_panel_trial import sha
from scripts.run_phase_extension_trial import PHASE_GUIDE, phase_schema
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.five_head_repair import (
    aggregate_five_heads,
    compile_joint,
    joint_schema,
    normalize_five_heads,
)
from surgical_agent.research.verification.phase_extension import (
    phase_apply,
    phase_choice_error,
)
from surgical_agent.research.verification.targeted_joint_repair import (
    apply_targeted_round,
)

PROFILE = "joint_phase_feedback_glm_v1"
SOURCE = ROOT / "artifacts/preflight/expanded_split_review_32_20260910_v1"
PROMPTS = ROOT / "src/surgical_agent/research/verification/prompts"
TASKS = ("instrument", "verb", "target", "ivt", "phase")
ARMS = ("h0", "control", "joint_r1", "joint_r2", "joint_r2_targeted")
LIMITS = {"openrouter_usd": "3", "aliyun_cny": "2", "xai_usd": "0"}
RULE = ("Current GLM/Qwen/GPT/Gemini/DeepSeek roster fixed. Shared cached H0, graph proposal and images. "
        "A=current compact four-head mean plus blind Phase majority; B=Phase repair recommendation, all-seven "
        "Phase and same four-head pool reviewed jointly; C=continue B once using specific fallible feedback. "
        "First repair does not modify the four-head candidate pool. Round2 can expand the pool via repair. "
        "All seven Phase scores need five valid judgments before Phase selection. Mean4 add/mean2 remove and "
        "IVT component protection unchanged. Five fresh judgments per candidate per round, no vote mixing. "
        "No new models/images/H0 prompt/graph fitting, retries, or automatic promotion. "
        "Benefit requires mean F1 above both control and H0, mean precision/Verb/IVT no worse than control, "
        "Phase no worse than H0 and control, fewer label errors than both. Report all task regressions. "
        "Development selection fixed by metadata positions, not scores. Confirmation required after selection. "
        "D=zero-call shadow of C admitting only changes also proposed by Repair, with IVT component protection; "
        "derived from development unproposed-edit drift and frozen before confirmation GT scoring.")


def four_pool(pool):
    return {"propositions": [deepcopy(p) for p in pool["propositions"] if p["task"] != "phase"]}


def joint_pool(pool):
    result = four_pool(pool)
    result["propositions"] += [{"id": f"phase_{p['id']}", "task": "phase", "label_id": p["id"],
        "name": p["name"], "components": None} for p in PHASE_GUIDE]
    return result


def packet_of(body):
    return json.loads(body["messages"][0]["content"][0]["text"])


def put_packet(body, packet):
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def phase_proposal(base, selected, initial):
    body = gemini_proposal(base, selected, initial["h0"], initial["pool"], [])
    packet = packet_of(body)
    packet.update(task="Propose the current workflow Phase for independent review; do not edit any interaction labels.",
        instructions=("Compare the current Phase hypothesis with all seven definitions using the current operative "
            "location and ongoing activity. The old prediction and relation hints may be wrong. Use history only "
            "as context; do not assume stage order. Recommend one Phase or null if indistinguishable. Cite current "
            "image 2 for a non-null answer. Return only phase_id, image_indices, observation as JSON data."),
        phase_definitions=PHASE_GUIDE, response_schema=phase_schema(3),
        candidate_relation_hints=initial["hints"]["packet"])
    body["response_format"] = {"type": "json_object"}
    return put_packet(body, packet)


def joint_wire(seat, base, selected, pool, current, recommendation, variant):
    body = roster.review_wire(seat, base, selected, pool)
    packet = packet_of(body)
    packet["task"] = "Independently rate all four interaction heads and all seven Phase alternatives."
    packet["instructions"] += "\n\n" + (PROMPTS / f"joint_phase_feedback_{variant}.txt").read_text(encoding="utf-8").strip()
    packet["phase_definitions"] = deepcopy(PHASE_GUIDE)
    packet["current_prediction_hypothesis"] = deepcopy(current)
    packet["phase_recommendation_hypothesis"] = recommendation
    return put_packet(body, packet)


def select_joint(current, pool, means):
    sub = four_pool(pool)
    out = common.panel.select(current, sub, {p["id"]: means[p["id"]] for p in sub["propositions"]}, threshold=4)
    phase_means = {f"phase_{i}": means[f"phase_{i}"] for i in range(7)}
    complete = all(v is not None for v in phase_means.values())
    if not complete:
        phase_means = {k: None for k in phase_means}
    out, decision = phase_apply(out, phase_means)
    decision["valid_panel"] = complete
    return out, decision


def feedback_for(result):
    """No GT, scores or reviewer identities in repair feedback; bounded by pool."""
    pool, means, final = result["pool"], result["means"], result["prediction"]
    issues = common.panel.unresolved(final, four_pool(pool),
        {p["id"]: means[p["id"]] for p in four_pool(pool)["propositions"]},
        result["diagnostics"], threshold=4)
    ids = {i["candidate_id"] for i in issues}
    phase_ids = [p["id"] for p in pool["propositions"] if p["task"] == "phase"]
    phase_values = [means[p] for p in phase_ids]
    phase_problem = (not result["phase_decision"]["valid_panel"]
        or means[f"phase_{final['phase'][0]}"] < 4
        or sum(v is not None and v >= 3.5 for v in phase_values) > 1)
    if phase_problem:
        ids.update(phase_ids)
    rows = []
    for p in pool["propositions"]:
        if p["id"] not in ids:
            continue
        observations = []
        for seat in SEATS:
            item = result["reviews"][seat]["judgments"].get(p["id"])
            if item:
                observation = {k: item[k] for k in ("finding", "scope", "image_indices", "observation")}
                if observation not in observations:
                    observations.append(observation)
        rows.append({"candidate": p, "currently_selected": p["label_id"] in final[p["task"]],
            "reason": "Unresolved visual support or competing workflow hypotheses.", "observations": observations})
    return rows


def repair_wire(base, selected, current, pool, feedback, hints, variant):
    body = gemini_proposal(base, selected, current, four_pool(pool), [])
    packet = packet_of(body)
    packet.update(task="Propose evidence-based repairs for unresolved five-head hypotheses.",
        instructions=(PROMPTS / f"joint_phase_repair_{variant}.txt").read_text(encoding="utf-8").strip(),
        response_schema=joint_schema(), review_feedback=feedback, phase_definitions=PHASE_GUIDE,
        candidate_relation_hints=hints["packet"])
    if variant in ("v2", "v3", "v4"):
        packet["task"] = "Independently reconstruct the current five-head scene to propose repairs."
        packet["tasks_to_reinspect"] = sorted({x["candidate"]["task"] for x in feedback})
        for key in ("current_prediction", "candidate_pool", "issues", "review_feedback", "candidate_relation_hints"):
            packet.pop(key, None)
    body["response_format"] = {"type": "json_object"}
    return put_packet(body, packet)


def panel_call(calls, key, stage, bodies):
    with ThreadPoolExecutor(max_workers=5) as executor:
        return dict(zip(SEATS, executor.map(lambda s: calls.call(key, stage, s, bodies[s]), SEATS), strict=True))


def review_joint(calls, base, selected, current, pool, recommendation, stage, variant):
    bodies = {s: joint_wire(s, base, selected, pool, current, recommendation, variant) for s in SEATS}
    if variant == "v3" and stage == "joint_r2":
        for body in bodies.values():
            packet = packet_of(body)
            packet.pop("current_prediction_hypothesis")
            packet.pop("phase_recommendation_hypothesis")
            put_packet(body, packet)
    raw = panel_call(calls, selected["key"], stage,
        bodies)
    reviews, formats = normalize_five_heads(raw, pool, image_count=3)
    means, diagnostics = aggregate_five_heads(reviews, pool, image_count=3)
    prediction, decision = select_joint(current, pool, means)
    return {"raw": raw, "pool": pool, "reviews": reviews, "format_diagnostics": formats,
        "means": means, "diagnostics": diagnostics, "prediction": prediction, "phase_decision": decision}


def run_case(calls, base, selected, initial, variant, index):
    h0, pool = initial["h0"], initial["pool"]
    record = {"h0": h0, "initial_pool": pool, "predictions": {"h0": h0}}

    def control():
        with ThreadPoolExecutor(max_workers=2) as executor:
            g = executor.submit(panel_call, calls, selected["key"], "control_graph",
                {s: roster.review_wire(s, base, selected, pool) for s in SEATS})
            p = executor.submit(panel_call, calls, selected["key"], "control_phase",
                {s: roster.phase_wire(s, selected) for s in SEATS})
            raw, phase = g.result(), p.result()
        reviews, formats = common.normalize_five(raw, pool, 3)
        means, diagnostics = common.panel.aggregate(reviews, pool, image_count=3)
        four = common.panel.select(h0, pool, means, threshold=4)
        final, decision = common.apply_phase_choices(four, phase, 3)
        record["control"] = {"raw": raw, "phase_raw": phase, "means": means, "reviews": reviews,
            "format_diagnostics": formats, "diagnostics": diagnostics, "phase_decision": decision}
        record["predictions"]["control"] = final

    def joint():
        raw = calls.call(selected["key"], "phase_recommendation", "base", phase_proposal(base, selected, initial))
        error = phase_choice_error(raw, 3)
        recommendation = None if error else raw["phase_id"]
        record["phase_proposal"] = {"raw": raw, "error": error}
        first = review_joint(calls, base, selected, h0, joint_pool(pool), recommendation, "joint_r1", variant)
        record["joint_r1"] = first
        record["predictions"]["joint_r1"] = first["prediction"]
        record["predictions"]["joint_r2"] = deepcopy(first["prediction"])
        record["predictions"]["joint_r2_targeted"] = deepcopy(first["prediction"])
        feedback = feedback_for(first)
        record["feedback"] = feedback
        if not feedback or calls.stopped:
            record["stop_reason"] = "NO_ISSUES" if not feedback else "CIRCUIT_STOP"
            return
        raw = calls.call(selected["key"], "repair_r2", "base",
            repair_wire(base, selected, first["prediction"], first["pool"], feedback, initial["hints"], variant))
        record["repair_r2"] = raw
        try:
            compiled = compile_joint(first["prediction"], four_pool(first["pool"]), raw)
        except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
            record["stop_reason"] = "REPAIR_INVALID:" + type(exc).__name__
            return
        # Re-evaluate a concrete repair hypothesis even if IDs are unchanged:
        # existing uncertainty and competing Phase hypotheses can still resolve.
        second = review_joint(calls, base, selected, first["prediction"], compiled["pool"],
            compiled["raw_prediction"]["phase"][0], "joint_r2", variant)
        record["joint_r2"] = second
        record["compiled_repair"] = compiled
        record["predictions"]["joint_r2"] = second["prediction"]
        targeted, rejected = apply_targeted_round(first["prediction"], compiled["paper_style"], second["prediction"])
        record["predictions"]["joint_r2_targeted"] = targeted
        record["targeted_rejections"] = rejected
        record["stop_reason"] = "STATE_REPEATED" if second["prediction"] == first["prediction"] else "ROUND_CAP_2"

    for action in ((control, joint) if index % 2 == 0 else (joint, control)):
        action()
    return record


def prepare(output, source, variant, full=False):
    if output.exists():
        raise ValueError("new output required")
    plan = read(source / "plan.json")
    done = read(source / "completion.json")
    if done.get("fatal_error"):
        raise ValueError("nonfatal upstream required")
    selection = [s for i, s in enumerate(plan["selection"]) if full or i % 8 in (2, 5)]
    if not full and len(selection) != 8:
        raise ValueError("dev protocol requires eight metadata-selected targets")
    initials = {}
    for s in selection:
        rel = f"targets/{s['key']}/result.json"
        if sha(source / rel) != done["hashes"][rel]:
            raise ValueError("upstream result seal mismatch")
        data = read(source / rel)
        initials[s["key"]] = {k: deepcopy(data[k]) for k in ("h0", "pool", "hints")}
        if data["hints"].get("excluded_video", s["video_id"]) != s["video_id"]:
            raise ValueError("wrong excluded video")
    save(output / "initials.json", initials)
    paths = sorted({*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "configs/perception/joint_openrouter_h0.yaml",
        ROOT / "DEFAULT_PIPELINE_VERSION.json", ROOT / "tests/unit/test_joint_phase_feedback.py"})
    paths = [p for p in paths if p.is_file()]
    for p in paths:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    manifest = {"profile": PROFILE, "variant": variant, "created_utc": now(), "rule": RULE,
        "selection": selection, "models": roster.MODELS, "families": roster.FAMILIES,
        "transports": roster.TRANSPORTS, "limits": LIMITS, "max_calls": len(selection) * 22,
        "source_archive": str(source.resolve()), "source_plan_sha256": sha(source / "plan.json"),
        "credential_root": str(ROOT.resolve()),
        "initials_sha256": sha(output / "initials.json"),
        "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "cohort": "previously examined Training development" if not full else "fixed confirmation source; see source protocol"}
    save(output / "plan.json", manifest)
    print(json.dumps({"prepared": True, "keys": [s["key"] for s in selection], "max_calls": manifest["max_calls"]}), flush=True)


def verify(output):
    p = read(output / "plan.json")
    assert p["profile"] == PROFILE and p["rule"] == RULE and p["models"] == roster.MODELS
    assert p["limits"] == LIMITS and p["initials_sha256"] == sha(output / "initials.json")
    for rel, digest in p["sources"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("frozen source changed: " + rel)
    for s in p["selection"]:
        for im in s["images"]:
            if sha(im["path"]) != im["sha256"]:
                raise ValueError("frozen image changed")
    return p


class BoundCalls(roster.GLMCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={a: Decimal(v) for a, v in LIMITS.items()}, rates=roster.RATES,
            providers=roster.PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"))
        self.attempted = set()
        self.targets = {s["key"] for s in plan["selection"]}

    def persist(self):
        for i in range(10):
            try:
                return super().persist()
            except PermissionError:
                if i == 9:
                    raise
                time.sleep(.05)

    def call(self, target, stage, seat, body):
        allowed = {(a, s) for a in ("control_graph", "control_phase", "joint_r1", "joint_r2") for s in SEATS}
        allowed |= {("phase_recommendation", "base"), ("repair_r2", "base")}
        with self.lock:
            ident = target, stage, seat
            if target not in self.targets or (stage, seat) not in allowed or ident in self.attempted:
                raise ValueError("undeclared or duplicate call")
            expected = PROPOSER if seat == "base" else roster.MODELS[seat]
            if body["model"] != expected:
                raise ValueError("wrong model")
            self.attempted.add(ident)
        result = super().call(target, stage, seat, body)
        with self.lock:
            records = [r for r in self.rows if (r["target"], r["stage"], r["seat"]) == ident]
            if records and records[0].get("exception_type") == "ProxyError":
                self.stopped = True
                self.persist()
        return result


def bases_for(adapter, plan):
    return {s["key"]: common.build_gemini_base(common.InferenceOnlyAdapter(adapter), s) for s in plan["selection"]}


@contextmanager
def credential_context(plan):
    previous = credential_loader.ROOT
    credential_loader.ROOT = Path(plan["credential_root"])
    try:
        # Read-only validation; secrets stay in their assigned original files.
        for seat in ("gpt", "qwen"):
            credential_loader.key_for(seat)
        yield
    finally:
        credential_loader.ROOT = previous


def execute(output, adapter):
    plan = verify(output)
    initials, bases = read(output / "initials.json"), bases_for(adapter, plan)
    with (output / "execution.lock").open("x") as f:
        f.write(sha(output / "plan.json"))
    start, fatal, rows = perf_counter(), None, []
    with credential_context(plan), roster.lightweight_protocol():
        calls = BoundCalls(output, plan)
        try:
            for index, s in enumerate(plan["selection"]):
                if calls.stopped:
                    break
                stamp = perf_counter()
                result = run_case(calls, bases[s["key"]], s, initials[s["key"]], plan["variant"], index)
                result["seconds"] = perf_counter() - stamp
                save(output / "targets" / s["key"] / "result.json", result)
                rows.append({**{k: s[k] for k in ("key", "video_id", "frame_id")}, **result["predictions"]})
                save(output / "predictions.json", rows)
                print(json.dumps({"done": len(rows), "target": s["key"], "calls": len(calls.rows),
                    "stop": result["stop_reason"], "occupied": {k: str(v) for k, v in calls.occupied.items()}}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            save(output / "predictions.json", rows)
            files = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal, "calls": len(calls.rows),
                "elapsed_seconds": perf_counter() - start, "hashes": {p.relative_to(output).as_posix(): sha(p) for p in files}})


def score(output, adapter):
    plan, done = verify(output), read(output / "completion.json")
    if done["fatal_error"]:
        raise ValueError("engineering-failed run; retain logs, do not claim efficacy")
    for rel, digest in done["hashes"].items():
        if sha(output / rel) != digest:
            raise ValueError("seal mismatch: " + rel)
    ledger, rows = read(output / "budget.json"), read(output / "predictions.json")
    if not ledger["stopped"] or len(rows) != len(plan["selection"]):
        raise ValueError("all targets and closed ledger required")
    bases, initials = bases_for(adapter, plan), read(output / "initials.json")
    replay = common.ReplayCalls(output, ledger["calls"])
    with roster.lightweight_protocol():
        for index, (s, row) in enumerate(zip(plan["selection"], rows, strict=True)):
            result = run_case(replay, bases[s["key"]], s, initials[s["key"]], plan["variant"], index)
            stored = read(output / "targets" / s["key"] / "result.json")
            stored.pop("seconds")
            if result != stored or any(row[a] != result["predictions"][a] for a in ARMS):
                raise ValueError("raw-response replay differs")
    if len(replay.rows) != len(ledger["calls"]):
        raise ValueError("replay call count differs")
    _, truths = common.score_saved(adapter, [{**r, "h1": None, "final": r["joint_r2"]} for r in rows])
    truth = {(r["video_id"], r["frame_id"]): r for r in truths}
    metrics = {a: common.compute_repair_comparison([{**truth[r['video_id'], r['frame_id']],
        "h0": r["h0"], "h1": None, "final": r[a]} for r in rows])["arms"]["final"] for a in ARMS}
    repaired = ("joint_r1", "joint_r2", "joint_r2_targeted")
    pairs = [(a, b) for a in ("h0", "control") for b in repaired] + [("joint_r1", b) for b in repaired[1:]]
    deltas = {f"{a}_to_{b}": [{"key": r["key"], **common.frame_delta(r[a], r[b], truth[r['video_id'], r['frame_id']]["gt"],
        truth[r['video_id'], r['frame_id']]["mask"])} for r in rows] for a, b in pairs}
    checks = {}
    for a in repaired:
        t, c, h = (metrics[k]["tasks"] for k in (a, "control", "h0"))
        checks[a] = {"mean_f1_above_control_and_h0": sum(x["micro_f1"] for x in t.values()) > max(sum(x["micro_f1"] for x in z.values()) for z in (c, h)),
            "mean_precision_not_lower": sum(x["micro_precision"] for x in t.values()) >= sum(x["micro_precision"] for x in c.values()),
            "verb_ivt_not_lower": all(t[k]["micro_f1"] >= c[k]["micro_f1"] for k in ("verb", "ivt")),
            "phase_not_lower": t["phase"]["micro_f1"] >= max(c["phase"]["micro_f1"], h["phase"]["micro_f1"]),
            "errors_reduced": all(common.summarize_deltas(deltas[f"{b}_to_{a}"])["net_errors_removed"] > 0 for b in ("h0", "control"))}
    costs = {account: {kind: str(sum(Decimal(c["charge"]) for c in ledger["calls"] if c["account"] == account and c["charge_kind"] == kind))
        for kind in ("native", "conservative_estimate", "unknown_reserved")} for account in LIMITS}
    report = {"metrics": metrics, "checks": checks, "success": {a: all(c.values()) for a, c in checks.items()},
        "changes": {k: common.summarize_deltas(v) for k, v in deltas.items()}, "costs": costs,
        "calls": len(ledger["calls"]), "statuses": dict(Counter(c["status"] for c in ledger["calls"])),
        "raw_replayed": True, "elapsed_seconds": done["elapsed_seconds"]}
    save(output / "scored_truth.json", truths)
    save(output / "frame_deltas.json", deltas)
    save(output / "metrics.json", report)
    print(json.dumps({"f1": {a: {t: round(h["micro_f1"] * 100, 3) for t, h in m["tasks"].items()} for a, m in metrics.items()},
        "checks": checks, "costs": costs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.source, args.variant, args.full)
    else:
        adapter = common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
        globals()[args.command](args.output, adapter)
