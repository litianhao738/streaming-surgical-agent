"""Original delivered blind H0 method, reproduced over the available OR route.

The delivered wire function/appendix is unchanged. Only transport is mapped to
OpenRouter; no H0 answers, priors, proposals, panel or repair acceptance calls.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests
from PIL import Image

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_disputed_relation_trial import verify_plan as verify_source
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_new_training_verb_guard_trial import InferenceOnlyAdapter
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import RATES_V2
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_disputed_relation_trial import exact_same as same
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.final_only import final_only_schema, validate_final_only
from surgical_agent.perception.main_h0 import load_main_h0_prompt, main_h0_input

REFERENCE = ROOT / "artifacts/external_reviews/delivery_blind_contact_audit_20260909_v1/run_gemini_blind_audit_ablation.py"
SOURCE = ROOT / "artifacts/preflight/disputed_relation_20260909_v1"
PROFILE = "original_blind_contact_h0_openrouter_transport_v1"
ARMS = ("control_h0", "blind_contact_audit")
VERSIONS = (*ARMS, "phase_frozen")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
MAX_CALLS = 48
LIMITS = {"openrouter_usd": "2"}
SUCCESS_RULE = (
    "On all 24 planned Training development targets, phase-frozen audit must improve IVT micro-F1 "
    "over freshly generated control H0, with no decrease in Verb Precision/F1 or Target Precision/F1 "
    "and no increase in total label errors. Report raw five-head audit too. Both arms must complete "
    "with valid predictions. No prompt/threshold changes after scoring. Not independent validation."
)


def reference_functions():
    """Load only the three fully inspected, pure wire-building source nodes."""
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {"jpeg_b64", "wire"})
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AUDIT_APPENDIX" for t in n.targets))]
    if len(nodes) != 3:
        raise ValueError("unexpected delivered reference functions")
    namespace = {"Any": Any, "Path": Path, "Image": Image, "io": io, "base64": base64, "json": json,
                 "load_main_h0_prompt": load_main_h0_prompt, "main_h0_input": main_h0_input,
                 "final_only_schema": final_only_schema}
    # Only inspected wire/JPEG functions and the constant; never the archive's entrypoint or API code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REFERENCE), "exec"), namespace)  # noqa: S102
    return namespace


def convert_wire(native):
    """Lossless request-content mapping, with explicit transport differences."""
    parts = native["contents"][0]["parts"]
    config = native["generationConfig"]
    content = []
    for p in parts:
        if "text" in p:
            content.append({"type": "text", "text": p["text"]})
        else:
            data = p["inlineData"]
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{data['mimeType']};base64," + data["data"]}})
    return {"model": PROPOSER, "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False,
                "require_parameters": True}, "temperature": config["temperature"],
        "max_tokens": config["maxOutputTokens"], "stream": False,
        "messages": [{"role": "system", "content": native["systemInstruction"]["parts"][0]["text"]},
                     {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "original_blind_h0_v1",
            "strict": True, "schema": deepcopy(config["responseJsonSchema"])}}}


def wire(row, arm):
    if arm not in ARMS:
        raise ValueError("unknown original experiment arm")
    return convert_wire(reference_functions()["wire"](row, arm=arm))


def decode(raw):
    if raw is None:
        return {"status": "API_UNAVAILABLE", "labels": None}
    try:
        validate_final_only(raw)
    except (ApiSchemaError, TypeError, ValueError):
        return {"status": "SCHEMA_FAILURE", "labels": None}
    return {"status": "OK", "labels": {
        t: [raw[t]["selected_id"]] if t == "phase" else list(raw[t]["selected_ids"]) for t in TASKS}}


def freeze_phase(control, audit):
    if control is None or audit is None:
        return None  # Failed required source is never silently a successful empty prediction.
    final = deepcopy(audit)
    final["phase"] = list(control["phase"])
    return final


def verify(output):
    plan = read(output / "plan.json")
    for k, v in {"profile": PROFILE, "model": PROPOSER, "arms": list(ARMS), "max_calls": MAX_CALLS,
                 "limits": LIMITS, "success_rule": SUCCESS_RULE, "automatic_retries": 0}.items():
        same(plan[k], v, k)
    for n, h in plan["source_sha256"].items():
        same(sha(ROOT / n), h, "frozen source")
    for n, h in plan["input_sha256"].items():
        same(sha(output / n), h, "frozen input")
    for s in plan["selection"]:
        for im in s["images"]:
            same(sha(im["path"]), im["sha256"], "source image")
    return plan


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new directory required")
    old = verify_source(SOURCE)
    done = read(SOURCE / "completion.json")
    if done["fatal_error"]:
        raise ValueError("source cohort must be closed")
    same(sha(SOURCE / "plan.json"), done["inference_artifact_sha256"]["plan.json"], "source selection")
    selected = deepcopy(old["selection"])
    same(len(selected), 24, "24 fixed targets")
    view = InferenceOnlyAdapter(adapter)
    preflight = {}
    for row in selected:
        if view.entries[row["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        sample = next(s for s in view.iter_inference_video(row["video_id"]) if s.target_frame_id == row["frame_id"])
        same(list(sample.causal_frame_ids), row["causal_frame_ids"], "canonical causal identities")
        if [Path(x).resolve() for x in sample.media_refs] != [Path(x["path"]).resolve() for x in row["images"]]:
            raise ValueError("canonical media paths differ")
        bodies = {a: wire(row, a) for a in ARMS}
        clean = deepcopy(bodies[ARMS[1]])
        same(clean["messages"][0]["content"], bodies[ARMS[0]]["messages"][0]["content"] + "\n\n" +
             reference_functions()["AUDIT_APPENDIX"], "verbatim delivered appendix")
        clean["messages"][0] = deepcopy(bodies[ARMS[0]]["messages"][0])
        same(clean, bodies[ARMS[0]], "only system appendix differs")
        preflight[row["key"]] = {a: {"fingerprint": fingerprint(b), "request": redact_images(b)} for a, b in bodies.items()}
    url = f"https://openrouter.ai/api/v1/models/{PROPOSER}/endpoints"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    endpoint = next(e for e in metadata["data"]["endpoints"] if e["tag"] == "google-ai-studio")
    if endpoint["status"] != 0 or not {"response_format", "temperature"} <= set(endpoint["supported_parameters"]):
        raise ValueError("fixed route unavailable")
    if any(Decimal(endpoint["pricing"][k]) > Decimal(v) for k, v in
           zip(("prompt", "completion"), RATES_V2["base"], strict=True)):
        raise ValueError("rate above reserve")
    save(output / "preflight.json", preflight)
    save(output / "model_metadata.json", {"queried_utc": now(), "url": url, "response": metadata, "paid_calls": 0})
    save(output / "selection.json", selected)
    (output / "original_audit_appendix.txt").write_text(reference_functions()["AUDIT_APPENDIX"], encoding="utf-8")
    dependencies = {ROOT / n for n in old["source_sha256"]} | {
        Path(__file__).resolve(), REFERENCE, ROOT / "tests/unit/test_original_blind_audit.py",
        ROOT / "scripts/score_disputed_relation_trial.py"}
    inputs = [output / n for n in ("preflight.json", "model_metadata.json", "selection.json", "original_audit_appendix.txt")]
    plan = {"profile": PROFILE, "created_utc": now(), "selection": selected, "model": PROPOSER,
        "arms": list(ARMS), "max_calls": MAX_CALLS, "limits": LIMITS, "rates": {"base": RATES_V2["base"]},
        "success_rule": SUCCESS_RULE, "automatic_retries": 0,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(dependencies)},
        "input_sha256": {p.relative_to(output).as_posix(): sha(p) for p in inputs},
        "source_cohort": str(SOURCE), "source_plan_sha256": sha(SOURCE / "plan.json"),
        "protocol": "Original delivered wire/appendix verbatim; fresh independent five-head control and audit; control then audit per target, one-second pause after calls; offline audit-four-heads plus control-Phase. No H0 answer, graph, prior, candidate pool, reviewers, extra reasoning/feedback instructions or cached predictions in either request.",
        "transport_difference": "Google official key unavailable; map original systemInstruction/contents/generationConfig to OpenRouter chat/structured output with fixed Google AI Studio route. Preserve original JPEG95 images as three parts. No per-image detail or reasoning override, as neither was explicit in delivered synchronous code. Provider default thinking can differ across APIs; not byte-identical native API replication or same-pixel protocol as the existing pipeline.",
        "gt_policy": "Same previously inspected 24 Training targets; no GT access before closed inference. Fresh controls, no reuse of cached labels. Not an independent holdout and not a reproduction of the colleague's unidentified eight VID31 frames.",
        "stop_http": [400, 401, 402, 403, 404, 429, 503]}
    save(output / "plan.json", plan)
    for p in dependencies:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    verify(output)
    print(json.dumps({"prepared": PROFILE, "targets": 24, "max_calls": MAX_CALLS,
                      "limits": LIMITS, "plan_sha256": sha(output / "plan.json")}), flush=True)


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=plan["rates"],
                         providers={"base": "Google AI Studio"}, max_calls=MAX_CALLS)
        self.known = {s["key"] for s in plan["selection"]}
        self.expected = read(output / "preflight.json")
        self.attempted = set()

    def call(self, target, stage, seat, body):
        if target not in self.known or stage not in ARMS or seat != "base":
            raise ValueError("undeclared call")
        same(fingerprint(body), self.expected[target][stage]["fingerprint"], "frozen request")
        if (target, stage) in self.attempted:
            raise ValueError("retry prohibited")
        self.attempted.add((target, stage))
        raw = super().call(target, stage, seat, body)
        if self.rows and self.rows[-1].get("http_status") in {400, 401, 402, 403, 404, 429, 503}:
            self.stopped = True
            self.persist()
        return raw


def execute(output):
    plan = verify(output)
    with (output / "execution.lock").open("x") as lock:
        lock.write(sha(output / "plan.json"))
    calls = BoundCalls(output, plan)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
             **dict.fromkeys(VERSIONS), "statuses": dict.fromkeys(ARMS, "NOT_ATTEMPTED")} for s in plan["selection"]]
    start, fatal = perf_counter(), None
    try:
        for selected, row in zip(plan["selection"], rows, strict=True):
            for arm in ARMS:
                if calls.stopped:
                    row["statuses"][arm] = "STOPPED_UNATTEMPTED"
                    continue
                result = decode(calls.call(row["key"], arm, "base", wire(selected, arm)))
                save(output / "targets" / row["key"] / f"{arm}.json", result)
                row[arm], row["statuses"][arm] = result["labels"], result["status"]
                row["phase_frozen"] = freeze_phase(row[ARMS[0]], row[ARMS[1]])
                save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "arm": arm, "status": result["status"], "calls": len(calls.rows)}), flush=True)
                sleep(1)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", {"targets": rows})
        try:
            verify(output)
        except Exception:
            fatal = fatal or "FROZEN_INPUT_CHANGED"
            raise
        finally:
            files = [output / n for n in ("plan.json", "budget.json", "predictions.json", "execution.lock")]
            files += [p for folder in ("calls", "targets") for p in (output / folder).rglob("*.json")]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                "elapsed_seconds": perf_counter() - start, "post_calls": len(calls.rows),
                "transport_statuses": dict(Counter(c["status"] for c in calls.rows)),
                "arm_statuses": {a: dict(Counter(r["statuses"][a] for r in rows)) for a in ARMS},
                "no_gt_during_inference": True, "inference_artifact_sha256": {
                    p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output):
    plan = verify(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal experiment required")
    for n, h in done["inference_artifact_sha256"].items():
        same(sha(output / n), h, "closed artifact")
    files = {p.relative_to(output).as_posix() for folder in ("calls", "targets") for p in (output / folder).rglob("*.json")}
    if files | {"plan.json", "budget.json", "predictions.json", "execution.lock"} != set(done["inference_artifact_sha256"]):
        raise ValueError("unfrozen/missing artifacts")
    calls = ledger["calls"]
    if len(calls) != done["post_calls"] or len(calls) > MAX_CALLS or len({(c['target'], c['stage']) for c in calls}) != len(calls):
        raise ValueError("invalid call count")
    for c in calls:
        if c["seat"] != "base" or c["stage"] not in ARMS or c["status"] == "DISPATCHED":
            raise ValueError("invalid call")
        same(c["model"], PROPOSER, "requested model")
        if c["status"] == "JSON_PARSED":
            folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_base"
            response = read(folder / "response.json")
            same(response["http_status"], 200, "successful transport")
            same(response["body"]["model"], PROPOSER, "returned model")
            same(response["body"].get("provider"), "Google AI Studio", "returned provider")
    charges = [Decimal(c["charge"]) for c in calls]
    if any(not c.is_finite() or c < 0 for c in charges):
        raise ValueError("invalid charges")
    total = sum(charges)
    same(total, Decimal(ledger["occupied"]["openrouter_usd"]), "ledger total")
    if total > Decimal(LIMITS["openrouter_usd"]):
        raise ValueError("budget exceeded")
    rows = read(output / "predictions.json")["targets"]
    same([(r["key"], r["video_id"], r["frame_id"]) for r in rows],
         [(s["key"], s["video_id"], s["frame_id"]) for s in plan["selection"]], "all target identities")
    replay = ReplayCalls(output, calls)
    for s, row in zip(plan["selection"], rows, strict=True):
        for a in ARMS:
            path = output / "targets" / row["key"] / f"{a}.json"
            if not path.exists():
                if row["statuses"][a] != "STOPPED_UNATTEMPTED" or row[a] is not None:
                    raise ValueError("missing attempted prediction")
                continue
            rebuilt = decode(replay.call(row["key"], a, "base", wire(s, a)))
            same(rebuilt, read(path), "raw answer replay")
            same(row[a], rebuilt["labels"], "prediction")
            same(row["statuses"][a], rebuilt["status"], "semantic status")
        same(row["phase_frozen"], freeze_phase(row[ARMS[0]], row[ARMS[1]]), "only Phase overwritten")
    same(len(replay.rows), len(calls), "all calls consumed")
    return plan, rows, ledger, done


def score(output, adapter):
    plan, rows, ledger, done = audit(output)
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r[ARMS[0]], "h1": None, "final": r["phase_frozen"]} for r in rows])
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics = {v: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r[ARMS[0]], "h1": None, "final": r[v]} for r in rows])["arms"]["final"] for v in VERSIONS}
    pairs = ((ARMS[0], ARMS[1]), (ARMS[0], "phase_frozen"), (ARMS[1], "phase_frozen"))
    details = {a + "_to_" + b: [{"key": r["key"], **frame_delta(r[a], r[b],
        truths[r["video_id"], r["frame_id"]]["gt"], truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows] for a, b in pairs}
    comparisons = {k: summarize_deltas(v) for k, v in details.items()}
    before, after = metrics[ARMS[0]]["tasks"], metrics["phase_frozen"]["tasks"]
    def compare(t, m, strict=False):
        x, y = before[t][m], after[t][m]
        return x is not None and y is not None and (y > x if strict else y >= x)
    checks = {"all_48_predictions_valid": all(r["statuses"][a] == "OK" for r in rows for a in ARMS),
        "IVT_F1_increases": compare("ivt", "micro_f1", True),
        **{f"{t}_{m}_not_decreased": compare(t, m) for t in ("verb", "target") for m in ("micro_precision", "micro_f1")},
        "total_errors_not_increased": comparisons[ARMS[0] + "_to_phase_frozen"]["net_errors_removed"] >= 0}
    costs = {a: {kind: str(sum(Decimal(c["charge"]) for c in ledger["calls"] if c["stage"] == a and c["charge_kind"] == kind))
                for kind in ("native", "unknown_reserved", "conservative_estimate")} for a in ARMS}
    report = {"profile": PROFILE, "metrics": metrics, "comparisons": comparisons, "costs_by_arm_usd": costs,
        "post_calls": len(ledger["calls"]), "elapsed_seconds": done["elapsed_seconds"],
        "transport_statuses": done["transport_statuses"], "arm_statuses": done["arm_statuses"],
        "predeclared_success": {"rule": SUCCESS_RULE, "checks": checks, "confirmed": all(checks.values())},
        "limitations": [plan["gt_policy"], plan["transport_difference"]],
        "audit": {"raw_answers_replayed": True, "only_Phase_replaced": True, "GT_after_closure": True}}
    save(output / "metrics.json", report)
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", details)
    print(json.dumps({k: report[k] for k in ("post_calls", "costs_by_arm_usd", "predeclared_success")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    if args.command == "execute":
        execute(args.output)
    else:
        adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
        prepare(args.output, adapter) if args.command == "prepare" else score(args.output, adapter)
