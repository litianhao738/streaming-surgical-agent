"""Five lightweight model score means + Qwen local repair, <=3 review rounds.

Separate developmental replay; never mutates H0 or previous experiment outputs.
"""
import argparse
import ast
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_prior_panel_trial as legacy
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.mean_panel import (
    full_universe,
    mean_scores,
    response_schema,
    run_arm,
)

read, save = legacy.read, legacy.save
SOURCE = ROOT / "artifacts/preflight/h0_prior_panel_openrouter_20260907_v7"
CONTRACTS = ROOT / "docs/protocols/h0_five_mean_v1"
ACADEMIC_CONTEXT = (
    "These images are frames from laparoscopic surgical videos used for academic medical video analysis. "
    "The task is research annotation of surgical instruments, actions, and interaction targets. "
    "Visible tissue, blood, and surgical instruments are part of the medical procedure being studied."
)
MODEL_TAGS = [
    ("google/gemini-2.5-flash-lite", "google-ai-studio"),
    ("qwen/qwen3-vl-8b-instruct", "alibaba"),
    ("mistralai/mistral-small-3.2-24b-instruct", "deepinfra/fp8"),
    ("google/gemini-2.5-flash", "google-ai-studio"),
    ("google/gemma-3-12b-it", "deepinfra/bf16"),
    ("qwen/qwen3.8-max-0902", "alibaba"),
]


def pricing_snapshot():
    original = legacy.MODEL_TAGS
    try:
        legacy.MODEL_TAGS = MODEL_TAGS
        return legacy.pricing_snapshot()
    finally:
        legacy.MODEL_TAGS = original


def request_body(base, model, schema, prompt, packet):
    body = legacy.request_body(base, model, schema, prompt, packet)
    if "reasoning" not in model["endpoint"]["supported_parameters"]:
        body.pop("reasoning", None)
    return body


def prepare(args):
    if args.output.exists():
        raise ValueError("fresh output required")
    completed = read(SOURCE / "prediction_completion.json")
    if sha256_file(SOURCE / "predictions.json") != completed["prediction_sha256"]:
        raise ValueError("source H0 changed")
    args.output.mkdir(parents=True)
    old_plan = read(SOURCE / "plan.json")
    rows = read(SOURCE / "predictions.json")
    h0_rows = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"]} for r in rows]
    if any(r["h0"] is None for r in h0_rows) or len(h0_rows) != 8:
        raise ValueError("eight frozen real H0 required")
    liability, count, predecessors = Decimal(0), 0, []
    previous_runs = [ROOT / f"artifacts/preflight/h0_prior_panel_openrouter_20260907_v{version}"
                     for version in (1, 2, 3, 4, 6, 7)]
    previous_mean = [p for p in sorted((ROOT / "artifacts/preflight").glob("h0_five_mean_openrouter_20260907_v*"))
                     if p.resolve() != args.output.resolve() and (p / "budget.json").is_file()]
    extra_calls = 0
    for p in [*previous_runs, *previous_mean]:
        ledger = read(p / "budget.json")
        for r in ledger["calls"].values():
            if r["state"] not in {"NOT_SENT", "RESERVED"}:
                count += 1
                extra_calls += int(p in previous_mean)
                liability += Decimal(r["cost_usd"] if r["cost_usd"] is not None else r["reserve_usd"])
        predecessors.append({"path": str(p), "ledger_sha256": sha256_file(p / "budget.json")})
    save(args.output / "h0_input.json", h0_rows)
    save(args.output / "smoke_input.json", read(SOURCE / "smoke_input.json"))
    paths = {ROOT / p for p in old_plan["runtime_source_sha256"]}
    paths.update({Path(__file__), ROOT / "src/surgical_agent/research/verification/mean_panel.py",
                  ROOT / "tests/unit/test_mean_panel.py"})
    paths.update(CONTRACTS.glob("*"))
    hashes = {}
    for p in sorted(paths):
        rel = str(p.relative_to(ROOT))
        dest = args.output / "frozen_source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        hashes[rel] = sha256_file(p)
    plan = {"version": "five_mean_runtime_v4", "created_utc": legacy.now(),
            "development_replay": True, "split": "Training", "selection": old_plan["selection"],
            "models": pricing_snapshot(), "source_sha256": hashes, "runtime_source_sha256": hashes,
            "data_sha256": old_plan["data_sha256"], "source_h0_sha256": completed["prediction_sha256"],
            "artifact_sha256": {n: sha256_file(args.output / n) for n in ("h0_input.json", "smoke_input.json")},
            "budget_usd": str(Decimal(15) - liability), "total_authorized_usd": "15",
            "carryover_liability_usd": str(liability), "carryover_calls": count, "predecessors": predecessors,
            "max_calls": 142 - extra_calls, "protocol": read(CONTRACTS / "protocol.json")}
    if args.resume_source:
        previous = args.resume_source.resolve()
        if previous not in {p.resolve() for p in previous_mean}:
            raise ValueError("resume source must be included in previous liabilities")
        old = read(previous / "plan.json")
        completion = read(previous / "prediction_completion.json")
        if sha256_file(previous / "predictions.json") != completion["prediction_sha256"]:
            raise ValueError("resume predictions changed")
        if old["selection"] != plan["selection"] or old["protocol"] != plan["protocol"]:
            raise ValueError("resume protocol differs")
        if [(m["model"], m["tag"]) for m in old["models"]] != MODEL_TAGS:
            raise ValueError("resume models differ")
        for rel, expected in old["runtime_source_sha256"].items():
            if rel != str(Path(__file__).relative_to(ROOT)) and sha256_file(ROOT / rel) != expected:
                raise ValueError("resume inference dependency differs: " + rel)

        def nodes(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef)}
        before = nodes(previous / "frozen_source/scripts/run_mean_panel_trial.py")
        after = nodes(Path(__file__))
        for name in ("pricing_snapshot", "request_body", "judge_packet"):
            if before[name] != after[name]:
                raise ValueError("resume request builder differs")
        smoke = read(previous / "smoke_result.json")
        if smoke["passed"] is not True:
            raise ValueError("resume needs successful same-contract smoke")
        history = legacy.merge_dispatch_history([read(p / "budget.json")["calls"] for p in previous_mean])
        save(args.output / "resume_input.json", {"rows": read(previous / "predictions.json"),
                                                "source_calls": history, "source": str(previous)})
        save(args.output / "smoke_result.json", {**smoke, "reused_from": str(previous)})
        shutil.copytree(previous / "targets", args.output / "targets")
        plan["resume_source"] = str(previous)
        plan["artifact_sha256"].update({n: sha256_file(args.output / n)
                                      for n in ("resume_input.json", "smoke_result.json")})
    # Validate exact cached image/request binding before any new paid call.
    import torch
    torch.set_num_threads(2)
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    for selected in plan["selection"]:
        base = legacy.build_base(adapter, selected)
        if canonical_request_metadata(base).to_mapping() != selected["request_metadata"]:
            raise ValueError("cached H0 metadata drift")
    save(args.output / "plan.json", plan)
    print(json.dumps({"status": "PREPARED", "targets": 8, "max_new_calls": plan["max_calls"],
                      "budget_remaining_usd": plan["budget_usd"], "models": MODEL_TAGS}), flush=True)


def judge_packet(base, selection, draft, round_no, previous=None):
    packet = legacy.packet_for(base, selection, full_universe(), draft)
    # Full ontology is in system prompt. Indexed score arrays cover all labels.
    packet.pop("propositions")
    packet["round"] = round_no
    if round_no == 1:
        packet.pop("current_prediction")
        packet["instruction"] = "Independent visual prediction. No initial answer is supplied."
    else:
        packet["instruction"] = ("Reinspect images to check this proposed revision, including missing labels. "
                                 "It is a fallible draft, not ground truth. Correct your earlier visual reading if warranted.")
        packet["your_previous_response"] = previous
    return {"academic_context": ACADEMIC_CONTEXT, **packet}


def execute(args, smoke=False):
    import torch
    torch.set_num_threads(2)
    plan = read(args.output / "plan.json")
    legacy.assert_frozen(args.output, plan)
    for pred in plan["predecessors"]:
        if sha256_file(Path(pred["path"]) / "budget.json") != pred["ledger_sha256"]:
            raise ValueError("earlier liability changed")
    live = pricing_snapshot()
    if any(a["endpoint"]["pricing"] != b["endpoint"]["pricing"] for a, b in zip(live, plan["models"], strict=True)):
        raise ValueError("live prices changed")
    with (args.output / ("smoke.lock" if smoke else "run.lock")).open("x") as f:
        f.write("One dispatch only; no automatic retry.")
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    budget = legacy.DirectBudget(args.output, plan, secret)
    models = plan["models"]
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    js = response_schema()
    ps = read(legacy.CONTRACTS / "repair_response.schema.json")
    jp = (CONTRACTS / "judge_prompt.txt").read_text(encoding="utf-8")
    pp = ACADEMIC_CONTEXT + "\n\n" + (legacy.CONTRACTS / "repair_prompt.txt").read_text(encoding="utf-8")
    pp += "\nThis experiment uses five 1-5 scores, not three-state votes. Issues are selected by arithmetic mean >=4 for ADD, <=2 for REMOVE. Fix only listed issues; component additions need their own listed issue.\n"
    if smoke:
        source = read(args.output / "smoke_input.json")
        s, h0 = source["selection"], source["h0"]
        base = legacy.build_base(adapter, s)
        names = [(f"smoke/judge_{i}", m["model"]) for i, m in enumerate(models[:5])]
        names.append(("smoke/repair", models[5]["model"]))
        if not budget.reserve(names):
            raise ValueError("insufficient smoke reservation")
        packet = judge_packet(base, s, h0, 1)
        raw = [budget.call(f"smoke/judge_{i}", request_body(base, m, js, jp, packet), js)
               for i, m in enumerate(models[:5])]
        try:
            mean_scores(raw)
            passed = True
        except (ValueError, TypeError):
            passed = False
        repair_packet = legacy.packet_for(base, s, full_universe(), h0)
        repair_packet.update(academic_context=ACADEMIC_CONTEXT, h0=h0, last_accepted=h0, issues=[],
                             allowed_edit_roles=legacy.repair_roles(full_universe()),
                             instruction="Engineering smoke: no issue; return empty edits.")
        patch = budget.call("smoke/repair", request_body(base, models[5], ps, pp, repair_packet), ps)
        save(args.output / "smoke_result.json", {"passed": passed and patch is not None and patch["edits"] == [],
                                                "five_full_score_arrays": passed, "engineering_only": True})
        budget.release_unsent()
    else:
        if not read(args.output / "smoke_result.json")["passed"]:
            raise ValueError("all five judges and repair smoke required")
        cached = {(r["video_id"], r["frame_id"]): r["h0"] for r in read(args.output / "h0_input.json")}
        resume = read(args.output / "resume_input.json") if plan.get("resume_source") else {"rows": [], "source_calls": {}}
        resumed = {(r["video_id"], r["frame_id"]): r for r in resume["rows"]}
        rows = []
        for s in plan["selection"]:
            key, h0 = s["key"], cached[s["video_id"], s["frame_id"]]
            if legacy.was_dispatched(resume["source_calls"], key):
                previous_row = resumed[s["video_id"], s["frame_id"]]
                if previous_row["h0"] != h0:
                    raise ValueError("resume H0 changed")
                rows.append(previous_row)
                save(args.output / "predictions.json", rows)
                continue
            base = legacy.build_base(adapter, s)
            destination = args.output / "targets" / key

            def panel(round_no, draft, previous, key=key, base=base, s=s):
                names = [(f"{key}/r{round_no}_judge_{i}", m["model"]) for i, m in enumerate(models[:5])]
                if round_no == 1 and not budget.reserve(names):
                    return None
                bodies = [request_body(base, m, js, jp, judge_packet(
                    base, s, draft, round_no, None if previous is None else previous[i]))
                    for i, m in enumerate(models[:5])]
                with ThreadPoolExecutor(max_workers=5) as pool:
                    futures = [pool.submit(budget.call, name, body, js)
                               for (name, _), body in zip(names, bodies, strict=True)]
                    return [f.result() for f in futures]

            def patch(round_no, state, issues, key=key, base=base, s=s, h0=h0):
                name = f"{key}/p{round_no}"
                reservations = [(name, models[5]["model"])] + [
                    (f"{key}/r{round_no + 1}_judge_{i}", m["model"]) for i, m in enumerate(models[:5])]
                if not budget.reserve(reservations):
                    return None
                packet = legacy.packet_for(base, s, full_universe(), state)
                packet.update(academic_context=ACADEMIC_CONTEXT, h0=h0, last_accepted=state, issues=issues,
                              allowed_edit_roles=legacy.repair_roles(full_universe()))
                return budget.call(name, request_body(base, models[5], ps, pp, packet), ps)

            result = run_arm(h0, panel, patch, ps, [im.identifier for im in base.images],
                             lambda stage, value, destination=destination: save(destination / f"{stage}.json", value))
            rows.append({"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0, "arm": result})
            budget.release_unsent()
            save(args.output / "predictions.json", rows)
        save(args.output / "prediction_completion.json", {"targets": len(rows),
             "prediction_sha256": sha256_file(args.output / "predictions.json"), "completed_utc": legacy.now()})
    assert_secret_absent(secret, args.output.rglob("*.json"))


def score(args):
    plan = read(args.output / "plan.json")
    legacy.assert_frozen(args.output, plan)
    done = read(args.output / "prediction_completion.json")
    if done["prediction_sha256"] != sha256_file(args.output / "predictions.json"):
        raise ValueError("saved predictions changed")
    rows = read(args.output / "predictions.json")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    reports = {}
    for i in range(3):
        values = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"],
                   "h1": None, "final": r["arm"]["snapshots"][i]} for r in rows]
        report, scored = legacy.score_saved(adapter, values)
        reports[f"D{i + 1}"] = report
        save(args.output / "scored" / f"D{i + 1}.json", scored)
    save(args.output / "metrics.json", reports)
    print(json.dumps({"status": "SCORED", "targets": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "docs/API.txt")
    parser.add_argument("--resume-source", type=Path,
                        help="Continue only never-dispatched targets under identical frozen inference contract")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "score":
        score(args)
    else:
        execute(args, smoke=args.mode == "smoke")
