"""Recover valid Qwen alias H0 responses without repeating any API request.

Freeze the public alias/snapshot identity evidence, then run only the twelve
previously undispatched mainline steps for each target. Never edit the first run.
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

import requests

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]

from scripts import compare_mainline_backbones as trial

ROOT, old, main = trial.ROOT, trial.old, trial.main
SNAPSHOT = "qwen/qwen3.8-max-0902"
STAGES = {k: v for k, v in main.STAGES.items() if k != "h0"}


def wire(body):
    result = trial.base_body(body, "qwen")
    result["model"] = SNAPSHOT
    return result


def verified_h0(source, row, base):
    folder = source / "qwen/calls" / f"{row['index']:03d}_{row['target']}_h0_base"
    expected = trial.base_body(old.gemini_h0_wire(base), "qwen")
    if old.read(folder / "request.json") != old.joint.roster.transport.redact_images(expected):
        raise ValueError("original paired H0 request mismatch")
    response = old.read(folder / "response.json")
    raw = response["body"]
    if (response["http_status"] != 200 or raw.get("error") or raw.get("model") != SNAPSHOT
            or raw.get("provider") != "Alibaba"):
        raise ValueError("not the verified Qwen snapshot")
    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
        raise ValueError("incomplete or refused H0")
    parsed = json.loads(choice["message"]["content"])
    old.gated.h0_from_raw(parsed)
    return {"raw": parsed, "source": str(folder), "seconds": row["elapsed_seconds"],
            "response_sha256": old.sha(folder / "response.json"), "request_sha256": old.sha(folder / "request.json")}


def prepare(source, output):
    if output.exists():
        raise ValueError("fresh output required")
    parent = trial.verify(source)
    done = old.read(source / "completion.json")
    if done["fatal_error"]:
        raise ValueError("parent must finish first")
    for rel, digest in done["evidence_sha256"].items():
        if old.sha(source / rel) != digest:
            raise ValueError("parent evidence changed")
    response = requests.get("https://openrouter.ai/api/v1/models/" + SNAPSHOT + "/endpoints", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    endpoint = next(e for e in metadata["data"]["endpoints"] if e["tag"] == "alibaba")
    original = parent["provider_metadata"]["qwen"]
    if (endpoint["status"] != 0 or endpoint["name"] != original["name"]
            or endpoint["provider_name"] != original["provider_name"] or endpoint["pricing"] != original["pricing"]):
        raise ValueError("public alias/snapshot identity not established")
    old.save(output / "snapshot_metadata.json", metadata)
    bases = trial.bases_for(parent)
    budget = old.read(source / "qwen/budget.json")
    if len(budget["calls"]) != 6 or any(c["stage"] != "h0" for c in budget["calls"]):
        raise ValueError("expected six original H0 calls and no dependent requests")
    cached = {r["target"]: verified_h0(source, r, bases[r["target"]]) for r in budget["calls"]}
    old.save(output / "verified_cached_h0.json", cached)
    sources = {p.relative_to(ROOT).as_posix(): old.sha(p) for p in trial.release.runtime_paths()}
    for rel in sources:
        dst = output / "frozen_source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    old.save(output / "plan.json", {"profile": "qwen_alias_recovery_training6_v1", "source": str(source),
             "created_utc": old.now(), "source_plan_sha256": old.sha(source / "plan.json"),
             "source_completion_sha256": old.sha(source / "completion.json"), "selection": parent["selection"],
             "snapshot": SNAPSHOT, "alias": trial.MODELS["qwen"]["model"],
             "identity_evidence": "The alias and snapshot endpoint metadata name the same qwen3.8-max-20260902 backend, Alibaba route and prices; all six original responses returned snapshot -0902.",
             "cached_h0_sha256": old.sha(output / "verified_cached_h0.json"),
             "metadata_sha256": old.sha(output / "snapshot_metadata.json"), "source_sha256": sources,
             "carried_occupied": budget["occupied"], "max_new_calls": 72, "stages": STAGES,
             "original_rank_rule_unchanged": trial.STANDARD["rank"], "gt_inspected_for_recovery": False,
             "no_repeated_requests": True, "gate_data_collected": False})
    print(json.dumps({"verified_cached_h0": len(cached), "new_h0_calls": 0, "max_remaining_calls": 72}), flush=True)


def verify(output):
    plan = old.read(output / "plan.json")
    source = Path(plan["source"])
    parent = trial.verify(source)
    for rel, h in plan["source_sha256"].items():
        if old.sha(ROOT / rel) != h or old.sha(output / "frozen_source" / rel) != h:
            raise ValueError("recovery source changed")
    for file, key in (("verified_cached_h0.json", "cached_h0_sha256"), ("snapshot_metadata.json", "metadata_sha256")):
        if old.sha(output / file) != plan[key]:
            raise ValueError("recovery evidence changed")
    if old.sha(source / "completion.json") != plan["source_completion_sha256"]:
        raise ValueError("parent changed")
    for cached in old.read(output / "verified_cached_h0.json").values():
        folder = Path(cached["source"])
        if old.sha(folder / "response.json") != cached["response_sha256"] or old.sha(folder / "request.json") != cached["request_sha256"]:
            raise ValueError("original H0 changed")
    return plan, parent


class Calls(main.MainlineCalls):
    def __init__(self, output, plan, parent):
        self.cached = old.read(output / "verified_cached_h0.json")
        self.h0_uses = []
        super().__init__(output, previous_budget=plan["carried_occupied"],
                         limits={k: Decimal(v) for k, v in trial.LIMITS.items()},
                         rates={**old.joint.roster.RATES, "base": parent["base_rates"]["qwen"]},
                         providers={**old.joint.roster.PROVIDERS, "base": "Alibaba", SNAPSHOT: "Alibaba"},
                         max_calls=72, reasoning_seats=("gemini",))

    def call(self, target, stage, seat, body):
        if stage == "h0":
            if seat != "base" or target in self.h0_uses:
                raise ValueError("invalid duplicate cached H0")
            self.h0_uses.append(target)
            return deepcopy(self.cached[target]["raw"])
        return super().call(target, stage, seat, wire(body) if seat == "base" else body)


def preflight(output):
    plan, parent = verify(output)
    cached, bases, checks = old.read(output / "verified_cached_h0.json"), trial.bases_for(parent), []
    class Mock(old.MockCalls):
        def call(self, target, stage, seat, body):
            if stage == "h0":
                return deepcopy(cached[target]["raw"])
            return super().call(target, stage, seat, wire(body) if seat == "base" else body)
    with old.joint.roster.lightweight_protocol():
        for s in plan["selection"]:
            c = Mock()
            main.run_target(c, bases[s["key"]], s, old.read(Path(plan["source"]) / "priors" / f"{s['video_id']}.json"), parent["gate"])
            if Counter(r["stage"] for r in c.rows) != Counter(STAGES):
                raise ValueError("dependent stage set changed")
            checks.append(s["key"])
    old.save(output / "preflight.json", {"plan_sha256": old.sha(output / "plan.json"), "targets": checks, "api_calls": 0})
    print(json.dumps({"preflight_targets": len(checks), "api_calls": 0}), flush=True)


def execute(output):
    plan, parent = verify(output)
    if old.read(output / "preflight.json")["plan_sha256"] != old.sha(output / "plan.json"):
        raise ValueError("preflight changed")
    bases, rows, fatal = trial.bases_for(parent), [], None
    with (output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(old.sha(output / "plan.json"))
    start = perf_counter()
    with old.joint.credential_context(parent), old.joint.roster.lightweight_protocol():
        c = Calls(output, plan, parent)
        try:
            for s in plan["selection"]:
                row = {k: s[k] for k in ("key", "video_id", "frame_id")}
                stamp = perf_counter()
                if c.stopped:
                    row.update(status="NOT_DISPATCHED", predictions={a: None for a in main.ARMS})
                else:
                    record = main.run_target(c, bases[s["key"]], s,
                             old.read(Path(plan["source"]) / "priors" / f"{s['video_id']}.json"), parent["gate"])
                    old.save(output / "targets" / s["key"] / "result.json", record)
                    row.update(status="PREDICTED", predictions=record["predictions"], timing=record["timing_seconds"])
                row["repair_seconds"] = perf_counter() - stamp
                row["original_h0_seconds"] = c.cached[s["key"]]["seconds"]
                row["seconds"] = row["repair_seconds"] + row["original_h0_seconds"]
                rows.append(row)
                old.save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": s["key"], "completed": len(rows), "new_calls": len(c.rows),
                                  "status": row["status"], "repair_seconds": row["repair_seconds"]}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            c.stopped = True
            try:
                c.persist()
            finally:
                c.close_ledger()
            old.save(output / "predictions.json", {"targets": rows})
            sealed = {p.relative_to(output).as_posix(): old.sha(p) for p in output.rglob("*.json")
                      if "frozen_source" not in p.parts and p.name != "completion.json"}
            old.save(output / "completion.json", {"seconds": perf_counter() - start, "fatal_error": fatal,
                     "calls": len(c.rows), "cached_h0_uses": c.h0_uses, "evidence_sha256": sealed,
                     "gate_data_collected": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    else:
        globals()[args.command](args.output.resolve())
