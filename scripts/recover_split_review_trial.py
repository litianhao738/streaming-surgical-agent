"""Independent, declared recovery of transport-affected paired targets only."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_split_review_trial as trial

SOURCE = ROOT / "artifacts/preflight/split_review_eight_20260909_v2"
MAX_NEW_CALLS = 60
NEW_LIMIT = Decimal(1)
CUMULATIVE_LIMIT = Decimal(3)
BASE_RUN_ARM = trial.run_arm


class RecoveryCalls(trial.BoundCalls):
    def __init__(self, output):
        super().__init__(output)
        cached = trial.read(output / "cached_ledger.json")
        self.rows = deepcopy(cached)
        self.occupied = {a: sum(Decimal(c["charge"]) for c in cached if c["account"] == a) for a in trial.LIMITS}
        self.cached_total = deepcopy(self.occupied)
        previous = trial.read(SOURCE / "budget.json")["occupied"]
        self.limits = {a: min(Decimal(trial.LIMITS[a]), self.occupied[a] + NEW_LIMIT,
            self.occupied[a] + CUMULATIVE_LIMIT - Decimal(previous[a])) for a in trial.LIMITS}
        self.archived = {(c["target"], c["stage"], c["seat"]) for c in cached}
        self.cached_targets = {c["target"] for c in cached}
        self.replay = trial.ReplayCalls(output, deepcopy(cached))

    def call(self, target, stage, seat, body):
        if (target, stage, seat) in self.archived:
            return self.replay.call(target, stage, seat, body)
        with self.lock:
            if len(self.rows) - len(self.archived) >= MAX_NEW_CALLS:
                self.stopped = True
        value = super().call(target, stage, seat, body)
        with self.lock:
            record = next((r for r in self.rows if (r["target"], r["stage"], r["seat"]) == (target, stage, seat)), {})
            if record.get("exception_type") == "ProxyError":
                self.stopped = True
                self.persist()
                trial.save(self.output / "network_circuit_breaker.json", {"target": target, "stage": stage,
                    "seat": seat, "reason": "ProxyError; stop subsequent dispatch, preserve in-flight calls"})
        return value


def run_arm(calls, key, initial, wire, arm):
    result = BASE_RUN_ARM(calls, key, initial, wire, arm)
    if isinstance(calls, RecoveryCalls) and key in calls.cached_targets:
        old = trial.read(SOURCE / "targets" / key / f"{arm}.json")
        trial.same({k: v for k, v in old.items() if k not in ("seconds", "request_seconds")},
                   {k: v for k, v in result.items() if k not in ("seconds", "request_seconds")})
        result = old  # Recorded API wait, not the very short replay time.
    return result


def prepare(output):
    if output.exists():
        raise ValueError("new recovery directory required")
    plan = trial.verify(SOURCE)
    done, ledger = trial.read(SOURCE / "completion.json"), trial.read(SOURCE / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed source required")
    for name, digest in done["hashes"].items():
        trial.same(trial.sha(SOURCE / name), digest)
    # Selection uses transport status only, not labels or whether scores improved.
    affected = {c["target"] for c in ledger["calls"] if not c["status"].startswith("JSON_PARSED")}
    cached = [c for c in ledger["calls"] if c["target"] not in affected]
    trial.same(len(affected), 4)
    trial.same(len(cached), 60)
    trial.same([c["index"] for c in cached], list(range(60)))
    output.mkdir(parents=True)
    for folder in ("requests", "frozen_source"):
        shutil.copytree(SOURCE / folder, output / folder)
    for name in ("initial_state.json", "metadata.json"):
        shutil.copyfile(SOURCE / name, output / name)
    for call in cached:
        name = f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        shutil.copytree(SOURCE / "calls" / name, output / "calls" / name)
    # v2's offline_preflight file was added after prepare, not a frozen plan input.
    trial.save(output / "cached_ledger.json", cached)
    plan = deepcopy(plan)
    code = Path(__file__).resolve()
    plan["sources"][code.relative_to(ROOT).as_posix()] = trial.sha(code)
    shutil.copyfile(code, output / "frozen_source" / code.relative_to(ROOT))
    plan["inputs"]["cached_ledger.json"] = trial.sha(output / "cached_ledger.json")
    plan["recovery"] = {"created_utc": trial.now(), "source_completion_sha256": trial.sha(SOURCE / "completion.json"),
        "source_budget_sha256": trial.sha(SOURCE / "budget.json"), "affected_targets": sorted(affected),
        "cached_calls": 60, "new_call_cap": MAX_NEW_CALLS, "additional_limits": {a: "1" for a in trial.LIMITS},
        "cumulative_limits": {a: "3" for a in trial.LIMITS}, "original_failed_calls_retained": True,
        "note": "Independent recovery after transport interruption, not an uninterrupted fresh 120-call trial. "
        "No answer-based selection or retry inside either batch. Review timing mixes original recorded waits "
        "for four complete targets with new waits for four transport-affected targets."}
    trial.save(output / "plan.json", plan)
    trial.verify(output)
    print(json.dumps(plan["recovery"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        trial.BoundCalls = RecoveryCalls
        trial.run_arm = run_arm
        adapter = trial.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
        getattr(trial, args.command)(args.output, adapter)
