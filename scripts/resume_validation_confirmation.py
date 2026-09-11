"""Resume a closed interrupted confirmation without repeating any saved attempt.

Replays every recorded request (including refusals), verifies request identity,
and spends only on previously unattempted tuples. File-write retries are local;
there are no API retries, roster changes, or ground-truth reads.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_validation_confirmation as trial
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_repair_revision_trial import parse_review_json


class ResumeCalls(trial.joint.roster.GLMCalls):
    def __init__(self, output, source, plan, *, offline=False):
        previous = trial.read(source / "budget.json")
        if not previous["stopped"] or previous["limits"] != plan["limits"]:
            raise ValueError("closed source and unchanged limits required")
        super().__init__(output, limits={a: Decimal(v) for a, v in plan["limits"].items()},
                         rates=trial.joint.roster.RATES, providers=trial.joint.roster.PROVIDERS,
                         max_calls=plan["max_calls"], reasoning_seats=("gemini",))
        self.rows = deepcopy(previous["calls"])
        self.occupied = {a: Decimal(v) for a, v in previous["occupied"].items()}
        self.source, self.offline = source, offline
        self.cached = {(r["target"], r["stage"], r["seat"]): r for r in self.rows}
        if len(self.cached) != len(self.rows):
            raise ValueError("duplicate source attempts")
        self.used = set()

    def persist(self):
        for attempt in range(40):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(.05)

    def call(self, target, stage, seat, body):
        key = target, stage, seat
        with self.lock:
            if key in self.used:
                raise ValueError("duplicate replay/dispatch")
            self.used.add(key)
        row = self.cached.get(key)
        if row is None:
            if self.offline:
                raise ValueError("offline replay cannot dispatch a new request")
            return super().call(target, stage, seat, body)
        folder = self.source / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
        if trial.read(folder / "request.json") != redact_images(body):
            raise ValueError("saved request differs: " + str(key))
        if row["status"] not in ("JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"):
            return None  # Keep failures and ambiguous attempts; never retry them.
        response = trial.read(folder / "response.json")
        raw = response["body"]
        expected = self.providers.get(body["model"], self.providers.get(seat))
        if (response["http_status"] != 200 or raw.get("error") or raw.get("model") != body["model"]
                or (expected is not None and raw.get("provider") != expected)
                or raw["choices"][0]["finish_reason"] != "stop"
                or raw["choices"][0]["message"].get("refusal")):
            raise ValueError("saved successful response identity invalid")
        import json
        text = raw["choices"][0]["message"]["content"]
        return json.loads(text) if seat == "base" else parse_review_json(text)[0]


def prepare(source, output, adapter):
    if output.exists():
        raise ValueError("new continuation directory required")
    plan = trial.read(source / "plan.json")
    trial.verify_plan(plan, source)
    completion = trial.read(source / "completion.json")
    if completion.get("fatal_error") != "PermissionError":
        raise ValueError("this recovery is for the recorded local file-write interruption")
    saved = trial.read(source / "predictions.json")["targets"]
    initials = trial.read(source / "initials.json")
    bases = trial.validation_bases(adapter, plan)
    ledger = ResumeCalls(output, source, plan, offline=True)
    calls_by_target = Counter(r["target"] for r in ledger.rows)
    replayed = []
    for index, selected in enumerate(plan["selection"]):
        key = selected["key"]
        if calls_by_target[key] != 16:
            break
        record = trial.run_target(ledger, bases[key], selected, initials[key], index)
        replayed.append({"key": key, "video_id": selected["video_id"],
                         "frame_id": selected["frame_id"], "predictions": record["predictions"]})
    if replayed[:len(saved)] != saved:
        raise ValueError("offline replay must exactly match every completed target")
    for name in ("plan.json", "initials.json", "offline_preflight.json", "wire_fingerprints.json"):
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, output / name)
    for name in ("calls", "frozen_source"):
        shutil.copytree(source / name, output / name)
    original_hashes = {p.relative_to(source).as_posix(): trial.sha(p)
                       for p in source.rglob("*.json") if "frozen_source" not in p.parts}
    trial.save(output / "recovery_plan.json", {
        "source": str(source.resolve()), "source_completion": completion,
        "source_hashes": original_hashes, "script_sha256": trial.sha(Path(__file__)),
        "cached_attempts": len(ledger.rows), "new_attempts_max": plan["max_calls"] - len(ledger.rows),
        "saved_targets_matched": len(saved), "complete_targets_reconstructed": len(replayed),
        "request_identity_checks": len(ledger.used), "api_calls_in_preflight": 0,
        "api_retries": 0, "local_write_retries": "up to 40 at 50 ms intervals",
        "limits_unchanged": True, "semantics_unchanged": True})
    shutil.copyfile(Path(__file__), output / "frozen_source/scripts/resume_validation_confirmation.py")
    print({"replayed_attempts": len(ledger.used), "saved_targets_matched": len(saved),
           "reconstructed_targets": len(replayed), "new_calls_max": plan["max_calls"] - len(ledger.rows)}, flush=True)


def execute(output, adapter):
    plan = trial.read(output / "plan.json")
    recovery = trial.read(output / "recovery_plan.json")
    source = Path(recovery["source"])
    trial.verify_plan(plan, output)
    if recovery["script_sha256"] != trial.sha(Path(__file__)):
        raise ValueError("recovery source changed")
    for name, digest in recovery["source_hashes"].items():
        if trial.sha(source / name) != digest:
            raise ValueError("interrupted archive changed: " + name)
    if (output / "execution.lock").exists():
        raise ValueError("single-use continuation")
    initials, bases = trial.read(output / "initials.json"), trial.validation_bases(adapter, plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as f:
        f.write(trial.sha(output / "recovery_plan.json"))
    rows, fatal, start = [], None, time.perf_counter()
    with trial.joint.credential_context(plan), trial.joint.roster.lightweight_protocol():
        calls = ResumeCalls(output, source, plan)
        try:
            for index, selected in enumerate(plan["selection"]):
                if calls.stopped:
                    break
                key = selected["key"]
                record = trial.run_target(calls, bases[key], selected, initials[key], index)
                trial.save(output / "targets" / key / "result.json", record)
                rows.append({"key": key, "video_id": selected["video_id"],
                             "frame_id": selected["frame_id"], "predictions": record["predictions"]})
                trial.save(output / "predictions.json", {"targets": rows})
                print({"target": key, "targets_completed": len(rows), "unique_attempts": len(calls.rows)}, flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            trial.save(output / "predictions.json", {"targets": rows})
            trial.save(output / "completion.json", {
                "closed_utc": trial.now(), "fatal_error": fatal, "calls": len(calls.rows),
                "cached_calls": recovery["cached_attempts"],
                "new_calls": len(calls.rows) - recovery["cached_attempts"],
                "statuses": dict(Counter(r["status"] for r in calls.rows)),
                "occupied": {k: str(v) for k, v in calls.occupied.items()},
                "seconds": time.perf_counter() - start + recovery["source_completion"]["seconds"],
                "api_retries": 0, "gt_not_loaded_during_inference": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = trial.joint.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.source, args.output, dataset)
    else:
        execute(args.output, dataset)
