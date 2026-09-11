"""Recover a closed local-write failure without repeating any inference POST.

The original 160-request protocol and bodies are unchanged. Copy the 60 closed
responses, reuse them by exact wire identity, and dispatch only missing calls.
"""
import argparse
import shutil
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from scripts import run_compact_verifier_trial as trial
from scripts.run_graph_review_trial import TimedCalls
from scripts.score_five_head_repair_trial import ReplayCalls

INTERRUPTED = trial.ROOT / "artifacts/preflight/compact_verifier_paired_20260909_v2"


class ResumedCalls(trial.BoundCalls):
    def __init__(self, output):
        old = trial.read(INTERRUPTED / "budget.json")
        TimedCalls.__init__(self, output, previous_budget=old["carried_occupied"],
                           limits={k: Decimal(v) for k, v in trial.LIMITS.items()},
                           rates=trial.RATES_V2, providers=trial.PROVIDERS,
                           max_calls=trial.MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.rows = deepcopy(old["calls"])
        self.occupied = {k: Decimal(v) for k, v in old["occupied"].items()}
        self.attempted = set()
        self.archived = {(r["target"], r["stage"], r["seat"]) for r in self.rows}
        self.replay = ReplayCalls(output, deepcopy(self.rows))

    def persist(self):
        for attempt in range(20):
            try:
                return super().persist()
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)  # Retry local atomic replacement only, never POST.

    def call(self, target, stage, seat, body):
        if (target, stage, seat) in self.archived:
            return self.replay.call(target, stage, seat, body)
        return super().call(target, stage, seat, body)


def prepare(output):
    if output.exists():
        raise ValueError("fresh continuation directory required")
    trial.verify(INTERRUPTED)
    old, done = (trial.read(INTERRUPTED / n) for n in ("budget.json", "completion.json"))
    if not old["stopped"] or done["fatal_error"] != "PermissionError":
        raise ValueError("only the audited closed local-write failure may resume")
    trial.same(len(old["calls"]), 60)
    for name, digest in done["hashes"].items():
        trial.same(trial.sha(INTERRUPTED / name), digest)
    if any(r["status"] == "DISPATCHED" or not r.get("finished_utc") for r in old["calls"]):
        raise ValueError("an unresolved call cannot be dispatched again")
    output.mkdir(parents=True)
    for name in ("plan.json", "initial_state.json", "metadata.json"):
        shutil.copyfile(INTERRUPTED / name, output / name)
    for folder in ("requests", "calls", "frozen_source"):
        shutil.copytree(INTERRUPTED / folder, output / folder)
    for r in old["calls"]:
        folder = output / "calls" / f"{r['index']:03d}_{r['target']}_{r['stage']}_{r['seat']}"
        trial.same(trial.read(folder / "record.json"), r)
    trial.save(output / "resume_manifest.json", {"source": str(INTERRUPTED),
               "source_completion_sha256": trial.sha(INTERRUPTED / "completion.json"),
               "resumer_sha256": trial.sha(Path(__file__)), "cached_calls": 60, "remaining_max_posts": 100,
               "timing_note": "Execution panel_seconds includes fast archive replay. For comparison use per-arm max seat elapsed for each target, and disclose this as reconstructed parallel API time."})
    trial.verify(output)
    print("Prepared exact-wire continuation: 60 cached calls, at most 100 new POSTs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        manifest = trial.read(args.output / "resume_manifest.json")
        trial.same(trial.sha(Path(__file__)), manifest["resumer_sha256"])
        trial.same(trial.sha(INTERRUPTED / "completion.json"), manifest["source_completion_sha256"])
        trial.BoundCalls = ResumedCalls
        adapter = trial.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
        getattr(trial, args.command)(args.output, adapter)
