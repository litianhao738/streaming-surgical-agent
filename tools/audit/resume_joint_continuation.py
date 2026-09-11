"""Resume only unattempted requests from an interrupted frozen continuation."""
import argparse
import shutil
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter


def resume(source, output):
    if output.exists() or (source / "completion.json").exists():
        raise ValueError("New output and incomplete source required")
    sys.path[:0] = [str(source / "frozen_source"), str(source / "frozen_source/src")]
    from scripts import run_joint_phase_continuation_trial as runner
    trial = runner.trial
    plan = runner.verify(source)
    old = trial.read(source / "budget.json")
    source_hashes = {p.relative_to(source).as_posix(): trial.sha(p) for p in source.rglob("*.json")
        if "frozen_source" not in p.parts}
    for c in old["calls"]:
        if c["status"] == "DISPATCHED":
            folder = source / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
            if (folder / "response.json").exists():
                raise ValueError("Saved response needs separate reconciliation")
    shutil.copytree(source, output)
    pending = []
    for c in old["calls"]:
        if c["status"] == "DISPATCHED":
            pending.append(c["index"])
            c.update(status="INTERRUPTED_UNKNOWN", recovery="No saved response; reserved cost retained; no retry")
            folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
            trial.save(folder / "record.json", c)
    trial.save(output / "budget.json", old)
    trial.save(output / "resumption.json", {"source": str(source.resolve()), "source_hashes": source_hashes,
        "started_utc": trial.now(), "existing_attempts": len(old["calls"]), "interrupted_indices": pending,
        "policy": "Replay every attempted request; send only never-attempted requests. No retries.",
        "completed_targets_preserved": len(list((source / "targets").glob("*/result.json")))})
    old_replay = trial.common.ReplayCalls(output, deepcopy(old["calls"]))
    identities = {(c["target"], c["stage"], c["seat"]) for c in old["calls"]}
    class ReplayOrRun:
        def __init__(self, underlying):
            self.underlying = underlying
        def __getattr__(self, name):
            return getattr(self.underlying, name)
        def call(self, target, stage, seat, body):
            if (target, stage, seat) in identities:
                return old_replay.call(target, stage, seat, body)
            return self.underlying.call(target, stage, seat, body)
    adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases, initials = trial.bases_for(adapter, plan), trial.read(output / "initials.json")
    start, rows, fatal = perf_counter(), [], None
    with trial.credential_context(plan), runner.reuse(output, "execute"), trial.roster.lightweight_protocol():
        wrapped = trial.BoundCalls(output, plan)
        actual = wrapped.actual
        actual.rows = deepcopy(old["calls"])
        actual.occupied = {k: Decimal(v) for k, v in old["occupied"].items()}
        actual.carried = old["carried_occupied"]
        actual.attempted = set(identities)
        actual.persist()
        calls = ReplayOrRun(wrapped)
        try:
            for i, s in enumerate(plan["selection"]):
                stamp = perf_counter()
                r = trial.run_case(calls, bases[s["key"]], s, initials[s["key"]], plan["variant"], i)
                dest = output / "targets" / s["key"] / "result.json"
                if dest.exists():
                    saved = trial.read(dest)
                    seconds = saved.pop("seconds")
                    if r != saved:
                        raise ValueError("Completed target changed on resumption")
                    r["seconds"] = seconds
                else:
                    r["seconds"] = perf_counter()-stamp
                trial.save(dest, r)
                rows.append({**{k:s[k] for k in ("key", "video_id", "frame_id")}, **r["predictions"]})
                trial.save(output / "predictions.json", rows)
                print({"done": len(rows), "total_attempts": len(actual.rows), "new_attempts": len(actual.rows)-len(old["calls"])}, flush=True)
            assert len(old_replay.rows) == len(old["calls"])
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            actual.stopped = True
            actual.persist()
            files = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
            trial.save(output / "completion.json", {"closed_utc": trial.now(), "fatal_error": fatal,
                "recovered_interrupted_run": True, "elapsed_is_lower_bound": True,
                "elapsed_time_note": "Resumption execution only; original run and interruption pause excluded.",
                "calls": len(actual.rows), "new_calls": len(actual.rows)-len(old["calls"]),
                "elapsed_seconds": perf_counter()-start,
                "hashes": {p.relative_to(output).as_posix(): trial.sha(p) for p in files}})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    resume(a.source, a.output)
