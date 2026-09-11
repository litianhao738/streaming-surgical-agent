"""Recover an interrupted frozen continuation without network calls or retries."""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def recover(source, output):
    if output.exists() or (source / "completion.json").exists():
        raise ValueError("new destination and incomplete source required")
    sys.path[:0] = [str(source / "frozen_source"), str(source / "frozen_source/src")]
    from scripts import run_joint_phase_continuation_trial as runner
    trial = runner.trial
    runner.verify(source)
    original = trial.read(source / "budget.json")
    pending = [c for c in original["calls"] if c["status"] == "DISPATCHED"]
    if not pending:
        raise ValueError("No dispatched calls to recover")
    for c in pending:
        folder = source / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        if (folder / "response.json").exists():
            raise ValueError("Saved response needs separate reconciliation")
    source_hashes = {p.relative_to(source).as_posix(): trial.sha(p) for p in source.rglob("*.json")
        if "frozen_source" not in p.parts}
    shutil.copytree(source, output)
    ledger = trial.read(output / "budget.json")
    for c in ledger["calls"]:
        if c["status"] == "DISPATCHED":
            c.update(status="INTERRUPTED_UNKNOWN", recovery="No saved response; reserved cost retained; no retry")
            folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
            trial.save(folder / "record.json", c)
    ledger["stopped"] = True
    trial.save(output / "budget.json", ledger)
    plan = runner.verify(output)
    adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases, initials = trial.bases_for(adapter, plan), trial.read(output / "initials.json")
    rows = []
    with patch("requests.post", side_effect=AssertionError("Recovery must not call API")), \
         patch.object(adapter, "iter_video", side_effect=AssertionError("Recovery must not read GT")), \
         runner.reuse(output, "score"), trial.roster.lightweight_protocol():
        replay = trial.common.ReplayCalls(output, ledger["calls"])
        for i, s in enumerate(plan["selection"]):
            r = trial.run_case(replay, bases[s["key"]], s, initials[s["key"]], plan["variant"], i)
            target = output / "targets" / s["key"] / "result.json"
            if target.exists():
                stored = trial.read(target)
                seconds = stored.pop("seconds")
                if r != stored:
                    raise ValueError("Completed target changed during recovery")
                r["seconds"] = seconds
            else:
                r["seconds"] = 0
            trial.save(target, r)
            rows.append({**{k:s[k] for k in ("key", "video_id", "frame_id")}, **r["predictions"]})
        assert len(replay.rows) == len(ledger["calls"])
    trial.save(output / "predictions.json", rows)
    trial.save(output / "interruption_recovery.json", {"source": str(source.resolve()), "source_hashes": source_hashes,
        "recovered_utc": trial.now(), "api_calls_added": 0, "gt_reads": 0,
        "interrupted_indices": [c["index"] for c in pending], "completed_targets_preserved": 15,
        "last_target_missing_votes_fallback": True, "unknown_reserved_charges_retained": True,
        "elapsed_time_note": "API timestamp span is a lower bound; interruption pause is excluded; last target seconds unavailable."})
    times = [datetime.fromisoformat(c[k]) for c in ledger["calls"] for k in ("started_utc", "finished_utc") if c.get(k)]
    files = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
    trial.save(output / "completion.json", {"closed_utc": trial.now(), "fatal_error": None,
        "recovered_interrupted_run": True, "interrupted_calls": len(pending), "elapsed_is_lower_bound": True,
        "elapsed_seconds": (max(times)-min(times)).total_seconds(), "calls": len(ledger["calls"]),
        "hashes": {p.relative_to(output).as_posix(): trial.sha(p) for p in files}})
    print({"recovered": len(rows), "unknown_calls": len(pending), "new_api_calls": 0}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    recover(a.source, a.output)
