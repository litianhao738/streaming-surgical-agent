"""Frozen two-target confirmation scheduler; predictions and model wires unchanged."""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_joint_phase_continuation_trial as continuation

trial = continuation.trial
SCHEDULER = {"workers": 2, "unit": "independent target", "output_order": "frozen selection order"}
CONFIRMATION = ("Predeclared same-video fresh16 confirmation. Primary v2 independent reconstruction, secondary v4 "
    "limited alternative proposals. Both ordinary and targeted admission reported. No further prompt/threshold tuning. "
    "Freeze both arms before reading this cohort's GT scores; preserve all failed calls and targets. "
    "Original strict success rule remains binding; partial task gains are reported separately.")


def prepare(output, source, parent, variant):
    if parent:
        if (parent / "metrics.json").exists() or (parent / "scored_truth.json").exists():
            raise ValueError("Secondary must be frozen before confirmation GT scoring")
        continuation.prepare(output, parent, variant)
    else:
        trial.prepare(output, source, variant, full=True)
    plan = trial.read(output / "plan.json")
    plan.update(execution_scheduler=SCHEDULER, confirmation_protocol=CONFIRMATION,
        cohort="Predeclared fresh Training targets from the same four videos; labels not scored before freeze")
    trial.save(output / "plan.json", plan)


def verify(output):
    p = trial.read(output / "plan.json")
    (continuation.verify if "parent_archive" in p else trial.verify)(output)
    assert p["execution_scheduler"] == SCHEDULER and p["confirmation_protocol"] == CONFIRMATION
    assert p["variant"] in ("v2", "v4")
    return p


def protocol(output, p, command):
    return continuation.reuse(output, command) if "parent_archive" in p else nullcontext()


def preflight(output, adapter):
    from tools.audit.preflight_joint_phase_feedback import MockCalls
    p = verify(output)
    initials, bases = trial.read(output / "initials.json"), trial.bases_for(adapter, p)
    parent = Path(p["parent_archive"]) if "parent_archive" in p else None
    prefix = continuation.ArchivedPrefix(parent, trial.common.ReplayCalls) if parent else None
    originals = {r["key"]: r for r in trial.read(parent / "predictions.json")} if parent else None
    def work(pair):
        i, s = pair
        calls = MockCalls(initials[s["key"]]["h0"])
        wrapped = continuation.ReusePrefix(calls, prefix, tuple(p["new_stages"])) if prefix else calls
        result = trial.run_case(wrapped, bases[s["key"]], s, initials[s["key"]], p["variant"], i)
        if originals:
            assert all(result["predictions"][a] == originals[s["key"]][a] for a in ("h0", "control", "joint_r1"))
            assert result["predictions"]["joint_r2"] == originals[s["key"]]["joint_r1"]
        else:
            assert all(result["predictions"][a] == initials[s["key"]]["h0"] for a in trial.ARMS)
        return s["key"], len(calls.rows)
    with patch("requests.post", side_effect=AssertionError("No API allowed")), \
         patch.object(adapter, "iter_video", side_effect=AssertionError("No GT allowed")), trial.roster.lightweight_protocol(), \
         ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(work, enumerate(p["selection"])))
    assert [r[0] for r in results] == [s["key"] for s in p["selection"]]
    trial.save(output / "offline_preflight.json", {"parallel_targets": 2, "new_wire_mock_calls": sum(r[1] for r in results),
        "reused_prefix_calls": len(prefix.rows) if prefix else 0, "api_calls": 0, "gt_reads": 0,
        "plan_sha256": trial.sha(output / "plan.json")})
    print({"preflight": True, "mock_calls": sum(r[1] for r in results)}, flush=True)


def execute(output, adapter):
    p = verify(output)
    initials, bases = trial.read(output / "initials.json"), trial.bases_for(adapter, p)
    with (output / "execution.lock").open("x") as f:
        f.write(trial.sha(output / "plan.json"))
    start, rows, fatal = perf_counter(), [], None
    with trial.credential_context(p), protocol(output, p, "execute"), trial.roster.lightweight_protocol():
        calls = trial.BoundCalls(output, p)
        def work(pair):
            i, s = pair
            if calls.stopped:
                raise RuntimeError("Circuit stopped before target")
            stamp = perf_counter()
            result = trial.run_case(calls, bases[s["key"]], s, initials[s["key"]], p["variant"], i)
            result["seconds"] = perf_counter()-stamp
            return s, result
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for s, result in executor.map(work, enumerate(p["selection"])):
                    trial.save(output / "targets" / s["key"] / "result.json", result)
                    rows.append({**{k:s[k] for k in ("key", "video_id", "frame_id")}, **result["predictions"]})
                    trial.save(output / "predictions.json", rows)
                    print({"done": len(rows), "target": s["key"], "calls": len(calls.rows)}, flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            files = [f for f in output.rglob("*.json") if "frozen_source" not in f.parts and f.name != "completion.json"]
            trial.save(output / "completion.json", {"closed_utc": trial.now(), "fatal_error": fatal,
                "calls": len(calls.rows), "elapsed_seconds": perf_counter()-start,
                "hashes": {f.relative_to(output).as_posix(): trial.sha(f) for f in files}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--variant", choices=("v2", "v4"), default="v2")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.source, args.parent, args.variant)
    else:
        adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
        if args.command == "score":
            plan = verify(args.output)
            with protocol(args.output, plan, "score"):
                trial.score(args.output, adapter)
        else:
            globals()[args.command](args.output, adapter)
