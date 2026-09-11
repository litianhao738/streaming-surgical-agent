"""Bounded Repair-only v2 experiment reusing the sealed control and round-one calls."""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_joint_phase_feedback_trial as trial

METHOD = ("Repair v2 independently reconstructs the original three images with full ontology and unresolved task names; "
    "current prediction, pool, relation hints and prior review answers are withheld from Repair only. "
    "Same JSON, model, generation settings, reviewers, first round and thresholds. Parent control/round1 reused exactly. "
    "This is development after a failed confirmation, not a fresh validation result.")
POST = ("repair_r2", "joint_r2")
METHOD_V3 = ("Reuse sealed v2 independent Repair proposals. Remove current_prediction_hypothesis and "
    "phase_recommendation_hypothesis from round-two Verifier only. Same review prompt, candidate pool, JSON, "
    "five models and mean thresholds. No ground truth or candidate-specific exceptions. "
    "Development on previously scored targets; fresh confirmation required.")
METHOD_V4 = ("Independent image-first Repair proposes at most two plausible IVT alternatives per visible instrument "
    "when action/contact is ambiguous, within the same five-head JSON. Same v2 withheld-answer packet. "
    "Restore original hypothesis-aware Verifier v1; thresholds, models, first round and all images unchanged. "
    "Development diagnosis: independent reconstruction added no missing true IVT. Fresh confirmation required.")
METHODS = {"v2": METHOD, "v3": METHOD_V3, "v4": METHOD_V4}


def prepare(output, parent, variant="v2"):
    parent_plan = trial.read(parent / "plan.json")
    trial.prepare(output, Path(parent_plan["source_archive"]), variant, full=True)
    p = trial.read(output / "plan.json")
    assert p["selection"] == parent_plan["selection"]
    assert p["initials_sha256"] == parent_plan["initials_sha256"]
    p.update(parent_archive=str(parent.resolve()), parent_plan_sha256=trial.sha(parent / "plan.json"),
        parent_completion_sha256=trial.sha(parent / "completion.json"), continuation_method=METHODS[variant],
        new_stages=list(("joint_r2",) if variant == "v3" else POST),
        max_calls=len(p["selection"]) * (5 if variant == "v3" else 6), cohort="Development on previously scored confirmation; no independent efficacy claim")
    trial.save(output / "plan.json", p)


def verify(output):
    p = trial.verify(output)
    assert p["variant"] in METHODS
    assert p["continuation_method"] == METHODS[p["variant"]]
    parent = Path(p["parent_archive"])
    assert trial.sha(parent / "plan.json") == p["parent_plan_sha256"]
    assert trial.sha(parent / "completion.json") == p["parent_completion_sha256"]
    done = trial.read(parent / "completion.json")
    assert done["fatal_error"] is None
    for rel, value in done["hashes"].items():
        assert trial.sha(parent / rel) == value, rel
    assert trial.read(parent / "plan.json")["selection"] == p["selection"]
    assert trial.sha(parent / "initials.json") == p["initials_sha256"]
    return p


class ReusePrefix:
    def __init__(self, actual, prefix, new_stages=POST):
        self.actual, self.prefix = actual, prefix
        self.new_stages = new_stages

    def __getattr__(self, key):
        return getattr(self.actual, key)

    @property
    def stopped(self):
        return self.actual.stopped

    @stopped.setter
    def stopped(self, value):
        self.actual.stopped = value

    def call(self, target, stage, seat, body):
        return (self.actual if stage in self.new_stages else self.prefix).call(target, stage, seat, body)


class ArchivedPrefix:
    def __init__(self, parent, replay_type):
        self.archives = []
        seen = set()
        while parent is not None:
            if parent.resolve() in seen:
                raise ValueError("cyclic replay parents")
            seen.add(parent.resolve())
            done = trial.read(parent / "completion.json")
            for rel, value in done["hashes"].items():
                assert trial.sha(parent / rel) == value, rel
            self.archives.append(replay_type(parent, trial.read(parent / "budget.json")["calls"]))
            previous = trial.read(parent / "plan.json").get("parent_archive")
            parent = Path(previous) if previous else None

    @property
    def rows(self):
        return [r for archive in self.archives for r in archive.rows]

    def call(self, target, stage, seat, body):
        for archive in self.archives:
            if any((c["target"], c["stage"], c["seat"]) == (target, stage, seat) for c in archive.available):
                return archive.call(target, stage, seat, body)
        raise ValueError("missing sealed prefix call")


@contextmanager
def reuse(output, command):
    plan = verify(output)
    parent = Path(plan["parent_archive"])
    original_live, original_replay = trial.BoundCalls, trial.common.ReplayCalls
    def prefix():
        return ArchivedPrefix(parent, original_replay)
    stages = tuple(plan.get("new_stages", POST))
    if command == "execute":
        with patch.object(trial, "BoundCalls", lambda *a, **k: ReusePrefix(original_live(*a, **k), prefix(), stages)):
            yield
    else:
        with patch.object(trial.common, "ReplayCalls", lambda *a, **k: ReusePrefix(original_replay(*a, **k), prefix(), stages)):
            yield


def preflight(output, adapter):
    from tools.audit.preflight_joint_phase_feedback import MockCalls
    p = verify(output)
    parent = Path(p["parent_archive"])
    originals = {r["key"]: r for r in trial.read(parent / "predictions.json")}
    prefix = ArchivedPrefix(parent, trial.common.ReplayCalls)
    initials, bases = trial.read(output / "initials.json"), trial.bases_for(adapter, p)
    count = 0
    with patch("requests.post", side_effect=AssertionError("No API allowed")), \
         patch.object(adapter, "iter_video", side_effect=AssertionError("No GT allowed")), trial.roster.lightweight_protocol():
        for i, s in enumerate(p["selection"]):
            mocked = MockCalls(initials[s["key"]]["h0"])
            result = trial.run_case(ReusePrefix(mocked, prefix, tuple(p.get("new_stages", POST))),
                bases[s["key"]], s, initials[s["key"]], p["variant"], i)
            assert all(result["predictions"][a] == originals[s["key"]][a] for a in ("h0", "control", "joint_r1"))
            assert result["predictions"]["joint_r2"] == originals[s["key"]]["joint_r1"]
            count += len(mocked.rows)
    trial.save(output / "offline_preflight.json", {"new_wire_mock_calls": count, "reused_prefix_calls": len(prefix.rows),
        "api_calls": 0, "gt_reads": 0, "plan_sha256": trial.sha(output / "plan.json")})
    print({"new_wire_mock_calls": count, "prefix_calls": len(prefix.rows)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--variant", choices=tuple(METHODS), default="v2")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.parent, args.variant)
    else:
        adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
        if args.command == "preflight":
            preflight(args.output, adapter)
        else:
            with reuse(args.output, args.command):
                getattr(trial, args.command)(args.output, adapter)
