"""Default graph/Phase repair with the measured compact verifier wording.

Reuse the frozen scheduler and selectors in a process-local context. Historical
runner files remain unchanged so their request hashes and replays stay valid.
The context spans all worker calls and restores its bindings on exit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_parallel_phase_trial as original
from scripts import score_parallel_phase_trial as original_score
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_prior_panel_trial import read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.compact_prompt import (
    PROFILE as PROMPT_PROFILE,
)
from surgical_agent.research.verification.compact_prompt import compact_review_wire

VERSION = "parallel-phase-repair-v1.1.0-compact-prompt"
PROFILE = "fixed_h0_parallel_graph_and_phase_compact_v2"
LEGACY_PROFILE = original.PROFILE
_REVIEW_WIRE = original.review_wire
_PHASE_WIRE = original.phase_wire
_VERIFY_PLAN = original.verify_plan
_CONTEXT_LOCK = RLock()
TEMPLATES = ("compact_interaction_v1.txt", "compact_phase_v2.txt")
PROMPT_FILES = [ROOT / "src/surgical_agent/research/verification/prompts" / n for n in TEMPLATES]
RUNTIME_FILES = [Path(__file__).resolve(),
                 ROOT / "src/surgical_agent/research/verification/compact_prompt.py", *PROMPT_FILES]


def compact_body(body, branch):
    # Actual provider tokens were checked in the paired trial. This local guard
    # bounds UTF-8 text length only; it is not advertised as an exact tokenizer.
    return compact_review_wire(body, branch, count_tokens=lambda text: len(text.encode("utf-8")))


def review_wire(seat, base, selected, pool):
    return compact_body(_REVIEW_WIRE(seat, base, selected, pool), "graph")


def phase_wire(seat, selected):
    return compact_body(_PHASE_WIRE(seat, selected), "phase")


def verify_plan(plan):
    if (plan.get("profile") != PROFILE or plan.get("repair_version") != VERSION
            or plan.get("verifier_prompt_profile") != PROMPT_PROFILE):
        raise ValueError("compact repair plan/version mismatch")
    for path in RUNTIME_FILES:
        if plan["source_sha256"].get(path.relative_to(ROOT).as_posix()) != sha(path):
            raise ValueError("compact runtime/template missing or changed")
    _VERIFY_PLAN(plan)  # Called while the original runner is bound to PROFILE.


@contextmanager
def compact_protocol():
    """Configure only this CLI process, without editing historical modules."""
    with _CONTEXT_LOCK:
        bindings = {name: getattr(original, name) for name in ("PROFILE", "review_wire", "phase_wire", "verify_plan")}
        scorer_verify = original_score.verify_plan
        try:
            original.PROFILE = PROFILE
            original.review_wire = review_wire
            original.phase_wire = phase_wire
            original.verify_plan = verify_plan
            original_score.verify_plan = verify_plan
            yield
        finally:
            for name, value in bindings.items():
                setattr(original, name, value)
            original_score.verify_plan = scorer_verify


def finalize_compact_plan(output):
    """Replace review fingerprints after original source/H0 checks finish."""
    plan = read(output / "plan.json")
    if plan["profile"] != LEGACY_PROFILE:
        raise ValueError("expected a newly prepared original repair plan")
    plan.update(profile=PROFILE, repair_version=VERSION, verifier_prompt_profile=PROMPT_PROFILE,
                base_repair_version="parallel-phase-repair-v1.0.0-experimental",
                prompt_budget_check="UTF-8 text length nonincrease; native token counts are provider-specific")
    for path in RUNTIME_FILES:
        plan["source_sha256"][path.relative_to(ROOT).as_posix()] = sha(path)
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # Proposal fingerprints stay original. Review pools are generated at run
    # time; run_graph records the actual compact review request fingerprints.
    for key, selected in plan["phase_inputs"].items():
        plan["wire_fingerprints"][key]["phase"] = {
            seat: fingerprint(phase_wire(seat, selected)) for seat in original.SEATS}
    plan["policy"] += " Four-head and Phase verifier wording uses the measured compact v2 JSON-mode prompt; all output contracts, proposals, models and selectors remain original."
    plan["limitations"] += ["Compact wording reduced input in 76 measurable paired calls; four-head F1 was unchanged. Phase gain was confounded by provider failures; no general accuracy or cost improvement is claimed."]
    save(output / "plan.json", plan)
    with compact_protocol():
        verify_plan(plan)
    return plan


def prepare(output, source, adapter):
    # The legacy preflight verifies its historical inputs with the legacy
    # fingerprints. Only then bind and freeze the new prompt's fingerprints.
    original.prepare(output, source, adapter)
    plan = finalize_compact_plan(output)
    print(json.dumps({"default_repair_version": VERSION, "prompt_profile": PROMPT_PROFILE,
                      "plan_sha256": sha(output / "plan.json"), "targets": len(plan["selection"])}), flush=True)


def dispatch(command, output, adapter, *, source=original.SOURCE):
    if command == "prepare":
        return prepare(output, source, adapter)
    if command not in ("execute", "score"):
        raise ValueError("unknown repair command")
    profile = read(output / "plan.json")["profile"]
    action = original.execute if command == "execute" else original_score.score
    if profile == LEGACY_PROFILE:
        # A changed default must not change an already prepared experiment.
        print(json.dumps({"using_saved_profile": LEGACY_PROFILE, "reason": "historical frozen plan"}), flush=True)
        return action(output, adapter)
    if profile != PROFILE:
        raise ValueError("unsupported saved repair profile; use its original experiment entrypoint")
    with compact_protocol():
        return action(output, adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=original.SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    dispatch(args.command, args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3), source=args.source)
