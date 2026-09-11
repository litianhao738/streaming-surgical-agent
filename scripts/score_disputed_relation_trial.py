"""Offline-only compatibility for Decimal equality in a frozen trial audit.

The inference implementation and closed artifacts remain unchanged. The generic
JSON-hash equality helper cannot serialize Decimal ledger totals. Compare those
numeric values exactly; delegate every other check to the original helper.
"""
import argparse
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_disputed_relation_trial as trial
from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_feedback_continuation import same as json_same
from scripts.run_prior_panel_trial import now, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter


def exact_same(actual, expected, label):
    if isinstance(actual, Decimal) or isinstance(expected, Decimal):
        if type(actual) not in (Decimal, int) or type(expected) not in (Decimal, int) or actual != expected:
            raise ValueError("exact numeric mismatch: " + label)
    else:
        json_same(actual, expected, label)


def score(output, adapter):
    original = trial.same
    try:
        trial.same = exact_same
        trial.score(output, adapter)
    finally:
        trial.same = original
    save(output / "offline_score_compatibility.json", {
        "created_utc": now(), "code": str(Path(__file__).relative_to(ROOT)), "code_sha256": sha(__file__),
        "closed_completion_sha256": sha(output / "completion.json"), "new_api_calls": 0,
        "frozen_inference_code_modified": False,
        "change": "Compare Decimal account totals by exact numeric equality; all nonnumeric checks use original JSON hash equality.",
        "reason": "Original scorer audit called a JSON-serialization helper on Decimal ledger totals before reading GT."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
