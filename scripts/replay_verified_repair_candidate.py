"""Replay published model outputs offline; no images, GT, credentials or API calls.

This validates the coordinator trajectory and arithmetic of archived aggregate
counts. It does not regenerate model answers or independently rescore GT.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.review_feedback import build_review_feedback
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)
from surgical_agent.research.verification.review_normalization import normalize_review

PROFILE = "verified_single_proposer_visual_feedback_offline_v1"
TASKS = ("instrument", "verb", "target", "ivt", "phase")
DEFAULT = ROOT / "docs/experiments/verified_repair_candidate_20260908.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_digest(path):
    """Portable source fingerprint, insensitive only to CRLF versus LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, context):
    # JSON canonical comparison preserves bool versus int and array ordering.
    require(digest(actual) == digest(expected), context + " differs from archive")


def parse(text):
    value, diagnostic = parse_review_json_compatible(text)
    require(value is not None, "archived model document rejected: " + str(diagnostic))
    return value


def score_counts(counts):
    names = {"tp", "fp", "fn", "exact_matches", "valid_targets"}
    require(set(counts) == names and all(type(x) is int and x >= 0 for x in counts.values()),
            "invalid aggregate counts")
    tp, fp, fn, exact, valid = (counts[k] for k in ("tp", "fp", "fn", "exact_matches", "valid_targets"))
    require(exact <= valid, "exact matches exceed valid targets")
    return {"micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "exact_set_accuracy": exact / valid if valid else None}


def verify_sources(hashes, root):
    require(isinstance(hashes, dict) and bool(hashes), "missing replay source hashes")
    for name, expected in hashes.items():
        require(isinstance(name, str) and "\\" not in name and not Path(name).is_absolute(),
                "source must be a portable relative path")
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), "missing source: " + name)
        require(source_digest(path) == expected, "replay source changed: " + name)


def replay(bundle, *, root=ROOT):
    require(set(bundle) == {"payload", "payload_sha256"}, "invalid bundle envelope")
    p = bundle["payload"]
    same(digest(p), bundle["payload_sha256"], "payload checksum")
    require(p.get("profile") == PROFILE and p.get("schema_version") == 1, "unsupported release profile")
    same(p["policy"], {"reviewer_seats": list(SEATS), "threshold": 4.0, "max_rounds": 2,
                       "feedback_arm": "evidence_feedback", "phase_unchanged": True}, "frozen policy")
    verify_sources(p["replay_source_sha256_lf"], root)
    require(len(p["targets"]) == 4, "release contains exactly four targets")
    seen, outputs, rounds = set(), [], 0
    prediction_totals = {arm: {task: 0 for task in TASKS}
                         for arm in ("h0", "shared_first_round", "evidence_feedback")}
    for target in p["targets"]:
        key, frame = target["key"], target["frame_id"]
        require(key == f"{target['video_id']}_{frame}" and key not in seen, "duplicate or wrong target binding")
        seen.add(key)
        require(type(frame) is int and target["split"] == "Training", "invalid target identity")
        same(target["causal_frame_ids"], [frame - 50, frame - 25, frame], "causal window")
        require(len(target["images"]) == 3, "three real image identities required")
        for index, image in enumerate(target["images"]):
            same(image["frame_id"], target["causal_frame_ids"][index], "image/frame binding")
            require(image["index"] == index and re.fullmatch(r"[0-9a-f]{64}", image["source_sha256"])
                    and re.fullmatch(r"[0-9a-f]{64}", image["request_sha256"]), "invalid image hash or index")
        same(target["gt_availability"], {q: True for q in TASKS}, "archived task masks")
        current = labels(parse(target["h0_model_text"]))
        same(current, target["h0"], "H0 labels")
        for task in TASKS:
            prediction_totals["h0"][task] += len(current[task])
        pool, issues, previous_reviews, status = make_pool(current), [], None, None
        require(1 <= len(target["rounds"]) <= 2, "round cap exceeded")
        for number, record in enumerate(target["rounds"], 1):
            require(number == record["round"] and (number == 1 or status == "UNRESOLVED"),
                    "continued after model pass or wrong round order")
            same(record["before"], current, "before state")
            same(record["pool_before"], pool, "previous candidate pool")
            same(record["input_issues"], issues, "proposer issue input")
            feedback = None if number == 1 else build_review_feedback(pool, previous_reviews, issues, image_count=3)
            same(record["review_evidence_feedback"], feedback, "proposer visual feedback")
            proposal = parse(record["proposal_model_text"])
            same(proposal, record["proposal"], "parsed proposal")
            pool = make_pool(current, proposal, pool)
            same(pool, record["pool"], "expanded candidate pool")
            require(set(record["reviewer_model_text"]) == set(SEATS), "five distinct named seats required")
            reviews, formats = {}, {}
            for seat in SEATS:
                value = parse(record["reviewer_model_text"][seat])
                same(value, record["raw_reviews"][seat], "raw parsed review " + seat)
                reviews[seat], formats[seat] = normalize_review(value, pool, seat=seat, image_count=3)
            same(reviews, record["reviews"], "normalized reviews")
            same(formats, record["format_diagnostics"], "format diagnostics")
            means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
            same(means, record["means"], "candidate means")
            same(diagnostics, record["diagnostics"], "candidate diagnostics")
            current = panel.select(current, pool, means, threshold=4.0)
            issues = panel.unresolved(current, pool, means, diagnostics, threshold=4.0)
            status = "UNRESOLVED" if issues else "MODEL_PASS"
            same(current, record["after"], "repaired state")
            same(issues, record["issues"], "unresolved items")
            same(status, record["status"], "round status")
            require(record["reviewed"] is True and record["review_calls_observed"] == 5,
                    "incomplete archived review panel")
            previous_reviews = reviews
            rounds += 1
            if number == 1:
                for task in TASKS:
                    prediction_totals["shared_first_round"][task] += len(current[task])
        require(len(target["rounds"]) == 2 or status == "MODEL_PASS", "unresolved trajectory truncated")
        same(current, target["final"], "final labels")
        same(current["phase"], target["h0"]["phase"], "unchanged phase")
        for task in TASKS:
            prediction_totals["evidence_feedback"][task] += len(current[task])
        outputs.append({"key": key, "rounds": len(target["rounds"]), "status": status, "final": current})
    scores = {}
    require(set(p["scores"]) == {"h0", "shared_first_round", "issues_only", "evidence_feedback"},
            "missing score arm")
    for arm, row in p["scores"].items():
        require(set(row) == set(TASKS), "missing score head")
        scores[arm] = {}
        for task, values in row.items():
            computed = score_counts(values["counts"])
            require(values["counts"]["valid_targets"] == 4, "aggregate mask count differs")
            if arm in prediction_totals:
                require(values["counts"]["tp"] + values["counts"]["fp"] == prediction_totals[arm][task],
                        "aggregate prediction count differs from replay")
            same(computed, values["metrics"], "metric arithmetic " + arm + "/" + task)
            scores[arm][task] = computed
    return {"verified": True, "api_calls": 0, "targets": 4, "replayed_rounds": rounds,
            "gt_independently_rescored": False, "scope": p["limitations"],
            "outputs": outputs, "metrics_from_archived_counts": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(replay(json.loads(args.bundle.read_text(encoding="utf-8"))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
