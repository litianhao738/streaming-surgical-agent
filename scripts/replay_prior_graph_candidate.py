"""Replay the frozen prior-graph candidate trial without data, keys or network.

Rebuilds leave-video-out priors, candidate hints and the five-seat coordinator.
Score checks validate archived aggregate arithmetic and prediction totals;
they do not independently rescore ground truth or regenerate model answers.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.replay_verified_repair_candidate import (
    TASKS,
    digest,
    parse,
    require,
    same,
    score_counts,
    verify_sources,
)
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import fit_prior, labels
from surgical_agent.research.verification.review_normalization import normalize_review

PROFILE = "prior_graph_candidate_offline_v1"
RELEASE = "prior-graph-repair-v1.0.0-experimental"
ARMS = ("control", "prior_graph")
DEFAULT = ROOT / "docs/experiments/prior_graph_candidate_replay_20260908.json"
REQUIRED_SOURCES = {
    "scripts/replay_prior_graph_candidate.py",
    "scripts/replay_verified_repair_candidate.py",
    "src/surgical_agent/research/retrieval/prior_candidates.py",
    "src/surgical_agent/research/verification/prior_panel.py",
    "src/surgical_agent/research/verification/candidate_coordinator.py",
    "src/surgical_agent/research/verification/recent_mean_panel.py",
    "src/surgical_agent/research/verification/review_json_compat.py",
    "src/surgical_agent/research/verification/review_normalization.py",
}


def replay(bundle, *, root=ROOT):
    """Validate one published archive; never call transport or load a dataset."""
    require(set(bundle) == {"payload", "payload_sha256"}, "invalid bundle envelope")
    p = bundle["payload"]
    same(digest(p), bundle["payload_sha256"], "payload checksum")
    require(p.get("profile") == PROFILE and p.get("schema_version") == 1,
            "unsupported release profile")
    require(p.get("release") == RELEASE, "unsupported release identity")
    same(p["policy"], {"arms": list(ARMS), "threshold": 4.0, "max_rounds": 1,
                       "max_hints": 2, "max_new_ivts": 4, "reviewer_seats": list(SEATS),
                       "phase_unchanged": True}, "frozen policy")
    require(REQUIRED_SOURCES <= set(p["replay_source_sha256_lf"]),
            "missing required replay source hashes")
    verify_sources(p["replay_source_sha256_lf"], root)
    require(len(p["targets"]) == 8, "release contains exactly eight targets")
    videos = {target["video_id"] for target in p["targets"]}
    require(len(videos) == 4 and set(p["priors"]) == videos, "four bound query priors required")
    require(videos <= set(p["training_counts"]), "query identities absent from count archive")
    for video in sorted(videos):
        rebuilt = fit_prior(p["training_counts"], video)
        same(rebuilt, p["priors"][video], "leave-video-out prior " + video)
        require(video == rebuilt["excluded_video"] and video not in rebuilt["fit_videos"],
                "query video leaked into prior")

    prediction_totals = {arm: {task: 0 for task in TASKS} for arm in ("h0", *ARMS)}
    seen, outputs, shared_panels = set(), [], 0
    for target in p["targets"]:
        key, frame, video = target["key"], target["frame_id"], target["video_id"]
        require(type(frame) is int and frame >= 51 and target["split"] == "Training",
                "invalid target identity")
        require(key == f"{video}_{frame}" and key not in seen, "duplicate or wrong target binding")
        seen.add(key)
        same(target["causal_frame_ids"], [frame - 50, frame - 25, frame], "causal window")
        hashes = target["image_sha256"]
        require(isinstance(hashes, list) and len(hashes) == 3
                and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes),
                "three source image hashes required")
        h0 = labels(parse(target["h0_model_text"]))
        same(h0, target["h0"], "H0 labels")
        hints = retrieve_candidate_hints(h0, p["priors"][video], video_id=video)
        same(hints, target["hints"], "retrieved prior hints")
        require(len(hints["audit"]["selected"]) <= 2, "hint cap exceeded")
        for task in TASKS:
            prediction_totals["h0"][task] += len(h0[task])
        require(set(target["arms"]) == set(ARMS), "missing comparison arm")
        reconstructed = {}
        for arm in ARMS:
            record = target["arms"][arm]
            proposal = parse(record["proposal_model_text"])
            same(proposal, record["proposal"], "parsed proposal " + arm)
            pool = make_pool(h0, proposal, make_pool(h0))
            same(pool, record["pool"], "expanded candidate pool " + arm)
            require(set(record["reviewer_model_text"]) == set(SEATS)
                    and set(record["raw"]) == set(SEATS), "five distinct named seats required")
            reviews, formatting = {}, {}
            for seat in SEATS:
                value = parse(record["reviewer_model_text"][seat])
                same(value, record["raw"][seat], "parsed review " + arm + "/" + seat)
                reviews[seat], formatting[seat] = normalize_review(value, pool, seat=seat, image_count=3)
            same(reviews, record["reviews"], "normalized reviews " + arm)
            same(formatting, record["format_diagnostics"], "format diagnostics " + arm)
            means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
            same(means, record["means"], "candidate means " + arm)
            same(diagnostics, record["diagnostics"], "candidate diagnostics " + arm)
            prediction = panel.select(h0, pool, means, threshold=4.0)
            issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4.0)
            status = "UNRESOLVED" if issues else "MODEL_PASS"
            same(prediction, record["prediction"], "repaired prediction " + arm)
            same(issues, record["issues"], "unresolved items " + arm)
            same(status, record["status"], "round status " + arm)
            same(prediction["phase"], h0["phase"], "unchanged phase " + arm)
            reconstructed[arm] = {"prediction": prediction, "status": status}
            for task in TASKS:
                prediction_totals[arm][task] += len(prediction[task])
        for arm in ARMS:
            record = target["arms"][arm]
            source = record.get("shared_from")
            if source is None:
                continue
            require(source in ARMS and source != arm, "invalid shared panel source")
            origin = target["arms"][source]
            require(origin.get("shared_from") is None, "cyclic shared panel source")
            same(record["pool"], origin["pool"], "shared panel pool")
            same(record["raw"], origin["raw"], "shared panel raw answers")
            same(record["reviewer_model_text"], origin["reviewer_model_text"],
                 "shared panel model text")
            shared_panels += 1
        outputs.append({"key": key, "hints": hints["audit"]["selected"], "arms": reconstructed})

    require(set(p["scores"]) == {"h0", *ARMS}, "missing score arm")
    scores = {}
    for arm, row in p["scores"].items():
        require(set(row) == set(TASKS), "missing score head")
        scores[arm] = {}
        for task, values in row.items():
            counts = values["counts"]
            computed = score_counts(counts)
            require(counts["valid_targets"] == 8, "aggregate mask count differs")
            require(counts["tp"] + counts["fp"] == prediction_totals[arm][task],
                    "aggregate prediction count differs from replay")
            h0_counts = p["scores"]["h0"][task]["counts"]
            require(counts["tp"] + counts["fn"] == h0_counts["tp"] + h0_counts["fn"],
                    "aggregate GT count differs between arms")
            same(computed, values["metrics"], "metric arithmetic " + arm + "/" + task)
            scores[arm][task] = computed
        same(row["phase"], p["scores"]["h0"]["phase"], "unchanged phase aggregate")
    return {"verified": True, "release": RELEASE, "api_calls": 0, "targets": 8,
            "priors_rebuilt": len(videos), "hints_rebuilt": 8, "replayed_panels": 16,
            "shared_panels": shared_panels, "gt_independently_rescored": False,
            "scope": p["limitations"], "outputs": outputs,
            "metrics_from_archived_counts": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = replay(json.loads(args.bundle.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
