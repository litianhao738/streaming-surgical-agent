"""Replay archived graph repair and independent Phase decisions without an API.

No images, per-frame ground truth, credentials or ignored artifacts are needed.
Model outputs and cached H0 are evidence inputs, not regenerated predictions.
Metrics verify archived aggregate arithmetic, not an independent GT rescore.
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
    require,
    same,
    score_counts,
    verify_sources,
)
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.parallel_phase import run_parallel_repair
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    phase_choice_error,
)
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)
from surgical_agent.research.verification.review_normalization import normalize_review

PROFILE = "parallel_graph_phase_offline_v1"
RELEASE = "parallel-phase-repair-v1.0.0-experimental"
DEFAULT = ROOT / "docs/experiments/parallel_phase_replay_20260909.json"
ARMS = ("h0", "previous_graph", "previous_final", "graph_only", "final")
IDENTITIES = (
    "VID103_18326", "VID103_33576", "VID23_13176", "VID23_28876",
    "VID31_40701", "VID31_73701", "VID96_15051", "VID96_26051",
)
TRAINING_VIDEOS = {"VID02", "VID04", "VID11", "VID13", "VID17", "VID23", "VID31", "VID37", "VID96", "VID103"}
MODELS = {
    "base": "google/gemini-3.8-flash", "grok": "grok-4.6", "qwen": "qwen3.8-flash",
    "gpt": "openai/gpt-5.6-luna", "gemini": "google/gemini-3.5-flash-lite",
    "deepseek": "deepseek/deepseek-v4-flash-vision-exp",
}
POLICY = {
    "reviewer_seats": list(SEATS), "graph_mean_threshold": 4.0,
    "graph_rounds": 1, "phase_majority": 3, "phase_all_five_valid": True,
    "phase_changes_interaction_heads": False, "causal_offsets_frames": [-50, -25, 0],
    "models": MODELS,
}
REQUIRED_SOURCES = {
    "scripts/replay_parallel_phase_repair.py",
    "scripts/replay_verified_repair_candidate.py",
    "src/surgical_agent/research/verification/phase_extension.py",
    "src/surgical_agent/research/verification/parallel_phase.py",
    "src/surgical_agent/research/verification/prior_panel.py",
    "src/surgical_agent/research/verification/candidate_coordinator.py",
    "src/surgical_agent/research/verification/recent_mean_panel.py",
    "src/surgical_agent/research/verification/review_json_compat.py",
    "src/surgical_agent/research/verification/review_normalization.py",
    "src/surgical_agent/research/verification/semantic_coordinator.py",
    "src/surgical_agent/research/verification/final_only_grounded.py",
    "src/surgical_agent/perception/final_only.py",
    "src/surgical_agent/perception/ontology_prompt.py",
    "src/surgical_agent/perception/prompts/perception_schema_gate_owned_compact.json",
    "src/surgical_agent/research/signals/resources/ivt_components_v1.csv",
}


def answer(record, seat):
    """Replay only the archived content; transport failures stay absent votes."""
    require(set(record) == {"model", "http_status", "finish_reason", "content", "response_sha256"},
            "invalid minimal response envelope")
    require(record["model"] == MODELS[seat], "response model/seat mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", record["response_sha256"] or ""), "invalid response provenance hash")
    status = record["http_status"]
    require(type(status) is int and 100 <= status <= 599, "invalid HTTP status")
    if status != 200:
        require(record["content"] is None and record["finish_reason"] is None,
                "failed request must not supply a model vote")
        return None
    require(record["finish_reason"] == "stop" and isinstance(record["content"], str),
            "incomplete model output")
    if seat == "base":
        return json.loads(record["content"])
    value, _ = parse_review_json_compatible(record["content"])
    return value


def phase_merge(current, record):
    require(set(record) == {"responses", "errors", "decision", "prediction"}, "invalid Phase archive")
    require(set(record["responses"]) == set(SEATS), "five distinct Phase seats required")
    raw = {seat: answer(record["responses"][seat], seat) for seat in SEATS}
    errors = {seat: phase_choice_error(raw[seat], 3) for seat in SEATS}
    same(errors, record["errors"], "Phase validation diagnostics")
    prediction, decision = apply_phase_choices(current, raw, 3)
    same(prediction, record["prediction"], "Phase merged prediction")
    same(decision, record["decision"], "Phase majority decision")
    for task in TASKS[:-1]:
        same(prediction[task], current[task], "Phase must preserve " + task)
    return prediction, decision


def replay(bundle, *, root=ROOT):
    require(set(bundle) == {"payload", "payload_sha256"}, "invalid bundle envelope")
    p = bundle["payload"]
    same(digest(p), bundle["payload_sha256"], "payload checksum")
    require(set(p) == {"profile", "release", "schema_version", "policy", "limitations", "provenance",
                       "replay_source_sha256_lf", "targets", "scores"}, "unexpected evidence fields")
    require(p["profile"] == PROFILE and p["schema_version"] == 1, "unsupported replay profile")
    require(p["release"] == RELEASE, "unsupported release identity")
    same(p["policy"], POLICY, "frozen policy")
    require(REQUIRED_SOURCES <= set(p["replay_source_sha256_lf"]), "missing required source hashes")
    verify_sources(p["replay_source_sha256_lf"], root)
    require(isinstance(p["targets"], list) and len(p["targets"]) == 8, "exactly eight targets required")
    same([t["key"] for t in p["targets"]], list(IDENTITIES), "fixed target identities")
    outputs, totals = [], {arm: {task: 0 for task in TASKS} for arm in ARMS}
    for target in p["targets"]:
        require(set(target) == {"key", "video_id", "frame_id", "split", "causal_frame_ids",
                               "image_sha256", "h0", "previous_graph", "previous_phase", "graph", "phase"},
                "unexpected target fields; no per-frame GT is published")
        frame, video, key = target["frame_id"], target["video_id"], target["key"]
        require(type(frame) is int and key == f"{video}_{frame}" and target["split"] == "Training",
                "invalid target binding or split")
        same(target["causal_frame_ids"], [frame - 50, frame - 25, frame], "causal three-frame window")
        hashes = target["image_sha256"]
        require(isinstance(hashes, list) and len(hashes) == 3 and
                all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes),
                "three original image hashes required")
        h0 = labels(target["h0"])
        same(h0, target["h0"], "canonical cached H0")
        graph = target["graph"]
        require(set(graph) == {"hints", "proposal_response", "pool", "responses", "means",
                              "format_errors", "prediction", "status"}, "invalid graph archive")
        hint_audit = graph["hints"]["audit"]
        require(hint_audit["excluded_video"] == video and video not in hint_audit["fit_videos"],
                "query video leaked into archived graph hint sources")
        same(sorted(hint_audit["fit_videos"]), sorted(TRAINING_VIDEOS - {video}), "nine other Training hint sources")
        require(hint_audit["prior_is_not_visual_evidence"] is True and len(hint_audit["selected"]) <= 2,
                "invalid graph hint contract")
        proposal = answer(graph["proposal_response"], "base")
        require(proposal is not None, "published graph proposal is unavailable")
        pool = make_pool(h0, proposal, make_pool(h0))
        same(pool, graph["pool"], "candidate pool rebuilt from cached H0 and proposal")
        require(set(graph["responses"]) == set(SEATS), "five distinct graph seats required")
        reviews, format_errors = {}, {}
        for seat in SEATS:
            raw = answer(graph["responses"][seat], seat)
            reviews[seat], diagnostic = normalize_review(raw, pool, seat=seat, image_count=3)
            format_errors[seat] = diagnostic["errors"]
        same(format_errors, graph["format_errors"], "graph normalization rejection reasons")
        means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
        same(means, graph["means"], "five-seat candidate means")
        graph_prediction = panel.select(h0, pool, means, threshold=4.0)
        issues = panel.unresolved(graph_prediction, pool, means, diagnostics, threshold=4.0)
        same("UNRESOLVED" if issues else "MODEL_PASS", graph["status"], "graph round status")
        same(graph_prediction, graph["prediction"], "graph local repair")
        same(graph_prediction["phase"], h0["phase"], "graph must preserve cached Phase")
        final, phase_decision = phase_merge(graph_prediction, target["phase"])

        def cached_interaction(private_h0, h0=h0, graph_prediction=graph_prediction):
            same(private_h0, h0, "parallel interaction receives original H0")
            return {"prediction": graph_prediction}

        def cached_phase(record=target["phase"]):
            return {"raw_reviews": {seat: answer(record["responses"][seat], seat) for seat in SEATS}}

        joined = run_parallel_repair(h0, cached_interaction, cached_phase, image_count=3)
        same(joined["final"], final, "production parallel merge final")
        same(joined["phase_decision"], phase_decision, "production parallel majority decision")
        previous_graph = labels(target["previous_graph"])
        same(previous_graph["phase"], h0["phase"], "previous graph Phase")
        previous_final, previous_decision = phase_merge(previous_graph, target["previous_phase"])
        predictions = dict(zip(ARMS, (h0, previous_graph, previous_final, graph_prediction, final), strict=True))
        for arm, prediction in predictions.items():
            for task in TASKS:
                totals[arm][task] += len(prediction[task])
        outputs.append({"key": key, "final": final, "previous_final": previous_final,
                        "phase_decision": phase_decision, "previous_phase_decision": previous_decision})

    require(set(p["scores"]) == set(ARMS), "missing score arm")
    scores = {}
    for arm in ARMS:
        require(set(p["scores"][arm]) == set(TASKS), "missing score head")
        scores[arm] = {}
        for task, item in p["scores"][arm].items():
            require(set(item) == {"counts", "metrics"}, "invalid aggregate score envelope")
            counts = item["counts"]
            computed = score_counts(counts)
            require(counts["valid_targets"] == 8, "aggregate task mask count differs")
            require(counts["tp"] + counts["fp"] == totals[arm][task], "prediction total differs from replay")
            h0_counts = p["scores"]["h0"][task]["counts"]
            require(counts["tp"] + counts["fn"] == h0_counts["tp"] + h0_counts["fn"],
                    "GT aggregate count differs between arms")
            same(computed, item["metrics"], "archived metric arithmetic " + arm + "/" + task)
            scores[arm][task] = computed
    for source, merged in (("graph_only", "final"), ("previous_graph", "previous_final")):
        for task in TASKS[:-1]:
            same(p["scores"][source][task], p["scores"][merged][task], "unchanged interaction aggregate")
    return {"verified": True, "profile": PROFILE, "release": RELEASE, "api_calls": 0, "targets": 8,
            "graph_panels_replayed": 8, "phase_panels_replayed": 16,
            "production_parallel_merges_replayed": 8,
            "gt_independently_rescored": False, "h0_regenerated": False, "priors_refitted": False,
            "limitations": p["limitations"], "outputs": outputs, "metrics_from_archived_counts": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(replay(json.loads(args.bundle.read_text(encoding="utf-8"))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
