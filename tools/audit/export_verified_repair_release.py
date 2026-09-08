"""Export a bounded, GT-free public replay from the completed local experiment.

Only allowlisted model text, inference fields, aggregate scores and hashes leave
the archive. This tool never reads credential files or makes network requests.
"""
import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.replay_verified_repair_candidate import (
    DEFAULT,
    PROFILE,
    SEATS,
    TASKS,
    digest,
    replay,
    require,
    score_counts,
    source_digest,
)

FIELDS = ("round", "before", "pool_before", "input_issues", "review_evidence_feedback",
          "proposal", "pool", "raw_reviews", "reviews", "format_diagnostics", "means",
          "diagnostics", "after", "issues", "status", "reviewed", "review_calls_observed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_output(archive, target, stage, seat):
    matches = list((archive / "calls").glob(f"*_{target}_{stage}_{seat}/response.json"))
    require(len(matches) == 1, "one actual source response required")
    path = matches[0]
    response = read(path)
    require(response["http_status"] == 200, "only completed model responses are in this release")
    choices = response["body"]["choices"]
    require(len(choices) == 1 and choices[0]["finish_reason"] == "stop", "model response incomplete")
    text = choices[0]["message"]["content"]
    require(isinstance(text, str), "expected original model text")
    return text, {"archive_response": path.relative_to(archive).as_posix(), "sha256": sha(path)}


def source_hashes():
    paths = {ROOT / "scripts/replay_verified_repair_candidate.py"}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("surgical_agent") and path:
            resolved = Path(path).resolve()
            if resolved.suffix == ".py" and resolved.is_relative_to(ROOT):
                paths.add(resolved)
    paths.add(ROOT / "src/surgical_agent/research/signals/resources/ivt_components_v1.csv")
    # final-only validation reads this package JSON via importlib.resources.
    paths.add(ROOT / "src/surgical_agent/perception/prompts/perception_schema_gate_owned_compact.json")
    return {path.relative_to(ROOT).as_posix(): source_digest(path)
            for path in sorted(paths) if path.exists()}


def export(archive):
    plan, complete, audit = (read(archive / name) for name in ("plan.json", "completion.json", "independent_audit.json"))
    outcome = read(archive / "independent_outcome_audit.json")
    require(plan["profile"] == "paired_visual_review_feedback_v1" and complete["completed"] is True,
            "wrong or unfinished source experiment")
    require(audit["verified"] is True and audit["completed"] is True and outcome["verified"] is True,
            "source independent audits did not pass")
    require(audit["plan_sha256"] == sha(archive / "plan.json")
            and outcome["state_sha256"] == sha(archive / "inference_states.json")
            and outcome["main_audit_sha256"] == sha(archive / "independent_audit.json"),
            "audited artifact changed")
    states = read(archive / "inference_states.json")
    sources = {name.replace("\\", "/"): value for name, value in plan["source_sha256"].items()}
    targets = []
    for row in plan["selection"]:
        key, state = row["key"], states[row["key"]]
        h0_text, h0_ref = model_output(archive, key, "h0", "base")
        target = {"key": key, "video_id": row["video_id"], "frame_id": row["frame_id"],
                  "split": row["source_split"], "causal_frame_ids": row["causal_frame_ids"],
                  "gt_availability": row["gt_availability_only"], "h0": state["h0"],
                  "h0_model_text": h0_text, "h0_source": h0_ref, "images": [], "rounds": [],
                  "final": state["arms"]["evidence_feedback"]["current"]}
        for i, frame in enumerate(row["causal_frame_ids"]):
            target["images"].append({"index": i, "frame_id": frame, "source_sha256": row["images"][i]["sha256"],
                                     "request_sha256": row["request_metadata"]["images"][i]["sha256"]})
        histories = state["shared"]["history"] + state["arms"]["evidence_feedback"]["history"]
        for old in histories:
            number = old["round"]
            arm = "shared" if number == 1 else "evidence_feedback"
            source_path = archive / "targets" / key / f"{arm}_round_{number}.json"
            require(digest(read(source_path)) == digest(old), "saved history differs from standalone record")
            record = {name: deepcopy(old[name]) for name in FIELDS}
            record["proposal_model_text"], proposal_ref = model_output(archive, key, f"{arm}_proposal_{number}", "base")
            record["reviewer_model_text"], review_refs = {}, {}
            source_arm = old.get("shared_review", {}).get("source_arm", arm)
            for seat in SEATS:
                text, ref = model_output(archive, key, f"{source_arm}_review_{number}", seat)
                record["reviewer_model_text"][seat], review_refs[seat] = text, ref
            record["source"] = {"archive_record": source_path.relative_to(archive).as_posix(),
                                "sha256": sha(source_path), "proposal": proposal_ref, "reviews": review_refs,
                                "review_source_arm": source_arm,
                                "one_response_per_named_seat": True,
                                "shared_with_control_is_not_an_independent_vote": True}
            target["rounds"].append(record)
        targets.append(target)
    report = read(archive / "scores/round_2.json")["reports"]
    score_rows = {"h0": report["evidence_feedback"]["arms"]["h0"]["tasks"]}
    score_rows.update({arm: report[arm]["arms"]["final"]["tasks"]
                       for arm in ("shared_first_round", "issues_only", "evidence_feedback")})
    scores = {}
    for arm, task_scores in score_rows.items():
        scores[arm] = {}
        for task in TASKS:
            original = task_scores[task]
            counts = {name: original[name] for name in ("tp", "fp", "fn", "exact_matches", "valid_targets")}
            metrics = score_counts(counts)
            require(all(metrics[k] == original[k] for k in metrics), "source scorer arithmetic differs")
            scores[arm][task] = {"counts": counts, "metrics": metrics}
    replay_hashes = source_hashes()
    sources = {name: value for name, value in sources.items() if name in replay_hashes}
    payload = {"schema_version": 1, "profile": PROFILE,
               "candidate_status": "experimental_candidate_not_proven_better_than_H0",
               "limitations": ["Archived model-output replay: zero API calls, not a new experiment.",
                               "Images and per-frame GT labels are not included.",
                               "Public replay verifies inference and aggregate arithmetic; it cannot independently rescore GT.",
                               "Local independent GT audit passed before export; hashes identify its source artifacts.",
                               "Four Training targets only; no Test or independent-surgery generalization.",
                               "Final feedback arm equals shared R1; its R2 avoids one control-only error, adds no IVT gain."],
               "policy": {"reviewer_seats": list(SEATS), "threshold": 4.0, "max_rounds": 2,
                          "feedback_arm": "evidence_feedback", "phase_unchanged": True},
               "models": {"h0": plan["selection"][0]["request_metadata"]["requested_model_identifier"],
                          "reviewers": plan["models"]},
               "provenance": {"experiment": archive.name,
                              "artifacts_sha256": {name: sha(archive / name) for name in
                                                   ("plan.json", "completion.json", "inference_states.json",
                                                    "independent_audit.json", "independent_outcome_audit.json", "scores/round_2.json")},
                              "inference_source_sha256_original_bytes": sources,
                              "local_independent_gt_audit_passed": True},
               "replay_source_sha256_lf": replay_hashes, "targets": targets, "scores": scores}
    bundle = {"payload": payload, "payload_sha256": digest(payload)}
    replay(bundle)
    encoded = json.dumps(bundle, ensure_ascii=False)
    require(not re.search(r"(?i)(?:sk-(?:ws-|or-|[a-z0-9]{12})|bearer\s|data:image|[A-Z]:\\|/root/|authorization)", encoded),
            "unsafe material in allowlisted public export")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/preflight/evidence_feedback_four_20260908_v1")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    bundle = export(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exported": str(args.output), "targets": 4, "api_calls": 0,
                      "source_files": len(bundle["payload"]["replay_source_sha256_lf"])}))


if __name__ == "__main__":
    main()
