"""Four new Training targets: original issues vs fallible visual feedback.

Shared H0 and first review; at most one more proposal/panel per arm. Identical
second-round review requests share the same responses, including failures.
GT scoring is a separate command invoked only after immutable snapshots exist.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import (
    MODELS,
    PROVIDERS,
    RATES_V2,
    ROUTES,
    review_wire,
)
from scripts.run_repair_revision_trial import RevisionCalls, normalize_five
from scripts.run_repair_revision_trial import frozen_sources as revision_sources
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool

PROFILE = "paired_visual_review_feedback_v1"
ARMS = ("issues_only", "evidence_feedback")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VIDEOS = ("VID103", "VID23", "VID31", "VID96")
ALLOWANCE = {"openrouter_usd": Decimal(2), "xai_usd": Decimal(2), "aliyun_cny": Decimal(1)}
SELECTION_PATH = ROOT / "artifacts/training/gate/final_only_preparation_20260908_v2/pilot_selection.json"
EARLIER_PLANS = (ROOT / "artifacts/preflight/repair_revision_development_20260908_v2/plan.json",
                 ROOT / "artifacts/preflight/repair_revision_confirmation_20260908_v1/plan.json")


def build_feedback(pool, reviews, issues, image_count):
    from surgical_agent.research.verification.review_feedback import (
        build_review_feedback,
    )
    return build_review_feedback(pool, reviews, issues, image_count=image_count)


def fingerprint(body):
    return hashlib.sha256(json.dumps(redact_images(body), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def proposal_wire(base, selected, current, pool, issues, feedback=None):
    body = gemini_proposal(base, selected, current, pool, issues)
    if feedback is not None:
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        packet["review_evidence_feedback"] = deepcopy(feedback)
        body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def select_targets(adapter, selection_path=SELECTION_PATH, earlier_plans=EARLIER_PLANS):
    manifest = read(selection_path)
    if manifest.get("no_gt_label_values_used_for_selection") is not True:
        raise ValueError("selection must use time and GT availability only")
    prior_ids = {(r["video_id"], r["frame_id"]) for path in earlier_plans for r in read(path)["selection"]}
    selected = []
    for video in VIDEOS:
        available = [r for r in manifest["selection"] if r["video_id"] == video]
        row = deepcopy(available[2])  # Frozen zero-based time position, never selected by score.
        if adapter.entries[video].split is not DatasetSplit.TRAINING or (video, row["frame_id"]) in prior_ids:
            raise ValueError("targets must be new Training identities")
        excluded = {available[i]["frame_id"] for i in (1, 6, 3, 8)}
        mask = row["gt_availability_only"]
        if (row["frame_id"] in excluded or set(mask) != set(TASKS)
                or any(type(value) is not bool or not value for value in mask.values())):
            raise ValueError("fixed target overlaps earlier time positions or lacks task masks")
        sample = next(s for s in adapter.iter_inference_video(video) if s.target_frame_id == row["frame_id"])
        if list(sample.causal_frame_ids) != row["causal_frame_ids"]:
            raise ValueError("selected causal window changed")
        for im, path in zip(row["images"], sample.media_refs, strict=True):
            if sha(path) != im["sha256"]:
                raise ValueError("selected image changed")
        row["key"] = f"{video}_{row['frame_id']}"
        row["request_metadata"] = canonical_request_metadata(build_gemini_base(adapter, row)).to_mapping()
        selected.append(row)
    return selected


def frozen_sources():
    return sorted({*revision_sources(), Path(__file__).resolve(),
                   ROOT / "src/surgical_agent/research/verification/review_feedback.py"})


def prepare(output, previous, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    prior, prior_plan = read(previous / "budget.json"), read(previous / "plan.json")
    if not prior["stopped"] or prior_plan.get("stage") != "confirmation":
        raise ValueError("previous must be the closed confirmation ledger")
    parser_version = "complete_document_identical_review_core_duplicates_v1"
    if prior_plan.get("review_json_parser") != parser_version or prior_plan["models"] != MODELS or prior_plan["h0"] != PROPOSER:
        raise ValueError("H0/reviewer/parser versions differ from confirmation")
    selected = select_targets(adapter)
    metadata = {}
    for seat, route in ROUTES.items():
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if endpoint["status"] != 0 or not {"response_format", "reasoning"} <= set(endpoint["supported_parameters"]):
            raise ValueError("unavailable fixed model route")
        if any(Decimal(endpoint["pricing"][field]) > Decimal(rate)
               for field, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("price exceeds reserved rate")
        metadata[seat] = data
    source_paths = frozen_sources()
    if any(not path.is_file() for path in source_paths):
        raise ValueError("all feedback dependencies must exist before freezing")
    plan = {"profile": PROFILE, "created_utc": now(), "selection": selected, "arms": list(ARMS),
            "selection_source": str(SELECTION_PATH), "selection_sha256": sha(SELECTION_PATH),
            "earlier_selection_sha256": {str(path): sha(path) for path in EARLIER_PLANS},
            "selection_rule": "zero-based index 2 of the previously fixed time/mask-only manifest for each of four Training videos",
            "models": MODELS, "h0": PROPOSER, "threshold": 4, "round_cap": 2,
            "review_json_parser": parser_version, "max_calls": len(selected) * 19,
            "stopping": "true shared MODEL_PASS stops both arms; otherwise each gets at most one second round; no extra round for GT score",
            "only_treatment_change": "add review_evidence_feedback to the second-round proposer packet; original issues retained",
            "shared_review_policy": "within target and second round only: all five complete request bodies must match; share original raw replies including failures",
            "gt_policy": "GT values are unavailable to inference; score every selected target only after all paid inference is closed and both snapshots are saved",
            "previous_budget": str(previous), "previous_budget_sha256": sha(previous / "budget.json"),
            "carried_occupied": prior["occupied"], "incremental_allowances": {k: str(v) for k, v in ALLOWANCE.items()},
            "limits": {k: str(Decimal(prior["occupied"][k]) + v) for k, v in ALLOWANCE.items()},
            "rates": RATES_V2, "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
            "primary_success_rule": "Both Target and IVT micro-F1 and exact-set accuracy nondecreasing vs shared H0 and issues-only; fewer total label errors; no extra frames with worsened heads.",
            "scope": "four new time targets within Training videos; development evidence, not Test or independent-surgery generalization"}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    for row in selected:
        save(output / "h0_preflight" / f"{row['key']}.json", redact_images(gemini_h0_wire(build_gemini_base(adapter, row))))
    for path in source_paths:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": [r["key"] for r in selected],
                      "max_calls": plan["max_calls"], "allowances": plan["incremental_allowances"]}), flush=True)


def verify_plan(plan):
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen inference source changed: " + name)
    if sha(Path(plan["selection_source"])) != plan["selection_sha256"]:
        raise ValueError("selection source changed")
    for name, digest in plan["earlier_selection_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("earlier selection changed")
    for row in plan["selection"]:
        for im in row["images"]:
            if sha(Path(im["path"])) != im["sha256"]:
                raise ValueError("selected image changed")


def round_step(calls, output, key, arm_name, number, base, selected, current, pool, issues,
               *, feedback=None, reusable=None):
    record = {"round": number, "before": deepcopy(current), "pool_before": deepcopy(pool),
              "input_issues": deepcopy(issues), "after": deepcopy(current), "pool": deepcopy(pool),
              "status": "NOT_ATTEMPTED", "reviewed": False, "issues": deepcopy(issues), "review_evidence_feedback": feedback}
    prefix = f"{arm_name}"
    proposal = calls.call(key, f"{prefix}_proposal_{number}", "base",
                          proposal_wire(base, selected, current, pool, issues, feedback))
    record["proposal"] = proposal
    try:
        if proposal is None:
            raise ValueError("proposal unavailable")
        expanded = make_pool(current, proposal, pool)
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        record["status"] = "PROPOSAL_FAILED"
        return record, None
    record["pool"] = expanded
    if not expanded["propositions"]:
        record["status"] = "EMPTY_POOL_UNVERIFIED"
        return record, None
    bodies = {seat: review_wire(seat, base, selected, expanded) for seat in SEATS}
    record["review_request_fingerprints"] = {seat: fingerprint(body) for seat, body in bodies.items()}
    reuse = reusable is not None and all(bodies[seat] == reusable["bodies"][seat] for seat in SEATS)
    if reuse:
        raw = deepcopy(reusable["raw"])
        source_record = Path(reusable["record_path"])
        record["shared_review"] = {"source_arm": reusable["arm"], "source_record": str(source_record),
                                   "source_record_sha256": sha(source_record), "full_request_bodies_exactly_equal": True,
                                   "request_fingerprints": record["review_request_fingerprints"],
                                   "responses": reusable["responses"], "includes_original_failures": True}
        review_count = reusable["review_count"]
    else:
        def one(seat):
            return calls.call(key, f"{prefix}_review_{number}", seat, bodies[seat])
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
        review_count = sum(r.get("target") == key and r.get("stage") == f"{prefix}_review_{number}" for r in calls.rows)
    reviews, format_diagnostics = normalize_five(raw, expanded, len(base.images))
    means, diagnostics = panel.aggregate(reviews, expanded, image_count=len(base.images))
    try:
        after = panel.select(current, expanded, means)
        unresolved = panel.unresolved(after, expanded, means, diagnostics)
        status = "UNRESOLVED" if unresolved else "MODEL_PASS"
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        after, unresolved, status = deepcopy(current), deepcopy(issues), "SELECTION_FAILED"
    if review_count < 5:
        status = "INCOMPLETE_REVIEW_DISPATCH"
    record.update(raw_reviews=raw, reviews=reviews, format_diagnostics=format_diagnostics, means=means,
                  diagnostics=diagnostics, after=after, issues=unresolved, status=status,
                  reviewed=review_count == 5, review_calls_observed=review_count, new_review_calls=0 if reuse else review_count)
    response_refs = {}
    if not reuse:
        for row in calls.rows:
            if row.get("target") != key or row.get("stage") != f"{prefix}_review_{number}":
                continue
            folder = output / "calls" / f"{row['index']:03d}_{key}_{row['stage']}_{row['seat']}"
            response_path = folder / "response.json"
            response_refs[row["seat"]] = {"status": row["status"], "path": str(response_path),
                                          "sha256": sha(response_path) if response_path.exists() else None}
    cache = {"arm": arm_name, "bodies": bodies, "raw": raw, "responses": response_refs,
             "review_count": review_count, "record_path": str(output / "targets" / key / f"{arm_name}_round_{number}.json")}
    return record, cache


def snapshot(states):
    return [{"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": s["h0"],
             "shared_first_round": {"final": s["shared"]["current"], "status": s["shared"]["status"],
                                    "reviewed": s["shared"]["reviewed"]},
             "arms": {a: {"final": arm["current"], "status": arm["status"],
                           "round2_attempted": arm["round2_attempted"], "round2_reviewed": arm["round2_reviewed"],
                           "actual_rounds": int(s["shared"]["reviewed"]) + int(arm["round2_reviewed"])}
                      for a, arm in s["arms"].items()}} for s in states.values()]


def write_snapshot(output, states, number):
    save(output / "inference_states.json", states)
    save(output / f"round_{number}_predictions.json", snapshot(states))


def grouped_costs(rows, states):
    shared_stages = {(key, f"{record['shared_review']['source_arm']}_review_2")
                     for key, state in states.items() for arm in state["arms"].values()
                     for record in arm["history"] if "shared_review" in record}
    groups = {}
    for row in rows:
        stage = row["stage"]
        group = ("shared_h0" if stage == "h0" else "shared_first_round" if stage.startswith("shared_")
                 else "shared_round2_review" if (row["target"], stage) in shared_stages
                 else next(a for a in ARMS if stage.startswith(a + "_")))
        counts = groups.setdefault(group, {"calls": 0, "costs": {}})
        counts["calls"] += 1
        account = counts["costs"].setdefault(row["account"], {})
        kind = row["charge_kind"]
        account[kind] = str(Decimal(account.get(kind, "0")) + Decimal(row["charge"]))
    return groups


def execute(output, adapter):
    plan = read(output / "plan.json")
    if plan["profile"] != PROFILE or (output / "execution.lock").exists():
        raise ValueError("wrong profile or paid replay prohibited")
    verify_plan(plan)
    if sha(Path(plan["previous_budget"]) / "budget.json") != plan["previous_budget_sha256"]:
        raise ValueError("previous ledger changed")
    bases = {r["key"]: build_gemini_base(adapter, r) for r in plan["selection"]}
    for row in plan["selection"]:
        if canonical_request_metadata(bases[row["key"]]).to_mapping() != row["request_metadata"]:
            raise ValueError("H0 request differs from preflight")
    with (output / "execution.lock").open("x") as marker:
        marker.write(sha(output / "plan.json"))
    calls = RevisionCalls(output, plan["carried_occupied"], limits={k: Decimal(v) for k, v in plan["limits"].items()},
                          rates=plan["rates"], providers=PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"))
    states = {r["key"]: {"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": None,
                        "shared": {"current": None, "pool": None, "issues": [], "status": "NOT_ATTEMPTED", "reviewed": False, "history": []},
                        "arms": {a: {"current": None, "status": "NOT_ATTEMPTED", "round2_attempted": False,
                                     "round2_reviewed": False, "history": []} for a in ARMS}} for r in plan["selection"]}
    for selected in plan["selection"]:
        key, base = selected["key"], bases[selected["key"]]
        state = states[key]
        if calls.stopped:
            break
        raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
        try:
            validate_final_only(raw)
        except (ApiSchemaError, TypeError, ValueError):
            state["shared"]["status"] = "H0_FAILED"
            for arm in state["arms"].values():
                arm["status"] = "H0_FAILED"
            calls.stopped = True
            break
        state["h0"] = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
        record, _ = round_step(calls, output, key, "shared", 1, base, selected, state["h0"], make_pool(state["h0"]), [])
        state["shared"].update(current=record["after"], pool=record["pool"], issues=record["issues"],
                               status=record["status"], reviewed=record["reviewed"], history=[record])
        save(output / "targets" / key / "shared_round_1.json", record)
        for arm in state["arms"].values():
            arm.update(current=deepcopy(record["after"]), status="SHARED_MODEL_PASS" if record["status"] == "MODEL_PASS"
                       else "AWAITING_ROUND_2" if record["status"] == "UNRESOLVED" else record["status"])
        save(output / "inference_states.json", states)
    write_snapshot(output, states, 1)
    verify_plan(plan)
    for index, selected in enumerate(plan["selection"]):
        if calls.stopped:
            break
        key, base = selected["key"], bases[selected["key"]]
        state = states[key]
        if state["shared"]["status"] != "UNRESOLVED":
            continue
        first = state["shared"]["history"][0]
        feedback_error = None
        try:
            feedback = build_feedback(first["pool"], first["reviews"], first["issues"], len(base.images))
        except (TypeError, ValueError, KeyError) as exc:
            feedback, feedback_error = None, type(exc).__name__
        reusable = None
        for name in ARMS if index % 2 == 0 else tuple(reversed(ARMS)):
            if calls.stopped:
                break
            if name == ARMS[1] and feedback_error is not None:
                state["arms"][name].update(status="FEEDBACK_BUILD_FAILED", feedback_exception_type=feedback_error)
                save(output / "inference_states.json", states)
                continue
            record, cache = round_step(calls, output, key, name, 2, base, selected,
                                       deepcopy(first["after"]), deepcopy(first["pool"]), deepcopy(first["issues"]),
                                       feedback=feedback if name == ARMS[1] else None, reusable=reusable)
            state["arms"][name].update(current=record["after"], status=record["status"], round2_attempted=True,
                                         round2_reviewed=record["reviewed"], history=[record])
            save(output / "targets" / key / f"{name}_round_2.json", record)
            save(output / "inference_states.json", states)
            if reusable is None and cache is not None:
                reusable = cache
            print(json.dumps({"target": key, "arm": name, "status": record["status"],
                              "shared_second_round_review": "shared_review" in record, "calls": len(calls.rows)}), flush=True)
    write_snapshot(output, states, 2)
    verify_plan(plan)
    calls.stopped = True
    calls.persist()
    # All paid inference is closed before any GT labels or scores are loaded.
    for number in (1, 2):
        subprocess.run([sys.executable, str(Path(__file__)), "score", "--output", str(output), "--round", str(number)],
                       cwd=ROOT, check=True)
    complete = all(s["h0"] is not None and all(a["status"] in {"MODEL_PASS", "UNRESOLVED", "SHARED_MODEL_PASS"}
                                                        for a in s["arms"].values()) for s in states.values())
    save(output / "completion.json", {"finished_utc": now(), "completed": complete, "targets": len(states),
         "post_calls": len(calls.rows), "last_snapshot": 2, "cost_groups": grouped_costs(calls.rows, states),
         "gt_scoring_started_after_paid_inference_closed": True,
         "call_statuses": dict(Counter(row["status"] for row in calls.rows)),
         "cost_note": "new calls only; shared H0/R1/R2 counted once; currencies and native/estimated/reserved costs separate",
         "shared_second_round_panels": sum("shared_review" in r for s in states.values() for a in s["arms"].values() for r in a["history"]),
         "actual_rounds": {r["key"]: {a: int(states[r["key"]]["shared"]["reviewed"]) + int(v["round2_reviewed"])
                                      for a, v in states[r["key"]]["arms"].items()} for r in plan["selection"]}})


def score(output, adapter, number):
    if not read(output / "budget.json")["stopped"]:
        raise ValueError("GT scoring requires the paid inference ledger to be closed")
    path = output / f"round_{number}_predictions.json"
    digest, rows = sha(path), read(path)
    reports = {}
    for name in ("shared_first_round", *ARMS):
        data = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"], "h1": None,
                 "final": r[name]["final"] if name == "shared_first_round" else r["arms"][name]["final"]} for r in rows]
        reports[name], truth = score_saved(adapter, data)
        save(output / "scores" / f"round_{number}_{name}_truth.json", truth)
    coverage = {"selected": len(rows), "h0_available": sum(r["h0"] is not None for r in rows),
                "shared_first_round_reviewed": sum(r["shared_first_round"]["reviewed"] for r in rows),
                "round2_attempted": {a: sum(r["arms"][a]["round2_attempted"] for r in rows) for a in ARMS},
                "round2_reviewed": {a: sum(r["arms"][a]["round2_reviewed"] for r in rows) for a in ARMS},
                "statuses": {a: dict(Counter(r["arms"][a]["status"] for r in rows)) for a in ARMS},
                "all_selected_targets_retained": True, "missing_GT_excluded_per_task_mask": True,
                "unattempted_repairs_are_not_negative_gate_labels": True}
    if sha(path) != digest:
        raise ValueError("predictions changed during independent scoring")
    save(output / "scores" / f"round_{number}.json", {"prediction_sha256": digest, "reports": reports, "coverage": coverage})
    print(json.dumps({"round": number, "coverage": coverage, "f1": {a: {t: m["micro_f1"] for t, m in report["arms"]["final"]["tasks"].items()}
                    for a, report in reports.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.previous, adapter)
    elif args.command == "execute":
        execute(args.output, adapter)
    else:
        score(args.output, adapter, args.round)
