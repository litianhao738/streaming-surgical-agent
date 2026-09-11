"""Independently audit and score a closed fixed-pool quorum/recheck factorial.

All policies retain all eight archived Training identities. GT is loaded only
after immutable call, source, queue, response and prediction replay checks pass.
The archived-feedback quorum arm is an offline policy replay, not a new rollout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import score_prior_feedback_continuation as archived
from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2
from scripts.run_repair_revision_trial import normalize_five
from scripts.run_semantic_candidate_trial import PROPOSER
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import flexible_quorum as quorum
from surgical_agent.research.verification import recent_mean_panel as original_panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.semantic_coordinator import item_error

PROFILE = "quorum_direct_recheck_v1"
TASKS = archived.TASKS
POLICIES = {"q5": 5, "q4": 4}
VERSIONS = ("h0", "graph_r1", "graph_feedback_r2", "quorum4_r1",
            "archived_feedback_quorum4", "direct_recheck_5", "direct_recheck_4")
PAIRS = tuple(dict.fromkeys([
    *(("h0", v) for v in VERSIONS[1:]),
    *(("graph_r1", v) for v in VERSIONS[3:]),
    *(("graph_feedback_r2", v) for v in VERSIONS[4:]),
    ("quorum4_r1", "direct_recheck_4"), ("direct_recheck_5", "direct_recheck_4"),
    ("archived_feedback_quorum4", "direct_recheck_4")]))
FALLBACK = {"NOT_ATTEMPTED", "BUDGET_STOPPED", "INCOMPLETE_PANEL", "SELECTION_FAILED"}


def validate(output):
    output = Path(output)
    plan, completion, ledger = (read(output / f"{n}.json") for n in ("plan", "completion", "budget"))
    if not completion.get("closed_utc") or ledger.get("stopped") is not True:
        raise ValueError("scoring requires closed inference")
    for name in ("plan", "predictions", "budget", "initial_state", "policy_state"):
        if sha(output / f"{name}.json") != completion.get(f"{name}_sha256"):
            raise ValueError("closed snapshot changed: " + name)
    if (plan.get("profile") != PROFILE or plan.get("threshold") != 4 or plan.get("round_cap") != 2
            or plan.get("models") != MODELS or plan.get("h0") != PROPOSER or plan.get("policies") != POLICIES
            or plan.get("rates") != json.loads(json.dumps(RATES_V2))
            or plan.get("limits") != {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}):
        raise ValueError("frozen quorum/recheck policy or models differ")
    if any(c.get("status") == "DISPATCHED" for c in ledger["calls"]):
        raise ValueError("outstanding API call")
    if len(ledger["calls"]) > plan["max_calls"]:
        raise ValueError("frozen call cap exceeded")
    if any(c["stage"] != "direct_review_2" or c["seat"] not in SEATS for c in ledger["calls"]):
        raise ValueError("unexpected H0/proposer/retry call in direct recheck")
    archived._hashes(ROOT, plan["source_sha256"], "frozen source")
    source = Path(plan["source_root"])
    archived._hashes(source, plan["source_snapshot_sha256"], "archived experiment")
    artifacts = completion["inference_artifact_sha256"]
    archived._hashes(output, artifacts, "closed inference artifact")
    required = {p.relative_to(output).as_posix() for folder in ("calls", "targets")
                for p in (output / folder).rglob("*.json")}
    if not required <= {Path(p).as_posix() for p in artifacts}:
        raise ValueError("closed manifest omits a new inference artifact")
    source_required = {"plan.json", "completion.json", "predictions.json", "budget.json", "initial_state.json"}
    source_required.update(p.relative_to(source).as_posix() for folder in ("calls", "targets")
                           for p in (source / folder).rglob("*.json"))
    if not source_required <= {Path(p).as_posix() for p in plan["source_snapshot_sha256"]}:
        raise ValueError("source manifest omits an archived inference artifact")
    source_plan, source_rows, source_initial, source_ledger = archived.validate(source)
    _, source_audit = archived.audit_records(source, source_plan, source_rows, source_initial, source_ledger)
    initial = read(output / "initial_state.json")["targets"]
    if initial != source_initial:
        raise ValueError("initial state differs from the exact archived source")
    rows = read(output / "predictions.json")["targets"]
    prepared = read(output / "policy_state.json")["targets"]
    identities = archived._identity(plan["selection"])
    if (len(identities) != 8 or len(set(identities)) != 8
            or len({key for key, _, _ in identities}) != 8
            or any(archived._identity(group) != identities for group in (source_rows, initial, rows))
            or [r["key"] for r in prepared] != [r["key"] for r in rows]
            or archived._identity(source_plan["selection"]) != identities):
        raise ValueError("all eight original target identities must remain in order")
    for row, first, old in zip(rows, initial, source_rows, strict=True):
        if (row["h0"] != first["h0"] or row["graph_r1"] != first["graph_r1"]["prediction"]
                or row["graph_feedback_r2"] != old["arms"]["evidence_feedback"]["prediction"]):
            raise ValueError("inherited baseline prediction changed")
        for version in VERSIONS:
            archived._labels(row[version])
            if row[version]["phase"] != row["h0"]["phase"]:
                raise ValueError("repair changed Phase")
    return plan, rows, initial, prepared, ledger, source_audit


def checked_aggregate(reviews, pool, minimum_valid, image_count):
    """Cross-check core aggregation against direct valid-evidence arithmetic."""
    means, diagnostics = quorum.aggregate(reviews, pool, minimum_valid=minimum_valid, image_count=image_count)
    expected_queue_inputs = {}
    for proposition in pool["propositions"]:
        pid, task = proposition["id"], proposition["task"]
        valid, invalid = {}, {}
        for seat in SEATS:
            item = reviews[seat]["judgments"].get(pid)
            reason = item_error(item, task, image_count)
            if reason:
                invalid[seat] = reason
            else:
                valid[seat] = item["rating"]
        expected = sum(valid.values()) / len(valid) if len(valid) >= minimum_valid else None
        if means[pid] != expected:
            raise ValueError("quorum average contains invalid evidence or has wrong denominator")
        expected_diagnostic = {"scores": [valid.get(seat) for seat in SEATS], "invalid": invalid,
            "valid_count": len(valid), "valid_seats": [s for s in SEATS if s in valid],
            "explicit_conflict": bool(valid and min(valid.values()) <= 2 and max(valid.values()) >= 4)}
        if diagnostics[pid] != expected_diagnostic:
            raise ValueError("quorum valid-seat diagnostics disagree with actual item validation")
        expected_queue_inputs[pid] = {"valid": valid, "invalid": invalid,
            "explicit_conflict": bool(valid and min(valid.values()) <= 2 and max(valid.values()) >= 4)}
    return means, diagnostics, expected_queue_inputs


def checked_queue(current, pool, means, diagnostics, clean):
    queue = quorum.recheck_queue(current, pool, means, diagnostics)
    expected = {p["id"] for p in pool["propositions"] if
                (p["label_id"] in current[p["task"]] and (means[p["id"]] is None or means[p["id"]] < 4))
                or clean[p["id"]]["explicit_conflict"]}
    actual = [r["candidate_id"] for r in queue]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("recheck queue omits or fabricates a low-support/conflicting proposition")
    return queue


def replay_initial(first, saved, image_count):
    old = first["graph_r1"]
    pool = make_pool(first["h0"], old["proposal"], make_pool(first["h0"]))
    reviews, formatting = normalize_five(old["raw"], pool, image_count)
    if reviews != old["reviews"] or formatting != old["format_diagnostics"] or pool != old["pool"]:
        raise ValueError("original graph pool/review replay differs")
    states = {}
    for name, minimum in POLICIES.items():
        means, diagnostics, clean = checked_aggregate(reviews, pool, minimum, image_count)
        prediction = original_panel.select(first["h0"], pool, means, threshold=4)
        queue = checked_queue(prediction, pool, means, diagnostics, clean)
        states[name] = {"minimum_valid": minimum, "prediction": prediction, "means": means,
                        "diagnostics": diagnostics, "queue": queue}
        if states[name] != saved["policies"][name]:
            raise ValueError("prepared first-round quorum state differs from raw evidence")
    if states["q5"]["prediction"] != old["prediction"] or states["q5"]["means"] != old["means"]:
        raise ValueError("quorum five did not reproduce the original baseline")
    return pool, states


def archived_feedback_replay(source, first, image_count):
    """Isolated q4 replay from original graph R1, never an adaptive new rollout."""
    record = read(source / "targets" / first["key"] / "evidence_feedback.json")
    prediction = deepcopy(first["graph_r1"]["prediction"])
    trace = {"applied": False, "means": None, "diagnostics": None,
             "pool": record["pool"], "before": deepcopy(prediction), "status": record["status"]}
    if record["reviewed"] and record["review_calls_observed"] == 5 and record["status"] in {"UNRESOLVED", "MODEL_PASS"}:
        reviews, _ = normalize_five(record["raw"], record["pool"], image_count)
        means, diagnostics, _ = checked_aggregate(reviews, record["pool"], 4, image_count)
        try:
            prediction = original_panel.select(prediction, record["pool"], means, threshold=4)
        except (ApiSchemaError, TypeError, ValueError, KeyError):
            trace["status"] = "SELECTION_FAILED"
        else:
            trace.update(applied=True, means=means, diagnostics=diagnostics)
    trace["prediction"] = prediction
    return prediction, trace


def audit_records(output, plan, rows, initial, prepared, ledger):
    output = Path(output)
    records, traces, audit = {}, {}, Counter()
    eligible_targets = []
    for selected, row, first, ready in zip(plan["selection"], rows, initial, prepared, strict=True):
        key, image_count = row["key"], len(selected["causal_frame_ids"])
        pool, states = replay_initial(first, ready, image_count)
        if row["quorum4_r1"] != states["q4"]["prediction"]:
            raise ValueError("quorum four first-round prediction differs")
        archived_prediction, archived_trace = archived_feedback_replay(Path(plan["source_root"]), first, image_count)
        if row["archived_feedback_quorum4"] != archived_prediction:
            raise ValueError("archived feedback q4 policy replay differs")
        record = read(output / "targets" / key / "direct.json")
        if record["pool"] != pool or set(record["policies"]) != set(POLICIES):
            raise ValueError("direct recheck changed candidate pool or policies")
        records[key] = record
        traces[key] = {"first_round": states, "archived_feedback_quorum4": archived_trace}
        eligible = any(state["queue"] for state in states.values())
        if eligible:
            eligible_targets.append(key)
        calls = [c for c in ledger["calls"] if c["target"] == key]
        if len({c["seat"] for c in calls}) != len(calls) or any(c["seat"] not in SEATS for c in calls):
            raise ValueError("duplicate or unknown reviewer seat in direct recheck")
        count = len(calls)
        if (count != record["review_calls_observed"] or record["reviewed"] != (count == 5)
                or record["attempted"] != bool(count) or (not eligible and calls)):
            raise ValueError("direct panel dispatches disagree with policy eligibility")
        clean_reviews = formatting = None
        if record["raw"] is not None:
            expected_raw = {seat: None for seat in SEATS}
            for call in calls:
                expected_raw[call["seat"]] = archived._parsed_call(output, call)
            if record["raw"] != expected_raw:
                raise ValueError("direct review differs from saved raw model response")
            clean_reviews, formatting = normalize_five(record["raw"], pool, image_count)
            if record["reviews"] != clean_reviews or record["format_diagnostics"] != formatting:
                raise ValueError("direct review normalization differs")
        elif calls:
            raise ValueError("dispatched review requires an explicit five-seat raw result")
        for name, minimum in POLICIES.items():
            old, final = states[name], record["policies"][name]
            if (final["before"] != old["prediction"] or final["queue"] != old["queue"]
                    or final["eligible"] != bool(old["queue"])):
                raise ValueError("policy starting point or queue differs")
            prediction, queue_after = old["prediction"], old["queue"]
            expected_means, expected_diagnostics = old["means"], old["diagnostics"]
            if not old["queue"]:
                expected_status = "NO_RECHECK_NEEDED"
            elif count < 5:
                expected_status = "INCOMPLETE_PANEL" if record["raw"] is not None else final["status"]
                if record["raw"] is None and expected_status not in {"NOT_ATTEMPTED", "BUDGET_STOPPED"}:
                    raise ValueError("unattempted eligible policy claims a completed review")
            else:
                means, diagnostics, clean = checked_aggregate(clean_reviews, pool, minimum, image_count)
                try:
                    prediction = original_panel.select(old["prediction"], pool, means, threshold=4)
                    queue_after = quorum.recheck_queue(prediction, pool, means, diagnostics)
                except (ApiSchemaError, TypeError, ValueError, KeyError):
                    prediction, queue_after, expected_status = old["prediction"], old["queue"], "SELECTION_FAILED"
                else:
                    checked_queue(prediction, pool, means, diagnostics, clean)
                    expected_means, expected_diagnostics = means, diagnostics
                    expected_status = "REVIEWED_PENDING" if queue_after else "REVIEWED_NO_PENDING"
            if final["means"] != expected_means or final["diagnostics"] != expected_diagnostics:
                raise ValueError("direct quorum arithmetic, fallback or diagnostics differ")
            version = "direct_recheck_" + str(minimum)
            if (final["prediction"] != prediction or row[version] != prediction
                    or final["queue_after"] != queue_after or final["status"] != expected_status):
                raise ValueError("direct selection, fallback or stopping replay differs")
            audit["direct_policy_states_replayed"] += 1
        audit["initial_policy_states_replayed"] += 2
        audit["archived_feedback_policies_replayed"] += 1
    if plan["max_calls"] != len(eligible_targets) * 5:
        raise ValueError("call cap differs from the union of eligible target queues")
    if plan.get("eligible_targets", eligible_targets) != eligible_targets:
        raise ValueError("planned eligibility differs from raw first-round queues")
    return records, traces, dict(audit)


def audit_wires(output, plan, initial, records, ledger, adapter):
    from scripts.check_candidate_panel_providers import redact_images
    from scripts.run_evidence_feedback_trial import fingerprint
    from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
    from scripts.run_recent_mean_panel_trial import review_wire

    verified = 0
    for selected, first in zip(plan["selection"], initial, strict=True):
        key = first["key"]
        record = records[key]
        if not record["attempted"]:
            continue
        base = build_gemini_base(adapter, selected)
        old = first["graph_r1"]
        pool = make_pool(first["h0"], old["proposal"], make_pool(first["h0"]))
        bodies = {s: review_wire(s, base, selected, pool) for s in SEATS}
        if record["request_fingerprints"] != {s: fingerprint(b) for s, b in bodies.items()}:
            raise ValueError("direct wire fingerprints differ from the original clean pool review")
        for call in (c for c in ledger["calls"] if c["target"] == key):
            body = bodies[call["seat"]]
            folder = output / "calls" / f"{call['index']:03d}_{key}_{call['stage']}_{call['seat']}"
            if read(folder / "request.json") != redact_images(body) or read(folder / "record.json") != call:
                raise ValueError("actual direct request or ledger binding differs")
            if call["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
                response = read(folder / "response.json")["body"]
                provider = PROVIDERS.get(body["model"], PROVIDERS.get(call["seat"]))
                if response.get("model") != body["model"] or (provider is not None and response.get("provider") != provider):
                    raise ValueError("response model or provider differs")
            verified += 1
    return {"actual_complete_requests_verified": verified, "new_proposer_or_h0_calls": 0}


def summarize(rows, truth, records, initial, traces):
    truth_by_id = {(r["video_id"], r["frame_id"]): r for r in truth}
    initial_by_key = {r["key"]: r for r in initial}
    metrics = {}
    for version in VERSIONS:
        data = [{**truth_by_id[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None,
                 "final": r[version]} for r in rows]
        metrics[version] = compute_repair_comparison(data)["arms"]["final"]
    details, changes = [], []
    coverage = {v: {t: Counter(dict.fromkeys(("valid_targets", "gt_positive", "pool_true", "pool_false",
                "pool_size", "selected_true", "missing_from_pool", "in_pool_not_selected"), 0))
                for t in TASKS[:4]} for v in VERSIONS[1:]}
    for row in rows:
        key = row["key"]
        truth_row = truth_by_id[row["video_id"], row["frame_id"]]
        details.append({"key": key, "comparisons": {f"{a}_to_{b}":
            frame_delta(row[a], row[b], truth_row["gt"], truth_row["mask"]) for a, b in PAIRS}})
        for version in VERSIONS[1:]:
            pool = (traces[key]["archived_feedback_quorum4"]["pool"] if version in
                    {"archived_feedback_quorum4", "graph_feedback_r2"} else initial_by_key[key]["graph_r1"]["pool"])
            for task, counts in coverage[version].items():
                if not truth_row["mask"][task]:
                    continue
                expected, selected = set(truth_row["gt"][task]), set(row[version][task])
                candidates = {p["label_id"] for p in pool["propositions"] if p["task"] == task}
                counts.update(valid_targets=1, gt_positive=len(expected), pool_true=len(expected & candidates),
                    pool_false=len(candidates - expected), pool_size=len(candidates), selected_true=len(selected & expected),
                    missing_from_pool=len(expected - candidates), in_pool_not_selected=len((expected & candidates) - selected))
        contexts = [("graph_r1", "quorum4_r1", traces[key]["first_round"]["q4"]),
                    ("graph_feedback_r2", "archived_feedback_quorum4", traces[key]["archived_feedback_quorum4"]),
                    ("graph_r1", "direct_recheck_5", records[key]["policies"]["q5"]),
                    ("quorum4_r1", "direct_recheck_4", records[key]["policies"]["q4"]),
                    ("direct_recheck_5", "direct_recheck_4", records[key]["policies"]["q4"])]
        for before, after, context in contexts:
            for task in TASKS[:4]:
                if not truth_row["mask"][task]:
                    continue
                for label in sorted(set(row[before][task]) ^ set(row[after][task])):
                    pid = f"{task}_{label}"
                    changes.append({"key": key, "comparison": f"{before}_to_{after}", "candidate_id": pid,
                        "operation": "ADD" if label in row[after][task] else "DELETE", "in_gt": label in truth_row["gt"][task],
                        "mean": (context.get("means") or {}).get(pid),
                        "diagnostic": (context.get("diagnostics") or {}).get(pid),
                        "note": "For direct q5-vs-q4, both initial state and eligibility can differ; this is not automatically a four-vote-only causal change."})
    for tasks in coverage.values():
        for counts in tasks.values():
            counts["pool_recall"] = counts["pool_true"] / counts["gt_positive"] if counts["gt_positive"] else None
    return {"metrics": metrics, "candidate_coverage": coverage, "comparisons": {f"{a}_to_{b}":
        summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details]) for a, b in PAIRS}}, details, changes


def runtime(records, ledger, completion):
    totals = defaultdict(lambda: {"calls": 0, "by_charge_kind": defaultdict(Decimal),
                                  "prompt_tokens": 0, "completion_tokens": 0, "statuses": Counter()})
    for call in ledger["calls"]:
        group = totals[call["account"]]
        group["calls"] += 1
        group["by_charge_kind"][call["charge_kind"]] += Decimal(call["charge"])
        group["statuses"][call["status"]] += 1
        for field in ("prompt_tokens", "completion_tokens"):
            group[field] += call.get("usage", {}).get(field, 0)
    arithmetic = Counter()
    for record in records.values():
        for policy in POLICIES:
            for diagnostic in (record["policies"][policy].get("diagnostics") or {}).values():
                valid = diagnostic.get("valid_count")
                if valid is not None:
                    arithmetic[f"{policy}_valid_{valid}"] += 1
    return {"new_calls": len(ledger["calls"]), "accounts": {a: {**v,
                "by_charge_kind": {k: str(c) for k, c in v["by_charge_kind"].items()},
                "total_including_unknown_reserved": str(sum(v["by_charge_kind"].values(), Decimal(0))),
                "statuses": dict(v["statuses"])} for a, v in totals.items()},
            "inference_seconds": completion.get("inference_seconds"),
            "panel_seconds": archived._distribution(r.get("panel_seconds") for r in records.values() if r["attempted"]),
            "policy_statuses": {name: dict(Counter(r["policies"][name]["status"] for r in records.values())) for name in POLICIES},
            "per_policy_eligible_targets": {name: sum(r["policies"][name]["eligible"] for r in records.values()) for name in POLICIES},
            "valid_review_count_distribution": dict(arithmetic),
            "note": "Both policies reuse each actual panel, including failures, counted once. Native provider charges, conservative estimates, and unknown reserved upper bounds are separate; reserves are not confirmed spending."}


def score(output, adapter):
    output = Path(output)
    plan, rows, initial, prepared, ledger, old_audit = validate(output)
    records, traces, audit = audit_records(output, plan, rows, initial, prepared, ledger)
    audit.update(audit_wires(output, plan, initial, records, ledger, adapter))
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["graph_r1"]} for r in rows])
    result, details, changes = summarize(rows, truth, records, initial, traces)
    result.update(profile=PROFILE, targets=len(rows), runtime=runtime(records, ledger, read(output / "completion.json")),
                  audit={**audit, "original_archive_replay": old_audit, "immutable_snapshots_verified": True,
                         "valid_evidence_means_independently_checked": True, "gt_opened_only_after_closed_inference": True},
                  limitations=["Same eight previously inspected Training targets; development diagnosis, not independent validation.",
                    "Direct recheck uses the original fixed graph R1 pool and fresh five-seat reviews; no new visual frames or candidate proposals.",
                    "Four valid reviews can admit a mean while the fifth is missing/invalid; invalid and missing evidence never receive fabricated scores.",
                    "Archived feedback q4 is an offline replay starting from the original q5 graph R1, not a q4-adaptive candidate rollout.",
                    "Policies may have different initial labels and eligible queues; final q5-vs-q4 differences combine these effects.",
                    "No pending recheck is a policy stop condition, not verified GT correctness; full Tracker/Gate integration is not measured."])
    validate(output)
    for name, value in (("metrics", result), ("frame_deltas", details), ("scored_truth", truth),
                        ("audit", result["audit"]), ("quorum_change_attribution", changes)):
        save(output / f"{name}.json", value)
    print(json.dumps({"f1": {a: {t: v["micro_f1"] for t, v in m["tasks"].items()} for a, m in result["metrics"].items()},
                      "comparisons": {k: v["categories"] for k, v in result["comparisons"].items()},
                      "runtime": result["runtime"], "audit": result["audit"]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
