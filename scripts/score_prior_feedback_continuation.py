"""Independently replay and task-mask-score the closed eight-target continuation.

No model is called here. The original H0 and both first-round arms are retained;
unattempted, failed and unchanged second rounds remain in all GT denominators.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import read, save
from scripts.run_repair_revision_trial import normalize_five, parse_review_json
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_feedback import build_review_feedback

PROFILE = "prior_graph_feedback_continuation_v1"
ARMS = ("evidence_feedback", "issues_only")
TASKS = tuple(TASK_ID_BOUNDS)
VERSIONS = ("h0", "control_r1", "graph_r1", *ARMS)
PAIRS = (("h0", "control_r1"), ("h0", "graph_r1"),
         *(("h0", a) for a in ARMS), ("control_r1", "graph_r1"),
         *(("graph_r1", a) for a in ARMS), ("issues_only", "evidence_feedback"))
FALLBACK_STATUSES = {"INHERITED_MODEL_PASS", "NO_NEW_CANDIDATES", "PROPOSAL_FAILED",
                     "NOT_ATTEMPTED", "BUDGET_STOPPED", "INCOMPLETE_PANEL", "SELECTION_FAILED"}


def _checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("snapshot path escapes its root")
    return path


def _hashes(root, mapping, description):
    if not mapping:
        raise ValueError(f"missing {description} hashes")
    for name, digest in mapping.items():
        path = _checked_path(root, name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"{description} changed: {name}")


def _identity(rows):
    return [(r["key"], r["video_id"], r["frame_id"]) for r in rows]


def _labels(prediction):
    if not isinstance(prediction, dict) or set(prediction) != set(TASKS):
        raise ValueError("all inherited and continued predictions require five heads")
    for task, (lower, upper) in TASK_ID_BOUNDS.items():
        values = prediction[task]
        if (not isinstance(values, list) or len(values) != len(set(values))
                or any(type(v) is not int or not lower <= v <= upper for v in values)):
            raise ValueError("invalid prediction label type, bounds or duplicates")


def validate(output):
    """Validate every immutable inference artifact before any GT is loaded."""
    output = Path(output)
    plan, done, ledger = (read(output / f"{n}.json") for n in ("plan", "completion", "budget"))
    if not done.get("closed_utc") or ledger.get("stopped") is not True:
        raise ValueError("scoring requires closed inference")
    for name in ("plan", "predictions", "budget", "initial_state"):
        if sha(output / f"{name}.json") != done.get(f"{name}_sha256"):
            raise ValueError("closed snapshot changed: " + name)
    if (plan.get("profile") != PROFILE or plan.get("arms") != list(ARMS)
            or plan.get("threshold") != 4 or plan.get("round_cap") != 2):
        raise ValueError("wrong frozen continuation policy")
    if any(c.get("status") == "DISPATCHED" for c in ledger["calls"]):
        raise ValueError("outstanding API call")
    if len(ledger["calls"]) > plan["max_calls"]:
        raise ValueError("call cap exceeded")
    _hashes(ROOT, plan["source_sha256"], "frozen source")
    source = Path(plan["source_root"])
    _hashes(source, plan["source_snapshot_sha256"], "original experiment")
    artifacts = done["inference_artifact_sha256"]
    _hashes(output, artifacts, "closed inference artifact")
    required = {p.relative_to(output).as_posix() for folder in ("calls", "targets")
                for p in (output / folder).rglob("*.json")}
    if not required <= {Path(p).as_posix() for p in artifacts}:
        raise ValueError("inference closure omits a call or target artifact")
    source_required = {"plan.json", "completion.json", "predictions.json", "budget.json"}
    source_required.update(p.relative_to(source).as_posix() for folder in ("calls", "targets", "priors")
                           for p in (source / folder).rglob("*.json"))
    if not source_required <= {Path(p).as_posix() for p in plan["source_snapshot_sha256"]}:
        raise ValueError("source closure omits an original inference artifact")
    initial = read(output / "initial_state.json")["targets"]
    rows = read(output / "predictions.json")["targets"]
    expected = _identity(plan["selection"])
    original = read(source / "predictions.json")["targets"]
    if (len(expected) != 8 or len(set(expected)) != 8 or len({k for k, _, _ in expected}) != 8
            or len({(v, f) for _, v, f in expected}) != 8
            or any(_identity(group) != expected for group in (initial, rows, original))
            or _identity(read(source / "plan.json")["selection"]) != expected):
        raise ValueError("all eight original planned targets must remain in frozen order")
    for row, first, old in zip(rows, initial, original, strict=True):
        if row["h0"] != first["h0"] or row["h0"] != old["h0"] or set(row["arms"]) != set(ARMS):
            raise ValueError("shared H0 or continuation arms differ")
        for name, old_name in (("control_r1", "control"), ("graph_r1", "prior_graph")):
            source_arm = read(source / "targets" / row["key"] / f"{old_name}.json")
            if (first[name] != source_arm or row[name] != source_arm["prediction"]
                    or row[name] != old["arms"][old_name]["prediction"]):
                raise ValueError("inherited first round changed")
        if first["hints"] != read(source / "targets" / row["key"] / "hints.json"):
            raise ValueError("inherited graph hints changed")
        for value in (row["h0"], row["control_r1"], row["graph_r1"],
                      *(row["arms"][a]["prediction"] for a in ARMS)):
            _labels(value)
            if value["phase"] != row["h0"]["phase"]:
                raise ValueError("phase changed in a repair arm")
    return plan, rows, initial, ledger


def _replay_panel(record, before, image_count, *, allow_selection_failure=False):
    reviews, formatting = normalize_five(record["raw"], record["pool"], image_count)
    means, diagnostics = panel.aggregate(reviews, record["pool"], image_count=image_count)
    for name, value in (("reviews", reviews), ("format_diagnostics", formatting),
                        ("means", means), ("diagnostics", diagnostics)):
        if record[name] != value:
            raise ValueError("panel replay differs: " + name)
    try:
        prediction = panel.select(before, record["pool"], means, threshold=4)
        issues = panel.unresolved(prediction, record["pool"], means, diagnostics, threshold=4)
    except (ApiSchemaError, TypeError, ValueError, KeyError):
        if not allow_selection_failure:
            raise
        return None, None
    return prediction, issues


def _parsed_call(output, call):
    if call["status"] not in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
        return None
    folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
    response = read(folder / "response.json")
    body = response["body"]
    choice = body["choices"][0]
    if (response["http_status"] != 200 or body.get("error") or choice["finish_reason"] != "stop"
            or choice["message"].get("refusal")):
        raise ValueError("accepted raw response is not a complete valid transport")
    text = choice["message"]["content"]
    return json.loads(text) if call["seat"] == "base" else parse_review_json(text)[0]


def audit_records(output, plan, rows, initial, ledger):
    """Rebuild decisions, feedback bindings and sharing from persisted raw data."""
    output = Path(output)
    records, audit = {}, Counter()
    for selected, row, first in zip(plan["selection"], rows, initial, strict=True):
        key, image_count = row["key"], len(selected["causal_frame_ids"])
        records[key] = {}
        for name in ("control_r1", "graph_r1"):
            old = first[name]
            if make_pool(row["h0"], old["proposal"], make_pool(row["h0"])) != old["pool"]:
                raise ValueError("original candidate pool replay differs")
            prediction, issues = _replay_panel(old, row["h0"], image_count)
            if prediction != row[name] or issues != old["issues"]:
                raise ValueError("original repair replay differs")
            audit["first_round_panels_replayed"] += 1
        r1 = first["graph_r1"]
        for arm in ARMS:
            short = row["arms"][arm]
            path = short.get("record", f"targets/{key}/{arm}.json")
            record = read(_checked_path(output, path))
            records[key][arm] = record
            if any(short[k] != record[k] for k in ("prediction", "status", "round2_attempted", "reviewed")):
                raise ValueError("continuation summary differs from full record")
            if (record["before"] != r1["prediction"] or record["pool_before"] != r1["pool"]
                    or record["input_issues"] != r1["issues"]
                    or record["candidate_relation_hints"] != first["hints"]["packet"]):
                raise ValueError("second round does not inherit graph first round")
            if arm == "evidence_feedback" and record["round2_attempted"]:
                expected = build_review_feedback(r1["pool"], r1["reviews"], r1["issues"], image_count=image_count)
                if record.get("review_evidence_feedback") != expected:
                    raise ValueError("feedback is not derived from the original reviewer evidence")
                audit["feedback_packets_rebuilt"] += 1
            elif arm == "issues_only" and record.get("review_evidence_feedback") is not None:
                raise ValueError("issues-only arm received evidence feedback")
            arm_calls = [c for c in ledger["calls"] if c["target"] == key and c["stage"].startswith(arm + "_")]
            proposal_calls = [c for c in arm_calls if "_proposal" in c["stage"]]
            if len(proposal_calls) != int(record["round2_attempted"]):
                raise ValueError("proposal attempt does not match ledger")
            if proposal_calls and _parsed_call(output, proposal_calls[0]) != record.get("proposal"):
                raise ValueError("proposal differs from raw model response")
            if r1["status"] == "MODEL_PASS" and (record["status"] != "INHERITED_MODEL_PASS" or arm_calls):
                raise ValueError("passed first round must not be called again")
            if record["status"] in FALLBACK_STATUSES and record["prediction"] != r1["prediction"]:
                raise ValueError("stopped or incomplete round must retain graph first round")
            if record["status"] in {"NOT_ATTEMPTED", "BUDGET_STOPPED", "INHERITED_MODEL_PASS"}:
                if record["round2_attempted"] or record["reviewed"]:
                    raise ValueError("unattempted round claims paid work")
                continue
            if record["status"] == "PROPOSAL_FAILED":
                continue
            rebuilt = make_pool(r1["prediction"], record["proposal"], r1["pool"])
            if rebuilt != record["pool"]:
                raise ValueError("continued candidate pool replay differs")
            audit["second_round_pools_rebuilt"] += 1
            if rebuilt == r1["pool"]:
                if record["status"] != "NO_NEW_CANDIDATES" or record["reviewed"] or len(arm_calls) != 1:
                    raise ValueError("unchanged pool must stop before a repeated review")
                audit["unchanged_pools_stopped"] += 1
                continue
            review_calls = [c for c in arm_calls if "_review" in c["stage"]]
            shared = record.get("shared_from")
            if shared:
                if shared not in records[key] or shared == arm or review_calls:
                    raise ValueError("invalid or cyclic panel sharing")
                parent = records[key][shared]
                if any(record[k] != parent[k] for k in ("pool", "raw", "review_request_fingerprints")):
                    raise ValueError("shared review request or raw response differs")
                observed = parent["review_calls_observed"]
                audit["panels_shared"] += 1
            else:
                observed = len(review_calls)
                if len({c["seat"] for c in review_calls}) != observed:
                    raise ValueError("duplicate reviewer seat")
                raw = {s: None for s in SEATS}
                for call in review_calls:
                    raw[call["seat"]] = _parsed_call(output, call)
                if raw != record["raw"]:
                    raise ValueError("review differs from original raw response")
            if (observed != record["review_calls_observed"] or record["reviewed"] != (observed == 5)
                    or record["new_review_calls"] != len(review_calls)):
                raise ValueError("review attempts differ from ledger")
            prediction, issues = _replay_panel(record, r1["prediction"], image_count,
                                                allow_selection_failure=record["status"] == "SELECTION_FAILED")
            if observed < 5:
                if record["status"] != "INCOMPLETE_PANEL" or record["issues"] != r1["issues"]:
                    raise ValueError("incomplete panel cannot grant a model pass")
            elif record["status"] == "SELECTION_FAILED":
                if prediction is not None or record["issues"] != r1["issues"]:
                    raise ValueError("claimed selection failure did not replay")
            elif (record["prediction"] != prediction or record["issues"] != issues
                  or record["status"] != ("UNRESOLVED" if issues else "MODEL_PASS")):
                raise ValueError("continued selection or stopping replay differs")
            audit["second_round_panels_replayed"] += 1
    return records, dict(audit)


def audit_wires(output, plan, initial, records, ledger, adapter):
    """Rebuild entire requests, including image bindings, before opening GT."""
    from scripts.check_candidate_panel_providers import redact_images
    from scripts.run_evidence_feedback_trial import fingerprint
    from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
    from scripts.run_prior_feedback_continuation import proposal_wire
    from scripts.run_recent_mean_panel_trial import PROVIDERS, review_wire

    verified = Counter()
    for selected, first in zip(plan["selection"], initial, strict=True):
        base = build_gemini_base(adapter, selected)
        key = selected["key"]
        expected_bodies = {}
        for arm in ARMS:
            record = records[key][arm]
            if record["round2_attempted"]:
                body = proposal_wire(base, selected, first, arm)
                if fingerprint(body) != record["proposal_request_fingerprint"]:
                    raise ValueError("proposal fingerprint differs from bound graph/feedback request")
                expected_bodies[(f"{arm}_proposal_2", "base")] = body
            if record.get("raw") is not None:
                # JSON on disk sorts dict keys. Rebuild ontology components in
                # their original insertion order before serializing wire text.
                pool = make_pool(record["before"], record["proposal"], record["pool_before"])
                bodies = {seat: review_wire(seat, base, selected, pool) for seat in SEATS}
                if {s: fingerprint(b) for s, b in bodies.items()} != record["review_request_fingerprints"]:
                    raise ValueError("review fingerprint differs from clean pool-only request")
                expected_bodies.update({(f"{arm}_review_2", s): b for s, b in bodies.items()})
                verified["review_requests_reconstructed_including_shared"] += 5
        for call in (c for c in ledger["calls"] if c["target"] == key):
            body = expected_bodies.get((call["stage"], call["seat"]))
            if body is None:
                raise ValueError("ledger call has no corresponding reconstructed request")
            folder = output / "calls" / f"{call['index']:03d}_{key}_{call['stage']}_{call['seat']}"
            if read(folder / "request.json") != redact_images(body):
                raise ValueError("actual request differs from frozen inference mechanism")
            if read(folder / "record.json") != call:
                raise ValueError("call record differs from the closed accounting ledger")
            if call["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
                response = read(folder / "response.json")["body"]
                provider = PROVIDERS.get(body["model"], PROVIDERS.get(call["seat"]))
                if response.get("model") != body["model"] or (provider is not None and response.get("provider") != provider):
                    raise ValueError("accepted response model or provider differs from frozen request")
            verified["actual_complete_requests_verified"] += 1
    return dict(verified)


def _distribution(values):
    numbers = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    return {"n": len(numbers), "sum": sum(numbers), "mean": statistics.mean(numbers) if numbers else None,
            "median": statistics.median(numbers) if numbers else None, "max": max(numbers) if numbers else None}


def runtime(records, ledger):
    arms = {}
    for arm in ARMS:
        values = [row[arm] for row in records.values()]
        arms[arm] = {"statuses": dict(Counter(r["status"] for r in values)),
                     "round2_attempted": sum(r["round2_attempted"] for r in values),
                     "round2_reviewed": sum(r["reviewed"] for r in values),
                     "actual_review_rounds": dict(Counter(1 + int(r["reviewed"]) for r in values)),
                     "shared_panels": sum(bool(r.get("shared_from")) for r in values),
                     "proposal_seconds": _distribution(r.get("proposal_seconds") for r in values),
                     "independent_panel_seconds": _distribution(r.get("panel_seconds") for r in values
                                                                  if not r.get("shared_from")),
                     "observed_incremental_seconds": _distribution(r.get("total_seconds") for r in values)}
    costs = defaultdict(lambda: {"calls": 0, "charges": defaultdict(Decimal), "charge_kinds": Counter(),
                                 "prompt_tokens": 0, "completion_tokens": 0, "statuses": Counter()})
    for call in ledger["calls"]:
        arm = next((a for a in ARMS if call["stage"].startswith(a + "_")), None)
        if arm is None:
            raise ValueError("continuation contains an unexpected paid stage")
        for group in (arm, "all"):
            item = costs[group]
            item["calls"] += 1
            item["charges"][call["account"]] += Decimal(call["charge"])
            item["charge_kinds"][call["charge_kind"]] += 1
            item["statuses"][call["status"]] += 1
            for field in ("prompt_tokens", "completion_tokens"):
                item[field] += call.get("usage", {}).get(field, 0)
    return {"arms": arms, "incremental_costs": {k: {**v, "charges": {a: str(b) for a, b in v["charges"].items()},
            "charge_kinds": dict(v["charge_kinds"]), "statuses": dict(v["statuses"])} for k, v in costs.items()},
            "note": "Only new calls charged; inherited H0/R1 cost is excluded. Shared panels have no independent latency. "
                    "All feedback requests precede issues-only requests, so timing and model variability are order-confounded."}


def summarize(rows, truth, records):
    truth_by_id = {(r["video_id"], r["frame_id"]): r for r in truth}
    metrics = {}
    for version in VERSIONS:
        data = [{**truth_by_id[r["video_id"], r["frame_id"]], "h0": r["h0"], "h1": None,
                 "final": r[version] if version in VERSIONS[:3] else r["arms"][version]["prediction"]} for r in rows]
        metrics[version] = compute_repair_comparison(data)["arms"]["final"]
    details = []
    coverage = {a: {t: Counter(dict.fromkeys(("valid_targets", "gt_positive", "pool_true", "pool_false",
                "pool_size", "selected_true", "missing_from_pool", "in_pool_not_selected"), 0))
                for t in TASKS[:4]} for a in VERSIONS[1:]}
    for row in rows:
        gt = truth_by_id[row["video_id"], row["frame_id"]]
        predictions = {v: row[v] if v in VERSIONS[:3] else row["arms"][v]["prediction"] for v in VERSIONS}
        details.append({"key": row["key"], "comparisons": {
            f"{a}_to_{b}": frame_delta(predictions[a], predictions[b], gt["gt"], gt["mask"]) for a, b in PAIRS}})
        for version, tasks in coverage.items():
            record = records[row["key"]][version]
            for task, counts in tasks.items():
                if not gt["mask"][task]:
                    continue
                expected = set(gt["gt"][task])
                pool = {p["label_id"] for p in record["pool"]["propositions"] if p["task"] == task}
                predicted = set(predictions[version][task])
                counts.update(valid_targets=1, gt_positive=len(expected), pool_true=len(pool & expected),
                              pool_false=len(pool - expected), pool_size=len(pool), selected_true=len(predicted & expected),
                              missing_from_pool=len(expected - pool), in_pool_not_selected=len((expected & pool) - predicted))
    for tasks in coverage.values():
        for counts in tasks.values():
            counts["pool_recall"] = counts["pool_true"] / counts["gt_positive"] if counts["gt_positive"] else None
    return {"metrics": metrics, "candidate_coverage": coverage, "comparisons": {
        f"{a}_to_{b}": summarize_deltas([d["comparisons"][f"{a}_to_{b}"] for d in details]) for a, b in PAIRS}}, details


def score(output, adapter):
    output = Path(output)
    plan, rows, initial, ledger = validate(output)
    records, audit = audit_records(output, plan, rows, initial, ledger)
    audit.update(audit_wires(output, plan, initial, records, ledger, adapter))
    for row in initial:
        records[row["key"]].update({a: row[a] for a in ("control_r1", "graph_r1")})
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"],
        "h0": r["h0"], "h1": None, "final": r["graph_r1"]} for r in rows])
    result, details = summarize(rows, truth, records)
    result.update(profile=PROFILE, targets=len(rows), runtime=runtime(records, ledger),
                  audit={**audit, "immutable_snapshots_verified": True, "gt_used_only_after_closed_validation": True},
                  limitations=["Eight previously inspected Training targets; this is a paired mechanism diagnostic, not held-out validation.",
                               "No new H0 or first-round calls. Unattempted and fallback targets remain scored with task masks.",
                               "A model pass is not GT correctness. Second round only triggers after an unresolved first round.",
                               "Identical candidate pools stop without another review; this policy cannot reconsider a high-scoring existing error without a new candidate.",
                               "All feedback requests precede issues-only requests; API variability and order are not isolated.",
                               "No complete Tracker/Gate runtime is measured."])
    validate(output)
    save(output / "metrics.json", result)
    save(output / "frame_deltas.json", details)
    save(output / "scored_truth.json", truth)
    save(output / "audit.json", result["audit"])
    print(json.dumps({"metrics": result["metrics"], "comparisons": result["comparisons"],
                      "runtime": result["runtime"], "audit": result["audit"]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
