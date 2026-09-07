"""Read-only completed Training temporal audit: semantics, exact wires and costs."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_temporal_memory_trial as runner
from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.temporal_memory import (
    extend_history_request,
    trigger_propositions,
)
from tools.audit import audit_verifier_variants as independent

VARIANTS = ("initial_names1000", "repeat1000", "memory1000")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_of(row):
    return f"{row['video_id']}_{row['frame_id']}"


def check_complete(run_dir):
    if not (run_dir / "summary.json").is_file():
        raise ValueError("completed summary required before reading any GT")
    plan, summary = read(run_dir / "plan.json"), read(run_dir / "summary.json")
    if (plan.get("source_split") != "Training" or summary.get("source_split") != "Training"
            or plan.get("schema_version") != "temporal_memory_trial_plan_v1"
            or plan.get("variants") != list(VARIANTS) or summary.get("variants") != list(VARIANTS)
            or summary.get("targets") != 16 or plan.get("target_count") != 16
            or summary.get("plan_sha256") != plan.get("plan_sha256")
            or summary.get("predictions_sha256") != sha(run_dir / "predictions.json")):
        raise ValueError("completed Training temporal plan/targets/predictions mismatch")
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != plan["plan_sha256"]:
        raise ValueError("plan self-hash mismatch")
    rows = read(run_dir / "predictions.json")
    if ([key_of(r) for r in rows] != plan["keys"] or len(set(plan["keys"])) != 16
            or any("gt" in r or "mask" in r for r in rows)):
        raise ValueError("unscored prediction identity/GT separation mismatch")
    runner.assert_frozen(plan)
    return plan, summary, rows


def checked_wire(saved, request):
    wire = shared.wire_body(request)
    wire_hash = hashlib.sha256(canonical_json_bytes(wire)).hexdigest()
    if (saved.get("metadata") != canonical_request_metadata(request).to_mapping()
            or saved.get("payload") != thaw_json(request.payload)
            or saved.get("wire") != shared.safe_wire(wire)
            or "wire_sha256" in saved and saved["wire_sha256"] != wire_hash):
        raise ValueError("request metadata/payload/complete wire mismatch")
    return wire_hash


def audit_wires(run_dir, plan, rows):
    """Reconstruct from real frozen images before reading scored labels."""
    source, batch = Path(plan["source"]), Path(plan["source_names_batch"])
    collection_plan, originals = runner.completed_training(source)
    _, batch_rows = runner.completed_training(batch)
    batch_rows = {key_of(r): r for r in batch_rows}
    selected = {r["key"]: r for r in collection_plan["selection"]}
    originals = {key_of(r): r for r in originals}
    current = {key_of(r): r for r in rows}
    target_details = {r["key"]: r for r in plan["targets"]}
    initial, expected, hashes, details = {}, {}, {}, []
    for key in plan["keys"]:
        item, old, row = selected[key], originals[key], current[key]
        if row["h0"] != old["h0"] or row["h1"] != old["h1"]:
            raise ValueError("temporal predictions changed frozen H0/H1")
        previous = batch_rows[key]["variants"]["names1000"]
        if any(row["variants"]["initial_names1000"].get(field) != value for field, value in previous.items()):
            raise ValueError("initial names arm differs from original completed batch")
        _, restored = shared.restore_target(source, item, old, hashes)
        if restored is None:
            continue
        names = shared.variant_request(restored[0], "names1000")
        initial[key] = names
        first_path = batch / "calls" / key / "names1000" / "request.json"
        original_wire = checked_wire(read(first_path), names)
        first = row["variants"]["initial_names1000"]
        source_result, source_http = (read(first_path.parent / name) for name in ("result.json", "http_response.json"))
        native = json.loads(source_http["body"])
        if (source_result.get("status") != "OK" or source_http["status_code"] != 200
                or source_result.get("request_hash") != canonical_request_metadata(names).request_hash
                or source_result.get("payload") != first["review"]
                or native.get("model") != shared.MODEL or native.get("provider") != "Alibaba"
                or not native.get("id") or native["id"] != source_result.get("provider_request_id")
                or json.loads(native["choices"][0]["message"]["content"]) != first["review"]):
            raise ValueError("first names arm differs from its actual native response")
        runner.checked_initial_result(row, names, first)
        triggers = trigger_propositions(names, first["review"], h0=row["h0"], h1=row["h1"])
        target = target_details[key]
        if triggers != target["trigger_propositions"] or bool(triggers) != target["trigger"]:
            raise ValueError("declared triggers differ from GT-free first review")
        if not target["dispatch_pair"]:
            continue
        frames = target["extra_frame_ids"]
        folder = Path(item["images"][-1]["path"]).resolve().parent
        found = [(frame, folder / f"{frame:06d}.png") for frame in frames]
        for _, path in found:
            if sha(path) != plan["source_artifact_sha256"].get(str(path)):
                raise ValueError("actual extra source image lacks frozen SHA binding")
        memory, _, _ = extend_history_request(names, runner.load_extra_images(row["video_id"], found), frames)
        expected[(key, "repeat1000")] = names
        expected[(key, "memory1000")] = memory
        details.append({"key": key, "first_names_wire_sha256": original_wire,
                        "extra_frame_ids": frames, "extra_images": len(frames),
                        "repeat_wire_identical": True, "memory_original_input_reversible": True})
    planned = {(item["key"], item["variant"]): item for item in plan["selection"]}
    if len(planned) != len(plan["selection"]) or set(planned) != set(expected):
        raise ValueError("planned calls differ from independently reconstructed trigger pairs")
    for identity, request in expected.items():
        key, variant = identity
        saved = read(run_dir / "requests" / key / f"{variant}.json")
        wire_hash = checked_wire(saved, request)
        if wire_hash != planned[identity]["wire_sha256"]:
            raise ValueError("planned wire hash differs from actual reconstructed image request")
    records = read(run_dir / "calls_summary.json") if (run_dir / "calls_summary.json").is_file() else []
    if [(r["key"], r["stage"]) for r in records] != [
            (item["key"], item["variant"]) for item in plan["selection"][:len(records)]]:
        raise ValueError("actual temporal call order differs from frozen alternating dispatch")
    actual = set()
    for record in records:
        identity = record["key"], record["stage"]
        if identity not in expected or identity in actual:
            raise ValueError("unexpected or duplicate actual temporal call")
        actual.add(identity)
        directory = run_dir / "calls" / identity[0] / identity[1]
        request = expected[identity]
        checked_wire(read(directory / "request.json"), request)
        if read(directory / "result.json") != record or not (directory / "dispatch.lock").is_file():
            raise ValueError("actual temporal result/dispatch evidence mismatch")
        metadata = canonical_request_metadata(request)
        if record["request_hash"] != metadata.request_hash:
            raise ValueError("actual temporal result request hash mismatch")
        row = current[identity[0]]
        arm = row["variants"][identity[1]]
        if arm.get("supplemental_review") != record.get("payload"):
            raise ValueError("saved supplementary review differs from actual local result")
        replayed = runner.apply_supplement(row, request, record.get("payload"), dispatched=True)
        if any(arm.get(field) != value for field, value in replayed.items()):
            raise ValueError("saved supplementary decision does not reproduce its whole-review fallback")
        if record["status"] == "OK":
            http = read(directory / "http_response.json")
            native = json.loads(http["body"])
            if (http["status_code"] != 200 or native.get("model") != shared.MODEL
                    or native.get("provider") != "Alibaba" or not native.get("id")
                    or native["id"] != record.get("provider_request_id")
                    or json.loads(native["choices"][0]["message"]["content"]) != record["payload"]):
                raise ValueError("valid supplementary result differs from native identity/content")
    for key, row in current.items():
        for variant in ("repeat1000", "memory1000"):
            if (key, variant) not in actual and any(
                    row["variants"][variant][field] != row["variants"]["initial_names1000"][field]
                    for field in ("review", "final_a", "final_b")):
                raise ValueError("unattempted supplemental arm did not keep first names result")
    return initial, {"expected_calls": len(expected), "actual_calls": len(actual),
                     "unattempted_calls": len(expected) - len(actual), "targets": details,
                     "all_actual_wires_bound": True, "new_api_calls_by_audit": 0}


def exact_rate(task, metric):
    if not task["valid_targets"]:
        return None
    if metric == "f1":
        denominator = 2 * task["tp"] + task["fp"] + task["fn"]
        return Fraction(2 * task["tp"], denominator) if denominator else Fraction(0)
    return Fraction(task["exact_matches"], task["valid_targets"])


def qualification(metrics):
    arms = metrics["all_targets"]["arms"]
    def compare(left, right, task, metric, strict=False):
        a, b = exact_rate(arms[left]["tasks"][task], metric), exact_rate(arms[right]["tasks"][task], metric)
        return a is not None and b is not None and (a > b if strict else a >= b)
    result = {}
    for variant in ("repeat1000", "memory1000"):
        arm = variant + "_final_b"
        edits = metrics["edit_application"][arm]["tasks"]["ivt"]
        conditions = {"ivt_f1_above_h0": compare(arm, "h0", "ivt", "f1", True),
            "ivt_accuracy_not_below_h0": compare(arm, "h0", "ivt", "accuracy"),
            "beneficial_ivt_applied_exceeds_harmful": edits["beneficial_applied"] > edits["harmful_applied"]}
        conditions.update({f"{task}_f1_not_below_h0": compare(arm, "h0", task, "f1")
                           for task in ("instrument", "verb", "target")})
        result[variant] = {"conditions_vs_h0": conditions, "passes_h0_conditions": all(conditions.values()),
                           "ivt_beneficial_applied": edits["beneficial_applied"],
                           "ivt_harmful_applied": edits["harmful_applied"]}
    increment = {"ivt_f1_above_repeat": compare("memory1000_final_b", "repeat1000_final_b", "ivt", "f1", True),
        "ivt_accuracy_not_below_repeat": compare("memory1000_final_b", "repeat1000_final_b", "ivt", "accuracy")}
    increment.update({f"{task}_f1_not_below_repeat": compare("memory1000_final_b", "repeat1000_final_b", task, "f1")
                      for task in ("instrument", "verb", "target")})
    result["memory1000"]["increment_conditions_vs_repeat"] = increment
    result["memory1000"]["passes_increment_conditions"] = all(increment.values())
    return {"primary": "final_b", "scope": "Training practical screening only; no significance or automatic selection",
            "exact_integer_count_comparisons": True, "variants": result}


def assessment_transitions(rows, initial_requests):
    counters = {variant: {task: Counter() for task in ("instrument", "verb", "target", "ivt")}
                for variant in ("repeat1000", "memory1000")}
    details = []
    for row in rows:
        key = key_of(row)
        if key not in initial_requests:
            continue
        propositions = json.loads(initial_requests[key].payload["input_text"])["propositions"]
        first = {a["proposition_id"]: a for a in row["variants"]["initial_names1000"]["review"]["assessments"]}
        for variant in counters:
            arm = row["variants"][variant]
            after = {a["proposition_id"]: a for a in arm["review"]["assessments"]}
            for prop in propositions:
                task, label, identity = prop["task"], prop["label_id"], prop["proposition_id"]
                count = counters[variant][task]
                before, final = first[identity]["presence"], after[identity]["presence"]
                count["propositions"] += 1
                count["initial_unclear_total"] += before == "UNCLEAR"
                if not row["mask"][task]:
                    count["masked_propositions"] += 1
                    count["masked_initial_unclear"] += before == "UNCLEAR"
                    continue
                present = label in row["gt"][task]
                correct = "PRESENT" if present else "ABSENT"
                if before == "UNCLEAR":
                    state = "still_unclear" if final == "UNCLEAR" else "resolved_correct" if final == correct else "resolved_wrong"
                elif before == correct:
                    state = "correct_stayed_correct" if final == correct else "correct_to_unclear" if final == "UNCLEAR" else "correct_to_wrong"
                else:
                    state = "wrong_to_correct" if final == correct else "wrong_to_unclear" if final == "UNCLEAR" else "wrong_stayed_wrong"
                count[state] += 1
                if before == "UNCLEAR" and task in ("verb", "ivt"):
                    details.append({"key": key, "variant": variant, "proposition_id": identity,
                        "task": task, "label_id": label, "gt_presence": correct, "after": final,
                        "state": state, "review_source": arm["review_source"],
                        "supplemental_status": arm["supplemental_status"]})
    return {"meaning": "label-presence correctness, separate from repair application; false GT masks excluded",
            "tasks": {v: {t: dict(c) for t, c in tasks.items()} for v, tasks in counters.items()},
            "initial_unclear_verb_ivt_details": details}


def independent_paired_counts(rows, before, after):
    """Count changes directly from sets; never call the production comparator.

    Missing predictions contribute empty-set errors but cannot be exact. An
    availability change with identical sets is an equal-loss change. Frame
    mixed status depends on per-head gains AND harms, regardless of net loss.
    """
    categories = ("improved", "worsened", "equal_loss_changed", "unchanged")
    frames = dict.fromkeys(("valid_targets", *categories, "mixed", "unscored",
                            "wrong_to_exact", "exact_to_wrong"), 0)
    tasks = {task: dict.fromkeys(("valid_targets", *categories, "wrong_to_exact", "exact_to_wrong"), 0)
             for task in independent.TASKS}
    frame_details = []
    for row in rows:
        left, right = row[before], row[after]
        observed = []
        left_exact, right_exact = left is not None, right is not None
        for task in independent.TASKS:
            if not row["mask"][task]:
                continue
            expected = set(row["gt"][task])
            left_set = set() if left is None else set(left[task])
            right_set = set() if right is None else set(right[task])
            left_errors = sum(value not in expected for value in left_set) + sum(value not in left_set for value in expected)
            right_errors = sum(value not in expected for value in right_set) + sum(value not in right_set for value in expected)
            if left_errors != right_errors:
                category = "improved" if right_errors < left_errors else "worsened"
            else:
                changed = left_set != right_set or (left is None) != (right is None)
                category = "equal_loss_changed" if changed else "unchanged"
            was_exact = left is not None and left_set == expected
            now_exact = right is not None and right_set == expected
            counts = tasks[task]
            counts["valid_targets"] += 1
            counts[category] += 1
            counts["wrong_to_exact"] += not was_exact and now_exact
            counts["exact_to_wrong"] += was_exact and not now_exact
            left_exact = left_exact and was_exact
            right_exact = right_exact and now_exact
            observed.append(category)
        if not observed:
            frame_category = "unscored"
        elif "improved" in observed and "worsened" in observed:
            frame_category = "mixed"
        elif "improved" in observed:
            frame_category = "improved"
        elif "worsened" in observed:
            frame_category = "worsened"
        elif "equal_loss_changed" in observed:
            frame_category = "equal_loss_changed"
        else:
            frame_category = "unchanged"
        gain_exact = bool(observed) and not left_exact and right_exact
        loss_exact = bool(observed) and left_exact and not right_exact
        frames["valid_targets"] += bool(observed)
        frames[frame_category] += 1
        frames["wrong_to_exact"] += gain_exact
        frames["exact_to_wrong"] += loss_exact
        frame_details.append({"video_id": row["video_id"], "frame_id": row["frame_id"],
                              "category": frame_category, "wrong_to_exact": gain_exact,
                              "exact_to_wrong": loss_exact})
    return {"counts": {"frames": frames, "tasks": tasks}, "frame_details": frame_details}


def assert_independent_pairs(rows, metrics, between):
    flattened = [{**row, **{f"{v}_{p}": row["variants"][v][p]
                            for v in VARIANTS for p in ("final_a", "final_b")}} for row in rows]
    checked = {}
    for variant in VARIANTS:
        for policy in ("final_a", "final_b"):
            arm = f"{variant}_{policy}"
            recomputed = independent_paired_counts(flattened, "h0", arm)
            recorded = metrics["all_targets"]["changes_vs_h0"][arm]
            if recomputed["counts"] != recorded:
                raise AssertionError(f"independent paired counts disagree: h0 -> {arm}")
            recorded_details = metrics["all_targets"]["change_details"][arm]
            for actual, expected in zip(recomputed["frame_details"], recorded_details, strict=True):
                if any(actual[field] != expected[field] for field in actual):
                    raise AssertionError(f"independent per-frame classification disagrees: {arm}")
            checked[f"h0_to_{arm}"] = recomputed
    for policy in ("final_a", "final_b"):
        recomputed = independent_paired_counts(flattened, f"repeat1000_{policy}", f"memory1000_{policy}")
        if recomputed["counts"] != between[policy]["paired"]["h0_to_final"]:
            raise AssertionError(f"independent paired counts disagree: repeat -> memory {policy}")
        for actual, expected in zip(recomputed["frame_details"], between[policy]["details"], strict=True):
            target = expected["pairs"]["h0_to_final"]
            if (actual["video_id"] != expected["video_id"] or actual["frame_id"] != expected["frame_id"]
                    or any(actual[field] != target[field] for field in ("category", "wrong_to_exact", "exact_to_wrong"))):
                raise AssertionError(f"independent per-frame classification disagrees: repeat -> memory {policy}")
        checked[f"repeat_to_memory_{policy}"] = recomputed
    return {"method": "independent direct-set error and exactness counts; no production comparator calls",
            "all_eight_pair_counts_and_frame_classifications_match": True, "pairs": checked}


def temporal_metrics(rows, initial_requests):
    metrics = independent.compute_variant_audit(rows, VARIANTS)
    between = {policy: compute_repair_comparison([{**r,
        "h0": r["variants"]["repeat1000"][policy], "h1": r["variants"]["memory1000"][policy],
        "final": r["variants"]["memory1000"][policy]} for r in rows]) for policy in ("final_a", "final_b")}
    paired_check = assert_independent_pairs(rows, metrics, between)
    return {"metrics": metrics, "memory_vs_repeat": between, "independent_paired_check": paired_check,
            "development_qualification": qualification(metrics),
            "assessment_transitions": assessment_transitions(rows, initial_requests),
            "supplemental_status": {v: dict(Counter(r["variants"][v]["supplemental_status"] for r in rows)) for v in VARIANTS},
            "fallback_note": "failure reusing the initial review is not a semantic improvement; all targets remain scored"}


def audit_run(run_dir):
    run_dir = Path(run_dir).resolve()
    plan, summary, rows = check_complete(run_dir)
    initial, wires = audit_wires(run_dir, plan, rows)
    # The generic audit independently checks saved GT association, TP/FP/FN,
    # final metrics, proposed/applied edits and every native-cost ledger entry.
    base_audit = independent.audit_run(run_dir)
    scored = read(run_dir / "scored_predictions.json")
    report = temporal_metrics(scored, initial)
    if report["metrics"] != base_audit["metrics"] or report["memory_vs_repeat"] != summary["memory_vs_repeat"]:
        raise ValueError("independent temporal metrics disagree with saved summary")
    if wires["actual_calls"] != summary["provider_calls"]:
        raise ValueError("actual wire count differs from accounting")
    return {"schema_version": "independent_temporal_memory_audit_v1", "source_split": "Training",
            **report, "wire_audit": wires, "budget": base_audit["budget"],
            "cost_representation_audit": base_audit["cost_representation_audit"],
            "original_counts_costs_and_between_arm_metrics_match": True,
            "source_sha256": base_audit["source_sha256"], "audit_script_sha256": sha(__file__),
            "new_provider_calls": 0, "new_cost_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_run(args.run_dir)
    destination = args.output or args.run_dir / "independent_temporal_audit.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(destination), "wire_audit": result["wire_audit"],
                      "qualification": result["development_qualification"]}))
