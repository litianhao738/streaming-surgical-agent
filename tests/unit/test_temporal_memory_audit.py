"""Synthetic offline audit checks; never access live runs or dataset labels."""

import json
from copy import deepcopy

import pytest

from scripts import run_temporal_memory_trial as runner
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from tests.unit.test_temporal_memory import fixture
from tools.audit import audit_temporal_memory as audit


def example_rows():
    request, source, first_review = fixture()
    first = {**runner.evaluate(request, source, first_review), "review": first_review, "status": "OK",
             "review_source": "initial_names1000", "supplemental_status": "BASELINE_REUSED_NO_NEW_CALL"}
    source["variants"] = {"initial_names1000": first}
    correct = deepcopy(first_review)
    props = {p["proposition_id"]: p for p in json.loads(request.payload["input_text"])["propositions"]}
    for assessment in correct["assessments"]:
        prop = props[assessment["proposition_id"]]
        assessment["presence"] = "PRESENT" if prop["label_id"] in source["h1"][prop["task"]] else "ABSENT"
    repeated = runner.apply_supplement(source, request, first_review, dispatched=True)
    improved = runner.apply_supplement(source, request, correct, dispatched=True)
    row = {"video_id": source["video_id"], "frame_id": source["frame_id"],
           "h0": source["h0"], "h1": source["h1"], "gt": source["h1"],
           "mask": dict.fromkeys(source["h1"], True),
           "variants": {"initial_names1000": first, "repeat1000": repeated, "memory1000": improved}}
    return [row], {audit.key_of(row): request}


def test_metrics_screening_and_unclear_resolution_use_actual_applied_edits():
    rows, initial = example_rows()
    result = audit.temporal_metrics(rows, initial)
    qualification = result["development_qualification"]["variants"]
    assert qualification["repeat1000"]["passes_h0_conditions"] is False
    assert qualification["memory1000"]["passes_h0_conditions"] is True
    assert qualification["memory1000"]["passes_increment_conditions"] is True
    assert qualification["memory1000"]["ivt_beneficial_applied"] == 2
    assert qualification["memory1000"]["ivt_harmful_applied"] == 0
    transitions = result["assessment_transitions"]["tasks"]
    assert transitions["repeat1000"]["ivt"]["still_unclear"] == 2
    assert transitions["memory1000"]["ivt"]["resolved_correct"] == 2
    assert result["metrics"]["coverage"]["complete_five_head_gt_targets"] == 1
    assert result["memory_vs_repeat"]["final_b"]["paired"]["h0_to_final"]["frames"]["improved"] == 1
    assert result["independent_paired_check"]["all_eight_pair_counts_and_frame_classifications_match"] is True


def test_missing_gt_stays_in_trigger_denominator_but_not_presence_correctness():
    rows, initial = example_rows()
    rows[0]["mask"]["ivt"] = False
    result = audit.temporal_metrics(rows, initial)
    details = result["assessment_transitions"]["tasks"]["memory1000"]["ivt"]
    assert details["initial_unclear_total"] == details["masked_initial_unclear"] == 2
    assert details.get("resolved_correct", 0) == 0
    assert result["metrics"]["coverage"]["targets"] == 1
    assert result["metrics"]["coverage"]["complete_five_head_gt_targets"] == 0
    assert result["development_qualification"]["variants"]["memory1000"]["passes_h0_conditions"] is False


def test_wrong_resolution_and_failed_fallback_are_not_counted_as_improvement():
    rows, initial = example_rows()
    row, request = rows[0], next(iter(initial.values()))
    wrong = deepcopy(row["variants"]["memory1000"]["review"])
    for item in wrong["assessments"]:
        item["presence"] = "ABSENT" if item["presence"] == "PRESENT" else "PRESENT"
    row["variants"]["memory1000"] = runner.apply_supplement(row, request, wrong, dispatched=True)
    row["variants"]["repeat1000"] = runner.apply_supplement(row, request, None, dispatched=True)
    result = audit.temporal_metrics(rows, initial)
    transitions = result["assessment_transitions"]["tasks"]
    assert transitions["memory1000"]["ivt"]["resolved_wrong"] == 2
    assert transitions["repeat1000"]["ivt"]["still_unclear"] == 2
    assert result["supplemental_status"]["repeat1000"]["REQUEST_FAILURE_FALLBACK"] == 1
    assert result["development_qualification"]["variants"]["memory1000"]["passes_h0_conditions"] is False


def test_incomplete_run_never_reads_plan_predictions_or_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "read", lambda path: pytest.fail(f"unexpected read: {path}"))
    monkeypatch.setattr(audit.independent, "audit_run", lambda *a: pytest.fail("scoring before completion"))
    with pytest.raises(ValueError, match="completed summary"):
        audit.audit_run(tmp_path)


@pytest.mark.parametrize("tamper", ["image_wire", "text", "metadata", "hash"])
def test_wire_audit_detects_any_saved_transport_change(tamper):
    request, _, _ = fixture()
    wire = audit.shared.wire_body(request)
    digest = audit.hashlib.sha256(audit.canonical_json_bytes(wire)).hexdigest()
    saved = {"metadata": canonical_request_metadata(request).to_mapping(),
             "payload": thaw_json(request.payload), "wire": audit.shared.safe_wire(wire),
             "wire_sha256": digest}
    assert audit.checked_wire(saved, request) == digest
    if tamper == "image_wire":
        saved["wire"]["messages"][1]["content"][1]["image_url"]["detail"] = "high"
    elif tamper == "text":
        saved["payload"]["system_text"] += " changed"
    elif tamper == "metadata":
        saved["metadata"]["request_hash"] = "0" * 64
    else:
        saved["wire_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="complete wire mismatch"):
        audit.checked_wire(saved, request)


def test_manual_pair_counts_cover_mixed_partial_equal_failure_and_masks(monkeypatch):
    tasks = ("instrument", "verb", "target", "ivt", "phase")
    def labels():
        return {task: [] for task in tasks}
    def row(index, before, after, gt, valid=("instrument", "verb")):
        return {"video_id": "synthetic", "frame_id": index, "before": before, "after": after,
                "gt": gt, "mask": {task: task in valid for task in tasks}}
    # One head improves while another worsens: mixed even though pooled loss falls.
    truth = {**labels(), "instrument": [0, 1], "verb": [0]}
    rows = [row(1, {**labels(), "verb": [0]}, {**labels(), "instrument": [0, 1]}, truth)]
    # Same loss but different labels; neither side is exact.
    rows.append(row(2, {**labels(), "instrument": [0]}, {**labels(), "instrument": [1]}, labels(), ("instrument",)))
    # A partial improvement must not count as wrong-to-exact.
    rows.append(row(3, labels(), {**labels(), "instrument": [0]}, {**labels(), "instrument": [0, 1]}, ("instrument",)))
    # None is not an exact empty prediction; availability changes are real changes.
    rows.append(row(4, None, labels(), labels(), ("instrument",)))
    rows.append(row(5, labels(), None, labels(), ("instrument",)))
    rows.append(row(6, None, None, labels(), ("instrument",)))
    # Masked changes contribute neither head scoring nor whole-frame exactness.
    rows.append(row(7, labels(), {**labels(), "ivt": [4]}, {}, ()))
    monkeypatch.setattr(audit, "compute_repair_comparison", lambda *a: pytest.fail("manual recount called production"))
    result = audit.independent_paired_counts(rows, "before", "after")
    assert result["counts"]["frames"] == {
        "valid_targets": 6, "improved": 1, "worsened": 0, "equal_loss_changed": 3,
        "unchanged": 1, "mixed": 1, "unscored": 1, "wrong_to_exact": 1, "exact_to_wrong": 1}
    assert result["counts"]["tasks"]["instrument"] == {
        "valid_targets": 6, "improved": 2, "worsened": 0, "equal_loss_changed": 3,
        "unchanged": 1, "wrong_to_exact": 2, "exact_to_wrong": 1}
    assert result["counts"]["tasks"]["verb"]["worsened"] == 1
    assert result["counts"]["tasks"]["ivt"]["valid_targets"] == 0
    assert result["frame_details"][0]["category"] == "mixed"
    assert result["frame_details"][2]["wrong_to_exact"] is False


@pytest.mark.parametrize("tamper", ["head_count", "frame_count", "frame_identity", "between"])
def test_independent_pair_recount_detects_wrong_recorded_counts_and_classifications(tamper):
    rows, initial = example_rows()
    report = audit.temporal_metrics(rows, initial)
    metrics, between = deepcopy(report["metrics"]), deepcopy(report["memory_vs_repeat"])
    if tamper == "head_count":
        metrics["all_targets"]["changes_vs_h0"]["memory1000_final_b"]["tasks"]["ivt"]["improved"] += 1
    elif tamper == "frame_count":
        metrics["all_targets"]["changes_vs_h0"]["memory1000_final_a"]["frames"]["mixed"] += 1
    elif tamper == "frame_identity":
        metrics["all_targets"]["change_details"]["memory1000_final_b"][0]["category"] = "mixed"
    else:
        between["final_b"]["paired"]["h0_to_final"]["tasks"]["ivt"]["wrong_to_exact"] += 1
    with pytest.raises(AssertionError, match="independent"):
        audit.assert_independent_pairs(rows, metrics, between)
