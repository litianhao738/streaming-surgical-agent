"""Continuation scoring must preserve masks, immutable baselines and stop rules."""
from copy import deepcopy

import pytest

from scripts import score_prior_feedback_continuation as scorer
from scripts.run_prior_panel_trial import save
from scripts.run_repair_revision_trial import normalize_five
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_feedback import build_review_feedback


def panel_record(before, proposal, prior_pool=None):
    pool = make_pool(before, proposal, prior_pool or make_pool(before))
    raw = {seat: {"judgments": {p["id"]: {
        "rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [2],
        "observation": "Contact is visible but the action is uncertain."}
        for p in pool["propositions"]}} for seat in SEATS}
    reviews, formatting = normalize_five(raw, pool, 3)
    means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
    prediction = panel.select(before, pool, means)
    issues = panel.unresolved(prediction, pool, means, diagnostics)
    return {"before": before, "proposal": proposal, "pool": pool, "raw": raw, "reviews": reviews,
            "format_diagnostics": formatting, "means": means, "diagnostics": diagnostics,
            "prediction": prediction, "issues": issues, "status": "UNRESOLVED"}


def row_fixture():
    h0 = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}
    empty = {t: [] for t in scorer.TASKS[:4]}
    old = panel_record(h0, empty)
    first = {"key": "VID103_101", "video_id": "VID103", "frame_id": 101, "h0": h0,
             "control_r1": deepcopy(old), "graph_r1": deepcopy(old), "hints": {"packet": {"hints": []}}}
    records = {}
    for arm in scorer.ARMS:
        records[arm] = {"round2": 2, "before": h0, "pool_before": old["pool"], "input_issues": old["issues"],
            "candidate_relation_hints": first["hints"]["packet"], "proposal": empty,
            "pool": old["pool"], "prediction": h0, "issues": old["issues"],
            "status": "NO_NEW_CANDIDATES", "round2_attempted": True, "reviewed": False,
            "review_calls_observed": 0, "new_review_calls": 0, "shared_from": None,
            "proposal_seconds": 1.0, "panel_seconds": None, "total_seconds": 1.0}
        if arm == "evidence_feedback":
            records[arm]["review_evidence_feedback"] = build_review_feedback(old["pool"], old["reviews"], old["issues"])
    row = {k: first[k] for k in ("key", "video_id", "frame_id", "h0")}
    row.update(control_r1=h0, graph_r1=h0, arms={})
    selected = {k: row[k] for k in ("key", "video_id", "frame_id")}
    selected["causal_frame_ids"] = [51, 76, 101]
    return selected, row, first, records


def persist_records(output, row, records):
    for arm, record in records.items():
        relative = f"targets/{row['key']}/{arm}.json"
        save(output / relative, record)
        row["arms"][arm] = {k: record[k] for k in ("prediction", "status", "round2_attempted", "reviewed")}
        row["arms"][arm]["record"] = relative


def calls_for(output, row, records):
    calls = []
    for arm, record in records.items():
        if not record["round2_attempted"]:
            continue
        values = [(f"{arm}_proposal_2", "base", record["proposal"])]
        if record.get("raw") is not None and not record.get("shared_from"):
            values.extend((f"{arm}_review_2", seat, record["raw"][seat]) for seat in SEATS)
        for stage, seat, raw in values:
            call = {"index": len(calls), "target": row["key"], "stage": stage, "seat": seat,
                    "status": "JSON_PARSED", "account": "openrouter_usd", "charge": "0.01",
                    "charge_kind": "provider_reported", "usage": {}}
            calls.append(call)
            folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{stage}_{seat}"
            save(folder / "response.json", {"http_status": 200, "body": {"choices": [{
                "finish_reason": "stop", "message": {"content": scorer.json.dumps(raw)}}]}})
    return {"stopped": True, "calls": calls}


def test_unchanged_pool_stops_without_another_panel_and_feedback_rebuilds(tmp_path):
    selected, row, first, records = row_fixture()
    persist_records(tmp_path, row, records)
    ledger = calls_for(tmp_path, row, records)
    _, audit = scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)
    assert audit["unchanged_pools_stopped"] == 2
    assert audit["feedback_packets_rebuilt"] == 1
    assert audit["first_round_panels_replayed"] == 2
    records["evidence_feedback"]["review_evidence_feedback"]["candidates"][0]["valid_observations"][0]["observation"] = "Invented."
    persist_records(tmp_path, row, records)
    with pytest.raises(ValueError, match="feedback"):
        scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)


def expanded_fixture(tmp_path):
    selected, row, first, records = row_fixture()
    proposal = {t: [] for t in scorer.TASKS[:4]}
    proposal["ivt"] = [7]
    for record in records.values():
        record.update(panel_record(first["graph_r1"]["prediction"], proposal, first["graph_r1"]["pool"]))
        record.update(reviewed=True, review_calls_observed=5, new_review_calls=5,
                      review_request_fingerprints={s: s for s in SEATS}, panel_seconds=2.0, total_seconds=3.0)
    persist_records(tmp_path, row, records)
    ledger = calls_for(tmp_path, row, records)
    return selected, row, first, records, ledger


def test_expanded_pool_replays_raw_scores_and_detects_tampered_decision(tmp_path):
    selected, row, first, records, ledger = expanded_fixture(tmp_path)
    _, audit = scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)
    assert audit["second_round_panels_replayed"] == 2
    records["evidence_feedback"]["means"]["instrument_0"] = 5
    persist_records(tmp_path, row, records)
    with pytest.raises(ValueError, match="means"):
        scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)


def test_shared_reviews_require_identical_pool_raw_and_request_fingerprints(tmp_path):
    selected, row, first, records, _ = expanded_fixture(tmp_path)
    records["issues_only"].update(shared_from="evidence_feedback", new_review_calls=0, panel_seconds=None)
    persist_records(tmp_path, row, records)
    ledger = calls_for(tmp_path, row, records)
    _, audit = scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)
    assert audit["panels_shared"] == 1
    records["issues_only"]["review_request_fingerprints"]["gpt"] = "different"
    persist_records(tmp_path, row, records)
    with pytest.raises(ValueError, match="shared review"):
        scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)


def test_incomplete_panel_cannot_claim_a_completed_review_or_apply_edits(tmp_path):
    selected, row, first, records, _ = expanded_fixture(tmp_path)
    record = records["evidence_feedback"]
    record["raw"]["qwen"] = None
    reviews, formats = normalize_five(record["raw"], record["pool"], 3)
    means, diagnostics = panel.aggregate(reviews, record["pool"], image_count=3)
    record.update(reviews=reviews, format_diagnostics=formats, means=means, diagnostics=diagnostics,
                  status="INCOMPLETE_PANEL", reviewed=False, review_calls_observed=4, new_review_calls=4,
                  prediction=first["graph_r1"]["prediction"], issues=first["graph_r1"]["issues"])
    persist_records(tmp_path, row, records)
    ledger = calls_for(tmp_path, row, records)
    ledger["calls"] = [c for c in ledger["calls"] if not (c["stage"] == "evidence_feedback_review_2" and c["seat"] == "qwen")]
    scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)
    record["prediction"] = deepcopy(first["graph_r1"]["prediction"])
    record["prediction"]["ivt"] = [7]
    persist_records(tmp_path, row, records)
    with pytest.raises(ValueError, match="incomplete round"):
        scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)


def test_selection_failure_must_reproduce_an_actual_exception(tmp_path, monkeypatch):
    selected, row, first, records, ledger = expanded_fixture(tmp_path)
    for record in records.values():
        record.update(status="SELECTION_FAILED", prediction=first["graph_r1"]["prediction"],
                      issues=first["graph_r1"]["issues"])
    persist_records(tmp_path, row, records)
    with pytest.raises(ValueError, match="claimed selection failure"):
        scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)
    actual = panel.select

    def failed(current, pool, means, **kwargs):
        if any(p["task"] == "ivt" for p in pool["propositions"]):
            raise ValueError("simulated failure")
        return actual(current, pool, means, **kwargs)

    monkeypatch.setattr(panel, "select", failed)
    scorer.audit_records(tmp_path, {"selection": [selected]}, [row], [first], ledger)


def test_masked_heads_do_not_affect_metrics_changes_or_candidate_coverage():
    _, row, first, records = row_fixture()
    row["arms"] = {a: {"prediction": deepcopy(row["h0"])} for a in scorer.ARMS}
    row["arms"]["evidence_feedback"]["prediction"]["verb"] = [1]
    row["arms"]["evidence_feedback"]["prediction"]["instrument"] = [0, 1]
    truth = [{"video_id": row["video_id"], "frame_id": row["frame_id"],
              "gt": {"instrument": [0, 1], "phase": [2]},
              "mask": {t: t in {"instrument", "phase"} for t in scorer.TASKS}}]
    records.update(control_r1=first["control_r1"], graph_r1=first["graph_r1"])
    result, details = scorer.summarize([row], truth, {row["key"]: records})
    assert result["metrics"]["h0"]["tasks"]["instrument"]["fn"] == 1
    assert result["metrics"]["evidence_feedback"]["tasks"]["instrument"]["micro_f1"] == 1
    assert result["metrics"]["evidence_feedback"]["tasks"]["verb"]["micro_f1"] is None
    assert result["candidate_coverage"]["evidence_feedback"]["verb"]["valid_targets"] == 0
    assert details[0]["comparisons"]["graph_r1_to_evidence_feedback"]["category"] == "fully_corrected"
    assert "verb" not in details[0]["comparisons"]["graph_r1_to_evidence_feedback"]["tasks"]


def test_runtime_keeps_shared_panels_out_of_independent_latency(tmp_path):
    _, row, _, records, _ = expanded_fixture(tmp_path)
    records["issues_only"].update(shared_from="evidence_feedback", new_review_calls=0, panel_seconds=None)
    ledger = calls_for(tmp_path, row, records)
    result = scorer.runtime({row["key"]: records}, ledger)
    assert result["arms"]["issues_only"]["independent_panel_seconds"]["n"] == 0
    assert result["arms"]["evidence_feedback"]["independent_panel_seconds"]["sum"] == 2
    assert result["incremental_costs"]["all"]["calls"] == 7
    assert result["incremental_costs"]["all"]["charges"]["openrouter_usd"] == "0.07"


def frozen_fixture(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    selected, row, first, records = row_fixture()
    selection, initial, rows, original = [], [], [], []
    for index in range(8):
        current, old, pick = deepcopy(row), deepcopy(first), deepcopy(selected)
        for item in (current, old, pick):
            item.update(key=f"VID103_{101 + index * 25}", frame_id=101 + index * 25)
        selection.append(pick)
        initial.append(old)
        persist_records(output, current, deepcopy(records))
        rows.append(current)
        original.append({**{k: current[k] for k in ("key", "video_id", "frame_id", "h0")}, "arms": {
            "control": {"prediction": current["control_r1"]}, "prior_graph": {"prediction": current["graph_r1"]}}})
        for arm, name in (("control", "control_r1"), ("prior_graph", "graph_r1")):
            save(source / "targets" / current["key"] / f"{arm}.json", old[name])
        save(source / "targets" / current["key"] / "hints.json", old["hints"])
    for name, data in (("plan", {"selection": selection}), ("completion", {}),
                       ("predictions", {"targets": original}), ("budget", {})):
        save(source / f"{name}.json", data)
    plan = {"profile": scorer.PROFILE, "arms": list(scorer.ARMS), "threshold": 4, "round_cap": 2,
            "max_calls": 84, "selection": selection, "source_root": str(source),
            "source_snapshot_sha256": {p.relative_to(source).as_posix(): scorer.sha(p) for p in source.rglob("*.json")},
            "source_sha256": {"scripts/score_prior_feedback_continuation.py": scorer.sha(scorer.ROOT / "scripts/score_prior_feedback_continuation.py")}}
    for name, data in (("plan", plan), ("predictions", {"targets": rows}),
                       ("initial_state", {"targets": initial}), ("budget", {"stopped": True, "calls": []})):
        save(output / f"{name}.json", data)
    done = {"closed_utc": "test", **{f"{n}_sha256": scorer.sha(output / f"{n}.json")
        for n in ("plan", "predictions", "initial_state", "budget")},
        "inference_artifact_sha256": {p.relative_to(output).as_posix(): scorer.sha(p) for p in (output / "targets").rglob("*.json")}}
    save(output / "completion.json", done)
    return output, plan, done


def test_snapshot_guard_rejects_changed_raw_artifact_and_omitted_target_even_if_resigned(tmp_path):
    output, _, done = frozen_fixture(tmp_path)
    assert len(scorer.validate(output)[1]) == 8
    saved = scorer.read(output / "predictions.json")
    saved["targets"].pop()
    save(output / "predictions.json", saved)
    with pytest.raises(ValueError, match="snapshot"):
        scorer.validate(output)
    done["predictions_sha256"] = scorer.sha(output / "predictions.json")
    save(output / "completion.json", done)
    with pytest.raises(ValueError, match="eight original"):
        scorer.validate(output)


def test_snapshot_guard_rejects_unclosed_inference_before_gt(tmp_path):
    output, _, _ = frozen_fixture(tmp_path)
    save(output / "budget.json", {"stopped": False, "calls": []})
    with pytest.raises(ValueError, match="closed"):
        scorer.validate(output)


def test_snapshot_guard_rejects_modified_initial_evidence(tmp_path):
    output, _, _ = frozen_fixture(tmp_path)
    path = next((output / "targets").rglob("*.json"))
    record = scorer.read(path)
    record["input_issues"] = []
    save(path, record)
    with pytest.raises(ValueError, match="artifact changed"):
        scorer.validate(output)
