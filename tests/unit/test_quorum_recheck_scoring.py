"""Audits isolate validity-floor changes from extra reviews and preserve masks."""
from copy import deepcopy

import pytest

from scripts import score_quorum_recheck_trial as scorer
from scripts.run_prior_panel_trial import save
from scripts.run_repair_revision_trial import normalize_five
from surgical_agent.research.verification import flexible_quorum as quorum
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def raw_for(pool):
    raw = {s: {"judgments": {p["id"]: {"rating": 4, "finding": "MATCH", "scope": "LOCAL_REGION",
        "image_indices": [2], "observation": "Instrument acts directly on the described tissue."}
        for p in pool["propositions"]}} for s in SEATS}
    for seat in SEATS:
        raw[seat]["judgments"]["instrument_0"].update(rating=5)
    raw["deepseek"]["judgments"]["instrument_0"].update(rating=1, finding="REFUTED", scope="WHOLE_FRAME")
    raw["qwen"]["judgments"]["ivt_7"].update(rating=2, finding="REFUTED", scope="LOCAL_REGION")
    return raw


def initial_fixture():
    h0 = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}
    proposal = {"instrument": [], "verb": [], "target": [], "ivt": [7]}
    pool = make_pool(h0, proposal, make_pool(h0))
    raw = raw_for(pool)
    reviews, formats = normalize_five(raw, pool, 3)
    means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
    prediction = panel.select(h0, pool, means)
    old = {"pool": pool, "proposal": proposal, "raw": raw, "reviews": reviews,
        "format_diagnostics": formats, "means": means, "diagnostics": diagnostics,
        "prediction": prediction, "issues": panel.unresolved(prediction, pool, means, diagnostics), "status": "UNRESOLVED"}
    first = {"key": "VID103_101", "video_id": "VID103", "frame_id": 101, "h0": h0,
             "graph_r1": old, "control_r1": old, "hints": {"packet": None}}
    states = {}
    for name, floor in scorer.POLICIES.items():
        means, diagnostics = quorum.aggregate(reviews, pool, minimum_valid=floor, image_count=3)
        prediction = panel.select(h0, pool, means)
        states[name] = {"minimum_valid": floor, "prediction": prediction, "means": means,
            "diagnostics": diagnostics, "queue": quorum.recheck_queue(prediction, pool, means, diagnostics)}
    return first, {"key": first["key"], "policies": states}


def audit_fixture(tmp_path):
    first, ready = initial_fixture()
    pool = first["graph_r1"]["pool"]
    raw = raw_for(pool)
    reviews, formats = normalize_five(raw, pool, 3)
    record = {"pool": pool, "raw": raw, "reviews": reviews, "format_diagnostics": formats,
        "panel_seconds": 3.0, "review_calls_observed": 5, "reviewed": True, "attempted": True, "policies": {}}
    for name, minimum in scorer.POLICIES.items():
        before = ready["policies"][name]
        means, diagnostics = quorum.aggregate(reviews, pool, minimum_valid=minimum)
        prediction = panel.select(before["prediction"], pool, means)
        queue = quorum.recheck_queue(prediction, pool, means, diagnostics)
        record["policies"][name] = {"minimum_valid": minimum, "before": before["prediction"],
            "queue": before["queue"], "eligible": bool(before["queue"]), "means": means,
            "diagnostics": diagnostics, "prediction": prediction, "queue_after": queue,
            "status": "REVIEWED_PENDING" if queue else "REVIEWED_NO_PENDING"}
    source = tmp_path / "source"
    save(source / "targets" / first["key"] / "evidence_feedback.json", {
        "reviewed": False, "review_calls_observed": 0, "status": "NO_NEW_CANDIDATES", "pool": pool})
    row = {k: first[k] for k in ("key", "video_id", "frame_id", "h0")}
    row.update(graph_r1=first["graph_r1"]["prediction"], graph_feedback_r2=first["graph_r1"]["prediction"],
        quorum4_r1=ready["policies"]["q4"]["prediction"], archived_feedback_quorum4=first["graph_r1"]["prediction"],
        direct_recheck_5=record["policies"]["q5"]["prediction"], direct_recheck_4=record["policies"]["q4"]["prediction"])
    selected = {k: first[k] for k in ("key", "video_id", "frame_id")}
    selected["causal_frame_ids"] = [51, 76, 101]
    plan = {"selection": [selected], "source_root": str(source), "max_calls": 5, "eligible_targets": [first["key"]]}
    calls = []
    for index, seat in enumerate(SEATS):
        call = {"index": index, "target": first["key"], "stage": "direct_review_2", "seat": seat,
            "status": "JSON_PARSED", "account": "openrouter_usd", "charge": "0.01", "charge_kind": "native", "usage": {}}
        calls.append(call)
        folder = tmp_path / "calls" / f"{index:03d}_{first['key']}_direct_review_2_{seat}"
        save(folder / "response.json", {"http_status": 200, "body": {"choices": [{"finish_reason": "stop",
            "message": {"content": scorer.json.dumps(raw[seat])}}]}})
    save(tmp_path / "targets" / first["key"] / "direct.json", record)
    return plan, row, first, ready, record, {"stopped": True, "calls": calls}


def test_independent_arithmetic_omits_invalid_votes_without_fabricating_scores():
    first, _ = initial_fixture()
    pool, reviews = first["graph_r1"]["pool"], first["graph_r1"]["reviews"]
    q4, diag4, _ = scorer.checked_aggregate(reviews, pool, 4, 3)
    q5, _, _ = scorer.checked_aggregate(reviews, pool, 5, 3)
    assert q4["ivt_7"] == 4
    assert q5["ivt_7"] is None
    assert diag4["ivt_7"]["scores"] == [4, None, 4, 4, 4]
    assert diag4["ivt_7"]["valid_count"] == 4
    assert "qwen" not in diag4["ivt_7"]["valid_seats"]


def test_independent_guard_detects_wrong_average_denominator(monkeypatch):
    first, _ = initial_fixture()
    actual = quorum.aggregate

    def broken(*args, **kwargs):
        means, diagnostics = actual(*args, **kwargs)
        means["ivt_7"] = 3.2
        return means, diagnostics

    monkeypatch.setattr(quorum, "aggregate", broken)
    with pytest.raises(ValueError, match="wrong denominator"):
        scorer.checked_aggregate(first["graph_r1"]["reviews"], first["graph_r1"]["pool"], 4, 3)


def test_initial_floor5_matches_archive_and_queue_retains_valid_disagreement():
    first, ready = initial_fixture()
    pool, states = scorer.replay_initial(first, ready, 3)
    assert states["q5"]["prediction"] == first["graph_r1"]["prediction"]
    assert states["q5"]["prediction"]["ivt"] == []
    assert states["q4"]["prediction"]["ivt"] == [7]
    assert [v["candidate_id"] for v in states["q5"]["queue"]] == ["instrument_0"]
    ready["policies"]["q5"]["queue"] = []
    with pytest.raises(ValueError, match="prepared first-round"):
        scorer.replay_initial(first, ready, 3)
    assert len(pool["propositions"]) == 4


def test_raw_replay_rejects_tampered_prediction_and_new_candidate(tmp_path):
    plan, row, first, ready, record, ledger = audit_fixture(tmp_path)
    _, _, audit = scorer.audit_records(tmp_path, plan, [row], [first], [ready], ledger)
    assert audit["direct_policy_states_replayed"] == 2
    record["policies"]["q5"]["prediction"] = deepcopy(record["policies"]["q5"]["prediction"])
    record["policies"]["q5"]["prediction"]["ivt"] = [7]
    save(tmp_path / "targets" / first["key"] / "direct.json", record)
    with pytest.raises(ValueError, match="direct selection"):
        scorer.audit_records(tmp_path, plan, [row], [first], [ready], ledger)
    record["pool"] = make_pool(first["h0"], {"instrument": [], "verb": [], "target": [], "ivt": [17]}, record["pool"])
    save(tmp_path / "targets" / first["key"] / "direct.json", record)
    with pytest.raises(ValueError, match="changed candidate pool"):
        scorer.audit_records(tmp_path, plan, [row], [first], [ready], ledger)


def test_incomplete_dispatch_retains_each_policys_own_initial_labels(tmp_path):
    plan, row, first, ready, record, ledger = audit_fixture(tmp_path)
    ledger["calls"].pop()
    record["raw"]["deepseek"] = None
    record["reviews"], record["format_diagnostics"] = normalize_five(record["raw"], record["pool"], 3)
    record.update(review_calls_observed=4, reviewed=False)
    for name, old in ready["policies"].items():
        record["policies"][name].update(prediction=old["prediction"], queue_after=old["queue"],
            means=old["means"], diagnostics=old["diagnostics"], status="INCOMPLETE_PANEL")
    save(tmp_path / "targets" / first["key"] / "direct.json", record)
    scorer.audit_records(tmp_path, plan, [row], [first], [ready], ledger)
    assert record["policies"]["q5"]["prediction"]["ivt"] == []
    assert record["policies"]["q4"]["prediction"]["ivt"] == [7]


def test_archived_feedback_replay_starts_original_graph_not_quorum4_initial(tmp_path):
    first, ready = initial_fixture()
    pool = first["graph_r1"]["pool"]
    save(tmp_path / "targets" / first["key"] / "evidence_feedback.json", {
        "reviewed": False, "review_calls_observed": 0, "status": "NO_NEW_CANDIDATES", "pool": pool})
    prediction, trace = scorer.archived_feedback_replay(tmp_path, first, 3)
    assert prediction["ivt"] == []
    assert ready["policies"]["q4"]["prediction"]["ivt"] == [7]
    assert trace["applied"] is False


def test_masks_exclude_invalid_gt_from_all_policies_and_attribution(tmp_path):
    plan, row, first, ready, _, ledger = audit_fixture(tmp_path)
    records, traces, _ = scorer.audit_records(tmp_path, plan, [row], [first], [ready], ledger)
    gt = {"video_id": row["video_id"], "frame_id": row["frame_id"],
        "mask": {t: t == "ivt" for t in scorer.TASKS}, "gt": {"ivt": [7]}}
    result, details, changes = scorer.summarize([row], [gt], records, [first], traces)
    assert result["metrics"]["direct_recheck_4"]["tasks"]["ivt"]["micro_f1"] == 1
    assert result["metrics"]["direct_recheck_5"]["tasks"]["ivt"]["fn"] == 1
    assert result["metrics"]["direct_recheck_4"]["tasks"]["target"]["micro_f1"] is None
    assert result["candidate_coverage"]["direct_recheck_4"]["ivt"]["pool_recall"] == 1
    assert result["candidate_coverage"]["direct_recheck_4"]["target"]["valid_targets"] == 0
    assert details[0]["comparisons"]["direct_recheck_5_to_direct_recheck_4"]["category"] == "fully_corrected"
    assert all(c["candidate_id"].startswith("ivt_") for c in changes)


def test_costs_keep_reserved_unknown_amount_separate_from_confirmed_spending():
    call = {"account": "openrouter_usd", "charge": "0.01", "charge_kind": "native", "status": "JSON_PARSED", "usage": {}}
    failed = {**call, "charge": "0.10", "charge_kind": "unknown_reserved", "status": "API_FAILED"}
    result = scorer.runtime({}, {"calls": [call, failed]}, {"inference_seconds": 3})
    assert result["new_calls"] == 2
    assert result["accounts"]["openrouter_usd"]["by_charge_kind"] == {"native": "0.01", "unknown_reserved": "0.10"}
    assert result["accounts"]["openrouter_usd"]["total_including_unknown_reserved"] == "0.11"


def closed_minimal(tmp_path):
    for name, value in (("plan", {}), ("predictions", {}), ("initial_state", {}),
                        ("policy_state", {}), ("budget", {"stopped": True, "calls": []})):
        save(tmp_path / f"{name}.json", value)
    done = {"closed_utc": "test", **{f"{name}_sha256": scorer.sha(tmp_path / f"{name}.json")
        for name in ("plan", "predictions", "initial_state", "policy_state", "budget")}}
    save(tmp_path / "completion.json", done)
    return done


def test_no_gt_scoring_before_closed_ledger_or_after_snapshot_changes(tmp_path, monkeypatch):
    closed_minimal(tmp_path)
    monkeypatch.setattr(scorer, "score_saved", lambda *a, **k: pytest.fail("GT opened before validation"))
    save(tmp_path / "budget.json", {"stopped": False, "calls": []})
    with pytest.raises(ValueError, match="closed inference"):
        scorer.score(tmp_path, object())
    closed_minimal(tmp_path)
    save(tmp_path / "policy_state.json", {"targets": []})
    with pytest.raises(ValueError, match="snapshot changed"):
        scorer.score(tmp_path, object())


def test_frozen_rates_accept_json_array_roundtrip_but_reject_price_change(tmp_path, monkeypatch):
    done = closed_minimal(tmp_path)
    plan = {"profile": scorer.PROFILE, "threshold": 4, "round_cap": 2,
        "models": scorer.MODELS, "h0": scorer.PROPOSER, "policies": scorer.POLICIES,
        "rates": scorer.json.loads(scorer.json.dumps(scorer.RATES_V2)),
        "limits": {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"},
        "max_calls": 35, "source_sha256": {"placeholder": "placeholder"}}

    def reached_hash_audit(*args):
        raise RuntimeError("reached hash audit")

    monkeypatch.setattr(scorer.archived, "_hashes", reached_hash_audit)
    save(tmp_path / "plan.json", plan)
    done["plan_sha256"] = scorer.sha(tmp_path / "plan.json")
    save(tmp_path / "completion.json", done)
    with pytest.raises(RuntimeError, match="reached hash audit"):
        scorer.validate(tmp_path)
    plan["rates"]["grok"][0] = "0.01"
    save(tmp_path / "plan.json", plan)
    done["plan_sha256"] = scorer.sha(tmp_path / "plan.json")
    save(tmp_path / "completion.json", done)
    with pytest.raises(ValueError, match="policy or models differ"):
        scorer.validate(tmp_path)
