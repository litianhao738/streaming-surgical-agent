"""Offline end-to-end regressions for the isolated paired repair experiment."""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts import run_repair_revision_trial as trial
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


@pytest.fixture(autouse=True)
def _offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this regression suite must never access a provider or credential")
    monkeypatch.setattr("scripts.run_candidate_panel_trial.key_for", forbidden)
    monkeypatch.setattr(trial.requests, "post", forbidden)
    monkeypatch.setattr(trial.requests, "get", forbidden)


def _labels(empty=False):
    return {"instrument": [] if empty else [0], "verb": [], "target": [], "ivt": [], "phase": [2]}


def _prepare_fake(monkeypatch, tmp_path, *, rounds=3, empty=False, missing_proposals=False):
    output, previous = tmp_path / "run", tmp_path / "previous"
    prior = {"stopped": True, "occupied": {k: "0" for k in trial.ALLOWANCE}}
    trial.save(previous / "budget.json", prior)
    selected = {"key": "VID103_51", "video_id": "VID103", "frame_id": 51,
                "request_metadata": {}, "images": []}
    trial.save(output / "plan.json", {
        "profile": trial.PROFILE, "source_sha256": {}, "selection": [selected],
        "previous_budget": str(previous), "previous_budget_sha256": trial.sha(previous / "budget.json"),
        "carried_occupied": prior["occupied"], "limits": {k: "10" for k in trial.ALLOWANCE},
        "rates": trial.RATES_V2, "max_calls": 34, "round_cap": rounds,
    })
    dispatched, scored = [], []

    class FakeCalls:
        def __init__(self, *args, **kwargs):
            self.rows, self.stopped = [], False

        def call(self, target, stage, seat, body):
            dispatched.append((target, stage, seat, deepcopy(body)))
            self.rows.append({"charge": "0", "charge_kind": "native", "account": "openrouter_usd"})
            if stage == "h0":
                labels = _labels(empty)
                return {"schema_version": "joint_perception_final_only_v1",
                        **{t: {"selected_ids": labels[t]} for t in trial.TASKS[:-1]},
                        "phase": {"selected_id": 2}}
            if "_proposal_" in stage:
                return {t: [] for t in trial.TASKS[:-1]}
            items = {p["id"]: {"rating": 5, "finding": "MATCH", "scope": "LOCAL_REGION",
                              "image_indices": [2], "observation": "Fixture support."}
                     for p in body["pool"]["propositions"]}
            answer = ({"rows": [{"candidate_id": pid, **v} for pid, v in items.items()]}
                      if seat == "gemini" else {"judgments": items})
            if stage.startswith(trial.ARMS[1]) and not missing_proposals:
                existing = set(items)
                answer["proposals"] = {"ivt": [] if "ivt_7" in existing else [7]}
            return answer

        def persist(self):
            pass

    monkeypatch.setattr(trial, "RevisionCalls", FakeCalls)
    monkeypatch.setattr(trial, "build_gemini_base", lambda *a: SimpleNamespace(images=[1, 2, 3]))
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda *a: SimpleNamespace(to_mapping=dict))
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda *a: {"stage": "h0"})
    monkeypatch.setattr(trial, "gemini_proposal", lambda *a: {"issues": a[-1]})
    monkeypatch.setattr(trial, "review_wire", lambda *a: {"pool": deepcopy(a[-1])})
    monkeypatch.setattr(trial, "review_and_propose_wire", lambda *a: {"pool": deepcopy(a[-1])})

    def score_subprocess(command, **kwargs):
        number = int(command[-1])
        scored.append(number)
        assert (output / f"round_{number}_predictions.json").exists()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(trial.subprocess, "run", score_subprocess)
    return output, selected["key"], dispatched, scored


@pytest.mark.parametrize("empty", [False, True])
def test_new_proposals_require_another_round_and_do_not_change_comparator(monkeypatch, tmp_path, empty):
    output, key, dispatched, scored = _prepare_fake(monkeypatch, tmp_path, empty=empty)
    trial.execute(output, None)
    states = trial.read(output / "inference_states.json")[key]
    first, second = states["arms"][trial.ARMS[1]]["history"]
    assert states["h0"] == _labels(empty)
    assert first["after"] == _labels(empty)
    assert first["pending"]["queued_ivt"] == [7]
    assert "ivt_7" not in first["means"]
    assert second["means"]["ivt_7"] == 5
    assert second["after"]["ivt"] == [7]
    assert second["after"]["phase"] == [2]
    assert states["arms"][trial.ARMS[0]]["current"] == _labels(empty)
    assert states["arms"][trial.ARMS[1]]["status"] == "MODEL_PASS"
    assert sum(stage == "h0" for _, stage, _, _ in dispatched) == 1
    assert scored == [1, 2]
    for number in (1, 2):
        votes = [(seat, body) for _, stage, seat, body in dispatched
                 if stage == f"{trial.ARMS[1]}_review_{number}"]
        assert {seat for seat, _ in votes} == set(SEATS)
        assert all(body["pool"] == votes[0][1]["pool"] for _, body in votes)
    with pytest.raises(ValueError, match="replay"):
        trial.execute(output, None)


def test_final_round_proposals_are_archived_but_never_accepted(monkeypatch, tmp_path):
    output, key, _, scored = _prepare_fake(monkeypatch, tmp_path, rounds=1)
    trial.execute(output, None)
    state = trial.read(output / "inference_states.json")[key]["arms"][trial.ARMS[1]]
    assert state["current"] == _labels()
    assert state["history"][0]["pending"]["queued_ivt"] == [7]
    assert state["status"] != "MODEL_PASS"
    assert not state["active"]
    assert scored == [1]


def test_missing_proposal_contract_cannot_establish_model_pass(monkeypatch, tmp_path):
    output, key, _, _ = _prepare_fake(monkeypatch, tmp_path, missing_proposals=True)
    trial.execute(output, None)
    state = trial.read(output / "inference_states.json")[key]["arms"][trial.ARMS[1]]
    assert state["current"] == _labels()
    assert state["status"] != "MODEL_PASS"
    assert all(d["proposal_status"] != "VALID"
               for d in state["history"][-1]["proposal_diagnostics"].values())


def test_suggestion_votes_only_bound_pool_and_never_accept_labels():
    h0 = _labels()
    pool = make_pool(h0)
    suggestions = {s: {"ivt": [7, 17]} for s in SEATS}
    larger, diagnostic = trial.merge_suggestions(h0, pool, suggestions)
    assert h0 == _labels()
    assert pool == make_pool(h0)
    assert {p["label_id"] for p in larger["propositions"] if p["task"] == "ivt"} == {7, 17}
    assert diagnostic["queued_ivt"] == [7, 17]
    assert diagnostic["counts_are_not_visual_confidence"] is True
    with pytest.raises(ValueError, match="invalid"):
        trial.merge_suggestions(h0, pool, {SEATS[0]: {"ivt": [True]}})


def test_candidate_cap_keeps_existing_pool_and_does_not_partially_insert_ivt(monkeypatch):
    h0 = _labels()
    pool = make_pool(h0)
    original = deepcopy(pool)
    real_make_pool = trial.make_pool

    def constrained_make_pool(current, proposal, previous):
        # Simulate an expanded IVT plus components exceeding the real pool cap.
        if proposal["ivt"] == [7]:
            raise ValueError("candidate pool cap exceeded")
        return real_make_pool(current, proposal, previous)

    monkeypatch.setattr(trial, "make_pool", constrained_make_pool)
    expanded, diagnostic = trial.merge_suggestions(h0, pool, {s: {"ivt": [7, 17]} for s in SEATS})
    assert pool == original and h0 == _labels()
    assert diagnostic["queued_ivt"] == [17]
    assert diagnostic["dropped"] == [{"ivt": 7, "reason": "POOL_CAP"}]
    assert "ivt_7" not in {p["id"] for p in expanded["propositions"]}
    assert "ivt_17" in {p["id"] for p in expanded["propositions"]}


def test_unreviewed_capped_proposals_cannot_establish_model_pass(monkeypatch, tmp_path):
    output, key, _, _ = _prepare_fake(monkeypatch, tmp_path, rounds=1)

    def full_pool(current, pool, suggestions):
        return deepcopy(pool), {"queued_ivt": [], "dropped": [{"ivt": 7, "reason": "POOL_CAP"}],
                                "proposer_count": {7: 5}, "counts_are_not_visual_confidence": True}

    monkeypatch.setattr(trial, "merge_suggestions", full_pool)
    trial.execute(output, None)
    state = trial.read(output / "inference_states.json")[key]["arms"][trial.ARMS[1]]
    assert state["current"] == _labels()
    assert state["status"] != "MODEL_PASS"


def test_source_or_image_drift_rejected_before_call_dispatch(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x = 1", encoding="utf-8")
    plan = {"source_sha256": {str(source): trial.sha(source)}, "selection": []}
    trial.verify_plan(plan)
    source.write_text("x = 2", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        trial.verify_plan(plan)
    image = tmp_path / "image.png"
    image.write_bytes(b"fixed")
    plan = {"source_sha256": {}, "selection": [{"images": [{"path": str(image), "sha256": trial.sha(image)}]}]}
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        trial.verify_plan(plan)


@pytest.mark.parametrize("content,initial_parsed,exception,expected", [
    ('{"judgments": {}}```', None, "JSONDecodeError", {"judgments": {}}),
    ('{"judgments": ', None, "JSONDecodeError", None),
    ('{"judgments": {}, "judgments": {}}', {"judgments": {}}, None, None),
    ('{"judgments": {}}```', None, "ValueError", None),
])
def test_review_recovery_preserves_transport_checks_and_json_ambiguity(
        monkeypatch, tmp_path, content, initial_parsed, exception, expected):
    def fake_transport(self, target, stage, seat, body):
        row = {"index": 0, "target": target, "stage": stage, "seat": seat,
               "http_status": 200, "status": "JSON_PARSED" if initial_parsed is not None else "FAILED",
               "account": "openrouter_usd", "charge": "0.01", "charge_kind": "native"}
        if exception:
            row["exception_type"] = exception
        self.rows.append(row)
        folder = self.output / "calls" / f"000_{target}_{stage}_{seat}"
        trial.save(folder / "response.json", {"http_status": 200, "body": {"choices": [
            {"finish_reason": "stop", "message": {"content": content}}]}})
        return initial_parsed

    monkeypatch.setattr(trial.Calls, "call", fake_transport)
    calls = trial.RevisionCalls(tmp_path)
    result = calls.call("VID103_51", "review_1", "gemini", {})
    assert result == expected
    assert len(calls.rows) == 1 and calls.rows[0]["charge"] == "0.01"
    if expected is None:
        assert not calls.rows[0]["status"].startswith("JSON_PARSED")


def test_score_separates_failed_h0_from_paired_repair_observations(monkeypatch, tmp_path):
    rows = [
        {"video_id": "VID103", "frame_id": 51, "h0": _labels(),
         "arms": {a: {"final": _labels(), "actual_rounds": 1, "status": "MODEL_PASS"}
                  for a in trial.ARMS}},
        {"video_id": "VID103", "frame_id": 76, "h0": None,
         "arms": {a: {"final": None, "actual_rounds": 0, "status": "H0_FAILED"}
                  for a in trial.ARMS}},
    ]
    trial.save(tmp_path / "round_1_predictions.json", rows)
    requested = []

    def score_saved(adapter, data):
        requested.append(deepcopy(data))
        return {"arms": {"final": {"tasks": {t: {"micro_f1": 0} for t in trial.TASKS}}}}, data

    monkeypatch.setattr(trial, "score_saved", score_saved)
    trial.score(tmp_path, None, 1)
    report = trial.read(tmp_path / "scores/round_1.json")
    assert report["coverage"]["selected"] == 2
    assert report["coverage"]["h0_available"] == 1
    assert report["coverage"]["paired_with_both_reviewed"] == 1
    assert report["coverage"]["unfinished_observations_are_not_gate_negative_labels"]
    assert sorted(len(x) for x in requested) == [1, 1, 2, 2]
    assert all(x[0]["h0"] is not None for x in requested if len(x) == 1)


def _cache_fixture(tmp_path, *, status="JSON_PARSED", stopped=True):
    source, output = tmp_path / "source", tmp_path / "new"
    key = ("VID103_51", "review_1", "gemini")
    folder = source / "calls" / f"000_{key[0]}_{key[1]}_{key[2]}"
    body = {"model": trial.MODELS["gemini"], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "fixed protocol"}]}]}
    row = {"index": 0, "target": key[0], "stage": key[1], "seat": key[2],
           "status": status, "http_status": 200 if status == "JSON_PARSED" else 400,
           "model": body["model"], "charge": "0.10", "charge_kind": "native",
           "account": "openrouter_usd"}
    occupied = {k: "0.50" for k in trial.ALLOWANCE}
    limits = {k: "2.00" for k in trial.ALLOWANCE}
    trial.save(folder / "request.json", body)
    trial.save(folder / "record.json", row)
    trial.save(folder / "response.json", {"http_status": row["http_status"], "body": {
        "model": body["model"], "provider": trial.PROVIDERS.get("gemini"),
        "choices": [{"finish_reason": "stop", "message": {"content": '{"judgments": {}}'}}]}})
    trial.save(source / "budget.json", {"stopped": stopped, "calls": [row],
                                        "occupied": occupied, "limits": limits})
    return source, output, key, body, occupied, limits


def test_exact_wire_cache_has_no_new_charge_or_budget_reset(tmp_path):
    source, output, key, body, occupied, limits = _cache_fixture(tmp_path)
    calls = trial.RevisionCalls(output, occupied, reuse_source=source,
                               limits={k: Decimal(v) for k, v in limits.items()})
    before = deepcopy(calls.occupied)
    assert calls.call(*key, body) == {"judgments": {}}
    assert calls.call(*key, body) == {"judgments": {}}
    assert calls.rows == [] and calls.occupied == before
    assert calls.occupied == {k: Decimal(v) for k, v in occupied.items()}
    assert calls.limits == {k: Decimal(v) for k, v in limits.items()}
    assert len(calls.cache_hits) == 2
    assert all(hit["request_sha256"] and hit["response_sha256"] for hit in calls.cache_hits)
    calls.persist()
    ledger = trial.read(output / "budget.json")
    assert ledger["calls"] == []
    assert ledger["occupied"] == occupied


def test_cache_request_difference_stops_without_paid_fallback(tmp_path):
    source, output, key, body, occupied, limits = _cache_fixture(tmp_path)
    calls = trial.RevisionCalls(output, occupied, reuse_source=source,
                               limits={k: Decimal(v) for k, v in limits.items()})
    changed = deepcopy(body)
    changed["messages"][0]["content"][0]["text"] = "different candidate protocol"
    with pytest.raises(ValueError, match="differs"):
        calls.call(*key, changed)
    assert not calls.rows and not calls.cache_hits


def test_failed_response_is_not_cached_or_silently_reused(monkeypatch, tmp_path):
    source, output, key, body, occupied, limits = _cache_fixture(tmp_path, status="API_FAILED")
    calls = trial.RevisionCalls(output, occupied, reuse_source=source,
                               limits={k: Decimal(v) for k, v in limits.items()})
    attempts = []

    def no_network(self, target, stage, seat, request):
        attempts.append((target, stage, seat))

    monkeypatch.setattr(trial.Calls, "call", no_network)
    assert calls.call(*key, body) is None
    assert attempts == [key] and not calls.cache_hits


def test_cache_cannot_read_a_ledger_still_running(tmp_path):
    source, output, _, _, occupied, limits = _cache_fixture(tmp_path, stopped=False)
    with pytest.raises(ValueError, match="closed"):
        trial.RevisionCalls(output, occupied, reuse_source=source,
                            limits={k: Decimal(v) for k, v in limits.items()})


@pytest.mark.parametrize("change", ["zero_previous_cost", "increase_limit"])
def test_cache_reuse_cannot_reset_budget_or_enlarge_original_limit(tmp_path, change):
    source, output, _, _, occupied, limits = _cache_fixture(tmp_path)
    if change == "zero_previous_cost":
        occupied["openrouter_usd"] = "0"
    else:
        limits["openrouter_usd"] = "4"
    with pytest.raises(ValueError, match="balance or limit"):
        trial.RevisionCalls(output, occupied, reuse_source=source,
                            limits={k: Decimal(v) for k, v in limits.items()})


@pytest.mark.parametrize("mutation", ["model", "provider", "finish", "refusal"])
def test_cached_response_does_not_bypass_transport_identity_checks(tmp_path, mutation):
    source, output, key, body, occupied, limits = _cache_fixture(tmp_path)
    response_path = next((source / "calls").glob("*/response.json"))
    response = trial.read(response_path)
    if mutation in ("model", "provider"):
        response["body"][mutation] = "different"
    elif mutation == "finish":
        response["body"]["choices"][0]["finish_reason"] = "length"
    else:
        response["body"]["choices"][0]["message"]["refusal"] = "refused"
    trial.save(response_path, response)
    calls = trial.RevisionCalls(output, occupied, reuse_source=source,
                               limits={k: Decimal(v) for k, v in limits.items()})
    with pytest.raises(ValueError, match="identity or completion"):
        calls.call(*key, body)
    assert calls.rows == [] and calls.cache_hits == []


def test_prepare_reuse_keeps_original_total_limits_and_only_remaining_allowance(monkeypatch, tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"mock input")
    occupied, limits = ({k: "0.75" for k in trial.ALLOWANCE}, {k: "2.00" for k in trial.ALLOWANCE})
    prior = {"stopped": True, "occupied": occupied, "limits": limits, "calls": []}
    trial.save(source / "budget.json", prior)
    manifest = {"no_gt_label_values_used_for_selection": True, "selection": []}
    selected = []
    entries, samples = {}, {}
    for video in ("VID103", "VID23", "VID31", "VID96"):
        entries[video] = SimpleNamespace(split=trial.DatasetSplit.TRAINING)
        samples[video] = []
        for index in range(10):
            frame = 51 + index * 25
            row = {"video_id": video, "frame_id": frame, "causal_frame_ids": [frame - 50, frame - 25, frame],
                   "images": [{"path": str(image), "sha256": trial.sha(image)}] * 3,
                   "gt_availability_only": {t: True for t in trial.TASKS}}
            manifest["selection"].append(row)
            samples[video].append(SimpleNamespace(target_frame_id=frame, causal_frame_ids=row["causal_frame_ids"],
                                                  media_refs=[image] * 3))
            if index in (1, 6):
                selected.append({**deepcopy(row), "key": f"{video}_{frame}", "request_metadata": {}})
    trial.save(tmp_path / "artifacts/training/gate/final_only_preparation_20260908_v2/pilot_selection.json", manifest)
    trial.save(source / "plan.json", {"selection": selected, "arms": list(trial.ARMS), "max_calls": 272})
    adapter = SimpleNamespace(entries=entries, iter_inference_video=lambda video: iter(samples[video]))
    monkeypatch.setattr(trial, "ROOT", tmp_path)
    monkeypatch.setattr(trial, "frozen_sources", list)
    monkeypatch.setattr(trial, "build_gemini_base", lambda *args: object())
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda *args: SimpleNamespace(to_mapping=dict))
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda *args: {"messages": []})
    endpoints = [{"tag": route, "status": 0, "supported_parameters": ["response_format", "reasoning"],
                  "pricing": {"prompt": "0", "completion": "0"}} for route in trial.ROUTES.values()]
    monkeypatch.setattr(trial.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data": {"endpoints": endpoints}}))
    trial.prepare(output, source, adapter, "development", reuse_source=source)
    plan = trial.read(output / "plan.json")
    assert plan["limits"] == limits
    assert plan["carried_occupied"] == occupied
    assert all(Decimal(v) == Decimal("1.25") for v in plan["incremental_allowances"].values())
    assert plan["reuse_sha256"]
    trial.verify_plan(plan)
    prior["occupied"]["openrouter_usd"] = "0.76"
    trial.save(source / "budget.json", prior)
    with pytest.raises(ValueError, match="changed"):
        trial.verify_plan(plan)
