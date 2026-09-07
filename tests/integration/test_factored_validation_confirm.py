"""Synthetic-only confirmation guards: no dataset GT or provider network access."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_factored_validation_confirm as runner
from surgical_agent.api.credentials import SecretValue
from surgical_agent.artifacts.manifest import atomic_write_json
from tests.integration.test_factored_verifier_trial import part_review, source_request
from tests.integration.test_presence_review_trial import base, stage_response
from tests.unit.test_grounded_pipeline import _responses


def choice_fixture(tmp_path, monkeypatch):
    targets = tmp_path / "Validation_candidates.json"
    selection = [{"video_id": "VID110", "frame_id": 101 + 25 * i, "source_split": "Validation"}
                 for i in range(8)]
    atomic_write_json(targets, {"source_split": "Validation", "selection": selection})
    training = tmp_path / "training"
    training.mkdir()
    plan = {"source_split": "Training", "variants": [runner.VARIANT]}
    plan["plan_sha256"] = runner.digest(plan)
    atomic_write_json(training / "plan.json", plan)
    atomic_write_json(training / "predictions.json", {"synthetic": "training predictions"})
    atomic_write_json(training / "summary.json", {"source_split": "Training", "targets": 16,
        "variants": [runner.VARIANT], "plan_sha256": plan["plan_sha256"],
        "predictions_sha256": runner.shared.sha(training / "predictions.json")})
    (training / "scored_predictions.json").write_text("Do not parse synthetic GT", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic operator-selected single confirmation", encoding="utf-8")
    monkeypatch.setattr(runner, "sources", list)
    return SimpleNamespace(targets_json=targets, training_evidence=training, protocol=protocol,
        choice_lock=tmp_path / "selection.lock.json", select_factored_for_validation=True,
        max_review_calls=40, timeout_seconds=180, budget_ledger=tmp_path / "budget.jsonl",
        goal_id="synthetic-validation-goal")


def test_selection_is_explicit_exclusive_and_hashes_training_gt_without_parsing(tmp_path, monkeypatch):
    args = choice_fixture(tmp_path, monkeypatch)
    args.select_factored_for_validation = False
    with pytest.raises(ValueError, match="explicit"):
        runner.create_choice_lock(args)
    args.select_factored_for_validation = True
    lock = runner.create_choice_lock(args)
    assert lock["primary"] == "final_b" and lock["variant"] == "factored_names1000"
    assert lock["selection_policy"] == runner.POLICY
    assert "positive" in lock["selection_policy"]
    assert lock["selection_uses_training_results"] is True
    assert lock["gt_values_read_by_lock_writer"] is False
    assert lock["validation_gt_values_used_for_selection"] is False
    assert "gt_values_read_for_selection" not in lock
    assert runner.check_choice(args) == lock
    with pytest.raises(FileExistsError):
        runner.create_choice_lock(args)


@pytest.mark.parametrize("split", ["Training", "Testing"])
def test_wrong_split_rejected_before_training_evidence_or_calls(tmp_path, monkeypatch, split):
    args = choice_fixture(tmp_path, monkeypatch)
    manifest = runner.shared.read(args.targets_json)
    manifest["source_split"] = split
    atomic_write_json(args.targets_json, manifest)
    args.training_evidence = tmp_path / "nonexistent"
    with pytest.raises(ValueError, match="Training/Testing forbidden"):
        runner.create_choice_lock(args)


@pytest.mark.parametrize("mutation", ["primary", "budget", "manifest", "protocol", "training", "incomplete"])
def test_selection_or_bound_evidence_changes_are_rejected(tmp_path, monkeypatch, mutation):
    args = choice_fixture(tmp_path, monkeypatch)
    if mutation == "incomplete":
        summary = runner.shared.read(args.training_evidence / "summary.json")
        summary["predictions_sha256"] = "wrong"
        atomic_write_json(args.training_evidence / "summary.json", summary)
        with pytest.raises(ValueError, match="completed"):
            runner.create_choice_lock(args)
        return
    lock = runner.create_choice_lock(args)
    if mutation == "primary":
        lock["primary"] = "final_a"
        lock.pop("choice_sha256")
        lock["choice_sha256"] = runner.digest(lock)
        atomic_write_json(args.choice_lock, lock)
    elif mutation == "budget":
        args.budget_ledger = tmp_path / "another.jsonl"
    else:
        path = {"manifest": args.targets_json, "protocol": args.protocol,
                "training": args.training_evidence / "scored_predictions.json"}[mutation]
        path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch|changed after preflight"):
        runner.check_choice(args)


def test_no_lock_cannot_enter_pricing_key_or_budget(tmp_path, monkeypatch):
    args = SimpleNamespace(choice_lock=tmp_path / "missing", execute=True)
    monkeypatch.setattr(runner.shared, "start_execution", lambda *a: pytest.fail("network boundary entered"))
    with pytest.raises(FileNotFoundError):
        runner.run(args)


def test_offline_preflight_is_repeatable_and_binds_both_saved_h0_inputs(tmp_path, monkeypatch):
    args = choice_fixture(tmp_path, monkeypatch)
    runner.create_choice_lock(args)
    args.output = tmp_path / "confirmation"
    args.pricing_snapshot = tmp_path / "prices.json"
    args.dataset_root = tmp_path / "unused-synthetic-dataset"
    atomic_write_json(args.pricing_snapshot, {"offline": True})
    def prepare_collect(nested_args):
        assert nested_args.confirmation is True and nested_args.variants == [runner.VARIANT]
        assert nested_args.max_calls == 24
        selection = runner.shared.read(args.targets_json)["selection"]
        selection = [{**s, "key": f"{s['video_id']}_{s['frame_id']}"} for s in selection]
        plan = {"selection": selection, "source_artifact_sha256": {}, "pricing": {}, "reservation": {}}
        if runner.shared.save_plan(nested_args.output, plan, []):
            for selected in selection:
                atomic_write_json(nested_args.output / "requests" / f"{selected['key']}.json", {
                    "metadata": {"synthetic_key": selected["key"]}, "payload": {"frozen": True}})
        return None, plan, {}
    monkeypatch.setattr(runner.shared, "prepare_collect", prepare_collect)
    first, _ = runner.prepare(args)
    root_hash = runner.shared.sha(args.output / "plan.json")
    second, _ = runner.prepare(args)
    assert first == second and len(first["selection"]) == 8
    assert runner.shared.sha(args.output / "plan.json") == root_hash
    assert first["primary"] == "final_b" and first["max_provider_calls"] == 64
    assert first["selection_uses_training_results"] is True
    assert first["gt_values_used_in_requests"] is False
    assert first["validation_gt_values_used_for_selection"] is False
    assert "gt_values_used_in_requests_or_selection" not in first
    key = first["keys"][0]
    path = args.output / "requests" / f"{key}.json"
    atomic_write_json(path, {"metadata": {}, "payload": "changed after preflight"})
    with pytest.raises(ValueError, match="source artifact changed"):
        runner.prepare(args)
    assert runner.shared.sha(args.output / "plan.json") == root_hash


@pytest.mark.parametrize("tamper", ["wire", "native_payload", "native_failure"])
def test_real_dispatch_binding_and_native_failures_are_preserved(tmp_path, tamper):
    names, row = source_request()
    request = runner.split_factored_presence_request(names)[0]
    key, stage = "VID110_101", "p001"
    directory = tmp_path / "calls" / key / stage
    directory.mkdir(parents=True)
    saved = runner.saved_request(request)
    response = part_review(request, row)
    result = {"key": key, "stage": stage, "status": "OK", "http_status": 200,
              "request_hash": saved["metadata"]["request_hash"], "payload": response,
              "provider_request_id": "offline-native"}
    native = {"model": runner.shared.MODEL, "provider": "Alibaba", "id": "offline-native",
              "choices": [{"message": {"content": json.dumps(response)}}]}
    if tamper == "wire":
        saved["wire"]["provider"]["allow_fallbacks"] = True
    elif tamper == "native_payload":
        native["choices"][0]["message"]["content"] = json.dumps({"different": True})
    else:
        result.update(status="FAILURE", error="ValueError")
        result.pop("payload")
    atomic_write_json(directory / "request.json", saved)
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "http_response.json", {"status_code": 200,
        "body": "malformed synthetic body" if tamper == "native_failure" else json.dumps(native)})
    (directory / "dispatch.lock").write_text("offline", encoding="utf-8")
    if tamper == "native_failure":
        assert len(runner.audit_actual_receipts(tmp_path, {(key, stage): request})) == 4
    else:
        with pytest.raises(ValueError, match="binding mismatch|differs from native"):
            runner.audit_actual_receipts(tmp_path, {(key, stage): request})


def test_factored_failure_keeps_entire_frame_for_both_arms():
    names, row = source_request()
    parts = runner.split_factored_presence_request(names)
    responses = [part_review(part, row) for part in parts]
    assert runner.merge_and_evaluate(row, names, responses)["final_a"] == row["h1"]
    responses[0] = None
    result = runner.merge_and_evaluate(row, names, responses)
    assert result["final_a"] == result["final_b"] == row["h0"]
    assert result["status"] == "REVIEW_FAILURE_KEEP"


@pytest.mark.parametrize("cap, fail_first", [(1, False), (80, False), (80, True)])
def test_all_eight_persist_before_scoring_only_factored_post_and_validation_ledger(
        tmp_path, monkeypatch, cap, fail_first):
    names, original = source_request()
    original_base = base()
    numeric, evidence, full_ref = runner.shared.build_review_request(original_base, original)
    output = tmp_path / "run"
    output.mkdir()
    selected = [{"key": f"VID110_{101 + i * 25}", "video_id": "VID110", "frame_id": 101 + i * 25}
                for i in range(8)]
    plan = {"source_split": "Validation", "selection": selected, "variants": [runner.VARIANT],
            "plan_sha256": "synthetic", "choice_sha256": "synthetic-choice", "max_review_calls": cap,
            "max_provider_calls": 24 + cap, "timeout_seconds": 180, "source_sha256": {}}
    atomic_write_json(output / "plan.json", plan)
    plan_hash = runner.shared.sha(output / "plan.json")
    monkeypatch.setattr(runner, "prepare", lambda args: (plan, {s["key"]: original_base for s in selected}))
    monkeypatch.setattr(runner, "check_choice", lambda args: {})
    monkeypatch.setattr(runner.shared, "assert_frozen", lambda plan: None)
    secret = SecretValue("synthetic-key-not-real")
    ledger = runner.shared.GoalBudgetLedger(tmp_path / "budget.jsonl", "fixture")
    events = []

    def sender(url, *, data, **kwargs):
        wire = json.loads(data)
        body = json.loads(wire["messages"][1]["content"][0]["text"])
        is_review = "propositions" in body
        if is_review:
            assert (output / "review_manifest.json").is_file()
            assert len(runner.shared.read(output / "collected_predictions.json")) == 8
            assert len(body["propositions"]) == 1
            assert len(events) >= 8
            part = next(p for p in runner.split_factored_presence_request(names)
                        if json.loads(p.payload["input_text"])["propositions"] == body["propositions"])
            payload = part_review(part, original)
            if fail_first and not any(event == "review" for event in events):
                payload = {"invalid": "synthetic failure, not model evidence"}
        else:
            payload = stage_response(_responses(), "h0", original_base)
        events.append("review" if is_review else "h0")
        native = {"id": f"offline-{len(events)}", "model": runner.shared.MODEL, "provider": "Alibaba",
                  "usage": {"cost": .01}, "choices": [{"message": {"content": json.dumps(payload)}}]}
        return SimpleNamespace(status_code=200, text=json.dumps(native), json=lambda: native)

    calls = runner.shared.BudgetedCalls(output, 24 + cap, secret, ledger, phase="validation", sender=sender)
    def start(args, provided_plan):
        assert provided_plan["source_split"] == "Validation"
        assert args.max_calls == 24 + cap
        return secret, ledger, calls
    monkeypatch.setattr(runner.shared, "start_execution", start)
    def collect(base_request, proxy, key):
        proxy.call(key, "h0", base_request)
        item = next(s for s in selected if s["key"] == key)
        row = {**deepcopy(original), "video_id": item["video_id"], "frame_id": item["frame_id"]}
        if item != selected[0] and item != selected[1]:
            row["h1"] = None
        return row
    monkeypatch.setattr(runner.shared, "collect_target", collect)
    monkeypatch.setattr(runner.shared, "restore_target", lambda source, item, row, hashes:
        (original_base, (numeric, evidence, full_ref) if row["h1"] is not None else None))
    def score(args, rows, variants):
        assert len(rows) == 8
        assert runner.shared.read(output / "predictions.json") == rows
        assert (output / "inference_completion_binding.json").is_file()
        assert calls.stopped and variants == [runner.VARIANT]
        assert runner.shared.sha(output / "plan.json") == plan_hash
        assert all(set(r["variants"]) == {runner.VARIANT} for r in rows)
        if cap == 1:
            assert rows[0]["variants"][runner.VARIANT]["final_b"] == rows[0]["h0"]
            assert rows[1]["variants"][runner.VARIANT]["status"] == "NOT_ATTEMPTED"
        if fail_first:
            assert rows[0]["variants"][runner.VARIANT]["final_b"] == rows[0]["h0"]
        return rows, {"synthetic_masked_scoring": True}
    monkeypatch.setattr(runner.shared, "score_saved_rows", score)
    report = runner.run(SimpleNamespace(output=output, execute=True, api_key_file=tmp_path / "unused"))
    assert report["targets"] == 8 and report["names_joint_api_calls"] == 0
    assert report["collection_calls"] == 8
    assert report["review_calls"] == min(cap, 2 * len(runner.split_factored_presence_request(names)))
    assert report["primary"] == "final_b" and report["unpriced_calls"] == 0
    records = [json.loads(line) for line in (tmp_path / "budget.jsonl").read_text().splitlines()]
    reservations = [r for r in records if r["event"] == "RESERVE"]
    assert len(reservations) == len(events) and all(r["phase"] == "validation" for r in reservations)
    assert all(r["variant"] != "names1000" for r in reservations)
    assert not (output / "calls" / selected[0]["key"] / "names1000").exists()


def test_stage_caps_are_disjoint_and_cannot_bypass_shared_boundary():
    names, _ = source_request()
    calls = SimpleNamespace(used=0, stopped=None)
    def call(*args):
        calls.used += 1
        return {"offline": True}
    calls.call = call
    collection = runner.StageCalls(calls, cap=1, collection=True)
    with pytest.raises(ValueError, match="forbidden"):
        collection.call("key", "p001", names)
    assert collection.call("key", "h0", names) == {"offline": True}
    assert collection.call("key", "locator", names) is None and calls.used == 1
    review = runner.StageCalls(calls, cap=1, collection=False)
    with pytest.raises(ValueError, match="forbidden"):
        review.call("key", "proposal", names)
    with pytest.raises(ValueError, match="single-proposition"):
        review.call("key", "names1000", names)
