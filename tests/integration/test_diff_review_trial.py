"""Replay provenance, offline isolation, masked scoring and dispatch boundaries."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import run_diff_review_trial as runner
from surgical_agent.api.contracts import ApiImageInput, thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.research.verification.grounded_pipeline import run_grounded_target
from tests.unit.test_grounded_pipeline import _base, _responses


def _fixture(tmp_path, *, target=101, unavailable=False):
    import numpy as np
    import torch
    from PIL import Image

    source = tmp_path / "source"
    source.mkdir(parents=True)
    base = _base()
    frames = [target - 50, target - 25, target]
    key = f"VID103_{target}"
    images, source_images = [], {}
    for index, image in enumerate(base.images):
        frame = frames[index]
        path = source / "Frames" / f"{frame:06d}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(image.content)
        source_images[str(path)] = runner._hash(path)
        with Image.open(path) as image_pixels:
            rgb = np.array(image_pixels.convert("RGB"), dtype=np.float32) / 255.0
        encoded = runner.encode_rgb_png(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        images.append(ApiImageInput(f"cholectrack20:VID103:frame:{frame}", "image/png", encoded))
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    data.update(target_frame_id=target, causal_frame_ids=frames, selected_image_frame_ids=frames)
    payload["input_text"] = json.dumps(data)
    payload["openrouter_routing_profile"] = "strict_google_ai_studio"
    base = replace(base, model_identifier=runner.MODEL, images=tuple(images), payload=payload)
    responses = _responses()
    if unavailable:
        responses["proposal"]["instances"][0]["support"] = "INSUFFICIENT"

    def record(stage, request):
        directory = source / "calls" / key / stage
        metadata = canonical_request_metadata(request).to_mapping()
        saved = {"metadata": metadata, "payload": thaw_json(request.payload)}
        atomic_write_json(directory / "request.json", saved)
        if stage == "h0":
            atomic_write_json(source / "requests" / f"{key}.json", saved)
        atomic_write_json(directory / "result.json", {
            "request_hash": metadata["request_hash"], "status": "OK",
            "payload": responses[stage], "provider_request_id": f"mock-origin-{stage}",
        })
        atomic_write_json(directory / "response_record.json", {
            "request_hash": metadata["request_hash"], "parsed_payload": responses[stage],
            "requested_model_identifier": runner.MODEL,
            "provider_request_id": f"mock-origin-{stage}",
        })
        return deepcopy(responses[stage])

    row = run_grounded_target(base, record, proposal_slot="FIRST")
    selected = {
        "key": key, "video_id": "VID103", "frame_id": target,
        "source_split": "Training", "causal_frame_ids": frames,
        "source_images": source_images, "request_metadata": canonical_request_metadata(base).to_mapping(),
    }
    atomic_write_json(source / "plan.json", {
        "source_split": "Training", "model": runner.MODEL, "selection": [selected],
    })
    atomic_write_json(source / "predictions.json", [row])
    gt = {"instrument": [2], "verb": None, "target": None, "ivt": None, "phase": [3]}
    mask = {task: task in ("instrument", "phase") for task in gt}
    atomic_write_json(source / "scored_predictions.json", [{
        "video_id": "VID103", "frame_id": target, "h0": row["h0"], "h1": row["h1"],
        "final": row["final"], "gt": gt, "mask": mask,
    }])
    pricing = tmp_path / "pricing.json"
    atomic_write_json(pricing, {"data": {"id": runner.MODEL, "endpoints": [
        {"tag": "google-ai-studio", "pricing": {"prompt": "0.00000075", "completion": "0.00000375"}},
    ]}})
    args = SimpleNamespace(source=source, output=tmp_path / "trial", pricing_snapshot=pricing,
                           max_targets=3, budget_usd=.20, reserve_usd=.05,
                           execute=False, mock=False, api_key_file=None)
    return args, selected, row


def test_plan_reconstructs_exact_images_and_requests_without_reading_gt_values(tmp_path, monkeypatch):
    args, selected, row = _fixture(tmp_path)
    original = runner._read

    def guarded(path):
        assert str(path) != str(args.source / "scored_predictions.json"), "preflight read GT values"
        return original(path)

    monkeypatch.setattr(runner, "_read", guarded)
    monkeypatch.setattr(runner, "AccountedCalls", lambda *a: pytest.fail("offline contacted API"))
    plan, requests, frozen = runner.prepare(args)
    assert plan["max_provider_calls"] == 1 and plan["gt_used_in_requests_or_selection"] is False
    request = requests["VID103_101"]
    data = json.loads(request.payload["input_text"])
    assert data["h0"] == row["h0"] and data["h1"] == row["h1"]
    assert data["full_frame_ref"] == "frame:101"
    assert data["allowed_evidence_refs"] == ["frame:51", "frame:76", "frame:101", "crop:1"]
    assert "gt" not in data and "mask" not in data
    assert data["proposal_instance_labels"] == [{"instance_id": 1, "instrument_id": 2, "ivt_ids": [59]}]
    assert "contact_observation" not in data["proposal_instance_labels"][0]
    assert [image.sha256 for image in request.images[:3]] == [
        image["sha256"] for image in selected["request_metadata"]["images"]]
    assert request.images[-1].sha256 == row["crop_manifest"][0]["crop_sha256"]
    assert frozen[0]["old_final"] == row["final"]
    # Same preflight is idempotent; changing anything requires a new directory.
    assert runner.prepare(args)[0] == plan


@pytest.mark.parametrize("tamper", ["image", "crop", "payload", "response_hash", "candidate", "testing"])
def test_preflight_rejects_broken_historical_binding(tmp_path, tamper):
    args, selected, row = _fixture(tmp_path)
    if tamper == "image":
        from pathlib import Path

        Path(next(iter(selected["source_images"]))).write_bytes(b"changed pixels")
    elif tamper == "crop":
        row["crop_manifest"][0]["crop_sha256"] = "a" * 64
        atomic_write_json(args.source / "predictions.json", [row])
    elif tamper in {"payload", "response_hash"}:
        path = args.source / "calls/VID103_101/proposal/response_record.json"
        record = runner._read(path)
        if tamper == "payload":
            record["parsed_payload"]["instances"][0]["ivt_ids"] = [60]
        else:
            record["request_hash"] = "a" * 64
        atomic_write_json(path, record)
    elif tamper == "candidate":
        row["h1"]["ivt"] = [58]
        atomic_write_json(args.source / "predictions.json", [row])
    else:
        plan = runner._read(args.source / "plan.json")
        plan["selection"][0]["source_split"] = "Testing"
        atomic_write_json(args.source / "plan.json", plan)
    with pytest.raises(ValueError):
        runner.prepare(args)
    assert not args.output.exists()


def test_mock_is_offline_keeps_h0_and_honors_task_masks(tmp_path, monkeypatch):
    args, _, row = _fixture(tmp_path)
    args.mock = True
    monkeypatch.setattr(runner, "AccountedCalls", lambda *a: pytest.fail("mock contacted API"))
    report = runner.run(args)
    assert report["mode"] == "OFFLINE_SYNTHETIC_MOCK_NOT_MODEL_RESULTS"
    assert report["provider_calls"] == 0 and report["added_cost_usd"] == 0
    assert report["known_added_cost_usd"] == 0 and report["accounting_complete"] is True
    saved = runner._read(args.output / "predictions.json")[0]
    assert saved["final_a"] == saved["final_b"] == row["h0"]
    metrics = report["comparisons"]["final_a"]["arms"]["final"]["tasks"]
    assert metrics["instrument"]["exact_set_accuracy"] == 1
    assert metrics["ivt"]["valid_targets"] == 0 and metrics["ivt"]["micro_f1"] is None
    with pytest.raises(ValueError, match="already dispatched"):
        runner.run(args)


def test_execute_without_credentials_never_dispatches(tmp_path, monkeypatch):
    args, _, _ = _fixture(tmp_path)
    args.execute = True
    monkeypatch.setattr(runner, "AccountedCalls", lambda *a: pytest.fail("missing credentials dispatched"))
    with pytest.raises(ValueError, match="api-key-file"):
        runner.run(args)
    assert not (args.output / "execution.lock").exists()


def test_source_scoring_is_bound_to_old_predictions(tmp_path):
    args, _, _ = _fixture(tmp_path)
    _, _, rows = runner.prepare(args)
    rows[0].update(final_a=rows[0]["h0"], final_b=rows[0]["h0"])
    path = args.source / "scored_predictions.json"
    scored = runner._read(path)
    scored[0]["final"]["ivt"] = [60]
    atomic_write_json(path, scored)
    with pytest.raises(ValueError, match="does not match frozen"):
        runner.score_saved(args.source, rows)


def test_freeze_detects_modified_inputs_before_execution(tmp_path):
    args, _, _ = _fixture(tmp_path)
    plan, _, _ = runner.prepare(args)
    (args.source / "predictions.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact changed"):
        runner._assert_frozen(plan)


def test_mutually_exclusive_modes_and_unsafe_output_are_rejected(tmp_path):
    args, _, _ = _fixture(tmp_path)
    args.execute = args.mock = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.run(args)
    args.execute = args.mock = False
    args.output = args.source / "nested_trial"
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(args)


def test_unavailable_candidate_remains_in_scoring_denominator_with_zero_calls(tmp_path, monkeypatch):
    import shutil

    args, _, _ = _fixture(tmp_path / "first")
    other, selected, row = _fixture(tmp_path / "second", target=126, unavailable=True)
    assert row["h1"] is None
    key = selected["key"]
    shutil.copytree(other.source / "calls" / key, args.source / "calls" / key)
    shutil.copyfile(other.source / "requests" / f"{key}.json", args.source / "requests" / f"{key}.json")
    plan = runner._read(args.source / "plan.json")
    plan["selection"].append(selected)
    atomic_write_json(args.source / "plan.json", plan)
    for name in ("predictions.json", "scored_predictions.json"):
        atomic_write_json(args.source / name, runner._read(args.source / name) + runner._read(other.source / name))
    args.mock = True
    monkeypatch.setattr(runner, "AccountedCalls", lambda *a: pytest.fail("mock contacted API"))
    report = runner.run(args)
    assert report["targets"] == report["source_target_count"] == 2
    assert report["review_targets"] == 1 and report["provider_calls"] == 0
    assert report["comparisons"]["final_a"]["candidate_availability"]["rate"] == .5
    assert report["comparisons"]["final_a"]["arms"]["final"]["tasks"]["instrument"]["valid_targets"] == 2
    unavailable = runner._read(args.output / "predictions.json")[1]
    assert unavailable["status"] == "NO_CANDIDATE_KEEP"
    assert unavailable["final_a"] == unavailable["final_b"] == unavailable["h0"]


def test_execute_uses_audited_transport_once_and_reads_gt_only_after_response(tmp_path, monkeypatch):
    from scripts import run_grounded_api_pipeline as paid
    from surgical_agent.api.credentials import SecretValue
    from surgical_agent.api.providers.openrouter import HttpResponse

    args, _, row = _fixture(tmp_path)
    args.execute, args.api_key_file = True, tmp_path / "unused_test_key_file"
    original_audit, original_read = paid.AuditSender, runner._read
    dispatched = []

    def sender(url, headers, body, timeout):
        wire = json.loads(body)
        assert wire["model"] == runner.MODEL
        assert wire["provider"]["only"] == ["google-ai-studio"]
        assert wire["provider"]["allow_fallbacks"] is False
        dispatched.append(wire)
        review = runner.mock_response(runner.build_change_claims(row["h0"], row["h1"]))
        content = {
            "id": "gen-diff-test", "object": "chat.completion", "model": runner.MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(review)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120, "cost": .001},
        }
        return HttpResponse(200, {}, json.dumps(content).encode())

    def guarded_read(path):
        if str(path) == str(args.source / "scored_predictions.json"):
            assert len(dispatched) == 1, "GT values read before the only model response"
        return original_read(path)

    monkeypatch.setattr(paid, "AuditSender", lambda directory, secret: original_audit(directory, secret, sender))
    monkeypatch.setattr(runner, "resolve_api_key", lambda **kw: SecretValue("unit-test-diff-only-secret"))
    monkeypatch.setattr(runner, "_read", guarded_read)
    report = runner.run(args)
    assert report["provider_calls"] == len(dispatched) == 1
    assert report["added_cost_usd"] == .001
    assert report["known_added_cost_usd"] == .001 and report["accounting_complete"] is True
    call = args.output / "calls/VID103_101/diff_review"
    assert (call / "wire_request.json").is_file()
    assert (call / "response_record.json").is_file()
    saved = original_read(args.output / "predictions.json")[0]
    assert saved["request_hash"] == original_read(call / "response_record.json")["request_hash"]


@pytest.mark.parametrize("failure", ["timeout", "unpriced"])
def test_unknown_cost_stops_once_preserves_four_target_denominator_and_reports_null(tmp_path, monkeypatch, failure):
    import shutil

    from scripts import run_grounded_api_pipeline as paid
    from surgical_agent.api.credentials import SecretValue
    from surgical_agent.api.providers.openrouter import HttpResponse

    args, _, first_row = _fixture(tmp_path / "first")
    for frame, unavailable in ((126, False), (151, False), (176, True)):
        other, selected, _ = _fixture(tmp_path / f"source_{frame}", target=frame, unavailable=unavailable)
        key = selected["key"]
        shutil.copytree(other.source / "calls" / key, args.source / "calls" / key)
        shutil.copyfile(other.source / "requests" / f"{key}.json", args.source / "requests" / f"{key}.json")
        plan = runner._read(args.source / "plan.json")
        plan["selection"].append(selected)
        atomic_write_json(args.source / "plan.json", plan)
        for name in ("predictions.json", "scored_predictions.json"):
            atomic_write_json(args.source / name, runner._read(args.source / name) + runner._read(other.source / name))
    args.execute, args.api_key_file = True, tmp_path / "unused_test_key_file"
    original_audit = paid.AuditSender
    dispatched = []

    def sender(url, headers, body, timeout):
        dispatched.append(json.loads(body))
        if failure == "timeout":
            raise TimeoutError("offline simulated provider timeout after dispatch")
        review = runner.mock_response(runner.build_change_claims(first_row["h0"], first_row["h1"]))
        content = {
            "id": "gen-unpriced-diff-test", "object": "chat.completion", "model": runner.MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(review)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120, "cost": None},
        }
        return HttpResponse(200, {}, json.dumps(content).encode())

    monkeypatch.setattr(paid, "AuditSender", lambda directory, secret: original_audit(directory, secret, sender))
    monkeypatch.setattr(runner, "resolve_api_key", lambda **kw: SecretValue("unit-test-diff-only-secret"))
    report = runner.run(args)
    assert report["status"] == "STOPPED" and report["stop_reason"] == "UNPRICED_CALL_STOP"
    assert report["targets"] == report["source_target_count"] == 4 and report["review_targets"] == 3
    assert report["provider_calls"] == report["unpriced_calls"] == len(dispatched) == 1
    assert report["added_cost_usd"] is None and report["known_added_cost_usd"] == 0
    assert report["accounting_complete"] is False
    rows = runner._read(args.output / "predictions.json")
    assert [row["status"] for row in rows] == [
        "REVIEW_FAILURE_KEEP", "NOT_ATTEMPTED", "NOT_ATTEMPTED", "NO_CANDIDATE_KEEP"]
    assert all(row["final_a"] == row["final_b"] == row["h0"] for row in rows)
    assert not (args.output / "calls/VID103_126").exists()
    assert not (args.output / "calls/VID103_151").exists()
    assert report["comparisons"]["final_a"]["arms"]["final"]["tasks"]["instrument"]["valid_targets"] == 4
