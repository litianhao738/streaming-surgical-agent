import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import run_grounded_api_pipeline as runner
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.providers.openrouter import HttpResponse
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from tests.unit.test_main_h0_protocol import _config, _context


def request():
    base = JointPerceptionRequestBuilder(config=_config(provider="openrouter")).build(_context())
    payload = thaw_json(base.payload)
    payload["openrouter_routing_profile"] = "strict_google_ai_studio"
    return replace(base, model_identifier=runner.MODEL, payload=payload)


def response_body(*, cost=.01, content=None, usage=None):
    payload = {"schema_version": "joint_perception_final_only_v1",
               **{t: {"selected_ids": [0]} for t in ("instrument", "verb", "target", "ivt")},
               "phase": {"selected_id": 1}}
    return json.dumps({
        "id": "gen-test", "object": "chat.completion", "model": runner.MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content":
                     json.dumps(payload) if content is None else content}, "finish_reason": "stop"}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": cost},
    }).encode()


def caller(tmp_path, monkeypatch, response):
    original = runner.AuditSender

    def sender(url, headers, body, timeout):
        wire = json.loads(body)
        assert wire["model"] == runner.MODEL
        assert wire["provider"]["only"] == ["google-ai-studio"]
        assert wire["provider"]["allow_fallbacks"] is False
        return HttpResponse(200, {}, response)

    monkeypatch.setattr(runner, "AuditSender", lambda directory, secret: original(directory, secret, sender))
    return runner.AccountedCalls(tmp_path, {
        "max_provider_calls": 4, "budget_usd": .045, "per_call_reserve_usd": .04,
    }, SecretValue("unit-test-only-secret"))


def test_success_persists_response_and_stops_before_exceeding_reserve(tmp_path, monkeypatch):
    calls = caller(tmp_path, monkeypatch, response_body())
    assert calls.call("VID103_25101", "h0", request())["phase"] == {"selected_id": 1}
    assert calls.budget.used == 1
    assert float(calls.spent) == .01
    assert (tmp_path / "calls/VID103_25101/h0/response_record.json").is_file()
    assert calls.call("VID103_25126", "h0", request()) is None
    assert calls.stopped == "BUDGET_STOP" and calls.budget.used == 1


@pytest.mark.parametrize("malformed_usage", [None, []])
def test_unpriced_response_stops_without_automatic_retry(tmp_path, monkeypatch, malformed_usage):
    body = response_body(cost=None)
    if malformed_usage == []:
        data = json.loads(body)
        data["usage"] = []
        body = json.dumps(data).encode()
    calls = caller(tmp_path, monkeypatch, body)
    assert calls.call("VID103_25101", "h0", request()) is None
    assert calls.budget.used == 1 and calls.stopped == "UNPRICED_CALL_STOP"
    assert calls.records[0]["cost_usd"] is None
    assert calls.call("VID103_25126", "h0", request()) is None
    assert calls.budget.used == 1


def test_invalid_json_still_records_native_paid_usage(tmp_path, monkeypatch):
    calls = caller(tmp_path, monkeypatch, response_body(content="not-json"))
    assert calls.call("VID103_25101", "h0", request()) is None
    assert calls.records[0]["status"] == "FAILURE"
    assert calls.records[0]["cost_usd"] == .01
    assert calls.records[0]["cost_source"] == "raw_response_usage_after_parse_failure"
    assert calls.budget.used == 1


def test_stream_prefix_and_generation_survive_timeout(tmp_path, monkeypatch):
    first = b'data: {"id":"gen-partial","model":"google/gemini-3.8-flash","choices":[]}\n'

    class InterruptedStream:
        status, headers = 200, {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield b": OPENROUTER PROCESSING\n"
            yield first
            raise TimeoutError("simulated stream interruption")

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: InterruptedStream())
    audit = runner.AuditSender(tmp_path, SecretValue("unit-test-only-secret"))
    with pytest.raises(TimeoutError):
        audit("https://example.invalid", {}, b'{"messages":[]}', 1)
    assert first in (tmp_path / "response_stream.sse").read_bytes()
    assert json.loads((tmp_path / "generation.json").read_text())["id"] == "gen-partial"
    assert (tmp_path / "dispatch.lock").is_file()


def test_offline_scoring_keeps_failure_denominator_and_task_masks(monkeypatch):
    monkeypatch.setattr(runner, "VIDEOS", ("VID02",))
    target = SimpleNamespace(
        mask=SimpleNamespace(instrument=True, verb=False, target=False, ivt=False, phase=True),
        instrument_ids=(0,), phase_id=3,
    )

    class Adapter:
        def iter_video(self, video, frame_ids):
            assert video == "VID02" and list(frame_ids) == [51]
            yield SimpleNamespace(inference=SimpleNamespace(target_frame_id=51),
                                  frame_supervision=target, evaluation=None)

    report, scored = runner.score_saved(Adapter(), [
        {"video_id": "VID02", "frame_id": 51, "h0": None, "h1": None, "final": None},
    ])
    metrics = report["arms"]["final"]["tasks"]
    assert metrics["phase"]["valid_targets"] == 1
    assert metrics["phase"]["exact_set_accuracy"] == 0
    assert metrics["verb"]["valid_targets"] == 0 and metrics["verb"]["micro_f1"] is None
    assert scored[0]["gt"]["phase"] == [3] and scored[0]["gt"]["verb"] is None


def _prepare_case(tmp_path, monkeypatch, *, require_all=True, missing=False):
    videos = ("VID103", "VID96")
    events, aggregate_calls = [], []
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"offline source-hash fixture")
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(json.dumps({"data": {"id": runner.MODEL, "endpoints": [
        {"tag": "google-ai-studio", "pricing": {"prompt": "0.00000025", "completion": "0.0000015"}},
    ]}}), encoding="utf-8")
    args = SimpleNamespace(
        videos=list(videos), require_all_task_gt=require_all,
        dataset_root=tmp_path / "dataset", output=tmp_path / "run",
        pricing_snapshot=pricing_path, budget_usd=1.0, reserve_usd=.04,
        execute=False, api_key_file=None,
    )

    class MaskOnlyTarget:
        def __init__(self, *, complete):
            self.mask = SimpleNamespace(**{task: complete for task in runner.TASK_ATTRS})

        def __getattr__(self, name):
            raise AssertionError("preflight inspected a GT label value: " + name)

    class Adapter:
        def iter_inference_video(self, video):
            assert video in videos
            events.append(("inference", video))
            for frame in range(51, 252, 25):
                yield SimpleNamespace(
                    video_id=video, target_frame_id=frame,
                    causal_frame_ids=(frame - 50, frame - 25, frame),
                    source_split=runner.DatasetSplit.TRAINING,
                    media_refs=(str(image_path),) * 3,
                )

        def iter_video(self, video, frame_ids):
            # Both videos and all four targets were fixed before availability.
            assert [event for event in events if event[0] == "inference"] == [
                ("inference", "VID103"), ("inference", "VID96")]
            assert list(frame_ids) == [126, 201]
            events.append(("availability", video, list(frame_ids)))
            for frame in frame_ids:
                yield SimpleNamespace(
                    inference=SimpleNamespace(target_frame_id=frame),
                    frame_supervision=MaskOnlyTarget(complete=False),
                    evaluation=SimpleNamespace(instance_supervision_available=True, video=video, frame=frame),
                )

    def aggregate(evaluation, *, source):
        aggregate_calls.append((evaluation.video, evaluation.frame, source))
        complete = not (missing and evaluation.video == "VID103" and evaluation.frame == 126)
        return MaskOnlyTarget(complete=complete)

    base = request()

    class Builder:
        def build(self, sample):
            payload = thaw_json(base.payload)
            data = json.loads(payload["input_text"])
            data.update(video_id=sample.video_id, target_frame_id=sample.target_frame_id,
                        causal_frame_ids=list(sample.causal_frame_ids),
                        selected_image_frame_ids=list(sample.causal_frame_ids))
            payload["input_text"] = json.dumps(data)
            return replace(base, payload=payload)

    monkeypatch.setattr(runner, "CholecTrack20DatasetAdapter", lambda *a, **k: Adapter())
    monkeypatch.setattr(runner, "CausalApiMediaLoader", lambda: SimpleNamespace(
        load=lambda sample: SimpleNamespace(runtime_sample=sample, frames=())))
    monkeypatch.setattr(runner, "CausalPerceptionContextBuilder", lambda **k: SimpleNamespace(
        build=lambda sample, frames, **kwargs: sample))
    monkeypatch.setattr(runner, "JointPerceptionRequestBuilder", lambda **k: Builder())
    monkeypatch.setattr(runner, "aggregate_evaluation_target", aggregate)
    return args, videos, events, aggregate_calls


def test_prepare_checks_masks_after_fixed_selection_without_using_label_values(tmp_path, monkeypatch):
    args, videos, _, aggregate_calls = _prepare_case(tmp_path, monkeypatch)
    _, plan, requests = runner.prepare(args, videos=videos)
    assert runner.VIDEOS == ("VID02", "VID04", "VID11", "VID17")
    assert plan["videos"] == list(videos)
    assert [(row["video_id"], row["frame_id"]) for row in plan["selection"]] == [
        ("VID103", 126), ("VID103", 201), ("VID96", 126), ("VID96", 201)]
    assert plan["target_count"] == 4 and plan["max_provider_calls"] == 16
    assert all(all(row["task_masks"].values()) for row in plan["selection"])
    assert len(aggregate_calls) == 4
    assert all(source == "grounded_e2e_offline" for _, _, source in aggregate_calls)
    coverage = plan["gt_availability_check"]
    assert coverage["require_all_tasks"] and coverage["fully_annotated_targets"] == 4
    assert not coverage["label_values_used_for_selection_or_requests"]
    assert not coverage["resampling_after_mask_check"]
    for value in requests.values():
        data = json.loads(value.payload["input_text"])
        assert not {"task_masks", "mask", "gt", "ground_truth"} & data.keys()


def test_missing_required_gt_fails_before_dispatch_and_does_not_resample(tmp_path, monkeypatch):
    args, _, events, _ = _prepare_case(tmp_path, monkeypatch, missing=True)
    args.execute = True

    def forbidden(*args, **kwargs):
        raise AssertionError("a paid-call boundary was reached")

    monkeypatch.setattr(runner, "AccountedCalls", forbidden)
    monkeypatch.setattr(runner, "resolve_api_key", forbidden)
    with pytest.raises(ValueError, match="fixed targets lack required task GT"):
        runner.run(args)
    assert not args.output.exists()
    assert len([event for event in events if event[0] == "inference"]) == 2


def test_optional_mask_check_records_missing_heads_without_moving_targets(tmp_path, monkeypatch):
    args, videos, _, _ = _prepare_case(tmp_path, monkeypatch, require_all=False, missing=True)
    _, plan, _ = runner.prepare(args, videos=videos)
    assert plan["selection"][0]["key"] == "VID103_126"
    assert not any(plan["selection"][0]["task_masks"].values())
    assert plan["gt_availability_check"]["fully_annotated_targets"] == 3
    assert all(count == 3 for count in plan["gt_availability_check"]["valid_targets_by_task"].values())


def test_explicit_scoring_videos_uses_supplemental_rows_without_global_change():
    target = SimpleNamespace(
        mask=SimpleNamespace(**{task: True for task in runner.TASK_ATTRS}),
        instrument_ids=(0,), verb_ids=(1,), target_ids=(2,), triplet_ids=(3,), phase_id=4,
    )
    labels = {"instrument": [0], "verb": [1], "target": [2], "ivt": [3], "phase": [4]}

    class Adapter:
        def iter_video(self, video, frame_ids):
            assert video == "VID103" and list(frame_ids) == [126]
            yield SimpleNamespace(inference=SimpleNamespace(target_frame_id=126),
                                  frame_supervision=target, evaluation=None)

    report, scored = runner.score_saved(Adapter(), [
        {"video_id": "VID103", "frame_id": 126, "h0": labels, "h1": None, "final": labels},
    ], videos=("VID103",))
    assert scored[0]["video_id"] == "VID103"
    assert report["arms"]["final"]["tasks"]["ivt"]["exact_set_accuracy"] == 1
