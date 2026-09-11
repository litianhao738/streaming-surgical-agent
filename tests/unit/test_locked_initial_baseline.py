"""Prevent drift of the user-selected H0 protocol and causal input boundary."""

from copy import deepcopy

import pytest

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.api.errors import ApiContractError
from surgical_agent.perception import locked_baseline as baseline


def request():
    return baseline.build_locked_batch_request(
        video_id="VID103", target_frame_id=25101,
        images=tuple(ApiImageInput(f"cholectrack20:VID103:frame:{f}", "image/png", b"png")
                     for f in (25051, 25076, 25101)),
    )


def test_default_is_official_batch_protocol_with_frozen_prompt():
    lock = baseline.load_baseline_lock()
    assert lock["baseline_id"] == "h0_qwen38_official_batch_v1_20260906"
    assert lock["transport"] == "batch_file"
    assert lock["model_identity"]["exact_backend_snapshot_known"] is False
    r = request()
    assert r["body"]["model"] == "qwen3.8-max"
    assert r["body"]["reasoning_effort"] == "low"
    assert r["body"]["max_tokens"] == 4096
    assert r["body"]["enable_thinking"] is True
    baseline.verify_locked_batch_request(r)


@pytest.mark.parametrize("change", [
    lambda r: r["body"].update(model="qwen3.8-max-0902"),
    lambda r: r["body"].update(reasoning_effort="high"),
    lambda r: r["body"].update(max_tokens=512),
    lambda r: r["body"].update(provider={"only": ["alibaba"]}),
    lambda r: r["body"]["messages"][0].update(content="shortened prompt"),
    lambda r: r["body"]["messages"][1]["content"][0].update(text='{"gt": [60]}'),
    lambda r: r["body"]["messages"][1]["content"][1]["image_url"].update(detail="high"),
    lambda r: r["body"]["response_format"]["json_schema"].update(strict=False),
    lambda r: r.update(custom_id="VID103_25126"),
])
def test_modified_request_is_rejected(change):
    r = deepcopy(request())
    change(r)
    with pytest.raises(ApiContractError):
        baseline.verify_locked_batch_request(r)


def test_prompt_change_requires_new_version(monkeypatch):
    monkeypatch.setattr(baseline, "load_main_h0_prompt", lambda: "changed")
    with pytest.raises(ApiContractError, match="prompt changed"):
        baseline.load_baseline_lock()


def test_future_image_cannot_be_labeled_as_history():
    with pytest.raises(ApiContractError, match="causal PNG identities"):
        baseline.build_locked_batch_request(video_id="VID103", target_frame_id=51,
            images=(ApiImageInput("cholectrack20:VID103:frame:76", "image/png", b"png"),))


@pytest.mark.parametrize("count", [1, 2])
def test_partial_history_uses_real_images_without_padding(count):
    ids = (1, 26)[:count]
    r = baseline.build_locked_batch_request(video_id="VID103", target_frame_id=ids[-1],
        images=tuple(ApiImageInput(f"cholectrack20:VID103:frame:{f}", "image/png", b"png")
                     for f in ids))
    assert len(r["body"]["messages"][1]["content"]) == count+1
    baseline.verify_locked_batch_request(r)
