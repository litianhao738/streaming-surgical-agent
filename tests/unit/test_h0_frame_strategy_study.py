import base64
import json
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import pytest

from scripts.run_h0_frame_strategy_study import (
    OFFSETS,
    spread_targets,
    validate_request,
    variant_request,
)
from scripts.run_pure_h0_smoke import FinalOnlyRequestBuilder
from surgical_agent.api.contracts import ApiImageInput, thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.config.loader import load_api_config
from tests.unit.test_joint_api_vlm import _context


def requests():
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    base = FinalOnlyRequestBuilder(config=config).build(_context())
    ids = [1, 26, 51, 76, 101, 126]
    images = tuple(ApiImageInput(f"cholectrack20:VID01:frame:{fid}", "image/png", str(fid).encode()) for fid in ids)
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    data["selected_image_frame_ids"] = ids
    payload["input_text"] = json.dumps(data)
    base = replace(base, images=images, payload=payload)
    return {group: variant_request(base, group, 126) for group in OFFSETS}


def test_real_wire_images_have_true_times_same_prompt_and_identical_target_pixels():
    built = requests()
    assert len({r.payload["system_text"] for r in built.values()}) == 1
    assert [len(r.images) for r in built.values()] == [1, 3, 3, 6]
    for group, request in built.items():
        transport = OpenRouterTransport(api_key=SecretValue("fixture"), endpoint_identifier=request.endpoint_identifier)
        transport._validate_request(request)
        body = json.loads(transport._request_body(request))
        content = body["messages"][1]["content"]
        data = json.loads(content[0]["text"])
        assert "group" not in data
        assert data["relative_seconds"] == list(OFFSETS[group])
        observed = [base64.b64decode(x["image_url"]["url"].split(",", 1)[1]).decode() for x in content[1:]]
        assert observed == [str(126 + 25 * offset) for offset in OFFSETS[group]]
        assert content[-1]["image_url"]["detail"] == "high"
        assert body["temperature"] == 0


def test_actual_image_reordering_or_false_times_are_rejected():
    request = requests()["D"]
    with pytest.raises(ValueError, match="identities"):
        validate_request(replace(request, images=tuple(reversed(request.images))), "D", 126)
    payload = thaw_json(request.payload)
    data = json.loads(payload["input_text"])
    data["relative_seconds"][-1] = 1
    payload["input_text"] = json.dumps(data)
    with pytest.raises(ValueError, match="timestamps"):
        validate_request(replace(request, payload=payload), "D", 126)


def test_selection_spreads_real_time_and_handles_gap_without_using_labels():
    samples = [SimpleNamespace(target_frame_id=f) for f in list(range(1, 5001, 25)) + list(range(10001, 15001, 25))]
    selected = spread_targets(samples, 20)
    ids = [s.target_frame_id for s in selected]
    assert len(ids) == len(set(ids)) == 20
    assert all(b - a >= 250 for a, b in pairwise(ids))
    assert min(ids) < 5001 and max(ids) > 10001


def test_selection_refuses_to_manufacture_sufficiently_spaced_targets():
    with pytest.raises(ValueError, match="ten seconds"):
        spread_targets([SimpleNamespace(target_frame_id=f) for f in range(1, 201, 25)], 3)
