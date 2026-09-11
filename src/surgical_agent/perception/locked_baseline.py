"""User-selected official Batch H0: immutable request recipe and drift checks.

No provider calls, experiment imports, local credentials or GT are used here.
The lock pins the request protocol, not the provider's mutable model alias.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from importlib.resources import files

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.api.errors import ApiContractError
from surgical_agent.perception.main_h0 import load_main_h0_prompt, main_h0_input

BASELINE_ID = "h0_qwen38_official_batch_v1_20260906"
PROTOCOL_SHA256 = "9185ab5eab5d029b1e3bd7a620b9eec3245780db5d510ac89489e0f7b97b1ab3"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def load_baseline_lock() -> dict:
    lock = json.loads(
        files("surgical_agent.perception.prompts")
        .joinpath("initial_baseline_lock_v1.json").read_text(encoding="utf-8")
    )
    digest = lock.pop("protocol_sha256")
    if (lock["baseline_id"] != BASELINE_ID or digest != PROTOCOL_SHA256
            or hashlib.sha256(canonical_bytes(lock)).hexdigest() != digest):
        raise ApiContractError("initial baseline lock changed; create a new version")
    lock["protocol_sha256"] = digest
    if hashlib.sha256(load_main_h0_prompt().encode()).hexdigest() != lock["system_prompt_sha256"]:
        raise ApiContractError("frozen initial prompt changed; create a new baseline version")
    for count in (1, 2, 3):
        probe = main_h0_input(video_id="VID00", target_frame_id=51,
                              frame_ids=(1, 26, 51)[-count:])
        if hashlib.sha256(canonical_bytes(probe)).hexdigest() != lock["input_recipe_hashes"][str(count)]:
            raise ApiContractError("frozen causal input recipe changed")
    return lock


def build_locked_batch_request(
    *, video_id: str, target_frame_id: int, images: tuple[ApiImageInput, ...]
) -> dict:
    """Build exactly one pure-visual five-head target request, with no overrides."""
    lock = load_baseline_lock()
    if (not re.fullmatch(r"VID[0-9]+", video_id)
            or type(target_frame_id) is not int or target_frame_id < 1
            or not 1 <= len(images) <= 3):
        raise ApiContractError("invalid baseline target or image count")
    frame_ids = tuple(target_frame_id + 25 * offset for offset in range(1-len(images), 1))
    if frame_ids[0] < 1:
        raise ApiContractError("baseline images must have positive frame IDs")
    content = [{"type": "text", "text": json.dumps(
        main_h0_input(video_id=video_id, target_frame_id=target_frame_id, frame_ids=frame_ids),
        sort_keys=True, separators=(",", ":"),
    )}]
    for index, (frame_id, image) in enumerate(zip(frame_ids, images, strict=True)):
        if (image.identifier != f"cholectrack20:{video_id}:frame:{frame_id}"
                or image.mime_type != "image/png"):
            raise ApiContractError("frozen baseline requires ordered causal PNG identities")
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image.content).decode("ascii"),
            "detail": "high" if index == len(images)-1 else "low",
        }})
    body = lock["request_static"]
    body["messages"] = [
        {"role": "system", "content": load_main_h0_prompt()},
        {"role": "user", "content": content},
    ]
    request = {"custom_id": f"{video_id}_{target_frame_id}", "method": "POST",
               "url": "/v1/chat/completions", "body": body}
    if len(canonical_bytes(request)) > lock["max_jsonl_line_bytes"]:
        raise ApiContractError("frozen baseline request exceeds the Batch File line limit")
    return request


def verify_locked_batch_request(request: dict) -> None:
    """Reject altered generation, text/schema, temporal evidence and extra fields."""
    try:
        video_id, raw_target = request["custom_id"].rsplit("_", 1)
        target = int(raw_target)
        parts = request["body"]["messages"][1]["content"][1:]
        frame_ids = tuple(target + 25 * offset for offset in range(1-len(parts), 1))
        images = []
        for frame_id, part in zip(frame_ids, parts, strict=True):
            if part["type"] != "image_url":
                raise ValueError("unexpected image block")
            prefix, encoded = part["image_url"]["url"].split(",", 1)
            if prefix != "data:image/png;base64":
                raise ValueError("unexpected image encoding")
            images.append(ApiImageInput(
                f"cholectrack20:{video_id}:frame:{frame_id}", "image/png",
                base64.b64decode(encoded, validate=True),
            ))
        expected = build_locked_batch_request(video_id=video_id, target_frame_id=target,
                                               images=tuple(images))
        if request != expected:
            raise ValueError("request differs from the frozen recipe")
    except (KeyError, TypeError, ValueError, IndexError, binascii.Error) as exc:
        raise ApiContractError("request does not match the frozen initial baseline") from exc
