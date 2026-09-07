"""Bounded, image-only extension of an existing neutral names1000 review.

No dataset access, GT, API transport, prior answer text or repair policy lives
here. The caller verifies real frame availability and binds source image bytes.
"""

import json
from copy import deepcopy
from dataclasses import replace

from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    canonical_json_bytes,
    thaw_json,
)
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.factored_presence import (
    NAMES1000_PROMPT_VERSION,
    _source_body,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    build_presence_review_input,
    validate_presence_review,
)

MEMORY1000_PROMPT_VERSION = NAMES1000_PROMPT_VERSION + "_two_past_frames_v1"


def _checked_source(request):
    body = _source_body(request)
    target, video = body.get("target_frame_id"), body.get("video_id")
    if type(target) is not int or target < 51 or not isinstance(video, str) or not video:
        raise ValueError("memory requires an identified target with three real causal frames")
    frames = [target - 50, target - 25, target]
    if (body.get("causal_frame_ids") != frames or body.get("selected_image_frame_ids") != frames
            or any(type(frame) is not int for frame in body["causal_frame_ids"])
            or body.get("source_fps") != 25 or type(body.get("source_fps")) is not int
            or body.get("relative_seconds") != [-2, -1, 0]
            or body.get("full_frame_ref") != f"frame:{target}"
            or not 3 <= len(request.images) <= 6
            or list(request.payload.get("image_details", ())) != ["low", "low"] + ["high"] * (len(request.images) - 2)):
        raise ValueError("source frame/time/detail context differs from frozen three-frame names")
    manifest = body.get("evidence_manifest")
    crops = body.get("crop_manifest")
    if (not isinstance(manifest, list) or len(manifest) != len(request.images)
            or not isinstance(crops, list) or len(crops) != len(request.images) - 3):
        raise ValueError("source image/evidence/crop counts disagree")
    refs = []
    for index, (entry, image) in enumerate(zip(manifest, request.images, strict=True)):
        if (not isinstance(entry, dict) or type(entry.get("image_index")) is not int
                or entry["image_index"] != index or entry.get("image_identifier") != image.identifier
                or entry.get("sha256") != image.sha256):
            raise ValueError("source evidence does not match actual ordered image bytes")
        if index < 3:
            expected = {"image_index": index, "image_identifier": image.identifier,
                        "sha256": image.sha256, "ref": f"frame:{frames[index]}",
                        "kind": "target_full_frame" if index == 2 else "history_full_frame"}
            if image.identifier != f"cholectrack20:{video}:frame:{frames[index]}" or entry != expected:
                raise ValueError("source full-frame identity/reference mismatch")
        else:
            crop = crops[index - 3]
            expected = {"image_index": index, "image_identifier": image.identifier,
                        "sha256": image.sha256, "ref": f"crop:{crop['instance_id']}",
                        "kind": "target_crop", "target_frame_id": target}
            if (entry != expected or crop.get("target_frame_id") != target
                    or crop.get("source_sha256") != request.images[2].sha256
                    or crop.get("crop_sha256") != image.sha256):
                raise ValueError("source target-crop binding mismatch")
        refs.append(entry["ref"])
    if refs != body["allowed_evidence_refs"] or len(refs) != len(set(refs)):
        raise ValueError("source allowed evidence differs from supplied ordered images")
    return body


def trigger_propositions(names_request, first_review, *, h0, h1):
    """Return uncertain V/IVT propositions only after full first-review binding.

    Unavailable/invalid candidates or invalid reviews do not trigger. A valid
    source request with propositions differing from H0/H1 is a provenance error.
    This function accepts no GT or availability mask.
    """
    if h0 is None or h1 is None:
        return []
    body = _checked_source(names_request)
    try:
        expected = build_presence_review_input(h0, h1,
            allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"])
    except (ApiSchemaError, TypeError, ValueError, OverflowError):
        return []
    if not expected["propositions"]:
        return []
    plain = [{key: value for key, value in proposition.items()
              if key not in ("decoded_components", "decoded_label_name")}
             for proposition in body["propositions"]]
    if plain != expected["propositions"]:
        raise ValueError("names propositions differ from the actual frozen H0/H1 changes")
    try:
        validate_presence_review(first_review, h0=h0, h1=h1,
            allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"],
            schema_version=PRESENCE_REVIEW_1000_VERSION)
    except (ApiSchemaError, TypeError, ValueError, OverflowError):
        return []
    assessments = {entry["proposition_id"]: entry for entry in first_review["assessments"]}
    return [{"proposition_id": proposition["proposition_id"], "task": proposition["task"]}
            for proposition in body["propositions"]
            if proposition["task"] in ("verb", "ivt")
            and assessments[proposition["proposition_id"]]["presence"] == "UNCLEAR"]


def _expected_extension(original, extra_images, extra_frame_ids):
    body = _checked_source(original)
    images, frames = tuple(extra_images), list(extra_frame_ids)
    target, video = body["target_frame_id"], body["video_id"]
    if (len(images) not in (1, 2) or len(frames) != len(images)
            or any(type(frame) is not int or frame < 1 for frame in frames)
            or frames != ([target - 75] if len(frames) == 1 else [target - 100, target - 75])):
        raise ValueError("extra history must be contiguous t-75 or ordered t-100,t-75")
    if len(original.images) + len(images) > 8:
        raise ValueError("extended request exceeds eight images")
    for frame, image in zip(frames, images, strict=True):
        if not isinstance(image, ApiImageInput) or image.identifier != f"cholectrack20:{video}:frame:{frame}":
            raise ValueError("extra image must match this video and exact raw frame identity")
    count = len(images)
    evidence = [{"ref": f"frame:{frame}", "image_index": index, "image_identifier": image.identifier,
                 "sha256": image.sha256, "kind": "history_full_frame"}
                for index, (frame, image) in enumerate(zip(frames, images, strict=True))]
    previous = deepcopy(body["evidence_manifest"])
    for entry in previous:
        entry["image_index"] += count
    extended = {**body, "causal_frame_ids": frames + body["causal_frame_ids"],
                "selected_image_frame_ids": frames + body["selected_image_frame_ids"],
                "relative_seconds": [(frame - target) // 25 for frame in frames] + body["relative_seconds"],
                "allowed_evidence_refs": [e["ref"] for e in evidence] + body["allowed_evidence_refs"],
                "evidence_manifest": evidence + previous}
    payload = thaw_json(original.payload)
    payload["input_text"] = json.dumps(extended, sort_keys=True, separators=(",", ":"))
    payload["image_details"] = ["low"] * count + list(original.payload["image_details"])
    return replace(original, payload=payload, images=images + original.images,
                   prompt_version=MEMORY1000_PROMPT_VERSION)


def assert_memory_extension(original: ApiRequest, extended: ApiRequest):
    """Check only allowed additions and exact reversible original request bytes.

    No scripts/transport import: unchanged outer fields, image suffix, and full
    payload after stripping imply unchanged original wire content.
    """
    original_body = _checked_source(original)
    if not isinstance(extended, ApiRequest) or extended.prompt_version != MEMORY1000_PROMPT_VERSION:
        raise ValueError("explicit memory request version required")
    count = len(extended.images) - len(original.images)
    if count not in (1, 2) or extended.images[count:] != original.images:
        raise ValueError("memory must retain every original image and crop byte in order")
    body = json.loads(extended.payload["input_text"])
    expected = _expected_extension(original, extended.images[:count], body["causal_frame_ids"][:count])
    if extended != expected:
        raise ValueError("memory request changed fields beyond the declared history extension")
    stripped = deepcopy(body)
    for field in ("causal_frame_ids", "selected_image_frame_ids", "relative_seconds", "allowed_evidence_refs"):
        stripped[field] = stripped[field][count:]
    stripped["evidence_manifest"] = stripped["evidence_manifest"][count:]
    for entry in stripped["evidence_manifest"]:
        entry["image_index"] -= count
    if canonical_json_bytes(stripped) != canonical_json_bytes(original_body):
        raise ValueError("removing added history does not recover the complete original input")
    payload = thaw_json(extended.payload)
    payload["input_text"] = original.payload["input_text"]
    payload["image_details"] = payload["image_details"][count:]
    restored = replace(extended, prompt_version=original.prompt_version,
                       payload=payload, images=extended.images[count:])
    if restored != original:
        raise ValueError("removing memory does not restore the original request exactly")
    return True


def extend_history_request(names_request, extra_images, extra_frame_ids):
    """Add only supplied real, causal past frames; caller handles gap detection."""
    extended = _expected_extension(names_request, extra_images, extra_frame_ids)
    assert_memory_extension(names_request, extended)
    body = json.loads(extended.payload["input_text"])
    return extended, body["evidence_manifest"], body["full_frame_ref"]
