"""Synthetic request invariants for conditional past-frame evidence only."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import ApiImageInput, canonical_json_bytes, thaw_json
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
)
from surgical_agent.research.verification.temporal_memory import (
    MEMORY1000_PROMPT_VERSION,
    assert_memory_extension,
    extend_history_request,
    trigger_propositions,
)
from tests.integration.test_factored_verifier_trial import source_request


def fixture():
    request, row = source_request()
    body = json.loads(request.payload["input_text"])
    images = list(request.images)
    for index, frame in enumerate(body["causal_frame_ids"]):
        identifier = f"cholectrack20:{body['video_id']}:frame:{frame}"
        images[index] = replace(images[index], identifier=identifier)
        body["evidence_manifest"][index]["image_identifier"] = identifier
    request = with_body(replace(request, images=images), body)
    review = {"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": [
        {"proposition_id": proposition["proposition_id"], "presence": "UNCLEAR",
         "observation": "Synthetic uncertainty without GT.", "evidence_refs": [body["full_frame_ref"]],
         "scope": "FRAME", "full_frame_reviewed": True} for proposition in body["propositions"]]}
    return request, row, review


def with_body(request, body):
    payload = thaw_json(request.payload)
    payload["input_text"] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return replace(request, payload=payload)


def extra(frame, video="VID103"):
    return ApiImageInput(identifier=f"cholectrack20:{video}:frame:{frame}",
                         mime_type="image/png", content=f"synthetic-real-source-bytes-{video}-{frame}".encode())


def test_trigger_uses_only_complete_valid_unclear_verb_or_ivt_with_no_mask_argument():
    request, row, review = fixture()
    before = deepcopy(review)
    triggered = trigger_propositions(request, review, h0=row["h0"], h1=row["h1"])
    assert triggered == [{"proposition_id": "p003", "task": "ivt"},
                         {"proposition_id": "p004", "task": "ivt"}]
    assert review == before
    review["assessments"][2]["presence"] = "PRESENT"
    assert trigger_propositions(request, review, h0=row["h0"], h1=row["h1"]) == triggered[1:]
    review["assessments"][3]["presence"] = "ABSENT"
    assert trigger_propositions(request, review, h0=row["h0"], h1=row["h1"]) == []
    assert trigger_propositions(None, None, h0=None, h1=None) == []
    assert trigger_propositions(request, None, h0=row["h0"], h1=row["h0"]) == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_id", "unprovided_ref", "version",
                                      "overlong", "malformed", "wrong_type"])
def test_invalid_first_review_cannot_trigger_even_if_an_unclear_entry_exists(mutation):
    request, row, review = fixture()
    if mutation == "missing":
        review["assessments"].pop(0)
    elif mutation == "duplicate":
        review["assessments"][0] = deepcopy(review["assessments"][1])
    elif mutation == "wrong_id":
        review["assessments"][0]["proposition_id"] = "p040"
    elif mutation == "unprovided_ref":
        review["assessments"][0]["evidence_refs"] = ["frame:26"]
    elif mutation == "version":
        review["schema_version"] = "frame_label_presence_review_v2"
    elif mutation == "overlong":
        review["assessments"][0]["observation"] = "x" * 1001
    elif mutation == "malformed":
        review = None
    else:
        review["assessments"][0]["full_frame_reviewed"] = 1
    assert trigger_propositions(request, review, h0=row["h0"], h1=row["h1"]) == []


def test_source_propositions_must_match_h0_h1_and_not_just_response_id_count():
    request, row, review = fixture()
    body = json.loads(request.payload["input_text"])
    body["propositions"][0]["statement"] += " This is a different proposition."
    changed = with_body(request, body)
    with pytest.raises(ValueError, match="actual frozen H0/H1"):
        trigger_propositions(changed, review, h0=row["h0"], h1=row["h1"])


@pytest.mark.parametrize("frames", [[26], [1, 26]])
def test_extension_is_reversible_and_preserves_actual_original_wire(frames):
    original, _, _ = fixture()
    before_wire = canonical_json_bytes(shared.wire_body(original))
    expanded, evidence, full_ref = extend_history_request(original, [extra(f) for f in frames], frames)
    assert expanded.prompt_version == MEMORY1000_PROMPT_VERSION and full_ref == "frame:101"
    assert assert_memory_extension(original, expanded)
    body, original_body = json.loads(expanded.payload["input_text"]), json.loads(original.payload["input_text"])
    assert body["causal_frame_ids"] == frames + [51, 76, 101]
    assert body["selected_image_frame_ids"] == body["causal_frame_ids"]
    assert body["relative_seconds"] == ([-4, -3, -2, -1, 0] if len(frames) == 2 else [-3, -2, -1, 0])
    assert body["propositions"] == original_body["propositions"]
    assert body["crop_manifest"] == original_body["crop_manifest"]
    assert body["temporal_evidence"] == original_body["temporal_evidence"]
    assert [entry["image_index"] for entry in evidence] == list(range(len(expanded.images)))
    assert list(expanded.payload["image_details"]) == ["low"] * len(frames) + ["low", "low", "high", "high"]
    # Remove only the two allowed wire differences: metadata text and prefixed images.
    wire = shared.wire_body(expanded)
    old_wire = shared.wire_body(original)
    content = wire["messages"][1]["content"]
    wire["messages"][1]["content"] = [old_wire["messages"][1]["content"][0]] + content[1 + len(frames):]
    assert canonical_json_bytes(wire) == before_wire
    assert canonical_json_bytes(shared.wire_body(original)) == before_wire


def test_three_original_crops_can_extend_to_eight_images_without_changing_crops():
    original, _, _ = fixture()
    body = json.loads(original.payload["input_text"])
    images = list(original.images)
    for identity in (2, 3):
        image = ApiImageInput(identifier=f"contact:101:instance:{identity}",
                              mime_type="image/png", content=f"synthetic-crop-{identity}".encode())
        crop = deepcopy(body["crop_manifest"][0])
        crop.update(instance_id=identity, crop_sha256=image.sha256)
        body["crop_manifest"].append(crop)
        body["evidence_manifest"].append({"ref": f"crop:{identity}", "kind": "target_crop",
            "image_index": len(images), "image_identifier": image.identifier,
            "sha256": image.sha256, "target_frame_id": 101})
        body["allowed_evidence_refs"].append(f"crop:{identity}")
        images.append(image)
    payload = thaw_json(original.payload)
    payload["image_details"] += ["high", "high"]
    original = with_body(replace(original, images=images, payload=payload), body)
    extended, evidence, _ = extend_history_request(original, [extra(1), extra(26)], [1, 26])
    assert len(extended.images) == 8 and [entry["image_index"] for entry in evidence] == list(range(8))
    assert extended.images[-3:] == original.images[-3:]
    assert assert_memory_extension(original, extended)


@pytest.mark.parametrize("frames", [[], [1], [1, 51], [26, 1], [26, 26], [0, 26], [True], [1, 26, 51], [126]])
def test_no_skipped_gap_reordered_repeated_future_or_invalid_history(frames):
    original, _, _ = fixture()
    with pytest.raises(ValueError, match="contiguous"):
        extend_history_request(original, [extra(f) for f in frames], frames)


@pytest.mark.parametrize("mutation", ["other_video", "wrong_frame", "missing_image"])
def test_extra_image_identity_must_match_exact_frame_and_video(mutation):
    original, _, _ = fixture()
    image = extra(26, "VID23") if mutation == "other_video" else extra(1 if mutation == "wrong_frame" else 26)
    images = [] if mutation == "missing_image" else [image]
    with pytest.raises(ValueError):
        extend_history_request(original, images, [26])


@pytest.mark.parametrize("mutation", ["system", "parameters", "schema", "propositions", "crops", "full_ref",
                                      "detail", "image_bytes", "image_order", "old_index", "relative_time", "extra_field"])
def test_any_change_beyond_declared_extra_history_is_rejected(mutation):
    original, _, _ = fixture()
    expanded, _, _ = extend_history_request(original, [extra(1), extra(26)], [1, 26])
    payload, body = thaw_json(expanded.payload), json.loads(expanded.payload["input_text"])
    if mutation == "system":
        payload["system_text"] += " changed"
        changed = replace(expanded, payload=payload)
    elif mutation == "parameters":
        changed = replace(expanded, generation_parameters={"temperature": 1})
    elif mutation == "schema":
        changed = replace(expanded, response_schema_version="frame_label_presence_review_v2")
    elif mutation == "detail":
        payload["image_details"][0] = "high"
        changed = replace(expanded, payload=payload)
    elif mutation == "image_bytes":
        images = list(expanded.images)
        images[-1] = replace(images[-1], content=b"changed original crop")
        changed = replace(expanded, images=images)
    elif mutation == "image_order":
        images = list(expanded.images)
        images[-1], images[-2] = images[-2], images[-1]
        changed = replace(expanded, images=images)
    else:
        if mutation == "propositions":
            body["propositions"].reverse()
        elif mutation == "crops":
            body["crop_manifest"][0]["pixel_box_xyxy"][0] += 1
        elif mutation == "full_ref":
            body["full_frame_ref"] = "frame:26"
        elif mutation == "old_index":
            body["evidence_manifest"][-1]["image_index"] -= 1
        elif mutation == "relative_time":
            body["relative_seconds"][0] = -5
        else:
            body["prior_review"] = "do not add previous answers"
        changed = with_body(expanded, body)
    with pytest.raises(ValueError):
        assert_memory_extension(original, changed)


@pytest.mark.parametrize("mutation", ["image_sha", "fps", "frame_order", "source_detail", "old_reference"])
def test_invalid_original_image_context_is_rejected_before_extension(mutation):
    original, _, _ = fixture()
    payload, body = thaw_json(original.payload), json.loads(original.payload["input_text"])
    if mutation == "image_sha":
        body["evidence_manifest"][0]["sha256"] = "0" * 64
    elif mutation == "fps":
        body["source_fps"] = 30
    elif mutation == "frame_order":
        body["causal_frame_ids"].reverse()
    elif mutation == "old_reference":
        body["allowed_evidence_refs"][0] = "frame:1"
    else:
        payload["image_details"][0] = "high"
        original = replace(original, payload=payload)
    changed = original if mutation == "source_detail" else with_body(original, body)
    with pytest.raises(ValueError):
        extend_history_request(changed, [extra(26)], [26])
