"""Independent offline checks of the frozen numeric/named verifier contrast."""

import csv
import json
from dataclasses import fields, replace
from importlib.resources import files

import pytest

from scripts import run_verifier_variant_trial as runner
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    build_presence_review_input,
)
from tests.integration.test_presence_review_trial import base, stage_response
from tests.unit.test_grounded_pipeline import _responses

# Explicit labels check the public ID/name contract independently of the helper.
NAMES = {
    "instrument": ("grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag"),
    "verb": ("grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack", "null_verb"),
    "target": ("gallbladder", "cystic_plate", "cystic_duct", "cystic_artery", "cystic_pedicle",
               "blood_vessel", "fluid", "abdominal_wall_or_cavity", "liver", "adhesion", "omentum",
               "peritoneum", "gut", "specimen_bag", "null_target"),
}


@pytest.fixture
def review_request():
    responses = _responses()

    class Calls:
        def call(self, key, stage, request):
            return stage_response(responses, stage, request)

    original = base()
    row = runner.collect_target(original, Calls(), "offline-target")
    request, _, _ = runner.build_review_request(original, row)
    assert request.response_schema_version == PRESENCE_REVIEW_1000_VERSION
    return request


def with_propositions(request, propositions):
    payload = thaw_json(request.payload)
    body = json.loads(payload["input_text"])
    body["propositions"] = propositions
    payload["input_text"] = json.dumps(body)
    return replace(request, payload=payload)


def test_named_only_changes_decoded_fields_and_prompt_version(review_request):
    before_payload = thaw_json(review_request.payload)
    before_hash = canonical_request_metadata(review_request).request_hash
    assert runner.variant_request(review_request, "numeric1000") is review_request
    named = runner.variant_request(review_request, "names1000")
    for field in fields(review_request):
        if field.name not in {"payload", "prompt_version"}:
            assert getattr(named, field.name) == getattr(review_request, field.name)
    numeric_payload = thaw_json(review_request.payload)
    named_payload = thaw_json(named.payload)
    numeric_body = json.loads(numeric_payload.pop("input_text"))
    named_body = json.loads(named_payload.pop("input_text"))
    assert named_payload == numeric_payload  # Includes byte-identical system/schema text.
    for proposition in named_body["propositions"]:
        proposition.pop("decoded_components", None)
        proposition.pop("decoded_label_name", None)
    assert named_body == numeric_body
    assert named.prompt_version == review_request.prompt_version + "_decoded_names_v1"
    assert thaw_json(review_request.payload) == before_payload
    assert canonical_request_metadata(review_request).request_hash == before_hash
    assert canonical_request_metadata(named).request_hash != before_hash


def test_all_100_ivts_match_independently_read_frozen_component_csv(review_request):
    resource = files("surgical_agent.research.signals.resources").joinpath("ivt_components_v1.csv")
    rows = list(csv.DictReader(resource.read_text(encoding="utf-8").splitlines()))
    assert [int(row["ivt"]) for row in rows] == list(range(100))
    decoded = {}
    # One valid opaque ID per synthetic request; do not manufacture >40 propositions.
    for row in rows:
        label = int(row["ivt"])
        original = with_propositions(review_request, [{
            "proposition_id": "p001", "task": "ivt", "label_id": label,
            "statement": f"The target frame contains ivt label {label}.",
        }])
        named = runner.variant_request(original, "names1000")
        proposition = json.loads(named.payload["input_text"])["propositions"][0]
        decoded[label] = proposition["decoded_components"]
        assert decoded[label] == {task: names[int(row[task])] for task, names in NAMES.items()}
        assert proposition["label_id"] == label and proposition["proposition_id"] == "p001"
    assert decoded[17] == {"instrument": "grasper", "verb": "retract", "target": "gallbladder"}
    assert decoded[59] == {"instrument": "hook", "verb": "dissect", "target": "cystic_plate"}
    assert all(decoded[label]["verb"] == "null_verb" and decoded[label]["target"] == "null_target"
               for label in range(94, 100))


@pytest.mark.parametrize("task", ["instrument", "verb", "target"])
def test_all_component_head_names_preserve_ids_and_statements(review_request, task):
    propositions = [{"proposition_id": f"p{index + 1:03d}", "task": task, "label_id": index,
                     "statement": f"The target frame contains {task} label {index}."}
                    for index in range(len(NAMES[task]))]
    original = with_propositions(review_request, propositions)
    named = runner.variant_request(original, "names1000")
    actual = json.loads(named.payload["input_text"])["propositions"]
    assert actual == [{**item, "decoded_label_name": NAMES[task][item["label_id"]]}
                      for item in propositions]


def test_swapping_hypotheses_cannot_reveal_edit_direction_in_named_input(review_request):
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    h1 = {"instrument": [0], "verb": [1], "target": [0], "ivt": [17], "phase": [1]}
    bodies = []
    for first, second in ((h0, h1), (h1, h0)):
        neutral = build_presence_review_input(first, second,
            allowed_evidence_refs=["frame:101", "crop:1"], full_frame_ref="frame:101")
        original = with_propositions(review_request, neutral["propositions"])
        named = runner.variant_request(original, "names1000")
        body = json.loads(named.payload["input_text"])
        bodies.append(body)
        for forbidden in ("h0", "h1", "hypotheses", "operation", "change_id", "ADD", "REMOVE",
                          "proposal_instance_labels", "predicted_instance_regions"):
            assert forbidden not in named.payload["input_text"]
    assert bodies[0] == bodies[1]
