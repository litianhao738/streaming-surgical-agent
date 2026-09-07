"""Pure v3 normalization changes text length only, never model judgments."""

from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.presence_observation_length import (
    normalize_presence_observation_length,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
)


def raw_review(length=1001):
    return {"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": [{
        "proposition_id": "p003", "presence": "ABSENT", "observation": "观" * length,
        "evidence_refs": ["frame:101", "crop:1"], "scope": "FRAME", "full_frame_reviewed": True,
    }]}


@pytest.mark.parametrize("length", [1, 999, 1000, 1001, 1024, 1296, 5000])
def test_only_observation_string_length_changes_and_input_is_not_mutated(length):
    raw = raw_review(length)
    before = deepcopy(raw)
    normalized, audit = normalize_presence_observation_length(raw)
    expected = deepcopy(raw)
    expected["assessments"][0]["observation"] = "观" * min(length, 1000)
    assert normalized == expected and raw == before
    assert len(audit["changes"]) == int(length > 1000)
    if length > 1000:
        change = audit["changes"][0]
        assert change["path"] == "/assessments/0/observation"
        assert change["original_characters"] == length and change["retained_characters"] == 1000
        assert len(change["original_text_sha256"]) == len(change["retained_text_sha256"]) == 64
    normalized["assessments"][0]["evidence_refs"].append("another-ref")
    assert raw == before


def test_every_overlong_observation_is_processed_without_selecting_ids_or_verdicts():
    raw = raw_review()
    for index, presence in enumerate(("PRESENT", "UNCLEAR"), start=1):
        item = deepcopy(raw["assessments"][0])
        item.update(proposition_id=f"p00{index}", presence=presence, observation="x" * (1020 + index))
        raw["assessments"].append(item)
    normalized, audit = normalize_presence_observation_length(raw)
    assert len(audit["changes"]) == 3
    assert [a["presence"] for a in normalized["assessments"]] == ["ABSENT", "PRESENT", "UNCLEAR"]
    again, second_audit = normalize_presence_observation_length(normalized)
    assert again == normalized and second_audit["changes"] == []


@pytest.mark.parametrize("problem", ["version", "missing_version", "missing_observation", "number",
                                     "presence", "id", "duplicate_id", "ref_type", "scope", "bool", "extra"])
def test_other_schema_errors_are_not_repaired_even_when_text_is_overlong(problem):
    raw = raw_review(1296)
    item = raw["assessments"][0]
    if problem == "version":
        raw["schema_version"] = "frame_label_presence_review_v2"
    elif problem == "missing_version":
        del raw["schema_version"]
    elif problem == "missing_observation":
        del item["observation"]
    elif problem == "number":
        item["observation"] = 1296
    elif problem == "presence":
        item["presence"] = "SUPPORTED"
    elif problem == "id":
        item["proposition_id"] = "ivt:REMOVE:59"
    elif problem == "duplicate_id":
        raw["assessments"].append(deepcopy(item))
    elif problem == "ref_type":
        item["evidence_refs"] = [3]
    elif problem == "scope":
        item["scope"] = "CROP"
    elif problem == "bool":
        item["full_frame_reviewed"] = 1
    elif problem == "extra":
        item["confidence"] = .8
    before = deepcopy(raw)
    with pytest.raises(ApiSchemaError):
        normalize_presence_observation_length(raw)
    assert raw == before
