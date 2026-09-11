"""Instrument-anchored extraction: strict rows, legal triplets, quorum only."""
from copy import deepcopy

import pytest

from surgical_agent.research.verification.instrument_extraction import (
    NULL_TARGET,
    NULL_VERB,
    TRIPLET_TO_IVT,
    combine,
    extraction_schema,
    merge_with_h0,
    response_error,
    row_error,
    seat_labels,
    triplets,
)

SEATS5 = ("grok", "qwen", "gpt", "gemini", "deepseek")
H0 = {"instrument": [0, 2], "verb": [2], "target": [0], "ivt": [1], "phase": [1]}
LEGAL = next(iter(TRIPLET_TO_IVT))


def row(instrument=0, present=True, verb=2, target=0, alt_verb=None, alt_target=None,
        indices=(2,), observation="Tip contacts the tissue in the current image."):
    return {"instrument_id": instrument, "present": present, "verb_id": verb,
            "target_id": target, "alternative_verb_id": alt_verb,
            "alternative_target_id": alt_target, "image_indices": list(indices),
            "observation": observation}


def response(rows):
    return {"instruments": rows}


def test_schema_pins_the_roster_and_row_count():
    schema = extraction_schema([0, 2], 3)
    items = schema["properties"]["instruments"]
    assert items["minItems"] == items["maxItems"] == 2
    assert items["items"]["properties"]["instrument_id"]["enum"] == [0, 2]
    assert items["items"]["additionalProperties"] is False
    with pytest.raises(ValueError):
        extraction_schema([], 3)
    with pytest.raises(ValueError):
        extraction_schema([99], 3)


@pytest.mark.parametrize("mutation,expected", [
    ({"instrument": 5}, "UNKNOWN_INSTRUMENT"),
    ({"present": "yes"}, "INVALID_PRESENCE"),
    ({"verb": 99}, "INVALID_VERB_ID"),
    ({"target": -1}, "INVALID_TARGET_ID"),
    ({"indices": (0, 0)}, "INVALID_IMAGE_REFERENCE"),
    ({"indices": (7,)}, "INVALID_IMAGE_REFERENCE"),
    ({"observation": "   "}, "INVALID_OBSERVATION"),
    ({"observation": "x" * 1001}, "INVALID_OBSERVATION"),
    ({"verb": None}, "PRESENT_WITHOUT_INTERACTION"),
    ({"indices": (0,)}, "CURRENT_IMAGE_REQUIRED"),
    ({"alt_verb": 2}, "ALTERNATIVE_REPEATS_VERB"),
    ({"alt_target": 0}, "ALTERNATIVE_REPEATS_TARGET"),
])
def test_every_row_rejection_is_named(mutation, expected):
    assert row_error(row(**mutation), [0, 2], 3) == expected


def test_a_valid_row_and_a_valid_absence_both_pass():
    assert row_error(row(), [0, 2], 3) is None
    assert row_error(row(present=False, verb=None, target=None, indices=()), [0, 2], 3) is None


def test_an_absent_instrument_cannot_carry_an_interaction():
    assert row_error(row(present=False, verb=2, target=None, indices=()),
                     [0, 2], 3) == "ABSENT_WITH_INTERACTION"


def test_unknown_or_missing_fields_are_rejected_rather_than_filled_in():
    extra = {**row(), "confidence": 0.9}
    assert row_error(extra, [0, 2], 3) == "INVALID_FIELDS"
    missing = {k: v for k, v in row().items() if k != "observation"}
    assert row_error(missing, [0, 2], 3) == "INVALID_FIELDS"


@pytest.mark.parametrize("payload,expected", [
    ({"rows": []}, "INVALID_ENVELOPE"),
    (response([row(0)]), "INCOMPLETE_ROSTER"),
    (response([row(0), row(0)]), "DUPLICATE_INSTRUMENT"),
    (response([row(0), row(2)]), None),
])
def test_response_envelope_contract(payload, expected):
    assert response_error(payload, [0, 2], 3) == expected


def test_null_labels_are_legal_when_the_ontology_cannot_express_the_scene():
    assert row_error(row(verb=NULL_VERB, target=NULL_TARGET), [0, 2], 3) is None


def test_triplets_keep_only_real_ivt_classes_and_record_the_rest():
    instrument, verb, target = LEGAL
    illegal_target = next(t for t in range(15) if (instrument, verb, t) not in TRIPLET_TO_IVT)
    raw = response([row(instrument, verb=verb, target=target, alt_target=illegal_target)])
    assembled = triplets(raw)
    assert assembled["ivt"] == {TRIPLET_TO_IVT[LEGAL]: ["primary"]}
    assert assembled["rejected"] == [{"instrument": instrument, "verb": verb,
                                      "target": illegal_target,
                                      "reading": "alternative_target",
                                      "reason": "NOT_AN_IVT_CLASS"}]


def test_an_absent_instrument_contributes_no_relation():
    raw = response([row(present=False, verb=None, target=None, indices=())])
    assert triplets(raw)["ivt"] == {}
    assert seat_labels(raw) == {"instrument": set(), "verb": set(), "target": set(), "ivt": set()}


def test_quorum_admits_only_what_three_seats_read_the_same_way():
    agree = response([row(0, verb=2, target=0)])
    dissent = response([row(0, verb=3, target=0)])
    responses = dict(zip(SEATS5, [agree, agree, agree, dissent, dissent], strict=True))
    out = combine(responses, [0], 3)
    assert out["status"] == "EXTRACTED" and out["valid_seats"] == 5
    assert out["prediction"]["verb"] == [2]
    assert out["votes"]["verb"] == {2: 3, 3: 2}


def test_an_invalid_seat_casts_no_vote_and_is_named():
    good = response([row(0)])
    broken = response([row(0, indices=(0,))])
    responses = dict(zip(SEATS5, [good, good, good, broken, broken], strict=True))
    out = combine(responses, [0], 3)
    assert out["valid_seats"] == 3
    assert set(out["errors"]) == set(SEATS5[3:])
    assert all(v == "CURRENT_IMAGE_REQUIRED" for v in out["errors"].values())
    assert out["prediction"]["target"] == [0]


def test_too_few_valid_seats_reports_insufficient_rather_than_guessing():
    good = response([row(0)])
    broken = response([row(0, indices=(0,))])
    responses = dict(zip(SEATS5, [good, good, broken, broken, broken], strict=True))
    out = combine(responses, [0], 3)
    assert out["status"] == "INSUFFICIENT_VALID_SEATS" and out["valid_seats"] == 2


def test_alternatives_are_excluded_unless_explicitly_enabled():
    instrument, verb, target = LEGAL
    other = next(((i, v, t) for (i, v, t) in TRIPLET_TO_IVT
                  if i == instrument and t == target and v != verb), None)
    if other is None:
        pytest.skip("this ontology has no alternative verb for the sampled relation")
    raw = response([row(instrument, verb=verb, target=target, alt_verb=other[1])])
    responses = dict.fromkeys(SEATS5, raw)
    without = combine(responses, [instrument], 3)
    with_alts = combine(responses, [instrument], 3, include_alternatives=True)
    assert TRIPLET_TO_IVT[other] not in without["prediction"]["ivt"]
    assert TRIPLET_TO_IVT[other] in with_alts["prediction"]["ivt"]


def test_a_seat_votes_at_most_once_per_relation():
    instrument, verb, target = LEGAL
    raw = response([row(instrument, verb=verb, target=target, alt_verb=None, alt_target=None)])
    out = combine(dict.fromkeys(SEATS5, raw), [instrument], 3, include_alternatives=True)
    assert out["votes"]["ivt"][TRIPLET_TO_IVT[LEGAL]] == len(SEATS5)


@pytest.mark.parametrize("mode,expected_ivt", [("union", [1, 60]), ("replace", [60])])
def test_merge_modes_and_phase_is_never_touched(mode, expected_ivt):
    extraction = {"status": "EXTRACTED", "prediction": {
        "instrument": [0], "verb": [2], "target": [0], "ivt": [60]}}
    out = merge_with_h0(H0, extraction, mode=mode)
    assert out["ivt"] == expected_ivt
    assert out["phase"] == H0["phase"]
    assert out["phase"] is not H0["phase"]


def test_a_failed_extraction_leaves_h0_untouched():
    extraction = {"status": "INSUFFICIENT_VALID_SEATS", "prediction": {
        "instrument": [], "verb": [], "target": [], "ivt": []}}
    for mode in ("union", "replace"):
        assert merge_with_h0(H0, extraction, mode=mode) == {
            **{t: sorted(H0[t]) for t in ("instrument", "verb", "target", "ivt")},
            "phase": H0["phase"]}


def test_merge_rejects_an_undeclared_mode_and_does_not_mutate_h0():
    snapshot = deepcopy(H0)
    with pytest.raises(ValueError):
        merge_with_h0(H0, {"status": "EXTRACTED", "prediction": {}}, mode="overwrite")
    assert H0 == snapshot


def test_assembled_components_invert_the_published_ontology_csv():
    """Check against the frozen CSV, not a module global another test may patch."""
    from surgical_agent.perception.ontology_prompt import _ivt_rows

    rows = list(_ivt_rows())
    expected = {(i, v, t): ivt for ivt, i, v, t in rows}
    assert len(expected) == len(rows), "the ontology must not repeat a triplet"
    assert TRIPLET_TO_IVT == expected
    assert len(set(TRIPLET_TO_IVT.values())) == len(TRIPLET_TO_IVT), "mapping must be injective"
