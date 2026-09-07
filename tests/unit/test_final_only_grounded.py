from copy import deepcopy

import pytest

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.final_only_grounded import (
    final_labels,
    finalize_grounded_repair,
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_repair import (
    LOCATOR_VERSION,
    PROPOSAL_VERSION,
    REVIEW_VERSION,
    admit,
    proposed_labels,
)


def _evidence(groups=((2, [59]),)):
    locator = {
        "schema_version": LOCATOR_VERSION,
        "all_visible_tools_covered": True,
        "instances": [
            {"instance_id": i, "tip_box": [.2, .2, .4, .4]}
            for i in range(1, len(groups) + 1)
        ],
    }
    proposal = {
        "schema_version": PROPOSAL_VERSION,
        "instances": [
            {"instance_id": i, "instrument_id": instrument, "ivt_ids": ivts,
             "support": "SUPPORTED", "contact_observation": "Visible contact region."}
            for i, (instrument, ivts) in enumerate(groups, 1)
        ],
    }
    review = {
        "schema_version": REVIEW_VERSION,
        "preferred": "FIRST",
        "all_visible_tools_covered": True,
        "instances": [
            {"instance_id": i, "crop_relevant": True,
             "instrument_identity_supported": True,
             "target_identity_or_oov_supported": True,
             "action_or_oov_supported": True,
             "distinguishing_observation": "Visible contact distinguishes the proposal."}
            for i in range(1, len(groups) + 1)
        ],
    }
    return locator, proposal, review


def _h0():
    return {"instrument": [2], "verb": [2], "target": [0], "ivt": [60], "phase": [3]}


def test_permuted_sets_skip_review_and_preserve_original_h0_order():
    h0 = {"instrument": [2, 0], "verb": [2, 1], "target": [0],
          "ivt": [60, 17], "phase": [3]}
    saved = deepcopy(h0)
    locator, proposal, review = _evidence(((0, [17]), (2, [60])))
    old_candidate = proposed_labels(h0, locator, proposal)
    assert old_candidate != h0  # Reproduces the unnecessary v1 review/acceptance.
    assert admit(h0, old_candidate, locator, proposal, review,
                 proposal_slot="FIRST")["decision"] == "ACCEPT"
    result = finalize_grounded_repair(
        h0, locator, proposal, review, proposal_slot="FIRST"
    )
    assert result["decision"] == "KEEP"
    assert result["reason"] == "NO_LABEL_CHANGE"
    assert not result["review_required"] and result["hypotheses"] is None
    assert result["final"] == h0 == saved


def test_individually_valid_instances_cannot_overflow_final_schema():
    h0 = _h0()
    locator, proposal, review = _evidence(((0, [0, 1, 2]), (0, [3, 4, 5]),
                                         (0, [6, 7, 8])))
    old_candidate = proposed_labels(h0, locator, proposal)
    assert len(old_candidate["ivt"]) == 9
    assert admit(h0, old_candidate, locator, proposal, review,
                 proposal_slot="FIRST")["decision"] == "ACCEPT"
    result = finalize_grounded_repair(
        h0, locator, proposal, review, proposal_slot="FIRST"
    )
    assert result["reason"] == "CANDIDATE_OUTSIDE_FINAL_ONLY_CONTRACT"
    assert result["decision"] == "KEEP" and result["h1"] is None
    assert not result["review_required"] and result["final"] == h0


def test_wire_h0_and_batch_labels_share_boundary_without_forcing_projection():
    # An existing valid final-only response need not have exact IVT projection.
    h0 = {"instrument": [2], "verb": [2, 1], "target": [0, 8],
          "ivt": [59, 64], "phase": [3]}
    wire = {"schema_version": "joint_perception_final_only_v1"}
    wire.update({task: {"selected_ids": values} for task, values in h0.items()
                 if task != "phase"})
    wire["phase"] = {"selected_id": 3}
    assert final_labels(wire) == final_labels(h0) == h0
    result = finalize_grounded_repair(wire, None, None, None, proposal_slot="FIRST")
    assert result["final"] == h0 and result["decision"] == "KEEP"


def test_valid_repair_preserves_phase_and_hypothesis_binding():
    h0 = _h0()
    locator, proposal, review = _evidence()
    saved = deepcopy((h0, locator, proposal, review))
    prepared = prepare_grounded_review(h0, locator, proposal, proposal_slot="FIRST")
    assert prepared["review_required"]
    assert prepared["hypotheses"] == {"FIRST": prepared["h1"], "SECOND": h0}
    result = finalize_grounded_repair(
        h0, locator, proposal, review, proposal_slot="FIRST"
    )
    assert result["decision"] == "ACCEPT" and result["final"]["ivt"] == [59]
    assert result["final"]["phase"] == h0["phase"]
    assert (h0, locator, proposal, review) == saved
    assert finalize_grounded_repair(h0, locator, proposal, review,
                                   proposal_slot="SECOND")["decision"] == "KEEP"


@pytest.mark.parametrize("review_change", ["missing", "duplicate", "unsupported", "tie"])
def test_incomplete_or_unsupported_review_keeps_h0(review_change):
    h0 = _h0()
    locator, proposal, review = _evidence()
    if review_change == "missing":
        del review["instances"][0]["crop_relevant"]
    elif review_change == "duplicate":
        review["instances"].append(deepcopy(review["instances"][0]))
    elif review_change == "unsupported":
        review["instances"][0]["action_or_oov_supported"] = False
    else:
        review["preferred"] = "TIE"
    result = finalize_grounded_repair(h0, locator, proposal, review, proposal_slot="FIRST")
    assert result["decision"] == "KEEP" and result["final"] == h0


def test_invalid_optional_localization_keeps_valid_baseline():
    locator, proposal, review = _evidence()
    locator["instances"][0]["tip_box"] = [.4, .2, .2, .4]
    result = finalize_grounded_repair(_h0(), locator, proposal, review, proposal_slot="FIRST")
    assert not result["review_required"] and result["final"] == _h0()


@pytest.mark.parametrize("stage", ["locator", "proposal", "review"])
def test_oversized_optional_numeric_id_cannot_abort_the_valid_h0(stage):
    locator, proposal, review = _evidence()
    {"locator": locator, "proposal": proposal, "review": review}[stage]["instances"][0][
        "instance_id"
    ] = 10 ** 1000
    result = finalize_grounded_repair(_h0(), locator, proposal, review, proposal_slot="FIRST")
    assert result["decision"] == "KEEP" and result["final"] == _h0()


@pytest.mark.parametrize("task,value", [
    ("phase", []), ("phase", [1, 2]), ("ivt", [True]), ("ivt", [100]),
    ("verb", [2, 2]), ("target", "0"),
])
def test_invalid_h0_cannot_be_silently_kept(task, value):
    h0 = _h0()
    h0[task] = value
    with pytest.raises(ApiSchemaError):
        finalize_grounded_repair(h0, None, None, None, proposal_slot="FIRST")
