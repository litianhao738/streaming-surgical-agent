"""Mechanism tests; synthetic examples never represent model accuracy."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError

from surgical_agent.research.verification.prior_panel import (
    COMPONENTS,
    admit,
    apply_patch_response,
    build_universe,
    fit_prior,
    normalize_review,
    panel_votes,
    run_arm,
    video_counts,
)

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "docs/protocols/h0_prior_panel_v1"
JS = json.loads((C / "judge_response.schema.json").read_text(encoding="utf-8"))
PS = json.loads((C / "repair_response.schema.json").read_text(encoding="utf-8"))
H0 = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]}
REFS = ["frame:1", "frame:26", "frame:51"]


def response(u, overrides=None):
    out = []
    for p in u["propositions"]:
        presence = (overrides or {}).get((p["task"], p["label_id"]), "UNCLEAR")
        out.append({"proposition_id": p["proposition_id"], "presence": presence, "scope": "FRAME",
                    "full_frame_reviewed": True, "other_instances_accounted_for": True,
                    "evidence_refs": [REFS[-1]], "witness": {"local_instance_id": "tool1",
                    "target_frame_ref": REFS[-1], "tool_point_xy": [0.3, 0.3],
                    "interaction_point_xy": [0.4, 0.4], "target_role": "INTERACTION_TARGET",
                    "same_interaction": "YES", "temporal_refs": REFS,
                    "ontology_boundary": "IN_VOCABULARY"}, "issue_codes": ["NONE"], "observation": "Synthetic."})
    return {"schema_version": "h0_prior_panel_judge_v1", "assessments": out}


def synthetic_counts():
    return video_counts([{"frame_id": i, "mask": dict.fromkeys((*H0,), True),
                          "gt": {"instrument": [0, 0], "verb": [0], "target": [0], "ivt": [0], "phase": [0]}}
                         for i in range(40)])


def test_video_weighting_masks_and_isolation():
    c = synthetic_counts()
    assert c["instrument"]["x"][0] == 40
    all_counts = {v: deepcopy(c) for v in ("a", "b", "c", "heldout")}
    first = fit_prior(all_counts, "heldout")
    all_counts["heldout"]["ivt"]["x"][0] = 99999
    assert fit_prior(all_counts, "heldout") == first
    assert first["tasks"]["ivt"]["global"][0]["eligible"]
    assert first["tasks"]["ivt"]["global"][0]["rate"] == 1
    masked = video_counts([{"frame_id": 1, "mask": dict.fromkeys(H0, False), "gt": {}}])
    assert masked["ivt"]["n"] == 0
    with pytest.raises(ValueError):
        build_universe(H0, video_id="another", prior=first)
    u = build_universe(H0, video_id="heldout", prior=first)
    assert len([p for p in u["propositions"] if p["task"] == "instrument"]) == 7
    assert any(p["task"] == "ivt" and p["label_id"] == 0 for p in u["propositions"])


def test_equal_video_weights_and_low_support():
    c = synthetic_counts()
    zero = deepcopy(c)
    zero["ivt"]["n"] = 400
    zero["ivt"]["x"][0] = 0
    prior = fit_prior({"short": c, "long": zero}, "heldout")
    assert prior["tasks"]["ivt"]["global"][0]["rate"] == 0.5
    assert not prior["tasks"]["ivt"]["global"][0]["eligible"]
    u = build_universe(H0, video_id="heldout", prior=prior)
    assert not any(p["task"] == "ivt" for p in u["propositions"])


def test_coverage_failure_vs_single_item_normalization():
    u = build_universe(H0, video_id="x")
    r = response(u, {("instrument", 0): "ABSENT"})
    r["assessments"][0]["scope"] = "LOCAL"
    n = normalize_review(r, u, REFS, JS)
    assert n["assessments"][0]["presence"] == "UNCLEAR"
    assert n["assessments"][0]["raw_presence"] == "ABSENT"
    r["assessments"].pop()
    with pytest.raises(ValueError, match="coverage"):
        normalize_review(r, u, REFS, JS)
    with pytest.raises(ValueError):
        panel_votes([n, n])


def test_null_ontology_exception_does_not_treat_uncertainty_as_null():
    h0 = {"instrument": [0], "verb": [9], "target": [14], "ivt": [94], "phase": [0]}
    u = build_universe(h0, video_id="x")
    r = response(u, {("ivt", 94): "PRESENT"})
    a = next(a for a in r["assessments"] if a["presence"] == "PRESENT")
    a["witness"].update(ontology_boundary="JOINT_OUT_OF_VOCABULARY", interaction_point_xy=None,
                         target_role="NOT_APPLICABLE", same_interaction="NOT_APPLICABLE")
    assert normalize_review(r, u, REFS, JS)["assessments"][-1]["presence"] == "PRESENT"
    a["witness"]["ontology_boundary"] = "UNCLEAR"
    assert normalize_review(r, u, REFS, JS)["assessments"][-1]["presence"] == "UNCLEAR"


def test_two_add_votes_but_three_remove_votes():
    u = build_universe(H0, video_id="x")
    draft = {**H0, "instrument": [1]}
    yes = response(u, {("instrument", 1): "PRESENT", ("instrument", 0): "ABSENT"})
    neutral = response(u)
    assert admit(H0, draft, u, [yes, yes, neutral])["instrument"] == [0, 1]
    assert admit(H0, draft, u, [yes, yes, yes])["instrument"] == [1]


def test_shared_component_rebuilt_for_new_parent_and_no_projection():
    # Two distinct IVTs sharing a verb; no frame-specific exceptions.
    a, b = next((a, b) for a in COMPONENTS for b in COMPONENTS if a < b
                and COMPONENTS[a]["verb"] == COMPONENTS[b]["verb"]
                and COMPONENTS[a]["verb"] != 9)
    h0 = {**H0, "target": [14]}
    draft = deepcopy(h0)
    draft["ivt"] = [a, b]
    for parent in (a, b):
        for q, c in COMPONENTS[parent].items():
            draft[q] = sorted(set(draft[q]) | {c})
    u = build_universe(draft, video_id="x")
    # Restore actual immutable eligibility: new V/T came only with IVTs.
    for p in u["propositions"]:
        if p["task"] in {"verb", "target"} and p["label_id"] not in h0[p["task"]]:
            u["eligibility"][p["proposition_id"]]["independent"] = False
    votes = {(q, c): "PRESENT" for q in ("instrument", "verb", "target") for c in draft[q]}
    votes["ivt", b] = "PRESENT"
    r = response(u, votes)
    out = admit(h0, draft, u, [r, r, r])
    assert out["ivt"] == [b]
    assert COMPONENTS[b]["verb"] in out["verb"]
    assert 14 in out["target"]  # independent H0 target not projected away


def test_invalid_patch_direction_and_outside_issue():
    u = build_universe(H0, video_id="x")
    pid = u["propositions"][1]["proposition_id"]
    issue = {"issue_id": "i1", "proposition_id": pid, "operation": "ADD"}
    edit = {"proposition_id": pid, "operation": "ADD", "edit_role": "INDEPENDENT_COMPONENT",
            "dependency_ivt_proposition_ids": [], "issue_refs": ["i1"], "evidence_refs": [REFS[-1]], "observation": "Synthetic."}
    p = {"schema_version": "h0_prior_panel_patch_v1", "edits": [edit]}
    assert apply_patch_response(H0, u, [issue], p, REFS, PS)["instrument"] == [0, 1]
    edit["operation"] = "REMOVE"
    with pytest.raises(ValueError, match="direction"):
        apply_patch_response(H0, u, [issue], p, REFS, PS)
    p["edits"] = [edit] * 17
    with pytest.raises(ValidationError):
        apply_patch_response(H0, u, [issue], p, REFS, PS)


def test_round_failure_preserves_last_accepted_and_no_extra_patch():
    u = build_universe(H0, video_id="x")
    pid1, pid2 = u["propositions"][1]["proposition_id"], u["propositions"][2]["proposition_id"]
    calls = []

    def panel(n, current):
        if n == 3:
            return [None, response(u), response(u)]
        pred = {("instrument", 1): "PRESENT"}
        if n == 2:
            pred["instrument", 2] = "PRESENT"
        return [response(u, pred)] * 3

    def patch(n, current, issues):
        calls.append(n)
        pid = pid1 if n == 1 else pid2
        issue = next(i for i in issues if i["proposition_id"] == pid)
        return {"schema_version": "h0_prior_panel_patch_v1", "edits": [{"proposition_id": pid,
                "operation": "ADD", "edit_role": "INDEPENDENT_COMPONENT", "dependency_ivt_proposition_ids": [],
                "issue_refs": [issue["issue_id"]], "evidence_refs": [REFS[-1]], "observation": "Synthetic."}]}
    result = run_arm(H0, u, panel, patch, JS, PS, REFS, lambda *args: None)
    assert result["final"]["instrument"] == [0, 1]
    assert result["snapshots"][0] == H0
    assert calls == [1, 2]
    assert result["stop_reason"].startswith("ROUND_INVALID")


def test_valid_last_round_can_revert_previous_edit():
    u = build_universe(H0, video_id="x")
    draft = {**H0, "instrument": [0, 1]}
    n = response(u)
    assert admit(H0, draft, u, [n, n, n]) == H0
