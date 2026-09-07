from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

from scripts import run_mean_panel_trial as runner
from surgical_agent.research.verification.mean_panel import (
    admit,
    full_universe,
    issues_for,
    mean_scores,
    run_arm,
)
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS


def response(value=3):
    return {"schema_version": "five_mean_scores_v1", "observation": "Academic surgical observation.",
            **{q: [value] * n for q, n in BOUNDS.items()}}


def baseline():
    return {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]}


def patch(issue):
    return {"schema_version": "h0_prior_panel_patch_v1", "edits": [{
        "proposition_id": issue["proposition_id"], "operation": issue["operation"],
        "edit_role": "INDEPENDENT_COMPONENT", "dependency_ivt_proposition_ids": [],
        "issue_refs": [issue["issue_id"]], "evidence_refs": ["target"], "observation": "Visible tool."}]}


def test_exact_five_arithmetic_mean_not_vote_count():
    raw = [response(v) for v in (5, 5, 4, 3, 3)]
    assert mean_scores(raw)["ivt"][0] == 4
    raw[-1]["ivt"][0] = 2
    assert mean_scores(raw)["ivt"][0] == 3.8
    for invalid in (raw[:4], raw + [response()], raw[:4] + [None]):
        with pytest.raises(ValueError):
            mean_scores(invalid)


@pytest.mark.parametrize("bad", [True, 4.5, 0, 6])
def test_invalid_ratings_rejected(bad):
    raw = [response() for _ in range(5)]
    raw[0]["ivt"][0] = bad
    with pytest.raises((ValueError, ValidationError)):
        mean_scores(raw)


def test_full_catalog_and_score_thresholds():
    u = full_universe()
    assert len(u["propositions"]) == 132
    assert len({p["proposition_id"] for p in u["propositions"]}) == 132
    raw = [response() for _ in range(5)]
    for r in raw:
        r["instrument"][0] = 2
        r["ivt"][99] = 4
    issues = issues_for(baseline(), mean_scores(raw), raw)
    assert {(i["operation"], i["mean_score"]) for i in issues} == {("ADD", 4), ("REMOVE", 2)}
    assert len(issues) == 2


def test_new_ivt_requires_separately_supported_components_and_shared_retention():
    h0 = baseline()
    draft = deepcopy(h0)
    draft["ivt"] = [0]
    for q, c in COMPONENTS[0].items():
        draft[q] = sorted(set(draft[q]) | {c})
    means = mean_scores([response() for _ in range(5)])
    means["ivt"][0] = 4
    assert admit(h0, draft, means)["ivt"] == []
    for q, c in COMPONENTS[0].items():
        means[q][c] = 4
    assert admit(h0, draft, means) == draft
    removal = deepcopy(draft)
    removal["instrument"] = []
    means["instrument"][0] = 1
    assert admit(draft, removal, means)["instrument"] == [0]


def test_three_rounds_and_failed_final_round_keep_last_reviewed_state():
    raw = [response() for _ in range(5)]
    for r in raw:
        r["instrument"][1] = 4
    raw2 = deepcopy(raw)
    for r in raw2:
        r["instrument"][2] = 4
    saved, called = {}, []

    def panel(round_no, draft, previous):
        called.append(round_no)
        return {1: raw, 2: raw2, 3: [None] * 5}[round_no]

    result = run_arm(baseline(), panel, lambda n, s, issues: patch(issues[0]),
                     runner.read(runner.legacy.CONTRACTS / "repair_response.schema.json"),
                     ["target"], lambda name, value: saved.update({name: deepcopy(value)}))
    assert called == [1, 2, 3]
    assert result["final"]["instrument"] == [0, 1]
    assert result["snapshots"][0] == baseline()
    assert result["snapshots"][2] == result["snapshots"][1]
    assert result["stop_reason"].startswith("ROUND_INVALID")
    assert saved["round_3"]["status"].startswith("ROUND_INVALID")


def test_later_round_reassesses_net_edits_without_vote_accumulation():
    h0 = baseline()
    draft = {**h0, "instrument": [0, 1]}
    high = mean_scores([response(4) for _ in range(5)])
    unclear = mean_scores([response(3) for _ in range(5)])
    assert admit(h0, draft, high) == draft
    assert admit(h0, draft, unclear) == h0


def test_five_model_wire_starts_with_academic_context_and_blind_first_round(monkeypatch):
    monkeypatch.setattr(runner.legacy, "packet_for", lambda *a: {
        "current_prediction": baseline(), "propositions": [], "video_id": "VIDX"})
    first = runner.judge_packet(None, None, baseline(), 1)
    assert next(iter(first)) == "academic_context"
    assert "current_prediction" not in first
    later = runner.judge_packet(None, None, baseline(), 2, response())
    assert later["your_previous_response"] == response()
    base = SimpleNamespace(images=[], payload={"image_details": []})
    prompt = (runner.CONTRACTS / "judge_prompt.txt").read_text(encoding="utf-8")
    for model, tag in runner.MODEL_TAGS[:5]:
        body = runner.legacy.request_body(base, {"model": model, "tag": tag, "max_output": 16384},
                                          runner.response_schema(), prompt, first)
        assert body["messages"][0]["content"].startswith(runner.ACADEMIC_CONTEXT)
        assert body["messages"][1]["content"][0]["text"].startswith('{"academic_context":')


def test_wrong_length_is_not_silently_padded():
    raw = [response() for _ in range(5)]
    raw[0]["ivt"].pop()
    with pytest.raises(ValidationError):
        mean_scores(raw)
