import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from scripts import run_five_head_repair_trial as runner
from scripts.score_five_head_repair_trial import TASKS, VERSIONS, summarize
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.five_head_repair import (
    aggregate_five_heads,
    compile_joint,
    joint_schema,
    normalize_five_heads,
    phase_hints,
    select_five_heads,
)
from surgical_agent.research.verification.prior_panel import digest


def prediction():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [2]}


def proposal(phase=3):
    state = prediction()
    state["phase"] = [phase]
    return {"prediction": state, "observations": {t: "Current scene was inspected." for t in TASKS}}


def compiled():
    state = prediction()
    return compile_joint(state, make_pool(state), proposal())


def reviews(pool):
    out = {s: {"judgments": {}} for s in SEATS}
    for s in SEATS:
        for p in pool["propositions"]:
            score = 5 if p["id"] == "phase_3" else 1 if p["task"] == "phase" else 3
            out[s]["judgments"][p["id"]] = {"rating": score,
                "finding": "MATCH" if score >= 4 else "REFUTED" if score <= 2 else "UNCLEAR",
                "scope": "WHOLE_FRAME", "image_indices": [2], "observation": "The current whole scene supports this assessment."}
    return out


def test_joint_rewrite_changes_phase_and_closes_ivt_components_without_mutating_input():
    old, raw = prediction(), proposal()
    raw["prediction"].update(instrument=[], verb=[], target=[], ivt=[60])
    snapshot = deepcopy((old, raw))
    result = compile_joint(old, make_pool(old), raw)
    assert result["paper_style"] == {"instrument": [2], "verb": [2], "target": [0], "ivt": [60], "phase": [3]}
    assert len(result["derived_components"]) == 3
    assert not result["issues_after"]
    assert {p["label_id"] for p in result["pool"]["propositions"] if p["task"] == "phase"} == set(range(7))
    assert (old, raw) == snapshot


def test_empty_four_guard_does_not_freeze_phase():
    raw = proposal()
    for t in TASKS[:4]:
        raw["prediction"][t] = []
    result = compile_joint(prediction(), make_pool(prediction()), raw)
    assert result["empty_four_head_guard"]
    assert result["paper_style"]["ivt"] == [0]
    assert result["paper_style"]["phase"] == [3]


@pytest.mark.parametrize("field,value", [("phase", [2, 3]), ("phase", [7]), ("phase", []), ("phase", [True]),
    ("instrument", [0, 1, 2, 3]), ("verb", list(range(5))), ("target", list(range(6))), ("ivt", list(range(9)))])
def test_invalid_or_overcapacity_joint_answer_rejected(field, value):
    raw = proposal()
    raw["prediction"][field] = value
    with pytest.raises((ValueError, TypeError)):
        compile_joint(prediction(), make_pool(prediction()), raw)


def test_phase_is_really_reviewed_and_atomically_updated():
    pool = compiled()["pool"]
    clean, _ = normalize_five_heads(reviews(pool), pool)
    means, _ = aggregate_five_heads(clean, pool)
    final, decision = select_five_heads(prediction(), pool, means)
    assert final["phase"] == [3]
    assert decision["reason"] == "UNIQUE_BETTER_SUPPORTED_PHASE"
    assert {t: final[t] for t in TASKS[:4]} == {t: prediction()[t] for t in TASKS[:4]}


@pytest.mark.parametrize("alter,reason", [({"phase_2": None}, "INVALID_CURRENT_PHASE_EVIDENCE"),
    ({"phase_1": 5}, "TIED_PHASE_SUPPORT"), ({"phase_3": 3}, "NO_SUPPORTED_ALTERNATIVE"),
    ({"phase_2": 5, "phase_3": 4}, "CURRENT_PHASE_BEST")])
def test_phase_ambiguity_keeps_current(alter, reason):
    pool = compiled()["pool"]
    clean, _ = normalize_five_heads(reviews(pool), pool)
    means, _ = aggregate_five_heads(clean, pool)
    means.update(alter)
    final, decision = select_five_heads(prediction(), pool, means)
    assert final["phase"] == [2]
    assert decision["reason"] == reason


def test_phase_row_metadata_and_local_negative_are_not_silently_validated():
    pool = compiled()["pool"]
    raw = reviews(pool)
    rows = [{"candidate_id": k, **v} for k, v in raw["gemini"]["judgments"].items()]
    rows.append(deepcopy(next(r for r in rows if r["candidate_id"] == "phase_3")))
    raw["gemini"] = {"rows": rows}
    raw["qwen"]["judgments"]["phase_2"]["scope"] = "LOCAL_REGION"
    clean, formatting = normalize_five_heads(raw, pool)
    means, _ = aggregate_five_heads(clean, pool)
    assert formatting["gemini"]["errors"]["phase_3"] == ["DUPLICATE_CANDIDATE_ID"]
    assert formatting["qwen"]["errors"]["phase_2"] == ["LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"]
    assert means["phase_3"] is None and means["phase_2"] is None
    assert select_five_heads(prediction(), pool, means)[0]["phase"] == [2]


def test_joint_run_shares_one_proposal_and_reviews_even_without_four_head_changes(monkeypatch):
    initial = {"graph_r1": {"prediction": prediction(), "pool": make_pool(prediction())}}
    monkeypatch.setattr(runner, "repair_wire", lambda *a: {"messages": []})
    monkeypatch.setattr(runner, "five_review_wire", lambda s, b, f, p: {"pool": p, "messages": []})

    class Calls:
        stopped = False

        def __init__(self):
            self.rows = []

        def call(self, target, stage, seat, body):
            self.rows.append({"target": target, "stage": stage, "seat": seat})
            return proposal() if seat == "base" else reviews(body["pool"])[seat]

    calls = Calls()
    result = runner.run_target(calls, SimpleNamespace(images=[1, 2, 3]), {"key": "fixed"}, initial)
    assert result["status"] == "REVIEWED"
    assert len(calls.rows) == 6
    assert result["paper_style"]["phase"] == result["panel_five"]["phase"] == [3]
    assert result["panel_four_shadow"]["phase"] == [2]
    assert result["llm_raw"] == result["raw_repair"]["prediction"]


def test_failed_repair_preserves_all_arms_and_makes_no_review_call(monkeypatch):
    monkeypatch.setattr(runner, "repair_wire", lambda *a: {"messages": []})
    calls = SimpleNamespace(stopped=False, call=lambda *a: None)
    result = runner.run_target(calls, SimpleNamespace(images=[1, 2, 3]), {"key": "fixed"},
        {"graph_r1": {"prediction": prediction(), "pool": make_pool(prediction())}})
    assert result["status"] == "REPAIR_FAILED"
    for arm in ("llm_raw", "paper_style", "panel_five", "panel_four_shadow"):
        assert result[arm] == prediction()


def test_masked_phase_is_excluded_not_scored_as_wrong():
    row = {"key": "x", "video_id": "VID00", "frame_id": 51, **{v: prediction() for v in VERSIONS}}
    row["panel_five"] = proposal()["prediction"]
    truth = [{"video_id": "VID00", "frame_id": 51, "gt": prediction(), "mask": {t: t != "phase" for t in TASKS}}]
    initial = {"graph_r1": {"pool": make_pool(prediction())}}
    record = {"status": "REVIEWED", "compiled": compiled(), "phase_decision": {}}
    summary, details = summarize([row], truth, [initial], {"x": record})
    assert summary["metrics"]["panel_five"]["tasks"]["phase"]["valid_targets"] == 0
    assert "phase" not in details[0]["comparisons"]["graph_r1_to_panel_five"]["tasks"]


def test_phase_prior_rejects_query_video_leakage():
    prior = {"excluded_video": "VID00", "fit_videos": ["VID00"], "tasks": {}}
    prior["table_sha256"] = digest(prior)
    with pytest.raises(ValueError, match="query-contaminated"):
        phase_hints(prediction(), prior, video_id="VID00")


def test_real_cached_request_contracts_have_all_heads_and_no_fixed_phase():
    from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
    from scripts.run_prior_panel_trial import read
    from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

    source = runner.SOURCE
    if not source.exists():
        pytest.skip("local closed eight-target archive unavailable")
    selected = read(source / "plan.json")["selection"][0]
    initial = read(source / "initial_state.json")["targets"][0]
    initial["phase_prior"] = read(runner.ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1/priors/VID103.json")
    base = build_gemini_base(CholecTrack20DatasetAdapter("D:/cholec_dataset", causal_window_size=3), selected)
    body = runner.repair_wire(base, selected, initial)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert "Do not change Phase" not in json.dumps(packet)
    assert packet["response_schema"] == joint_schema()
    assert Draft202012Validator(joint_schema()).is_valid(proposal())
    pool = compile_joint(initial["graph_r1"]["prediction"], initial["graph_r1"]["pool"], proposal())["pool"]
    for seat in SEATS:
        wire = runner.five_review_wire(seat, base, selected, pool)
        text = wire["messages"][0]["content"][0]["text"]
        assert "phase_6" in text and "phase_semantics" in text
        assert "academic" in text.lower()
        assert len([c for c in wire["messages"][0]["content"] if c["type"] == "image_url"]) == 3
