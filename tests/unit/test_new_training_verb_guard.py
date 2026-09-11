"""Offline checks for the isolated shared-answer novel-Verb comparison."""
from copy import deepcopy

import pytest

from scripts import run_new_training_verb_guard_trial as trial
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.verb_guard import select_with_verb_guard


def case(scores=(3, 5, 5, 5, 5)):
    h0 = {"instrument": [0], "verb": [2], "target": [0], "ivt": [1], "phase": [1]}
    pool = make_pool(h0, {"instrument": [], "verb": [0], "target": [], "ivt": [7]})
    means = {p["id"]: 5.0 for p in pool["propositions"]}
    diagnostics = {pid: {"scores": [5] * 5, "invalid": {}, "explicit_conflict": False} for pid in means}
    means["verb_0"] = sum(scores) / 5
    diagnostics["verb_0"]["scores"] = list(scores)
    return h0, pool, means, diagnostics


def comparison(record, expected, mask=None):
    row = {"key": "VID23_101", "video_id": "VID23", "frame_id": 101,
           **{v: record[v] for v in trial.VERSIONS}}
    truth = {"video_id": "VID23", "frame_id": 101, "gt": expected,
             "mask": mask or dict.fromkeys(trial.TASKS, True)}
    metrics = {v: compute_repair_comparison([{**truth, "h0": row["h0"], "h1": None,
                "final": row[v]}])["arms"]["final"] for v in trial.VERSIONS}
    return trial.diagnostic_summary([row], {("VID23", 101): truth}, {row["key"]: record}, metrics)


def guarded_record(scores=(3, 5, 5, 5, 5)):
    h0, pool, means, diagnostics = case(scores)
    guard = select_with_verb_guard(h0, pool, means, diagnostics)
    return {"h0": h0, "original": panel.select(h0, pool, means), "verb_guard": guard["prediction"],
            "guard": guard, "guard_error": None, "graph": {"pool": pool}}


def test_blocks_false_new_verb_and_dependent_ivt_with_a_real_beneficial_change():
    h0, pool, means, diagnostics = case()
    snapshot = deepcopy((h0, pool, means, diagnostics))
    record = guarded_record()
    assert record["original"]["verb"] == [0, 2]
    assert record["original"]["ivt"] == [1, 7]
    assert record["verb_guard"] == h0
    assert record["verb_guard"]["verb"] is not h0["verb"]
    guard = select_with_verb_guard(h0, pool, means, diagnostics)
    assert guard["guarded_means"]["verb_0"] is None
    assert guard["guarded_means"]["ivt_7"] == means["ivt_7"]
    assert (h0, pool, means, diagnostics) == snapshot
    result = comparison(record, h0)
    assert result["blocked_actual_new_labels"]["verb"]["false_additions_blocked"] == 1
    assert result["blocked_actual_new_labels"]["ivt"]["false_additions_blocked"] == 1
    assert result["predeclared_success"]["confirmed"] is True
    assert result["predeclared_success"]["beneficial_successful_guard_label_changes"] == 2


def test_correct_additions_can_be_blocked_and_must_be_reported_as_harm():
    record = guarded_record()
    result = comparison(record, record["original"])
    assert result["blocked_actual_new_labels"]["verb"]["true_additions_blocked"] == 1
    assert result["blocked_actual_new_labels"]["ivt"]["true_additions_blocked"] == 1
    assert result["predeclared_success"]["confirmed"] is False
    assert result["predeclared_success"]["beneficial_successful_guard_label_changes"] == 0


def test_all_five_support_admits_new_labels_without_rechecking_existing_ivt():
    h0, pool, means, diagnostics = case((4, 4, 5, 5, 5))
    # The new condition never rechecks existing Verb/IVT ratings or overwrites Phase.
    means["verb_2"] = None
    diagnostics["verb_2"] = {"scores": [None] * 5, "invalid": {"grok": "missing"}}
    guard = select_with_verb_guard(h0, pool, means, diagnostics)
    assert guard["prediction"] == panel.select(h0, pool, means)
    assert guard["prediction"]["verb"] == [0, 2]
    assert guard["prediction"]["ivt"] == [1, 7]
    assert guard["prediction"]["phase"] == h0["phase"]
    assert guard["guarded_means"]["verb_2"] is None
    assert [d["candidate_id"] for d in guard["new_verb_decisions"]] == ["verb_0"]


@pytest.mark.parametrize("bad_scores,invalid", [
    ([4, 5, 5, 5], {}), ([True, 5, 5, 5, 5], {}), ([4.0, 5, 5, 5, 5], {}),
    ([None, 5, 5, 5, 5], {"grok": "invalid evidence"}),
    ([4, 5, 5, 5, 5], {"grok": "invalid evidence"}),
])
def test_invalid_or_incomplete_votes_never_pass_new_verb(bad_scores, invalid):
    h0, pool, means, diagnostics = case((4, 5, 5, 5, 5))
    diagnostics["verb_0"].update(scores=bad_scores, invalid=invalid)
    result = select_with_verb_guard(h0, pool, means, diagnostics)
    assert result["prediction"] == h0
    assert result["new_verb_decisions"][0]["reason"] == "INVALID_OR_INCOMPLETE_EVIDENCE"


def test_guard_exception_keeps_h0_but_cannot_count_as_a_guard_benefit(monkeypatch):
    record = guarded_record()
    h0 = record["h0"]
    raw = {"schema_version": FINAL_ONLY_SCHEMA_VERSION,
           **{t: {"selected_id": h0[t][0]} if t == "phase" else {"selected_ids": h0[t]}
              for t in trial.TASKS}}
    _, pool, means, diagnostics = case()
    graph = {"prediction": record["original"], "status": "UNRESOLVED", "pool": pool,
             "means": means, "diagnostics": diagnostics}

    class CachedCalls:
        def call(self, *args):
            return deepcopy(raw)

    def selection_failure(*args):
        raise ValueError("synthetic output-cap failure")

    monkeypatch.setattr(trial, "gemini_h0_wire", lambda base: {})
    monkeypatch.setattr(trial, "run_graph", lambda *args: deepcopy(graph))
    monkeypatch.setattr(trial, "select_with_verb_guard", selection_failure)
    rebuilt = trial.run_target(CachedCalls(), None, {"key": "VID23_101"}, {})
    assert rebuilt["verb_guard"] == h0
    assert rebuilt["original"] == record["original"]
    assert rebuilt["status"] == "UNRESOLVED"
    assert rebuilt["guard_status"] == "SELECTION_FAILED"
    result = comparison(rebuilt, h0)
    assert result["guard_selection_failures"] == 1
    assert result["blocked_actual_new_labels"]["verb"]["false_additions_blocked"] == 0
    assert result["blocked_actual_new_labels"]["ivt"]["false_additions_blocked"] == 0
    assert result["predeclared_success"]["confirmed"] is False
    assert result["predeclared_success"]["beneficial_successful_guard_label_changes"] == 0


def test_masked_gt_cannot_contribute_to_blocked_label_or_success_counts():
    record = guarded_record()
    mask = dict.fromkeys(trial.TASKS, True)
    mask.update(verb=False, ivt=False)
    result = comparison(record, record["h0"], mask)
    assert result["blocked_actual_new_labels"]["verb"]["valid_targets"] == 0
    assert result["blocked_actual_new_labels"]["ivt"]["valid_targets"] == 0
    assert result["predeclared_success"]["confirmed"] is False
    assert result["predeclared_success"]["beneficial_successful_guard_label_changes"] == 0


def test_existing_execution_lock_prevents_any_second_dispatch(tmp_path, monkeypatch):
    (tmp_path / "execution.lock").write_text("closed attempt", encoding="utf-8")
    monkeypatch.setattr(trial, "verify_plan", lambda output: {})

    def forbidden_calls(*args):
        pytest.fail("a second execution reached the API transport")

    monkeypatch.setattr(trial, "BoundCalls", forbidden_calls)
    with pytest.raises(ValueError, match="single-use"):
        trial.execute(tmp_path, None)
    assert (tmp_path / "execution.lock").read_text(encoding="utf-8") == "closed attempt"


def test_failed_audit_prevents_scoring_from_reading_query_gt(tmp_path, monkeypatch):
    def failed_audit(*args):
        raise ValueError("source changed after freeze")

    def forbidden_gt(*args):
        pytest.fail("GT was read before immutable inference audit completed")

    monkeypatch.setattr(trial, "audit", failed_audit)
    monkeypatch.setattr(trial, "score_saved", forbidden_gt)
    with pytest.raises(ValueError, match="source changed"):
        trial.score(tmp_path, None)
