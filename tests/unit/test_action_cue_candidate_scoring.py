"""Synthetic-only checks: no current experiment labels, credentials or API calls."""
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import score_action_cue_candidate_trial as scorer


def labels(verbs=(), *, phase=(1,)):
    return {"instrument": [0], "verb": list(verbs), "target": [0], "ivt": [0], "phase": list(phase)}


def fixture(count=1):
    rows, truth, records, calls = [], [], {}, []
    for i in range(count):
        key = f"SYNTHETIC_{i}"
        row = {"key": key, "video_id": "SYNTHETIC", "frame_id": i,
               "h0": labels([0]), "control": labels([0]), "action_cue": labels([0, 1]),
               "statuses": {a: "OK" for a in scorer.ARMS}}
        rows.append(row)
        truth.append({"video_id": "SYNTHETIC", "frame_id": i, "gt": labels([0, 1]),
                      "mask": {t: True for t in scorer.TASKS}})
        records[key] = {}
        for arm in scorer.ARMS:
            prediction = row[arm]
            pool = {"propositions": [{"id": f"{task}_{label}", "task": task, "label_id": label}
                                     for task in scorer.TASKS[:-1] for label in prediction[task]]}
            reviews = {seat: {"judgments": {p["id"]: {"rating": 4, "finding": "MATCH", "scope": "LOCAL_REGION",
                "image_indices": [2], "observation": "Synthetic visual evidence."} for p in pool["propositions"]}}
                for seat in scorer.SEATS}
            records[key][arm] = {"status": "OK", "pool": pool, "reviews": reviews,
                "means": {p["id"]: 4 for p in pool["propositions"]}, "proposal_seconds": 1.0, "panel_seconds": 3.0}
        for stage, tokens in (("h0", 100), ("control_proposal", 80), ("action_cue_proposal", 79)):
            calls.append({"target": key, "stage": stage, "seat": "base", "account": "openrouter_usd",
                          "charge": "0.01", "charge_kind": "native", "status": "JSON_PARSED",
                          "usage": {"prompt_tokens": tokens, "completion_tokens": 10}, "elapsed_seconds": 1.0})
    plan = {"profile": "synthetic_action_cue", "arms": list(scorer.ARMS),
            "selection": [{k: r[k] for k in ("key", "video_id", "frame_id")} for r in rows]}
    ledger = {"calls": calls, "stopped": True}
    done = {"closed_utc": "synthetic_closed", "fatal_error": None, "post_calls": len(calls), "elapsed_seconds": 5.0}
    return plan, rows, records, truth, ledger, done


def test_predeclared_success_requires_final_benefit_and_native_tokens():
    report, deltas, coverage = scorer.summarize(*fixture(24))
    assert report["predeclared_success"]["confirmed"] is True
    assert report["metrics"]["control"]["tasks"]["verb"]["tp"] == 24
    assert report["metrics"]["action_cue"]["tasks"]["verb"]["tp"] == 48
    assert coverage["arms"]["control"]["verb"]["pool_recall"] == 0.5
    assert coverage["arms"]["action_cue"]["verb"]["pool_recall"] == 1.0
    assert report["comparisons"]["control_to_action_cue"]["fixed_label_errors"] == 24
    assert len(deltas["control_to_action_cue"]) == 24


def test_failed_h0_empty_fallback_is_never_exact_even_on_empty_gt():
    data = fixture()
    row, truth = data[1][0], data[3][0]
    empty = {t: [] for t in scorer.TASKS}
    row.update(h0=None, control=deepcopy(empty), action_cue=deepcopy(empty))
    truth["gt"] = empty
    for arm in scorer.ARMS:
        data[2][row["key"]][arm] = {"status": "H0_FAILED", "pool": {"propositions": []}}
    report, _, coverage = scorer.summarize(*data)
    for version in scorer.VERSIONS:
        task = report["metrics"][version]["tasks"]["verb"]
        assert task["failed_predictions"] == 1
        assert task["exact_matches"] == 0
        assert task["valid_targets"] == 1
        assert report["metrics"][version]["all_valid_heads_exact"]["exact_matches"] == 0
    assert coverage["arms"]["control"]["verb"]["failed_h0_targets"] == 1


def test_fn_inside_outside_pool_and_per_verb_counts_are_distinct():
    data = fixture()
    data[3][0]["gt"]["verb"] = [0, 1, 2]
    data[1][0]["action_cue"]["verb"] = [0, 3]
    report, _, coverage = scorer.summarize(*data)
    counts = coverage["arms"]["action_cue"]["verb"]
    assert (counts["fn_inside_pool"], counts["fn_outside_pool"], counts["selected_false"]) == (1, 1, 1)
    per_verb = report["per_verb"]["action_cue"]
    assert per_verb["1"]["fn_inside_pool"] == 1
    assert per_verb["2"]["fn_outside_pool"] == 1
    assert per_verb["3"]["fp"] == 1
    assert sum(v["fn"] for v in per_verb.values()) == report["metrics"]["action_cue"]["tasks"]["verb"]["fn"]


def test_candidate_coverage_alone_cannot_pass():
    data = fixture(24)
    for row in data[1]:
        row["action_cue"] = deepcopy(row["control"])
    report, _, _ = scorer.summarize(*data)
    checks = report["predeclared_success"]["checks"]
    assert checks["verb_candidate_recall_strictly_increases"] is True
    assert checks["verb_f1_strictly_increases"] is False
    assert report["predeclared_success"]["confirmed"] is False


@pytest.mark.parametrize("new_value", [None, True, -1, 81])
def test_missing_invalid_or_increased_native_tokens_prevent_success(new_value):
    data = fixture(24)
    call = next(c for c in data[4]["calls"] if c["stage"] == "action_cue_proposal")
    call["usage"]["prompt_tokens"] = new_value
    report, _, _ = scorer.summarize(*data)
    assert report["predeclared_success"]["confirmed"] is False
    token_counts = report["runtime"]["proposal_native_prompt_tokens"]
    assert token_counts["verified_pairs"] == (24 if new_value == 81 else 23)
    assert token_counts["increased_pairs"] == (1 if new_value == 81 else 0)


def test_shared_panel_is_not_an_independent_latency_observation():
    data = fixture()
    data[2][data[1][0]["key"]]["action_cue"]["shared_from"] = "control"
    costs = scorer.accounting(data[1], data[2], data[4], data[5])
    assert costs["timing"]["action_cue"]["fresh_panel_seconds"]["n"] == 0
    assert costs["timing"]["action_cue"]["shared_panels_excluded"] == 1
    assert costs["dispatched"]["all"]["charges_by_account_and_kind"]["openrouter_usd"] == {"native": "0.03"}


def test_charges_preserve_currency_provenance_and_missing_token_usage():
    data = fixture()
    extra = deepcopy(data[4]["calls"][0])
    extra.update(stage="shared_phase", account="aliyun_cny", charge="0.004", charge_kind="conservative_estimate", usage={})
    data[4]["calls"].append(extra)
    costs = scorer.accounting(data[1], data[2], data[4], data[5])["dispatched"]["all"]
    assert costs["calls"] == 4
    assert costs["calls_with_prompt_tokens"] == 3
    assert costs["charges_by_account_and_kind"] == {
        "openrouter_usd": {"native": "0.03"}, "aliyun_cny": {"conservative_estimate": "0.004"}}


def test_invalid_evidence_is_counted_per_seat_and_candidate():
    data = fixture()
    record = data[2][data[1][0]["key"]]["action_cue"]
    record["reviews"][scorer.SEATS[0]]["judgments"]["verb_1"]["scope"] = "UNCERTAIN"
    record["means"]["verb_1"] = None
    result = scorer.review_validity(data[1], data[2])["action_cue"]
    assert result["seat_item_counts"][scorer.SEATS[0]]["UNCERTAIN_SUPPORT"] == 1
    assert result["seat_response_counts"][scorer.SEATS[0]]["INVALID"] == 1
    assert result["unavailable_candidate_means"] == 1


def test_shared_phase_change_is_rejected():
    data = fixture()
    data[1][0]["action_cue"]["phase"] = [2]
    with pytest.raises(ValueError, match="shared Phase"):
        scorer.summarize(*data)


def test_invalid_snapshot_cannot_trigger_gt_read(tmp_path, monkeypatch):
    def audit(*args):
        raise ValueError("frozen snapshot differs")
    monkeypatch.setitem(sys.modules, "scripts.run_action_cue_candidate_trial", SimpleNamespace(audit=audit))
    def gt_forbidden(*args):
        pytest.fail("query GT read before successful audit")
    monkeypatch.setattr(scorer, "score_saved", gt_forbidden)
    with pytest.raises(ValueError, match="snapshot"):
        scorer.score(tmp_path, object())


def test_scoring_audits_before_gt_and_again_before_writing(tmp_path, monkeypatch):
    plan, rows, records, truth, ledger, done = fixture()
    events = []
    def audit(*args):
        events.append("audit")
        return plan, rows, records, ledger, done
    def load_gt(*args):
        events.append("gt")
        return {}, truth
    monkeypatch.setitem(sys.modules, "scripts.run_action_cue_candidate_trial", SimpleNamespace(audit=audit))
    monkeypatch.setattr(scorer, "score_saved", load_gt)
    scorer.score(tmp_path, object())
    assert events == ["audit", "gt", "audit"]
    assert {p.name for p in tmp_path.glob("*.json")} == {
        "metrics.json", "summary.json", "scored_truth.json", "frame_deltas.json", "candidate_coverage.json"}


def test_selection_failure_or_unattempted_target_prevents_success():
    data = fixture(24)
    key = data[1][0]["key"]
    data[2][key]["action_cue"]["status"] = "SELECTION_FAILED"
    data[4]["calls"] = [c for c in data[4]["calls"] if not (c["target"] == key and c["stage"] == "h0")]
    report, _, _ = scorer.summarize(*data)
    assert report["predeclared_success"]["checks"]["no_selector_failure"] is False
    assert report["predeclared_success"]["checks"]["all_24_planned_targets_retained_and_attempted"] is False
