"""Synthetic, no-paid-calls archive replay using real action-cue/v1.1 wires.

This is an engineering preflight. All H0, proposals, reviews and Phase choices
are deliberately fabricated, and cannot be used as model-quality evidence.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from threading import RLock
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_action_cue_candidate_trial as trial
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_prior_panel_trial import read, save
from scripts.score_five_head_repair_trial import ReplayCalls
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

OUTPUT = ROOT / "artifacts/research/action_cue_offline_replay_20260909"
SOURCE = ROOT / "artifacts/preflight/default_compact_repair_ready_20260909_v1/plan.json"
PRIOR = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1/priors/VID103.json"
H0 = {"schema_version": "joint_perception_final_only_v1", "instrument": {"selected_ids": [2]},
      "verb": {"selected_ids": [2]}, "target": {"selected_ids": [0]},
      "ivt": {"selected_ids": [60]}, "phase": {"selected_id": 3}}


class SyntheticCalls:
    """Write the same request/response/record envelope consumed by ReplayCalls."""

    def __init__(self, output, *, different_pool, failed_seat):
        self.output, self.different_pool, self.failed_seat = output, different_pool, failed_seat
        self.rows, self.lock = [], RLock()

    def call(self, target, stage, seat, body):
        if stage == "h0":
            parsed = deepcopy(H0)
        elif stage == "shared_phase":
            parsed = {"phase_id": 1, "image_indices": [2],
                      "observation": "Synthetic current-frame phase evidence for engineering replay only."}
        elif stage.endswith("_proposal"):
            parsed = {t: [] for t in trial.TASKS[:-1]}
            if self.different_pool and stage.startswith("action_cue"):
                parsed["verb"] = [1]
        elif stage.endswith("_review"):
            if seat == self.failed_seat:
                parsed = None
            else:
                packet = json.loads(body["messages"][0]["content"][0]["text"])
                item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [2],
                        "observation": "Synthetic uncertain evidence for engineering replay only."}
                parsed = ({"rows": [{"candidate_id": p["id"], **item} for p in packet["propositions"]]}
                          if seat == "gemini" else
                          {"judgments": {p["id"]: deepcopy(item) for p in packet["propositions"]}})
        else:
            raise AssertionError("unexpected synthetic stage")
        with self.lock:
            row = {"index": len(self.rows), "target": target, "stage": stage, "seat": seat,
                   "model": body["model"], "status": "JSON_PARSED" if parsed is not None else "FAILED",
                   "http_status": 200 if parsed is not None else 503,
                   "synthetic": True, "no_paid_calls": True, "charge": "0", "charge_kind": "synthetic_zero"}
            folder = self.output / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
            response = {"http_status": row["http_status"], "synthetic": True, "no_paid_calls": True,
                        "body": {"model": body["model"], "synthetic": True,
                                 "choices": [{"finish_reason": "stop", "message": {
                                     "content": json.dumps(parsed, ensure_ascii=False)}}]}
                        if parsed is not None else {"error": "SYNTHETIC_FAILED_SEAT"}}
            save(folder / "request.json", redact_images(body))
            save(folder / "response.json", response)
            save(folder / "record.json", row)
            self.rows.append(row)
        return parsed


def run_case(name, base, selected, prior, *, different_pool, failed_seat):
    output = OUTPUT / name
    calls = SyntheticCalls(output, different_pool=different_pool, failed_seat=failed_seat)
    row, records, shared = trial.run_target(calls, base, selected, prior)
    expected = 18 if different_pool else 13
    if len(calls.rows) != expected:
        raise AssertionError("unexpected synthetic call count")
    if bool(records["action_cue"].get("shared_from")) == different_pool:
        raise AssertionError("complete-panel sharing differs from the planned case")
    if failed_seat is not None:
        for arm in trial.ARMS:
            if records[arm]["raw"][failed_seat] is not None:
                raise AssertionError("the failed reviewer was not preserved in both arms")
    for arm in trial.ARMS:
        if "rows" not in records[arm]["raw"]["gemini"]:
            raise AssertionError("synthetic Gemini must exercise the rows output contract")
    save(output / "synthetic_budget.json", {"synthetic": True, "no_paid_calls": True,
                                           "calls": calls.rows, "stopped": True})
    save(output / "synthetic_result.json", {"synthetic": True, "no_paid_calls": True,
                                           "row": row, "records": records, "shared": shared})
    # Roundtrip all stored JSON, including sorted archive mappings and the prior.
    saved = read(output / "synthetic_result.json")
    replay = ReplayCalls(output, read(output / "synthetic_budget.json")["calls"])
    rebuilt_row, rebuilt_records, rebuilt_shared = trial.run_target(replay, base, deepcopy(selected), deepcopy(prior))
    trial.same(rebuilt_row, saved["row"], "synthetic final predictions did not replay")
    trial.same(trial.without_timing(rebuilt_records), trial.without_timing(saved["records"]),
               "synthetic graph records did not replay")
    trial.same(trial.without_timing(rebuilt_shared), trial.without_timing(saved["shared"]),
               "synthetic shared H0/Phase records did not replay")
    trial.same(len(replay.rows), len(calls.rows), "not every synthetic response was consumed")
    return {"case": name, "synthetic": True, "no_paid_calls": True, "real_http_calls": 0,
            "synthetic_calls": expected, "all_responses_consumed": True, "exact_replay_without_timing": True,
            "different_pool": different_pool, "failed_seat": failed_seat,
            "shared_from": records["action_cue"].get("shared_from"), "gemini_rows_verified": True,
            "shared_phase_equal": row["control"]["phase"] == row["action_cue"]["phase"],
            "proposal_token_proxy": shared["proposal_token_proxy"]}


def main():
    plan = read(SOURCE)
    selected = deepcopy(plan["phase_inputs"][plan["selection"][0]["key"]])
    selected["arm_order"] = list(trial.ARMS)
    prior = read(PRIOR)
    adapter = trial.InferenceOnlyAdapter(CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3))
    with patch("requests.post", side_effect=AssertionError("HTTP POST forbidden in synthetic offline replay")) as post:
        base = trial.build_gemini_base(adapter, selected)
        cases = [run_case("same_pool_failed_seat", base, selected, prior, different_pool=False, failed_seat="gpt"),
                 run_case("different_pool_complete_panels", base, selected, prior, different_pool=True, failed_seat=None)]
        post.assert_not_called()
    result = {"synthetic": True, "no_paid_calls": True, "real_http_calls": 0,
              "source_plan": str(SOURCE), "selected_key": selected["key"],
              "gt_labels_read": False, "purpose": "engineering archive compatibility only, not model accuracy",
              "passed": True, "cases": cases}
    save(OUTPUT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
