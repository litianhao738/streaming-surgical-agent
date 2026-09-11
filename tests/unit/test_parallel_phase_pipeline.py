"""Real concurrent scheduling and result isolation for the Phase side branch."""
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.parallel_phase import run_parallel_repair


def h0_prediction():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def graph_prediction():
    return {"instrument": [0, 2], "verb": [0, 2], "target": [0, 1], "ivt": [7, 60], "phase": [1]}


def phase_reviews(choices=(3, 3, 3, 1, None)):
    return {seat: {"phase_id": choice, "image_indices": [2] if choice is not None else [],
                   "observation": "Observed workflow at the target image."}
            for seat, choice in zip(SEATS, choices, strict=True)}


def test_callbacks_really_overlap_without_racing_against_a_wall_clock_threshold():
    rendezvous = Barrier(2)
    originals = h0_prediction()

    def interaction(current):
        assert current == originals
        rendezvous.wait(timeout=5)
        return {"prediction": graph_prediction(), "source": "new graph output"}

    def phase():
        rendezvous.wait(timeout=5)
        return {"raw_reviews": phase_reviews(), "source": "independent phase output"}

    result = run_parallel_repair(originals, interaction, phase)
    timing = result["timing"]
    assert max(timing["graph_start"], timing["phase_start"]) <= min(timing["graph_end"], timing["phase_end"])
    assert result["final"] == {**graph_prediction(), "phase": [3]}
    assert result["graph"]["source"] == "new graph output"
    assert result["phase"]["source"] == "independent phase output"


@pytest.mark.parametrize("first", ["graph", "phase"])
def test_merge_uses_new_graph_four_heads_regardless_of_branch_completion_order(first):
    rendezvous = Barrier(2)
    first_finished = Event()
    completion_order, order_lock = [], Lock()
    initial = h0_prediction()
    initial_snapshot = deepcopy(initial)

    def finish(name):
        rendezvous.wait(timeout=5)
        if name != first:
            assert first_finished.wait(timeout=5)
        with order_lock:
            completion_order.append(name)
        if name == first:
            first_finished.set()

    def interaction(current):
        # Mutation simulates an internal callback working on its own state.
        # Neither the user's H0 nor the phase callback may acquire this state.
        current["target"].append(1)
        finish("graph")
        return {"prediction": graph_prediction(), "extra": {"callback": "graph"}}

    def phase():
        finish("phase")
        return {"raw_reviews": phase_reviews(), "extra": {"callback": "phase"}}

    result = run_parallel_repair(initial, interaction, phase)
    assert completion_order == [first, "phase" if first == "graph" else "graph"]
    assert initial == initial_snapshot
    assert result["final"] == {**graph_prediction(), "phase": [3]}
    assert result["graph"]["prediction"] == graph_prediction()
    assert result["phase_decision"]["reason"] == "MAJORITY_PHASE_SWITCH"
    for task in initial:
        assert result["final"][task] is not result["graph"]["prediction"][task]


def test_phase_fallback_is_based_on_new_graph_prediction_not_cached_h0():
    updated = graph_prediction()
    updated["phase"] = [2]
    raw = phase_reviews()
    raw[SEATS[0]] = None
    result = run_parallel_repair(
        h0_prediction(), lambda current: {"prediction": deepcopy(updated)},
        lambda: {"raw_reviews": raw})
    assert result["final"] == updated
    assert result["phase_decision"]["reason"] == "INVALID_PANEL"
    assert result["phase_decision"]["before"] == 2
    assert result["phase_decision"]["after"] == 2


@pytest.mark.parametrize("choice", [None, {}, {"phase_id": 3, "image_indices": [0], "observation": "No target reference"}])
def test_phase_api_or_validation_failure_does_not_undo_graph_repair(choice):
    raw = phase_reviews()
    raw[SEATS[1]] = choice
    result = run_parallel_repair(
        h0_prediction(), lambda current: {"prediction": graph_prediction(), "status": "REVIEWED"},
        lambda: {"raw_reviews": raw, "status": "INVALID_PANEL"})
    assert result["final"] == graph_prediction()
    assert result["graph"]["status"] == "REVIEWED"
    assert result["phase_decision"]["reason"] == "INVALID_PANEL"


def test_expected_graph_api_failure_can_keep_h0_heads_while_phase_succeeds():
    h0 = h0_prediction()
    result = run_parallel_repair(
        h0, lambda current: {"prediction": current, "status": "PROPOSAL_FAILED"},
        lambda: {"raw_reviews": phase_reviews()})
    assert result["final"] == {**h0, "phase": [3]}
    assert result["graph"]["prediction"] == h0
    assert h0 == h0_prediction()


@pytest.mark.parametrize("choices", [(1, 1, 2, 2, None), (None, None, None, None, None)])
def test_abstention_or_no_majority_preserves_graph_phase_and_all_four_heads(choices):
    result = run_parallel_repair(
        h0_prediction(), lambda current: {"prediction": graph_prediction()},
        lambda: {"raw_reviews": phase_reviews(choices)})
    assert result["final"] == graph_prediction()
    assert result["phase_decision"]["reason"] == "NO_MAJORITY"


@pytest.mark.parametrize("failing_branch", ["graph", "phase"])
def test_unexpected_callback_exception_propagates_after_other_branch_finishes(failing_branch):
    rendezvous = Barrier(2)
    other_finished = Event()

    def interaction(current):
        rendezvous.wait(timeout=5)
        if failing_branch == "graph":
            raise RuntimeError("unexpected graph programming error")
        other_finished.set()
        return {"prediction": graph_prediction()}

    def phase():
        rendezvous.wait(timeout=5)
        if failing_branch == "phase":
            raise RuntimeError("unexpected phase programming error")
        other_finished.set()
        return {"raw_reviews": phase_reviews()}

    with pytest.raises(RuntimeError, match=f"unexpected {failing_branch} programming error"):
        run_parallel_repair(h0_prediction(), interaction, phase)
    assert other_finished.is_set()


def test_timing_reports_branch_intervals_and_join_without_summing_parallel_latency():
    result = run_parallel_repair(
        h0_prediction(), lambda current: {"prediction": graph_prediction()},
        lambda: {"raw_reviews": phase_reviews()})
    timing = result["timing"]
    assert set(timing) == {"graph_start", "graph_end", "graph_seconds", "phase_start", "phase_end",
                           "phase_seconds", "join_seconds", "merge_seconds", "parallel_seconds"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timing.values())
    for branch in ("graph", "phase"):
        assert timing[f"{branch}_end"] >= timing[f"{branch}_start"]
        assert timing[f"{branch}_seconds"] == pytest.approx(timing[f"{branch}_end"] - timing[f"{branch}_start"])
        assert timing["join_seconds"] >= timing[f"{branch}_end"]
    assert timing["parallel_seconds"] >= timing["join_seconds"]
    assert timing["parallel_seconds"] >= timing["merge_seconds"]


@pytest.fixture
def archived_graph_and_phase_case():
    """Read one local development archive; absent external datasets skip this check."""
    from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
    from scripts.run_prior_panel_trial import read
    from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

    root = Path(__file__).resolve().parents[2]
    graph_root = root / "artifacts/preflight/prior_candidate_eight_20260908_v1"
    phase_root = root / "artifacts/preflight/phase_extension_eight_20260909_v1"
    if not all((p / "plan.json").is_file() for p in (graph_root, phase_root)):
        pytest.skip("optional raw-response experiment archives are unavailable")
    key = "VID96_15051"
    graph_plan, phase_plan = read(graph_root / "plan.json"), read(phase_root / "plan.json")
    selected = next(r for r in graph_plan["selection"] if r["key"] == key)
    if not all(Path(i["path"]).is_file() for i in selected["images"]):
        pytest.skip("optional archived image paths are unavailable")
    graph_row = next(r for r in read(graph_root / "predictions.json")["targets"] if r["key"] == key)
    phase_row = next(r for r in read(phase_root / "predictions.json")["targets"] if r["key"] == key)
    adapter = CholecTrack20DatasetAdapter(Path(selected["images"][0]["path"]).parents[3], causal_window_size=3)
    return {"key": key, "graph_root": graph_root, "phase_root": phase_root, "selected": selected,
            "base": build_gemini_base(adapter, selected), "h0": graph_row["h0"],
            "prior": read(graph_root / "priors" / f"{selected['video_id']}.json"),
            "graph_expected": read(graph_root / "targets" / key / "prior_graph.json"),
            "hints_expected": read(graph_root / "targets" / key / "hints.json"),
            "phase_selected": phase_plan["phase_inputs"][key]["phase_short"],
            "phase_expected": read(phase_root / "targets" / key / "phase_short.json"),
            "final_expected": phase_row["phase_short"]}


class MappedArchiveCalls:
    """Map stage names only; ReplayCalls verifies each original wire and raw reply."""

    def __init__(self, root, key, stages):
        from scripts.run_prior_panel_trial import read
        from scripts.score_five_head_repair_trial import ReplayCalls

        calls = [c for c in read(root / "budget.json")["calls"]
                 if c["target"] == key and c["stage"] in stages.values()]
        self.replay = ReplayCalls(root, calls)
        self.stages, self.rows = stages, []

    def call(self, target, stage, seat, body):
        old_stage = self.stages[stage]
        result = self.replay.call(target, old_stage, seat, body)
        original = next(c for c in self.replay.available
                        if (c["target"], c["stage"], c["seat"]) == (target, old_stage, seat))
        self.rows.append({**original, "stage": stage})
        return result


def test_real_graph_archive_replays_from_original_h0_prior_and_initial_pool(archived_graph_and_phase_case):
    from scripts.run_parallel_phase_trial import STAGES, run_graph

    case = archived_graph_and_phase_case
    expected = case["graph_expected"]
    # This target reused the control panel because the original request bodies
    # were identical. Replay the actual paid requests, not a fictional graph call.
    review_arm = expected.get("shared_from", "prior_graph")
    proposal_arm = expected.get("proposal_shared_from") or "prior_graph"
    calls = MappedArchiveCalls(case["graph_root"], case["key"],
        {STAGES["proposal"]: f"{proposal_arm}_proposal", STAGES["review"]: f"{review_arm}_review"})
    snapshot = deepcopy((case["h0"], case["prior"]))

    actual = run_graph(calls, case["base"], case["selected"], case["h0"], case["prior"])

    assert actual["hints"] == case["hints_expected"]
    assert actual["proposal"] == expected["proposal"]
    assert actual["pool"] == expected["pool"]
    assert actual["raw_reviews"] == expected["raw"]
    for field in ("prediction", "means", "diagnostics", "reviews", "format_diagnostics", "issues", "status"):
        assert actual[field] == expected[field]
    assert actual["request_fingerprints"]["proposal"] == expected["proposal_fingerprint"]
    assert actual["request_fingerprints"]["reviews"] == expected["request_fingerprints"]
    assert actual["prediction"]["phase"] == case["h0"]["phase"]
    assert len(calls.rows) == len(calls.replay.rows) == 6
    assert {r["stage"] for r in calls.rows} == {STAGES["proposal"], STAGES["review"]}
    assert (case["h0"], case["prior"]) == snapshot


def test_real_phase_archive_replays_blind_requests_and_merges_only_after_graph(archived_graph_and_phase_case):
    from scripts.run_parallel_phase_trial import STAGES, collect_phase
    from scripts.run_phase_extension_trial import phase_wire

    case = archived_graph_and_phase_case
    calls = MappedArchiveCalls(case["phase_root"], case["key"], {STAGES["phase"]: "phase_short"})
    expected = case["phase_expected"]

    result = run_parallel_repair(case["h0"],
        lambda current: {"prediction": deepcopy(case["graph_expected"]["prediction"])},
        lambda: collect_phase(calls, case["phase_selected"]))

    assert result["phase"]["raw_reviews"] == expected["raw_reviews"]
    assert result["phase"]["errors"] == expected["errors"]
    assert result["phase"]["request_fingerprints"] == expected["request_fingerprints"]
    assert result["phase_decision"] == expected["decision"]
    assert result["final"] == case["final_expected"]
    assert len(calls.rows) == len(calls.replay.rows) == 5
    assert result["final"]["phase"] != case["h0"]["phase"]
    for task in ("instrument", "verb", "target", "ivt"):
        assert result["final"][task] == case["graph_expected"]["prediction"][task]
    import json
    packet = json.loads(phase_wire(SEATS[0], case["phase_selected"])["messages"][0]["content"][0]["text"])
    assert not {"h0", "current_prediction", "candidate_pool", "candidate_relation_hints", "review_feedback"} & packet.keys()
