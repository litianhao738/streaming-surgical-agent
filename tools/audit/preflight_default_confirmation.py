"""Exercise fresh H0/shared Phase branches with mock replies and no GT/network."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_default_improvement_trial import (
    GRAPH,
    INPUTS,
    TASKS,
    InferenceOnlyAdapter,
    build_gemini_base,
    read,
    roi,
    run_case,
    tracker_rows,
)
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter


class MockCalls:
    def __init__(self):
        self.rows = []

    def call(self, target, stage, seat, body):
        assert (target, stage, seat) not in [(r["target"], r["stage"], r["seat"]) for r in self.rows]
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        if stage == "h0":
            return {"schema_version": "joint_perception_final_only_v1",
                "instrument": {"selected_ids": [0]}, "verb": {"selected_ids": [1]},
                "target": {"selected_ids": [0]}, "ivt": {"selected_ids": [17]}, "phase": {"selected_id": 1}}
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        if stage == "phase":
            assert "current_prediction" not in packet and "propositions" not in packet
            return {"phase_id": 2, "image_indices": [2], "observation": "Mock stage observation."}
        if stage.endswith("_proposal"):
            return {t: [] for t in TASKS[:4]}
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [], "observation": "Mock unclear evidence."}
        if seat == "gemini":
            return {"rows": [{"candidate_id": p["id"], **item} for p in packet["propositions"]]}
        return {"judgments": {p["id"]: dict(item) for p in packet["propositions"]}}


def main():
    selected = read(INPUTS / "selection.json")["selection"][0]
    adapter = InferenceOnlyAdapter(CholecTrack20DatasetAdapter("D:/cholec_dataset", causal_window_size=3))
    tracks, _ = tracker_rows([selected])
    base = build_gemini_base(adapter, selected)
    checks = []
    with tempfile.TemporaryDirectory(prefix="default-confirmation-") as directory:
        views = roi.create_views(selected, tracks[selected["key"]]["tracks"], Path(directory) / "views")
        for candidate in ("roi_current", "roi_temporal", "verifier_current", "selector_only"):
            arms = ["control"] if candidate == "selector_only" else ["control", candidate]
            plan = {"arms": arms, "versions": ["h0", *arms, "selector_only"]}
            sample = {**selected, "arm_order": [a for a in arms if a != "verifier_current"]}
            calls = MockCalls()
            record = run_case(calls, base, sample, {"h0": None, "phase_raw": None, "views": views},
                              read(GRAPH / "priors" / f"{selected['video_id']}.json"), plan)
            for arm in plan["versions"]:
                assert record["predictions"][arm]["phase"] == ([1] if arm == "h0" else [2])
                assert all(record["predictions"][arm][t] == record["h0"][t] for t in TASKS[:4])
            assert sum(c["stage"] == "phase" for c in calls.rows) == 5
            assert sum(c["stage"] == "h0" for c in calls.rows) == 1
            checks.append({"candidate": candidate, "mock_calls": len(calls.rows), "shared_phase": True})
    print(json.dumps({"checks": checks, "actual_api_calls": 0, "query_gt_reads": 0}))


if __name__ == "__main__":
    main()
