from copy import deepcopy

import pytest

from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.reflective_repair import accept_reflection


def fixture():
    h0 = {"instrument": [2], "verb": [2], "target": [1], "ivt": [59], "phase": [3]}
    pool = make_pool(h0, {"instrument": [], "verb": [], "target": [], "ivt": [60]})
    prediction = {"instrument": [2], "verb": [2], "target": [0], "ivt": [60]}
    changes = [{"candidate_id": pid, "operation": op, "rating": 4 if op == "ADD" else 1,
                "finding": "MATCH" if op == "ADD" else "REFUTED",
                "scope": "LOCAL_REGION" if op == "ADD" else "WHOLE_FRAME",
                "image_indices": [2], "observation": "Observed contact relation across the current frame."}
               for pid, op in [("target_0", "ADD"), ("target_1", "REMOVE"), ("ivt_59", "REMOVE"), ("ivt_60", "ADD")]]
    return h0, pool, {"prediction": prediction, "changes": changes}


def test_coherent_candidate_bound_patch_keeps_phase_and_h0():
    h0, pool, raw = fixture()
    original = deepcopy(h0)
    final = accept_reflection(h0, pool, raw)
    assert final == {**raw["prediction"], "phase": [3]}
    assert h0 == original


@pytest.mark.parametrize("kind", ["missing", "duplicate", "local_delete", "wrong_direction", "unknown"])
def test_invalid_diff_cannot_change_prediction(kind):
    h0, pool, raw = fixture()
    if kind == "missing":
        raw["changes"].pop()
    elif kind == "duplicate":
        raw["changes"].append(raw["changes"][0])
    elif kind == "local_delete":
        raw["changes"][1]["scope"] = "LOCAL_REGION"
    elif kind == "wrong_direction":
        raw["changes"][0]["operation"] = "REMOVE"
    else:
        raw["prediction"]["target"] = [8]
    with pytest.raises(ValueError):
        accept_reflection(h0, pool, raw)


def test_change_cannot_remove_component_of_retained_ivt():
    h0, pool, raw = fixture()
    raw["prediction"]["ivt"] = [59, 60]
    raw["changes"] = [c for c in raw["changes"] if c["candidate_id"] != "ivt_59"]
    with pytest.raises(ValueError, match="removed component"):
        accept_reflection(h0, pool, raw)
