import json
from copy import deepcopy

import pytest

from surgical_agent.research.verification import split_review as split
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.semantic_coordinator import review_schema


@pytest.fixture
def state():
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]}
    return h0, make_pool(h0)


def replies(pool, rating):
    return {s: {"judgments": {p["id"]: {"rating": rating, "finding": "MATCH" if rating >= 4 else "UNCLEAR",
        "scope": "LOCAL_REGION" if rating >= 4 else "UNCERTAIN", "image_indices": [2],
        "observation": "fixture"} for p in pool["propositions"]}} for s in SEATS}


def test_split_schema_and_images_preserve_disjoint_exact_coverage(state):
    _, pool = state
    schema = review_schema(pool, 3)
    packet = {"academic_context": "academic", "task": "audit", "propositions": pool["propositions"], "response_schema": schema}
    body = {"messages": [{"content": [{"text": json.dumps(packet)}, {"image_url": {"url": "fixture"}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    before, groups = deepcopy(body), []
    for group in split.GROUPS:
        out = split.split_wire(body, group)
        data = json.loads(out["messages"][0]["content"][0]["text"])
        ids = {p["id"] for p in data["propositions"]}
        assert ids == set(data["response_schema"]["properties"]["judgments"]["required"])
        assert out["response_format"]["json_schema"]["schema"] == data["response_schema"]
        assert out["messages"][0]["content"][1:] == before["messages"][0]["content"][1:]
        groups.append(ids)
    assert not groups[0] & groups[1]
    assert groups[0] | groups[1] == {p["id"] for p in pool["propositions"]}
    assert body == before


def test_missing_relation_reply_does_not_invalidate_component_votes(state):
    h0, pool = state
    raw = {g: replies(split.subpool(pool, g), 5) for g in split.GROUPS}
    raw["relations"][SEATS[0]] = None
    result = split.combine(h0, pool, raw)
    assert result["means"]["ivt_0"] is None
    assert result["means"]["instrument_0"] == 5
    assert result["prediction"]["ivt"] == h0["ivt"]
    # Other valid component opinions still take effect, including components
    # inserted into the pool from the ontology for an existing IVT.
    for task in ("instrument", "verb", "target"):
        assert result["prediction"][task] == sorted(p["label_id"] for p in pool["propositions"] if p["task"] == task)


def test_relation_support_cannot_bypass_component_admission(state):
    h0, pool = state
    without_relation = {**h0, "ivt": []}
    raw = {"components": replies(split.subpool(pool, "components"), 3),
           "relations": replies(split.subpool(pool, "relations"), 5)}
    assert split.combine(without_relation, pool, raw)["prediction"]["ivt"] == []
