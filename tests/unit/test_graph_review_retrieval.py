"""Binding, budget and provenance checks for the opt-in local retrieval layer."""

import ast
import hashlib
import json
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import pytest

from surgical_agent.research.retrieval.graph_review import (
    RetrievalPolicy,
    knowledge_manifest,
    retrieve,
)
from surgical_agent.research.retrieval.review_corpus import CORPUS
from surgical_agent.research.verification.candidate_coordinator import make_pool

ROOT = Path(__file__).resolve().parents[2]


def case():
    current = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    return make_pool(current, {"instrument": [], "verb": [], "target": [8], "ivt": [19]})


def serialized(value):
    return json.dumps(value, ensure_ascii=False)


def test_sources_are_existing_text_not_fabricated_annotation_definitions():
    for entry in CORPUS:
        path = ROOT / entry["source_locator"].split("::")[0]
        literals = [node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert any(entry["text"] in literal for literal in literals), entry["id"]
        assert entry["source_kind"] in {"PROJECT_INTERPRETATION", "ONTOLOGY"}
        assert entry["original_prompt_contains"] is True


def test_manifest_preserves_full_ivt_and_does_not_expose_mutable_cache():
    manifest = knowledge_manifest()
    assert len([n for n in manifest["nodes"].values() if n["task"] == "ivt"]) == 100
    assert manifest["nodes"]["ivt_19"]["components"] == {"instrument": 0, "verb": 1, "target": 8}
    assert ["ivt_19", "HAS_COMPONENT", "target_8"] in manifest["edges"]
    assert ["ivt_19", "HAS_COMPONENT", "target_0"] not in manifest["edges"]
    assert manifest["knowledge_scope"] == "EXISTING_PROMPT_REORGANIZATION"
    assert not any(manifest[k] for k in ("uses_gt_statistics", "uses_reference_images", "uses_video_memory"))
    manifest["nodes"]["ivt_19"]["components"]["target"] = 0
    assert knowledge_manifest()["nodes"]["ivt_19"]["components"]["target"] == 8


@pytest.mark.parametrize("mode", ["none", "flat", "graph"])
def test_deterministic_inputs_unchanged_and_packet_has_no_decision_power(mode):
    pool = case()
    before = deepcopy(pool)
    first, second = retrieve(pool, mode), retrieve(pool, mode)
    assert pool == before
    assert first["reference_knowledge"] == second["reference_knowledge"]
    assert first["audit"] == second["audit"]
    assert first["stats"]["extra_model_calls"] == 0
    assert first["stats"]["novel_knowledge_items"] == 0
    assert not first["audit"]["changes_candidate_pool"]
    assert not first["audit"]["establishes_current_presence"]
    if mode == "none":
        assert first["reference_knowledge"] is None
        assert first["stats"]["wire_chars"] == 0
    else:
        assert first["reference_knowledge"]["items"]
        for item in first["reference_knowledge"]["items"]:
            assert not {"rating", "confidence", "finding", "accept", "present"} & set(item)
            assert item["source"] and item["original_prompt_contains"]
            assert set(item["for"]) <= {p["id"] for p in pool["propositions"]}


@pytest.mark.parametrize("mode", ["flat", "graph"])
@pytest.mark.parametrize("max_chars", [256, 800, 1600, 2500])
def test_whole_entries_and_whole_wire_fit_budget(mode, max_chars):
    result = retrieve(case(), mode, RetrievalPolicy(max_chars=max_chars))
    packet = result["reference_knowledge"]
    assert result["stats"]["wire_chars"] <= max_chars
    assert result["stats"]["unique_items"] <= 4
    if packet is not None:
        assert len(serialized(packet)) == result["stats"]["wire_chars"]
        originals = {entry["id"]: entry for entry in CORPUS}
        for item in packet["items"]:
            assert item["text"] == originals[item["id"]]["text"]
            source = result["audit"]["sources"][item["id"]]
            assert source["source_hash"] == hashlib.sha256(item["text"].encode()).hexdigest()
        assert len({i["id"] for i in packet["items"]}) == len(packet["items"])
    assert all(len(ids) <= 2 for ids in result["audit"]["by_candidate"].values())


def test_large_pool_keeps_all_queries_and_reports_budget_exclusions():
    nodes = knowledge_manifest()["nodes"]
    props = [v for v in nodes.values() if v["task"] != "ivt"]
    props.extend(nodes[f"ivt_{i}"] for i in range(32))
    result = retrieve({"propositions": props}, "graph")
    assert result["stats"]["candidates"] == 64
    assert len(result["audit"]["by_candidate"]) == 64
    assert result["stats"]["wire_chars"] <= 2500
    assert result["audit"]["excluded"]


def test_graph_audit_paths_follow_exact_ontology_and_sources():
    result = retrieve(case(), "graph")
    manifest = knowledge_manifest()
    valid_edges = {tuple(e) for e in manifest["edges"]}
    for pid, items in result["audit"]["paths"].items():
        for eid, hit in items.items():
            path = hit["path"]
            assert path[0][0] == pid
            assert path[-1][2] == "text:" + eid
            assert len(path) <= 2
            assert all(tuple(edge) in valid_edges for edge in path)
            assert all(a[2] == b[0] for a, b in pairwise(path))


def test_flat_uses_same_corpus_but_no_graph_paths():
    result = retrieve(case(), "flat")
    assert all(not hit["path"] for hits in result["audit"]["paths"].values() for hit in hits.values())
    shallow = retrieve(case(), "flat", RetrievalPolicy(max_hops=1))
    assert result["reference_knowledge"] == shallow["reference_knowledge"]
    assert result["audit"]["knowledge_sha256"] == retrieve(case(), "graph")["audit"]["knowledge_sha256"]


def test_null_mapping_and_empty_pool_have_explicit_behavior():
    current = {"instrument": [0], "verb": [9], "target": [14], "ivt": [94], "phase": [0]}
    result = retrieve(make_pool(current), "graph")
    assert "visible_null_relation" in result["audit"]["by_candidate"]["ivt_94"]
    empty = retrieve({"propositions": []}, "graph")
    assert empty["reference_knowledge"] is None
    assert empty["audit"]["fallback"] == "EXISTING_VISUAL_REVIEW"


@pytest.mark.parametrize("mutation", ["id", "tuple", "bool", "name", "duplicate", "component_missing", "extra"])
def test_misbound_candidates_fail_before_retrieval(mutation):
    # make_pool components reference the shared ontology. Corrupt only this
    # test input, so rejected payloads cannot poison later tests' label table.
    pool = deepcopy(case())
    relation = next(p for p in pool["propositions"] if p["id"] == "ivt_19")
    if mutation == "id":
        relation["id"] = "ivt_999"
    elif mutation == "tuple":
        relation["components"]["target"] = 0
    elif mutation == "bool":
        relation["components"]["verb"] = True
    elif mutation == "name":
        relation["name"] = "untrusted model claim"
    elif mutation == "duplicate":
        pool["propositions"].append(deepcopy(relation))
    elif mutation == "component_missing":
        pool["propositions"] = [p for p in pool["propositions"] if p["id"] != "target_8"]
    else:
        relation["gt"] = [19]
    with pytest.raises(ValueError):
        retrieve(pool)


@pytest.mark.parametrize("kwargs", [
    {"max_chars": 2501}, {"max_chars": 255}, {"max_unique_items": 5},
    {"max_items_per_candidate": 3}, {"max_hops": 3}, {"max_hops": True},
])
def test_policy_cannot_silently_expand_runtime_budget(kwargs):
    with pytest.raises(ValueError):
        RetrievalPolicy(**kwargs)


def test_modes_and_policy_types_are_explicit():
    with pytest.raises(ValueError):
        retrieve(case(), "memory")
    with pytest.raises(TypeError):
        retrieve(case(), "graph", {"max_chars": 2500})
