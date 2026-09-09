"""Bounded graph/flat retrieval for the existing five-seat visual review.

The service receives only an already frozen candidate pool. It neither proposes
candidates nor changes ratings/labels. It uses the shipped ontology and a small
attributable text corpus, without a model, embedding service, network, or GT.
Only ``reference_knowledge`` is intended for the model input; the longer audit
is local. The wire packet deliberately has no confidence or presence score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter

from surgical_agent.perception.ontology_prompt import _TASK_NAMES, _ivt_rows

from .review_corpus import CORPUS, CORPUS_VERSION

VERSION = "graph_review_retrieval_20260908_v1"
_TASKS = ("instrument", "verb", "target", "ivt")
_QUERY_ORDER = {"ivt": 0, "target": 1, "verb": 2, "instrument": 3}
_GUIDANCE = (
    "Reference definitions only, not evidence that any candidate occurs in this frame. "
    "Check the current images; preserve the existing rating and output rules."
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _wire_chars(value):
    # The existing reviewer wrapper serializes with json.dumps' default spaces.
    # Hashes remain compact, but the latency cap must cover the actual text.
    return len(json.dumps(value, ensure_ascii=False))


@dataclass(frozen=True)
class RetrievalPolicy:
    """Latency-oriented hard caps; expanding them needs a new protocol version."""

    max_chars: int = 2500
    max_unique_items: int = 4
    max_items_per_candidate: int = 2
    max_hops: int = 2

    def __post_init__(self):
        for name, low, high in (
            ("max_chars", 256, 2500),
            ("max_unique_items", 1, 4),
            ("max_items_per_candidate", 1, 2),
            ("max_hops", 1, 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")


@lru_cache(maxsize=1)
def _snapshot():
    nodes = {}
    for task in _TASKS[:3]:
        for label, name in enumerate(_TASK_NAMES[task]):
            nodes[f"{task}_{label}"] = {
                "id": f"{task}_{label}", "task": task, "label_id": label,
                "name": name, "components": None,
            }
    for ivt, instrument, verb, target in _ivt_rows():
        components = dict(zip(_TASKS[:3], (instrument, verb, target), strict=True))
        nodes[f"ivt_{ivt}"] = {
            "id": f"ivt_{ivt}", "task": "ivt", "label_id": ivt,
            "name": "/".join(_TASK_NAMES[t][v] for t, v in components.items()),
            "components": components,
        }
    entries = {}
    edges = []
    for node in nodes.values():
        if node["task"] == "ivt":
            edges.extend((node["id"], "HAS_COMPONENT", f"{t}_{v}")
                         for t, v in node["components"].items())
    for source in CORPUS:
        entry = dict(source)
        entry["source_hash"] = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        entry["source_hash_scope"] = "excerpt_utf8"
        entries[entry["id"]] = entry
        attached = set(entry["nodes"])
        attached.update(n["id"] for n in nodes.values() if n["task"] in entry["tasks"])
        if not attached <= set(nodes):
            raise ValueError("corpus references an unknown ontology node")
        edges.extend((nid, "DEFINED_BY", "text:" + entry["id"]) for nid in sorted(attached))
    edges.extend((a, "DISTINGUISHED_FROM", b)
                 for a, b in (("verb_0", "verb_1"), ("verb_1", "verb_0")))
    adjacency = {}
    for source, relation, destination in sorted(edges):
        adjacency.setdefault(source, []).append((relation, destination))
    version_hash = _hash({"version": VERSION, "corpus_version": CORPUS_VERSION,
                          "nodes": nodes, "entries": entries, "edges": sorted(edges)})
    return nodes, entries, adjacency, sorted(edges), version_hash


def knowledge_manifest():
    """Return a detached public snapshot suitable for a pre-call frozen manifest."""
    nodes, entries, _, edges, digest = _snapshot()
    return json.loads(_json({
        "version": VERSION, "corpus_version": CORPUS_VERSION, "sha256": digest,
        "nodes": nodes, "edges": edges, "entries": entries,
        "knowledge_scope": "EXISTING_PROMPT_REORGANIZATION",
        "uses_gt_statistics": False, "uses_reference_images": False,
        "uses_video_memory": False, "extra_model_calls": 0,
    }))


def _validated(pool):
    if not isinstance(pool, dict) or set(pool) != {"propositions"}:
        raise ValueError("expected the frozen make_pool object")
    propositions = pool["propositions"]
    if not isinstance(propositions, list) or len(propositions) > 64:
        raise ValueError("candidate pool must have at most 64 propositions")
    nodes = _snapshot()[0]
    result, seen = [], set()
    for candidate in propositions:
        if not isinstance(candidate, dict) or set(candidate) != {"id", "task", "label_id", "name", "components"}:
            raise ValueError("candidate must match the make_pool contract")
        pid = candidate["id"]
        if not isinstance(pid, str) or pid not in nodes or pid in seen:
            raise ValueError("unknown or duplicate candidate ID")
        expected = nodes[pid]
        components = candidate["components"]
        if (type(candidate["label_id"]) is not int or candidate != expected
                or (components is not None and any(type(v) is not int for v in components.values()))):
            raise ValueError("candidate ID, name or full IVT tuple differs from the ontology")
        seen.add(pid)
        result.append(expected)
    for candidate in result:
        if candidate["components"] and any(f"{t}_{v}" not in seen for t, v in candidate["components"].items()):
            raise ValueError("the frozen pool is missing an IVT component proposition")
    return sorted(result, key=lambda p: (_QUERY_ORDER[p["task"]], p["label_id"]))


def _graph_hits(candidate, max_hops):
    _, entries, adjacency, _, _ = _snapshot()
    queue = deque([(candidate["id"], [], 0)])
    visited, hits = {candidate["id"]}, []
    while queue:
        node, path, distance = queue.popleft()
        if distance >= max_hops:
            continue
        for relation, destination in adjacency.get(node, ()):
            next_path = [*path, [node, relation, destination]]
            if destination.startswith("text:"):
                eid = destination[5:]
                # Candidate-specific text wins ties over task-wide text, then
                # stable IDs. The generic deletion rule remains a lower tie.
                priority = (distance + 1, not bool(entries[eid]["nodes"]),
                            eid in {"frame_level_absence", "independent_component_support"}, eid)
                hits.append((priority, eid, next_path))
            elif destination not in visited:
                visited.add(destination)
                queue.append((destination, next_path, distance + 1))
    unique = {}
    for rank, eid, path in sorted(hits):
        unique.setdefault(eid, {"id": eid, "path": path, "rank": list(rank[:-1])})
    return list(unique.values())


def _tokens(text):
    return Counter(re.findall(r"[a-z]+|\d+", text.lower()))


def _flat_hits(candidate):
    """BM25-like lexical retrieval over exactly the graph's source excerpts.

    Query words come only from this candidate's task/name. There is no graph
    traversal, relationship metadata, embedding, or extra generated query.
    """
    entries = _snapshot()[1]
    documents = {eid: _tokens(entry["text"]) for eid, entry in entries.items()}
    query = _tokens(candidate["task"] + " " + candidate["name"])
    mean_length = sum(sum(d.values()) for d in documents.values()) / len(documents)
    scores = []
    for eid, document in documents.items():
        score = 0.0
        length = sum(document.values())
        for token in query:
            if not document[token]:
                continue
            frequency = sum(token in other for other in documents.values())
            idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
            term = document[token]
            score += idf * term * 2.2 / (term + 1.2 * (0.25 + 0.75 * length / mean_length))
        if score > 0:
            scores.append((-score, eid))
    return [{"id": eid, "path": [], "rank": [round(-score, 8)]} for score, eid in sorted(scores)]


def _wire(bindings):
    entries = _snapshot()[1]
    return {"use": _GUIDANCE, "items": [
        {"id": eid, "for": sorted(candidates), "text": entries[eid]["text"],
         "source_kind": entries[eid]["source_kind"],
         "source": entries[eid]["source_locator"],
         "original_prompt_contains": entries[eid]["original_prompt_contains"]}
        for eid, candidates in bindings.items()
    ]}


def retrieve(pool, mode="graph", policy=None):
    """Return a bounded model packet plus local-only audit and timing.

    ``mode='none'`` returns no model packet. The packet does not identify the
    arm, so exact matching graph/flat packets can share the same model call.
    Neither ``pool`` nor any global snapshot is exposed for mutation.
    """
    started = perf_counter()
    if mode not in {"none", "flat", "graph"}:
        raise ValueError("retrieval mode must be none, flat or graph")
    policy = RetrievalPolicy() if policy is None else policy
    if not isinstance(policy, RetrievalPolicy):
        raise TypeError("policy must be RetrievalPolicy")
    candidates = _validated(pool)
    _, entries, _, _, kb_hash = _snapshot()
    queries = {p["id"]: (_graph_hits(p, policy.max_hops) if mode == "graph" else _flat_hits(p))
               for p in candidates} if mode != "none" else {p["id"]: [] for p in candidates}
    bindings, by_candidate, paths, exclusions = {}, {p["id"]: [] for p in candidates}, {}, []
    # One best available item per candidate before second items. Relations come
    # first, then Target, Verb and Instrument; this ordering is frozen, not
    # chosen from annotation availability, model ratings, or observed results.
    for turn in range(policy.max_items_per_candidate):
        for candidate in candidates:
            pid = candidate["id"]
            if len(by_candidate[pid]) > turn:
                continue
            for hit in queries[pid]:
                eid = hit["id"]
                if eid in by_candidate[pid]:
                    continue
                if eid not in bindings and len(bindings) >= policy.max_unique_items:
                    exclusions.append({"candidate_id": pid, "item_id": eid, "reason": "ITEM_CAP"})
                    continue
                trial = {key: list(value) for key, value in bindings.items()}
                trial.setdefault(eid, []).append(pid)
                if _wire_chars(_wire(trial)) > policy.max_chars:
                    exclusions.append({"candidate_id": pid, "item_id": eid, "reason": "WIRE_CHAR_CAP"})
                    continue
                bindings = trial
                by_candidate[pid].append(eid)
                paths.setdefault(pid, {})[eid] = hit
                break
    packet = _wire(bindings) if bindings else None
    wire_chars = _wire_chars(packet) if packet is not None else 0
    audit = {
        "schema_version": VERSION, "mode": mode, "policy": asdict(policy),
        "knowledge_sha256": kb_hash, "knowledge_scope": "EXISTING_PROMPT_REORGANIZATION",
        "candidate_pool_sha256": _hash({"propositions": candidates}),
        "by_candidate": by_candidate, "paths": paths, "excluded": exclusions,
        "sources": {eid: {k: entries[eid][k] for k in (
            "source_locator", "source_kind", "source_hash", "source_hash_scope", "original_prompt_contains"
        )} for eid in bindings},
        "changes_candidate_pool": False, "establishes_current_presence": False,
        "fallback": "EXISTING_VISUAL_REVIEW" if packet is None else None,
    }
    return {
        "reference_knowledge": packet,
        "stats": {"mode": mode, "wire_chars": wire_chars, "unique_items": len(bindings),
                  "candidates": len(candidates), "covered_candidates": sum(bool(v) for v in by_candidate.values()),
                  "uncovered_candidates": [pid for pid, values in by_candidate.items() if not values],
                  "novel_knowledge_items": sum(not entries[eid]["original_prompt_contains"] for eid in bindings),
                  "extra_model_calls": 0, "retrieval_ms": (perf_counter() - started) * 1000,
                  "reference_sha256": _hash(packet)},
        "audit": audit,
    }
