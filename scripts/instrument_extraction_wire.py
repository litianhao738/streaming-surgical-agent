"""Request builder for instrument-anchored extraction.

Reuses the current default roster, transport and images unchanged, and replaces
only the question and the response contract. No historical wire is edited, so
every published experiment keeps its request hashes.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_glm_parallel_repair as roster
from scripts.check_candidate_panel_providers import ACADEMIC
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.instrument_extraction import (
    NULL_TARGET,
    NULL_VERB,
    VERSION,
    extraction_schema,
)

CONTRACT = "instrument_anchored_extraction_v1"
TASK = ("For each listed instrument, state the tissue its tip is acting on and the action it "
        "performs in the CURRENT image.")
INSTRUCTIONS = (
    "Work instrument by instrument. Find that tool's tip in the current image, then name the "
    "tissue it directly acts on and the action it performs. Do not start from a phase, a habit "
    "or the tool's identity: a grasper is not automatically grasping, and a hook is not "
    "automatically dissecting.\n"
    "Target means the tissue the tip actually acts on. Anatomy that is merely visible, nearby, "
    "deformed at a distance, or pulled through something else is NOT the target. Gripping tissue "
    "alone does not prove retract; visible displacement alone does not identify the tissue at "
    "the tip.\n"
    "Set present=false when that instrument is not visible in the current image, and then leave "
    "verb_id, target_id and both alternatives null.\n"
    f"When the acting tissue is outside the supplied ontology use target_id {NULL_TARGET}; when "
    f"the action is outside it use verb_id {NULL_VERB}. Do not substitute a nearby legal label "
    "for something the ontology cannot express.\n"
    "Only when the evidence genuinely allows two readings, supply ONE alternative verb or ONE "
    "alternative target. Leave both null when the primary reading is clear. Alternatives are "
    "competing hypotheses, not additional simultaneous labels.\n"
    "Earlier images are causal context only. Any present=true row must cite the current image "
    "index. Give one short English observation naming the tip location and the contact "
    "(1..1000 characters). Return only the JSON object in response_schema."
)


def ontology_block():
    return {
        "verbs": [{"id": i, "name": n} for i, n in enumerate(_TASK_NAMES["verb"])],
        "targets": [{"id": i, "name": n} for i, n in enumerate(_TASK_NAMES["target"])],
        "instruments": [{"id": i, "name": n} for i, n in enumerate(_TASK_NAMES["instrument"])],
    }


def extraction_wire(seat, base, selected, instruments, *, override=None):
    """One request per seat over the same three images the panel already sees.

    `override` swaps only the model binding, so a stronger backbone can answer
    the identical question. Omitting it reproduces the roster wire byte for byte.
    """
    ids = sorted(instruments)
    # A minimal pool keeps the frozen transport builder happy; its candidate
    # list is replaced wholesale below, so no rating contract survives.
    stub = make_pool({"instrument": ids, "verb": [], "target": [], "ivt": [],
                      "phase": [0]})
    body = roster.review_wire(seat, base, selected, stub)
    image_count = len(base.images)
    schema = extraction_schema(ids, image_count)
    packet = {
        "academic_context": ACADEMIC,
        "contract_version": VERSION,
        "task": TASK,
        "instructions": INSTRUCTIONS,
        "instruments_to_report": [{"id": i, "name": _TASK_NAMES["instrument"][i]} for i in ids],
        "ontology": ontology_block(),
        "label_boundaries": _LABEL_BOUNDARY,
        "target_frame_id": selected["frame_id"],
        "images": [{"index": n, "frame_id": f,
                    "seconds_relative_to_target": (f - selected["frame_id"]) / 25}
                   for n, f in enumerate(selected["causal_frame_ids"])],
        "current_image_index": image_count - 1,
        "response_schema": schema,
    }
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body["max_tokens"] = 4096
    # None removes a key the template seat set (e.g. a reasoning switch another
    # provider rejects); any other value replaces it. No override is a no-op.
    # Keys starting with "_" configure this builder and are never sent.
    for key, value in (override or {}).items():
        if key.startswith("_"):
            continue
        if value is None:
            body.pop(key, None)
        else:
            body[key] = deepcopy(value)
    if body.get("response_format", {}).get("type") == "json_schema":
        if (override or {}).get("_schema_dialect") == "anthropic":
            schema = anthropic_schema(schema)
        body["response_format"]["json_schema"] = {"name": CONTRACT, "strict": True,
                                                  "schema": schema}
    return body


def anthropic_schema(schema):
    """Anthropic rejects an enum under a multi-type declaration, so a nullable
    enum becomes anyOf; array length bounds are dropped because the exact roster
    is enforced locally by `response_error` either way."""
    def walk(node):
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        node = {k: walk(v) for k, v in node.items() if k not in ("minItems", "maxItems")}
        kinds = node.get("type")
        if isinstance(kinds, list) and "null" in kinds and "enum" in node:
            rest = {k: v for k, v in node.items() if k not in ("type", "enum")}
            values = [v for v in node["enum"] if v is not None]
            kind = next(k for k in kinds if k != "null")
            return {**rest, "anyOf": [{"type": kind, "enum": values}, {"type": "null"}]}
        return node
    return walk(schema)
