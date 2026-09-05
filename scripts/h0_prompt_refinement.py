"""Isolated H0 prompt candidates; never mutate the production or frozen baseline."""

from __future__ import annotations

import json
from dataclasses import replace

from scripts.run_h0_conservative_input_smoke import conservative_request
from scripts.run_h0_frame_strategy_study import validate_request
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.schema import schema_for

ARMS = ("baseline", "schema_only", "tuned")

ORIGINAL_ASSOCIATION = (
    "Internally associate each currently visible instrument with its action and the\n"
    "anatomy at its tip or contact point. Build only ontology-valid IVT tuples from\n"
    "those associations, then select one phase consistent with the target scene and\n"
    "causal progression. Do this check internally and do not expose reasoning."
)

# These are evidence-use instructions, not new definitions of dataset classes.
# In particular, no motion threshold or grasp/retract conflict rule is invented.
REFINED_ASSOCIATION = (
    "Resolve each tool separately before combining frame-level labels:\n"
    "1. Establish its presence in the target image; earlier visibility alone is insufficient.\n"
    "2. Assess its current action from the tool's interaction and causal changes. "
    "Distinguish tissue motion from camera/viewpoint change; sparse images do not reveal "
    "every intervening action.\n"
    "3. Identify the anatomical structure, material or region actually acted on by that tool, "
    "using contact or other visible action evidence. Use surrounding anatomy as context, "
    "not as another active target merely because it is nearby.\n"
    "4. Match supported tool-action-target associations to the exact IVT tuples. Collect "
    "the supported frame-level instrument, verb and target labels, then select phase "
    "from the target scene and causal context. A typical phase alone does not establish "
    "a specific interaction.\n"
    "Perform these checks internally and output only the required JSON."
)


def remove_duplicate_schema(system_text: str, schema_version: str) -> str:
    """Remove only the known suffix, retaining the full wire response_format."""
    suffix = "Return exactly this JSON schema:\n" + json.dumps(
        schema_for(schema_version), sort_keys=True
    )
    if not system_text.endswith(suffix) or system_text.count(suffix) != 1:
        raise ValueError("unexpected source schema suffix; refusing heuristic deletion")
    return system_text[:-len(suffix)] + "Follow the supplied response_format JSON schema."


def variant_request(base, arm):
    if arm not in ARMS:
        raise ValueError("unknown prompt refinement arm")
    data = json.loads(base.payload["input_text"])
    validate_request(base, "B", data["target_frame_id"])
    if thaw_json(base.generation_parameters).get("reasoning") != {"effort": "low"}:
        raise ValueError("prompt study requires unchanged low reasoning")
    if arm == "baseline":
        return base
    candidate = conservative_request(base) if arm == "tuned" else base
    payload = thaw_json(candidate.payload)
    system = remove_duplicate_schema(payload["system_text"], base.response_schema_version)
    if arm == "tuned":
        if system.count(ORIGINAL_ASSOCIATION) != 1:
            raise ValueError("unexpected source association instruction")
        system = system.replace(ORIGINAL_ASSOCIATION, REFINED_ASSOCIATION)
    payload["system_text"] = system
    return replace(
        candidate,
        payload=payload,
        prompt_version=f"joint_final_only_prompt_refinement_{arm}_v1",
    )
