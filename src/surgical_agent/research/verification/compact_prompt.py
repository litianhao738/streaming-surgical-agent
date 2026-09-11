"""Opt-in wording revision; preserve evidence, output contracts and transport.

Pass the target provider's text token counter to enforce its input budget.
Local surrogate encodings support an offline audit, not a billing guarantee.
This module performs no API calls and is not enabled in the frozen pipeline.
"""
import json
from copy import deepcopy
from importlib.resources import files

PROFILE = "verifier_compact_wording_v2_json_mode_candidate"


def compact_review_wire(body, branch, *, count_tokens):
    """Return a private request with only selected instruction text replaced.

    Reject unknown input contracts and any text-token increase. Images, schema,
    candidate data, phase definitions, model and generation settings stay exact.
    """
    if branch not in ("graph", "phase"):
        raise ValueError("branch must be graph or phase")
    out = deepcopy(body)
    messages = out["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("expected the frozen single-user review request")
    content = messages[0]["content"]
    if content[0]["type"] != "text":
        raise ValueError("expected the review packet first")
    before = content[0]["text"]
    packet = json.loads(before)
    schema_keys = set(packet["response_schema"]["properties"])
    if branch == "graph":
        if schema_keys not in ({"judgments"}, {"rows"}) or "propositions" not in packet:
            raise ValueError("expected the four-head evidence review contract")
        required = {"instructions", "proposition_semantics", "output_contract_clarification"}
        if not required <= packet.keys():
            raise ValueError("expected the explicit semantic review prompt")
        template = "compact_interaction_v1.txt"
        # The replacement consolidates these two repeated instruction sections.
        del packet["proposition_semantics"]
        del packet["output_contract_clarification"]
    else:
        if schema_keys != {"phase_id", "image_indices", "observation"}:
            raise ValueError("expected the single-choice Phase contract")
        template = "compact_phase_v2.txt"
    text = files("surgical_agent.research.verification.prompts").joinpath(template).read_text(encoding="utf-8")
    packet["instructions"] = text.strip().replace("{current_image_index}", str(len(content) - 2))
    after = json.dumps(packet, ensure_ascii=False)
    if count_tokens(after) > count_tokens(before):
        raise ValueError("revised prompt exceeds the original text-token budget")
    content[0]["text"] = after
    return out
