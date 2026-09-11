"""Disjoint component/relation panels, with unchanged item and admission rules."""
import json
from copy import deepcopy

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.review_normalization import normalize_review

PROFILE = "compact_components_relation_split_v1"
GROUPS = {"components": ("instrument", "verb", "target"), "relations": ("ivt",)}


def subpool(pool, group):
    if group not in GROUPS:
        raise ValueError("unknown review group")
    return {"propositions": [deepcopy(p) for p in pool["propositions"] if p["task"] in GROUPS[group]]}


def split_wire(body, group):
    """Filter candidate/schema IDs only; keep images and semantic rules intact."""
    out = deepcopy(body)
    packet = json.loads(out["messages"][0]["content"][0]["text"])
    pool = subpool({"propositions": packet["propositions"]}, group)
    ids = [p["id"] for p in pool["propositions"]]
    if not ids:
        raise ValueError("empty group must skip the API call")
    packet["propositions"] = pool["propositions"]
    packet["task"] = ("Audit only the listed instrument, verb and target candidates independently."
        if group == "components" else
        "Audit only the listed complete instrument-verb-target relations; verify that all three belong to the same interaction.")
    schema = packet["response_schema"]
    if set(schema["properties"]) == {"judgments"}:
        nested = schema["properties"]["judgments"]
        nested["properties"] = {pid: nested["properties"][pid] for pid in ids}
        nested["required"] = ids
    elif set(schema["properties"]) == {"rows"}:
        nested = schema["properties"]["rows"]
        nested["items"]["properties"]["candidate_id"]["enum"] = ids
        nested["minItems"] = nested["maxItems"] = len(ids)
    else:
        raise ValueError("unexpected compact response contract")
    if out["response_format"]["type"] == "json_schema":
        out["response_format"]["json_schema"]["schema"] = deepcopy(schema)
    out["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return out


def combine(h0, pool, raw, image_count=3):
    if set(raw) != set(GROUPS):
        raise ValueError("both groups required")
    normalized = {s: {"judgments": {}} for s in SEATS}
    formatting = {}
    for group in GROUPS:
        selected = subpool(pool, group)
        formatting[group] = {}
        if set(raw[group]) != set(SEATS):
            raise ValueError("five named seats per group required")
        for seat in SEATS:
            cleaned, diagnostic = normalize_review(raw[group][seat], selected, seat=seat, image_count=image_count)
            if set(normalized[seat]["judgments"]) & set(cleaned["judgments"]):
                raise ValueError("overlapping review groups")
            normalized[seat]["judgments"].update(cleaned["judgments"])
            formatting[group][seat] = diagnostic
    means, diagnostics = panel.aggregate(normalized, pool, image_count=image_count)
    return {"prediction": panel.select(h0, pool, means, threshold=4), "reviews": normalized,
        "means": means, "diagnostics": diagnostics, "formatting": formatting}
