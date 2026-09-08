"""Remove harmless review metadata without repairing a model's judgment.

The output is consumed by the existing semantic coordinator. Missing, ambiguous,
or semantically invalid items remain absent and therefore cannot cast a vote.
An optional JSON string preserves duplicate-key detection before dict parsing;
duplicates already discarded by an upstream JSON decoder cannot be recovered.
"""

import json
import re
from copy import deepcopy

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.semantic_coordinator import item_error

CORE_FIELDS = frozenset({"rating", "finding", "scope", "image_indices", "observation"})
_ALIASES = {
    "score": "rating", "rating_score": "rating", "judgment_rating": "rating",
    "score_value": "rating", "rating_value": "rating", "rating_correction": "rating",
    "verdict": "finding", "finding_status": "finding", "finding_correction": "finding",
    "evidence_scope": "scope", "scope_correction": "scope", "scope_status": "scope",
    "indices": "image_indices", "frame_indices": "image_indices",
    "evidence_image_indices": "image_indices", "image_indices_string": "image_indices",
    "image_indices_correction": "image_indices",
}
_SCOPES = frozenset({"WHOLE_FRAME", "LOCAL_REGION", "UNCERTAIN"})
_CANDIDATE_KEY = re.compile(r"^(instrument|verb|target|ivt)_\d+$")


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def parse_review_json(text):
    """Parse one JSON object, allowing only complete boundary Markdown fences.

    A trailing-only ````` `` fence is allowed because it occurs in saved review
    responses. No object extraction, truncation repair, or in-JSON rewriting is
    performed. Return ``(None, diagnostics)`` for malformed/ambiguous input.
    Transport callers remain responsible for checking completion status and
    preserving the original response text.
    """
    diagnostics = {"removed_fences": [], "error": None}
    if not isinstance(text, str):
        diagnostics["error"] = "REVIEW_TEXT_REQUIRED"
        return None, diagnostics
    candidate = text.strip()
    if candidate.startswith("```"):
        first_line, separator, remainder = candidate.partition("\n")
        if (not separator or first_line.strip().lower() not in ("```", "```json")
                or not remainder.rstrip().endswith("```")):
            diagnostics["error"] = "INVALID_MARKDOWN_FENCE"
            return None, diagnostics
        candidate = remainder.rstrip()[:-3].strip()
        diagnostics["removed_fences"] = ["leading", "trailing"]
    elif candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
        diagnostics["removed_fences"] = ["trailing"]
    try:
        parsed = json.loads(candidate, object_pairs_hook=_unique_object)
    except _DuplicateKey:
        diagnostics["error"] = "DUPLICATE_JSON_KEY"
        return None, diagnostics
    except (ValueError, TypeError):
        diagnostics["error"] = "INVALID_JSON"
        return None, diagnostics
    if not isinstance(parsed, dict):
        diagnostics["error"] = "JSON_OBJECT_REQUIRED"
        return None, diagnostics
    return parsed, diagnostics


def _same_value(left, right):
    """Avoid treating True, 1 and 1.0 as identical metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    return left == right


def _alias_name(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower().replace("-", "_")


def _extra_error(field, value, item, proposition):
    alias = _alias_name(field)
    if alias in ("candidate_id", "id", "proposition_id"):
        return None if _same_value(value, proposition["id"]) else "CANDIDATE_ID_CONFLICT"
    if alias in ("task", "label_id"):
        return None if _same_value(value, proposition[alias]) else "CANDIDATE_BINDING_CONFLICT"
    # Observed Gemini metadata contained an unrelated candidate key with null.
    # A non-null nested candidate must not be silently discarded as metadata.
    if _CANDIDATE_KEY.fullmatch(field) and value is not None:
        return "NESTED_CANDIDATE_CONTENT"
    canonical = alias if alias in CORE_FIELDS else _ALIASES.get(alias)
    # scope_notes is prose metadata unless it states an explicit enum value.
    if alias == "scope_notes" and isinstance(value, str) and value in _SCOPES:
        canonical = "scope"
    if canonical is None or value is None:
        return None
    comparable = value
    if alias == "image_indices_string" and isinstance(value, str):
        try:
            comparable = json.loads(value)
        except (ValueError, TypeError):
            return "ALIAS_CONFLICT:" + field
    if not _same_value(comparable, item.get(canonical)):
        return "ALIAS_CONFLICT:" + field
    return None


def normalize_review(raw, pool, *, seat, image_count=3):
    """Return ``({'judgments': valid_items}, diagnostics)`` without mutation.

    ``errors`` maps candidate IDs to error-code lists. ``envelope_errors`` lists
    unbound/unknown rows and malformed envelopes. A duplicate bound ID rejects
    every occurrence of that ID; unrelated valid candidates remain usable.
    Harmless dropped fields are recorded by path, without copying their values.
    The function never coerces core values or invents a missing field. It applies
    the existing ``semantic_coordinator.item_error`` before accepting each item.
    """
    if seat not in SEATS:
        raise ValueError("unknown reviewer seat")
    if type(image_count) is not int or not 1 <= image_count <= 3:
        raise ValueError("only real causal short histories accepted")
    if not isinstance(pool, dict) or not isinstance(pool.get("propositions"), list):
        raise TypeError("invalid proposition pool")
    propositions = {}
    for p in pool["propositions"]:
        if (not isinstance(p, dict) or not isinstance(p.get("id"), str)
                or p["id"] in propositions
                or p.get("task") not in ("instrument", "verb", "target", "ivt")
                or type(p.get("label_id")) is not int
                or p["id"] != f"{p['task']}_{p['label_id']}"):
            raise ValueError("invalid or duplicate proposition binding")
        propositions[p["id"]] = p
    output = {"judgments": {}}
    diagnostics = {"seat": seat, "format": None, "ignored_fields": [],
                   "errors": {}, "envelope_errors": [], "accepted_ids": []}

    def error(pid, code):
        diagnostics["errors"].setdefault(pid, []).append(code)

    def fail_envelope(code):
        diagnostics["envelope_errors"].append(code)
        for pid in propositions:
            error(pid, code)
        return output, diagnostics

    if isinstance(raw, str):
        raw, diagnostics["parse"] = parse_review_json(raw)
        if diagnostics["parse"]["error"]:
            return fail_envelope(diagnostics["parse"]["error"])
    if not isinstance(raw, dict):
        return fail_envelope("INVALID_REVIEW_ENVELOPE")
    envelopes = set(raw) & {"judgments", "rows"}
    if len(envelopes) != 1:
        return fail_envelope("MISSING_OR_AMBIGUOUS_REVIEW_ENVELOPE")
    kind = next(iter(envelopes))
    diagnostics["format"] = kind
    for field in raw.keys() - {kind}:
        diagnostics["ignored_fields"].append({"path": "$", "field": str(field)})
    if kind == "judgments":
        if not isinstance(raw[kind], dict):
            return fail_envelope("INVALID_JUDGMENTS")
        entries = [(pid, item, "judgments." + str(pid)) for pid, item in raw[kind].items()]
    else:
        if not isinstance(raw[kind], list):
            return fail_envelope("INVALID_ROWS")
        entries = []
        for index, item in enumerate(raw[kind]):
            if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
                diagnostics["envelope_errors"].append(f"INVALID_ROW_ID:{index}")
                continue
            entries.append((item["candidate_id"], item, f"rows[{index}]"))
    seen = set()
    for pid, item, path in entries:
        if not isinstance(pid, str) or pid not in propositions:
            diagnostics["envelope_errors"].append("UNKNOWN_CANDIDATE_ID:" + str(pid))
            continue
        if pid in seen:
            error(pid, "DUPLICATE_CANDIDATE_ID")
            output["judgments"].pop(pid, None)
            continue
        seen.add(pid)
        if not isinstance(item, dict) or any(not isinstance(k, str) for k in item):
            error(pid, "SCHEMA_INVALID")
            continue
        core = {k: deepcopy(v) for k, v in item.items() if k in CORE_FIELDS}
        missing = CORE_FIELDS - core.keys()
        if missing:
            error(pid, "MISSING_CORE_FIELDS:" + ",".join(sorted(missing)))
        for field in item.keys() - CORE_FIELDS:
            code = _extra_error(field, item[field], core, propositions[pid])
            if code:
                error(pid, code)
            elif kind != "rows" or field != "candidate_id":
                diagnostics["ignored_fields"].append({"path": path, "field": field})
        semantic_error = item_error(core, propositions[pid]["task"], image_count)
        if semantic_error:
            error(pid, semantic_error)
        if pid not in diagnostics["errors"]:
            output["judgments"][pid] = core
    for pid in propositions.keys() - seen:
        error(pid, "MISSING_CANDIDATE_ID")
    diagnostics["accepted_ids"] = sorted(output["judgments"])
    diagnostics["ignored_fields"].sort(key=lambda x: (x["path"], x["field"]))
    return output, diagnostics
