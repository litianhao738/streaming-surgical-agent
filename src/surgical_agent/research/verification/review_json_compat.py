"""Versioned JSON compatibility for identical duplicate review core fields.

This module does not change the original parser or any frozen experiment. Only
an item directly inside ``rows`` or ``judgments`` may repeat a non-binding core
field, and only with identical types and values. Every occurrence is audited;
no conflicting duplicate is silently resolved by first/last-value selection.
"""

import json
import re

from surgical_agent.research.verification.review_normalization import (
    CORE_FIELDS,
    parse_review_json,
)


class _ObjectPairs(list):
    """Distinguish an object retaining duplicate keys from a JSON array."""


def _same_json_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left)
    return left == right


def _path_text(path):
    result = "$"
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part):
            result += "." + part
        else:
            result += "[" + json.dumps(part, ensure_ascii=True) + "]"
    return result


def _is_review_item(path):
    return len(path) == 2 and (
        (path[0] == "rows" and type(path[1]) is int)
        or (path[0] == "judgments" and isinstance(path[1], str)))


def parse_review_json_compatible(text):
    """Return ``(object_or_none, diagnostics)`` with audited duplicate handling.

    The original parser still controls complete single-object JSON and boundary
    Markdown fences. On duplicate-key rejection only, reparse the full document
    preserving every object pair; allow identical duplicates of rating, finding,
    scope, image_indices, or observation solely within a review item. Binding,
    container, unknown, nested, or conflicting duplicates remain rejected.

    JSON validity is not evidence validity: the caller must still use the normal
    candidate binding and semantic validator, including whole-frame deletion.
    """
    parsed, original_diagnostics = parse_review_json(text)
    diagnostics = {**original_diagnostics, "identical_duplicates": [], "duplicate_errors": []}
    if diagnostics["error"] != "DUPLICATE_JSON_KEY":
        return parsed, diagnostics

    # Fence boundaries were already validated by the original parser. Its hook
    # may detect a duplicate before the decoder sees a trailing second document,
    # so json.loads below must still consume the complete stripped document.
    candidate = text.strip()
    if "leading" in diagnostics["removed_fences"]:
        candidate = candidate.partition("\n")[2].rstrip()
    if "trailing" in diagnostics["removed_fences"]:
        candidate = candidate[:-3].strip()
    try:
        tree = json.loads(candidate, object_pairs_hook=_ObjectPairs)
    except (ValueError, TypeError):
        diagnostics["error"] = "INVALID_JSON"
        return None, diagnostics
    if not isinstance(tree, _ObjectPairs):
        diagnostics["error"] = "JSON_OBJECT_REQUIRED"
        return None, diagnostics

    def convert(value, path):
        if isinstance(value, _ObjectPairs):
            occurrences = {}
            for field, child in value:
                occurrences.setdefault(field, []).append(convert(child, (*path, field)))
            result = {}
            for field, versions in occurrences.items():
                if len(versions) > 1:
                    detail = {"path": _path_text((*path, field)), "field": field,
                              "occurrences": len(versions)}
                    if not _is_review_item(path) or field not in CORE_FIELDS:
                        diagnostics["duplicate_errors"].append({**detail, "reason": "FORBIDDEN_FIELD_OR_PATH"})
                    elif not all(_same_json_value(versions[0], other) for other in versions[1:]):
                        diagnostics["duplicate_errors"].append({**detail, "reason": "CONFLICTING_TYPE_OR_VALUE"})
                    else:
                        diagnostics["identical_duplicates"].append(detail)
                result[field] = versions[0]
            return result
        if isinstance(value, list):
            return [convert(child, (*path, index)) for index, child in enumerate(value)]
        return value

    parsed = convert(tree, ())
    if diagnostics["duplicate_errors"]:
        return None, diagnostics
    diagnostics["error"] = None
    return parsed, diagnostics
