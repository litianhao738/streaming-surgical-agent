"""Explicit v3, length-only normalization for optional engineering sensitivity.

This module has no default integration. It never changes model judgments or
repairs invalid IDs, references, scopes, missing fields, or other schema errors.
"""

from copy import deepcopy
from hashlib import sha256

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    validate_presence_review_shape,
)

OBSERVATION_LENGTH_NORMALIZATION_VERSION = "presence_observation_length_normalization_v1"
MAX_OBSERVATION_CHARACTERS = 1000


def normalize_presence_observation_length(review):
    """Return a deep-copied valid v3 review and an audit of text-only truncations.

    Reject every other response version and any remaining schema failure. The
    source must be an ordinary JSON object, as parsed from the native response.
    This function uses no label truth, request text, image, or admission rule.
    """
    if not isinstance(review, dict) or review.get("schema_version") != PRESENCE_REVIEW_1000_VERSION:
        raise ApiSchemaError("Length normalization requires an explicit presence v3 response")
    normalized, changes = deepcopy(review), []
    assessments = normalized.get("assessments")
    if isinstance(assessments, (list, tuple)):
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict):
                continue
            text = assessment.get("observation")
            if isinstance(text, str) and len(text) > MAX_OBSERVATION_CHARACTERS:
                retained = text[:MAX_OBSERVATION_CHARACTERS]
                assessment["observation"] = retained
                changes.append({
                    "path": f"/assessments/{index}/observation",
                    "original_characters": len(text), "retained_characters": len(retained),
                    "original_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "retained_text_sha256": sha256(retained.encode("utf-8")).hexdigest(),
                })
    validate_presence_review_shape(normalized, schema_version=PRESENCE_REVIEW_1000_VERSION)
    return normalized, {"normalization_version": OBSERVATION_LENGTH_NORMALIZATION_VERSION,
                        "changes": changes}
