"""Experimental instance-grounded proposals and conservative visual admission.

No GT, no learned scores, and no claim that model-assessed evidence is truth.
Not wired into the default production pipeline.
"""

import io
import math
from collections.abc import Mapping
from copy import deepcopy
from functools import partial

from PIL import Image

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _ivt_rows

LOCATOR_VERSION = "contact_locator_v1"
PROPOSAL_VERSION = "instance_interaction_proposal_v1"
REVIEW_VERSION = "contact_contrast_review_v1"
REVIEW_1000_VERSION = "contact_contrast_review_v2"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def integer(low, high):
    return {"type": "integer", "minimum": low, "maximum": high}


def array(items, maximum, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def enum(*values):
    return {"type": "string", "enum": list(values)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 300}
BOOL = {"type": "boolean"}
BOX = array({"type": "number", "minimum": 0, "maximum": 1}, 4, 4)
LOCATOR_SCHEMA = obj({"schema_version": {"type": "string", "const": LOCATOR_VERSION},
    "instances": array(obj({"instance_id": integer(1, 3), "tip_box": BOX}), 3),
    "all_visible_tools_covered": BOOL})
PROPOSAL_SCHEMA = obj({"schema_version": {"type": "string", "const": PROPOSAL_VERSION},
    "instances": array(obj({"instance_id": integer(1, 3), "instrument_id": integer(0, 6),
        "ivt_ids": array(integer(0, 99), 3), "support": enum("SUPPORTED", "INSUFFICIENT"),
        "contact_observation": TEXT}), 3)})
REVIEW_SCHEMA = obj({"schema_version": {"type": "string", "const": REVIEW_VERSION},
    "preferred": enum("FIRST", "SECOND", "TIE", "INSUFFICIENT"),
    "all_visible_tools_covered": BOOL,
    "instances": array(obj({"instance_id": integer(1, 3), "crop_relevant": BOOL,
        "instrument_identity_supported": BOOL, "target_identity_or_oov_supported": BOOL,
        "action_or_oov_supported": BOOL, "distinguishing_observation": TEXT}), 3)})

# The proposal still references the historical 300-character TEXT contract.
# A deep copy prevents the new review limit from changing that shared object.
REVIEW_1000_SCHEMA = deepcopy(REVIEW_SCHEMA)
REVIEW_1000_SCHEMA["properties"]["schema_version"]["const"] = REVIEW_1000_VERSION
REVIEW_1000_SCHEMA["properties"]["instances"]["items"]["properties"]["distinguishing_observation"]["maxLength"] = 1000


def validate_shape(value, schema):
    def invalid():
        raise ApiSchemaError("Grounded repair response violates the strict contract")
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, Mapping) or set(value) != set(schema["required"]):
            invalid()
        for key, spec in schema["properties"].items():
            validate_shape(value[key], spec)
    elif kind == "array":
        if not isinstance(value, (list, tuple)) or not schema["minItems"] <= len(value) <= schema["maxItems"]:
            invalid()
        for item in value:
            validate_shape(item, schema["items"])
    elif kind == "boolean":
        if type(value) is not bool:
            invalid()
    elif kind in ("integer", "number"):
        if type(value) not in ((int,) if kind == "integer" else (int, float)):
            invalid()
        if not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]:
            invalid()
    elif kind == "string":
        if not isinstance(value, str) or not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 1000):
            invalid()
        if "const" in schema and value != schema["const"]:
            invalid()
        if "enum" in schema and value not in schema["enum"]:
            invalid()


def validate_grounded(value, *, schema):
    validate_shape(value, schema)
    ids = [x["instance_id"] for x in value["instances"]]
    if len(ids) != len(set(ids)):
        raise ApiSchemaError("Duplicate instance identities")
    if value["schema_version"] == LOCATOR_VERSION:
        for item in value["instances"]:
            x0, y0, x1, y1 = item["tip_box"]
            if x1 <= x0 or y1 <= y0 or x1-x0 < .015 or y1-y0 < .015:
                raise ApiSchemaError("Invalid or degenerate localization box")
    if value["schema_version"] == PROPOSAL_VERSION:
        components = {row[0]: row[1:] for row in _ivt_rows()}
        for item in value["instances"]:
            if len(set(item["ivt_ids"])) != len(item["ivt_ids"]):
                raise ApiSchemaError("Duplicate IVT hypothesis")
            if any(components[i][0] != item["instrument_id"] for i in item["ivt_ids"]):
                raise ApiSchemaError("IVT belongs to a different instrument")


GROUNDED_CONTRACTS = {schema["properties"]["schema_version"]["const"]:
    (schema, partial(validate_grounded, schema=schema))
    for schema in (LOCATOR_SCHEMA, PROPOSAL_SCHEMA, REVIEW_SCHEMA, REVIEW_1000_SCHEMA)}


def make_contact_crops(target_image, locator, *, target_frame_id):
    """Deterministic pixel crops from the TARGET only, never GT boxes or synthesis."""
    validate_grounded(locator, schema=LOCATOR_SCHEMA)
    crops, manifest = [], []
    with Image.open(io.BytesIO(target_image.content)) as source:
        source = source.convert("RGB")
        width, height = source.size
        for item in locator["instances"]:
            x0, y0, x1, y1 = item["tip_box"]
            # Keep surrounding anatomy, not an isolated metal tip.
            pad_x, pad_y = max((x1-x0)*.75, .10), max((y1-y0)*.75, .10)
            box = (max(0, math.floor((x0-pad_x)*width)), max(0, math.floor((y0-pad_y)*height)),
                   min(width, math.ceil((x1+pad_x)*width)), min(height, math.ceil((y1+pad_y)*height)))
            pixels = source.crop(box)
            buffer = io.BytesIO()
            pixels.save(buffer, format="PNG")
            crop = ApiImageInput(f"contact:{target_frame_id}:instance:{item['instance_id']}", "image/png", buffer.getvalue())
            crops.append(crop)
            manifest.append({"instance_id": item["instance_id"], "target_frame_id": target_frame_id,
                "source_sha256": target_image.sha256, "crop_sha256": crop.sha256,
                "pixel_box_xyxy": list(box), "crop_size": list(pixels.size)})
    return tuple(crops), manifest


def proposed_labels(h0, locator, proposal):
    """Derive components from grounded instance interactions, not independent votes."""
    validate_grounded(locator, schema=LOCATOR_SCHEMA)
    validate_grounded(proposal, schema=PROPOSAL_SCHEMA)
    located = {x["instance_id"] for x in locator["instances"]}
    if not located or {x["instance_id"] for x in proposal["instances"]} != located:
        return None
    if not locator["all_visible_tools_covered"]:
        return None
    if any(x["support"] != "SUPPORTED" or not x["ivt_ids"] for x in proposal["instances"]):
        return None
    components = {row[0]: row[1:] for row in _ivt_rows()}
    ivts = sorted({v for x in proposal["instances"] for v in x["ivt_ids"]})
    labels = {"ivt": ivts, "phase": list(h0["phase"])}
    for column, task in enumerate(("instrument", "verb", "target")):
        labels[task] = sorted({components[v][column] for v in ivts})
    return labels


def admit(h0, h1, locator, proposal, review, *, proposal_slot,
          review_schema_version=REVIEW_VERSION):
    """Uncalibrated safety policy; semantic benefit must be evaluated offline."""
    if h1 is None or h1 != proposed_labels(h0, locator, proposal):
        return {"decision": "KEEP", "reason": "INCOMPLETE_OR_INVALID_GROUNDING"}
    if h1 == h0:
        return {"decision": "KEEP", "reason": "NO_CHANGE"}
    if review is None:
        return {"decision": "KEEP", "reason": "NO_VALID_REVIEW"}
    if review_schema_version not in (REVIEW_VERSION, REVIEW_1000_VERSION):
        raise ValueError("unsupported contrast-review schema version")
    validate_grounded(review, schema=GROUNDED_CONTRACTS[review_schema_version][0])
    if proposal_slot not in {"FIRST", "SECOND"}:
        raise ValueError("invalid blinded hypothesis slot")
    ids = {x["instance_id"] for x in locator["instances"]}
    if {x["instance_id"] for x in review["instances"]} != ids:
        return {"decision": "KEEP", "reason": "REVIEW_MISSING_INSTANCE"}
    if review["preferred"] != proposal_slot or not review["all_visible_tools_covered"]:
        return {"decision": "KEEP", "reason": "NO_PREFERENCE_OR_INCOMPLETE_COVERAGE"}
    for item in review["instances"]:
        if not all(item[key] for key in ("crop_relevant", "instrument_identity_supported",
                "target_identity_or_oov_supported", "action_or_oov_supported")):
            return {"decision": "KEEP", "reason": "INSUFFICIENT_DISCRIMINATING_EVIDENCE"}
        if len(item["distinguishing_observation"].strip()) < 12:
            return {"decision": "KEEP", "reason": "MISSING_VISUAL_OBSERVATION"}
    return {"decision": "ACCEPT", "reason": "MODEL_ASSESSED_LOCAL_EVIDENCE_PREFERENCE"}
