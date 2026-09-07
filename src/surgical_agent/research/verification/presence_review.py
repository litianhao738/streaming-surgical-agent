"""Neutral visual-presence review with the existing bounded A/B repair policies.

The model sees propositions about the current frame, never hypotheses or edit
directions. Python alone maps a presence assessment back to an addition/removal.
This removes a known interface ambiguity; it does not verify visual correctness.
"""

from copy import deepcopy

from surgical_agent.research.verification import diff_review as _diff

PRESENCE_REVIEW_VERSION = "frame_label_presence_review_v2"
PRESENCE_REVIEW_1000_VERSION = "frame_label_presence_review_v3"
_TASK_ORDER = ("instrument", "verb", "target", "ivt")
_PRESENCE_TO_VERDICT = {
    "PRESENT": "SUPPORTED", "ABSENT": "CONTRADICTED", "UNCLEAR": "INSUFFICIENT",
}

# Preserve the established evidence/observation limits while versioning the
# semantic interface. This copy never mutates the historical v1 schema.
PRESENCE_REVIEW_SCHEMA = deepcopy(_diff.DIFF_REVIEW_SCHEMA)
PRESENCE_REVIEW_SCHEMA["properties"]["schema_version"]["const"] = PRESENCE_REVIEW_VERSION
_ASSESSMENT = PRESENCE_REVIEW_SCHEMA["properties"]["assessments"]["items"]
_ASSESSMENT["properties"].pop("change_id")
_ASSESSMENT["properties"].pop("verdict")
_ASSESSMENT["properties"]["proposition_id"] = {
    "type": "string", "pattern": r"^p0(?:0[1-9]|[1-3][0-9]|40)$",
}
_ASSESSMENT["properties"]["presence"] = {
    "type": "string", "enum": list(_PRESENCE_TO_VERDICT),
}
_ASSESSMENT["required"] = list(_ASSESSMENT["properties"])

PRESENCE_REVIEW_1000_SCHEMA = deepcopy(PRESENCE_REVIEW_SCHEMA)
PRESENCE_REVIEW_1000_SCHEMA["properties"]["schema_version"]["const"] = PRESENCE_REVIEW_1000_VERSION
PRESENCE_REVIEW_1000_SCHEMA["properties"]["assessments"]["items"]["properties"]["observation"]["maxLength"] = 1000


def presence_review_schema(version=PRESENCE_REVIEW_VERSION):
    """Copy an explicit contract; historical defaults match the validators."""
    schemas = {PRESENCE_REVIEW_VERSION: PRESENCE_REVIEW_SCHEMA,
               PRESENCE_REVIEW_1000_VERSION: PRESENCE_REVIEW_1000_SCHEMA}
    if version not in schemas:
        _diff._invalid("Unsupported presence-review schema version")
    return deepcopy(schemas[version])

PRESENCE_REVIEW_INSTRUCTION = """Assess each supplied proposition about the TARGET
frame independently from the actual images and the supplied ontology. A
proposition asserts that a particular frame-level label is present somewhere in
the target frame. Return exactly one assessment for each opaque proposition_id.

PRESENT means visible target-frame evidence supports the label's presence.
ABSENT means sufficient target-frame evidence establishes that the frame-level
label is absent. UNCLEAR means neither presence nor absence is established.
These values describe the label's presence only. They are not judgments about
an answer, a proposed change, or whether any operation should be performed.

For IVT, the instrument, action and target must belong to the SAME interaction;
separate objects or interactions cannot collectively establish one IVT. Check
all visible instances before asserting frame-level absence: one instance not
performing an action does not establish that no instance performs it. ABSENT
requires FRAME scope, inspection of the full target frame, and a citation of
that full frame. Missing detail, occlusion, uncertain anatomy, insufficient
motion evidence or uncertainty about another visible instance mean UNCLEAR,
not ABSENT. Crops provide local evidence and cannot establish global absence.
PRESENT must cite the full target frame to anchor the evidence in the current
frame; relevant crops may also be cited. Preceding frames provide causal context
only and do not establish that an interaction continues in the target frame.

For null_verb, null_target and null IVTs, apply the supplied ontology and
label-boundary definitions. These labels do not mean uncertainty or poor image
quality. Use UNCLEAR when the images do not establish the defined category.
Do not infer presence from label frequency, tool compatibility alone, another
proposition, or a desire for matching heads. Phase is fixed and not assessed.

Briefly describe the discriminating observation and relevant counterevidence.
Cite only evidence_refs identifying images actually supplied, and accurately
state the scope and whether the full target frame was inspected. A reference
or explanation is not independent proof of visual correctness. Do not predict
outside the supplied propositions. Return only JSON matching
frame_label_presence_review_v2. No ground-truth labels are supplied.
"""


def _neutral_claims(h0, h1):
    claims = _diff.build_change_claims(h0, h1)
    return sorted(claims, key=lambda claim: (
        _TASK_ORDER.index(claim["task"]), claim["label_id"]))


def _propositions(h0, h1):
    return [{"proposition_id": f"p{index:03d}", "task": claim["task"],
             "label_id": claim["label_id"],
             "statement": (f"The target frame contains {claim['task']} label "
                           f"{claim['label_id']}, as defined in the supplied ontology.")}
            for index, claim in enumerate(_neutral_claims(h0, h1), start=1)]


def build_presence_review_input(h0, h1, *, allowed_evidence_refs, full_frame_ref):
    """Build the neutral model body; no baseline, candidate or edit is exposed."""
    refs = _diff._evidence_context(allowed_evidence_refs, full_frame_ref)
    return {"propositions": _propositions(h0, h1),
            "allowed_evidence_refs": refs, "full_frame_ref": full_frame_ref}


def validate_presence_review_shape(review, *, schema_version=PRESENCE_REVIEW_VERSION):
    """Validate wire structure; exact proposition/image binding needs context."""
    _diff._validate_shape(review, presence_review_schema(schema_version))
    identities = [item["proposition_id"] for item in review["assessments"]]
    if len(set(identities)) != len(identities):
        _diff._invalid("Duplicate presence-review propositions")


def validate_presence_review(review, *, h0, h1, allowed_evidence_refs, full_frame_ref,
                             schema_version=PRESENCE_REVIEW_VERSION):
    """Require one response per actual proposition and only supplied images."""
    validate_presence_review_shape(review, schema_version=schema_version)
    expected = build_presence_review_input(
        h0, h1, allowed_evidence_refs=allowed_evidence_refs, full_frame_ref=full_frame_ref)
    identities = {item["proposition_id"] for item in expected["propositions"]}
    if {item["proposition_id"] for item in review["assessments"]} != identities:
        _diff._invalid("Review must assess every neutral proposition exactly once")
    refs = set(expected["allowed_evidence_refs"])
    if any(not set(item["evidence_refs"]) <= refs for item in review["assessments"]):
        _diff._invalid("Presence review cites evidence that was not supplied")


def evaluate_presence_review(h0, h1, review, *, allowed_evidence_refs, full_frame_ref,
                             schema_version=PRESENCE_REVIEW_VERSION):
    """Translate neutral presence to established A/B policies without changing them.

    Invalid baseline labels raise. Invalid optional candidates/reviews preserve
    H0, as in v1. PRESENT never authorizes a removal; ABSENT never authorizes an
    addition. Model statements and image references remain unverified evidence.
    """
    from surgical_agent.api.errors import ApiSchemaError

    presence_review_schema(schema_version)
    diff_version = (_diff.DIFF_REVIEW_1000_VERSION if schema_version == PRESENCE_REVIEW_1000_VERSION
                    else _diff.DIFF_REVIEW_VERSION)
    result = _diff.evaluate_diff_review(
        h0, h1, None, allowed_evidence_refs=allowed_evidence_refs,
        full_frame_ref=full_frame_ref, schema_version=diff_version)
    result["schema_version"] = schema_version
    if result["reason_a"] in {"INVALID_CANDIDATE", "NO_LABEL_CHANGE"}:
        return result
    try:
        validate_presence_review(
            review, h0=h0, h1=h1, allowed_evidence_refs=allowed_evidence_refs,
            full_frame_ref=full_frame_ref, schema_version=schema_version)
    except (ApiSchemaError, OverflowError):
        return result
    claims_by_id = {f"p{index:03d}": claim
                    for index, claim in enumerate(_neutral_claims(h0, h1), start=1)}
    translated = {"schema_version": diff_version, "assessments": []}
    for assessment in review["assessments"]:
        claim = claims_by_id[assessment["proposition_id"]]
        translated["assessments"].append({
            "change_id": claim["change_id"],
            "verdict": _PRESENCE_TO_VERDICT[assessment["presence"]],
            **{key: deepcopy(assessment[key]) for key in (
                "observation", "evidence_refs", "scope", "full_frame_reviewed")},
        })
    result = _diff.evaluate_diff_review(
        h0, h1, translated, allowed_evidence_refs=allowed_evidence_refs,
        full_frame_ref=full_frame_ref, schema_version=diff_version)
    result["schema_version"] = schema_version
    return result
