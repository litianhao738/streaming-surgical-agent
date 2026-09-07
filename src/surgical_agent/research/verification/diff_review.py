"""Versioned label-change review and bounded repair, without GT or API calls.

Evidence references are checked against supplied inputs. A valid reference and
a model's assessment do not establish that its visual interpretation is true.
Imports of the existing final-label contract are delayed to avoid its legacy
API/schema registration cycle during import of this standalone module.
"""

import re
from collections.abc import Mapping
from copy import deepcopy

DIFF_REVIEW_VERSION = "frame_label_diff_review_v1"
DIFF_REVIEW_1000_VERSION = "frame_label_diff_review_v2"
_TASKS = ("instrument", "verb", "target", "ivt")
_BOUNDS = {"instrument": 6, "verb": 9, "target": 14, "ivt": 99}
# Two disjoint final-only sets can differ by at most 2*(3+4+5+8).
MAX_CHANGE_CLAIMS = 40
_CLAIM_PATTERN = r"^(instrument|verb|target|ivt):(ADD|REMOVE):(0|[1-9][0-9]?)$"


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


DIFF_REVIEW_SCHEMA = _object({
    "schema_version": {"type": "string", "const": DIFF_REVIEW_VERSION},
    "assessments": {"type": "array", "minItems": 0,
                    "maxItems": MAX_CHANGE_CLAIMS, "items": _object({
        "change_id": {"type": "string", "pattern": _CLAIM_PATTERN},
        "verdict": {"type": "string", "enum": [
            "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]},
        "observation": {"type": "string", "minLength": 1, "maxLength": 300},
        "evidence_refs": {"type": "array", "minItems": 0, "maxItems": 16,
                          "uniqueItems": True,
                          "items": {"type": "string", "minLength": 1,
                                    "maxLength": 200}},
        "scope": {"type": "string", "enum": ["FRAME", "INSTANCE"]},
        "full_frame_reviewed": {"type": "boolean"},
    })},
})

DIFF_REVIEW_1000_SCHEMA = deepcopy(DIFF_REVIEW_SCHEMA)
DIFF_REVIEW_1000_SCHEMA["properties"]["schema_version"]["const"] = DIFF_REVIEW_1000_VERSION
DIFF_REVIEW_1000_SCHEMA["properties"]["assessments"]["items"]["properties"]["observation"]["maxLength"] = 1000


def diff_review_schema(version=DIFF_REVIEW_VERSION):
    """Copy an explicit contract; historical defaults match the validators."""
    schemas = {DIFF_REVIEW_VERSION: DIFF_REVIEW_SCHEMA,
               DIFF_REVIEW_1000_VERSION: DIFF_REVIEW_1000_SCHEMA}
    if version not in schemas:
        _invalid("Unsupported diff-review schema version")
    return deepcopy(schemas[version])

DIFF_REVIEW_INSTRUCTION = """Review only the supplied label-change claims for the
TARGET frame. Each claim asks whether its label is present anywhere in that
frame, regardless of whether the proposed operation is ADD or REMOVE. For IVT,
the instrument, action and target must belong to the SAME interaction; separate
objects or interactions cannot collectively establish one IVT. Use preceding
frames only as causal context, never as proof an event remains in the target.

Return exactly one assessment per supplied change_id. SUPPORTED means visible
evidence supports the label's presence; CONTRADICTED means the target-frame
evidence contradicts its presence; INSUFFICIENT means neither is established.
Briefly describe the discriminating observation and any relevant counterevidence.
Cite only evidence_refs from the manifest of images actually supplied. ADD
assessments must cite the full target frame, optionally alongside relevant crops,
to anchor presence to the target rather than to a past frame. Crops
are local evidence: not finding an event in one crop does not refute its presence
elsewhere. A REMOVE can be justified only by a FRAME assessment that reviewed
and cites the full target frame, accounts for other visible instances, and
contradicts the original frame-level label. Occlusion, poor resolution, absent
motion cues or not seeing a label clearly are INSUFFICIENT, not evidence for
deletion. Do not infer correctness from either hypothesis, candidate legality,
label frequency, tool compatibility alone, or a desire for internally matching
heads. Phase is fixed. Do not predict labels outside the supplied claims.

These are model-assessed observations, not confidence scores or verified truth.
Return only JSON matching frame_label_diff_review_v1; no GT is supplied.
"""


def _invalid(message="Diff review violates the strict contract"):
    from surgical_agent.api.errors import ApiSchemaError

    raise ApiSchemaError(message)


def _labels(payload):
    from surgical_agent.research.verification.final_only_grounded import final_labels

    return final_labels(payload)


def _same(left, right):
    return all(set(left[task]) == set(right[task]) for task in (*_TASKS, "phase"))


def build_change_claims(h0, h1):
    """Build a deterministic symmetric difference of valid final-label sets."""
    baseline, candidate = _labels(h0), _labels(h1)
    if baseline["phase"] != candidate["phase"]:
        _invalid("Diff review must preserve H0 phase")
    claims = []
    for task in _TASKS:
        old, new = set(baseline[task]), set(candidate[task])
        for operation, values in (("REMOVE", old - new), ("ADD", new - old)):
            claims.extend({"change_id": f"{task}:{operation}:{label_id}",
                           "task": task, "label_id": label_id,
                           "operation": operation}
                          for label_id in sorted(values))
    return claims


def _validate_shape(value, schema):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, Mapping) or set(value) != set(schema["required"]):
            _invalid()
        for key, spec in schema["properties"].items():
            _validate_shape(value[key], spec)
    elif kind == "array":
        if not isinstance(value, (list, tuple)):
            _invalid()
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            _invalid()
        for item in value:
            _validate_shape(item, schema["items"])
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            _invalid()
    elif kind == "boolean":
        if type(value) is not bool:
            _invalid()
    elif kind == "string":
        if not isinstance(value, str):
            _invalid()
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 500):
            _invalid()
        if schema.get("minLength", 0) and not value.strip():
            _invalid()
        if "const" in schema and value != schema["const"]:
            _invalid()
        if "enum" in schema and value not in schema["enum"]:
            _invalid()
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            _invalid()


def validate_diff_review_shape(review, *, schema_version=DIFF_REVIEW_VERSION):
    """Provider registry validator; context binding happens before admission."""
    _validate_shape(review, diff_review_schema(schema_version))
    identities = [item["change_id"] for item in review["assessments"]]
    if len(set(identities)) != len(identities):
        _invalid("Duplicate diff-review claims")
    for identity in identities:
        task, _, label = identity.split(":")
        if int(label) > _BOUNDS[task]:
            _invalid("Diff-review label outside ontology")


def _evidence_context(allowed_evidence_refs, full_frame_ref):
    if isinstance(allowed_evidence_refs, (str, bytes)):
        _invalid("Evidence references require a collection")
    try:
        refs = list(allowed_evidence_refs)
    except TypeError:
        _invalid("Missing evidence references")
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        _invalid("Invalid evidence reference")
    if len(set(refs)) != len(refs):
        _invalid("Duplicate input evidence references")
    if not isinstance(full_frame_ref, str) or full_frame_ref not in refs:
        _invalid("Full target frame is missing from the evidence inputs")
    return refs


def build_diff_review_input(h0, h1, *, allowed_evidence_refs, full_frame_ref):
    """JSON prompt body; the runner supplies actual images and their manifest."""
    refs = _evidence_context(allowed_evidence_refs, full_frame_ref)
    return {"h0": _labels(h0), "h1": _labels(h1),
            "claims": build_change_claims(h0, h1),
            "allowed_evidence_refs": refs, "full_frame_ref": full_frame_ref}


def validate_diff_review(review, *, claims, allowed_evidence_refs, full_frame_ref,
                         schema_version=DIFF_REVIEW_VERSION):
    """Require exactly the real changes and references to supplied images."""
    validate_diff_review_shape(review, schema_version=schema_version)
    refs = set(_evidence_context(allowed_evidence_refs, full_frame_ref))
    expected = [claim["change_id"] for claim in claims]
    if len(set(expected)) != len(expected):
        _invalid("Duplicate expected change claims")
    if {item["change_id"] for item in review["assessments"]} != set(expected):
        _invalid("Review must assess every actual change exactly once")
    if any(not set(item["evidence_refs"]) <= refs for item in review["assessments"]):
        _invalid("Review cites evidence that was not supplied")


def _claim_admission(claim, assessment, full_frame_ref):
    if not assessment["evidence_refs"]:
        return False, "NO_REFERENCED_EVIDENCE"
    if claim["operation"] == "ADD":
        if assessment["verdict"] != "SUPPORTED":
            return False, "ADDITION_NOT_SUPPORTED"
        if full_frame_ref not in assessment["evidence_refs"]:
            return False, "ADDITION_REQUIRES_TARGET_FRAME_EVIDENCE"
        return True, "MODEL_ASSESSED_PRESENCE"
    if assessment["verdict"] != "CONTRADICTED":
        return False, "REMOVAL_NOT_CONTRADICTED"
    if (assessment["scope"] != "FRAME" or not assessment["full_frame_reviewed"]
            or full_frame_ref not in assessment["evidence_refs"]):
        return False, "REMOVAL_REQUIRES_FULL_FRAME_REVIEW"
    return True, "MODEL_ASSESSED_FRAME_ABSENCE"


def evaluate_diff_review(h0, h1, review, *, allowed_evidence_refs, full_frame_ref,
                         schema_version=DIFF_REVIEW_VERSION):
    """Evaluate whole-candidate A and bounded local B from one review.

    Invalid H0 raises: KEEP cannot repair invalid baseline data. Invalid optional
    candidates/reviews keep H0. GT availability never participates in admission;
    task masks belong solely to the downstream scorer. Neither policy uses the
    length or confidence of an explanation as a semantic correctness guarantee.
    """
    from surgical_agent.api.errors import ApiSchemaError
    from surgical_agent.perception.ontology_prompt import _ivt_rows

    diff_review_schema(schema_version)
    baseline = _labels(h0)
    result = {"schema_version": schema_version, "h0": baseline, "h1": None,
              "claims": [], "final_a": deepcopy(baseline),
              "final_b": deepcopy(baseline), "decision_a": "KEEP",
              "decision_b": "KEEP", "reason_a": "INVALID_CANDIDATE",
              "reason_b": "INVALID_CANDIDATE", "claim_decisions": []}
    try:
        candidate = _labels(h1)
        claims = build_change_claims(baseline, candidate)
    except (ApiSchemaError, OverflowError):
        return result
    result.update(h1=candidate, claims=claims)
    if not claims:
        result.update(reason_a="NO_LABEL_CHANGE", reason_b="NO_LABEL_CHANGE")
        return result
    try:
        validate_diff_review(review, claims=claims,
                             allowed_evidence_refs=allowed_evidence_refs,
                             full_frame_ref=full_frame_ref, schema_version=schema_version)
    except (ApiSchemaError, OverflowError):
        result.update(reason_a="INVALID_REVIEW", reason_b="INVALID_REVIEW")
        return result

    assessments = {item["change_id"]: item for item in review["assessments"]}
    decisions = {}
    for claim in claims:
        identity = claim["change_id"]
        passed, reason = _claim_admission(claim, assessments[identity], full_frame_ref)
        decisions[identity] = {**claim, "approved": passed, "reason": reason,
                               "applied_b": False, "reason_b": reason}
    result["claim_decisions"] = list(decisions.values())
    if all(item["approved"] for item in decisions.values()):
        result.update(final_a=deepcopy(candidate), decision_a="ACCEPT",
                      reason_a="ALL_CHANGE_CLAIMS_APPROVED")
    else:
        result["reason_a"] = "AT_LEAST_ONE_CHANGE_NOT_APPROVED"

    components = {row[0]: row[1:] for row in _ivt_rows()}
    updated = {task: set(baseline[task]) for task in _TASKS}

    def approved(task, operation, label):
        return decisions.get(f"{task}:{operation}:{label}", {}).get("approved", False)

    # IVT removal never cascades into component deletion.
    for label in set(baseline["ivt"]) - set(candidate["ivt"]):
        if approved("ivt", "REMOVE", label):
            updated["ivt"].discard(label)
            decisions[f"ivt:REMOVE:{label}"]["applied_b"] = True

    # An added IVT and all missing components form one dependency group. Existing
    # H0 components can be reused; no new component is invented without review.
    for label in sorted(set(candidate["ivt"]) - set(baseline["ivt"])):
        identity = f"ivt:ADD:{label}"
        if not decisions[identity]["approved"]:
            continue
        group = tuple(zip(_TASKS[:3], components[label], strict=True))
        if any(component not in baseline[task] and not approved(task, "ADD", component)
               for task, component in group):
            decisions[identity]["reason_b"] = "IVT_COMPONENT_NOT_SUPPORTED"
            continue
        updated["ivt"].add(label)
        decisions[identity]["applied_b"] = True
        for task, component in group:
            updated[task].add(component)

    # Independent heads retain their meaning, including instrument 6, which has
    # no component in the 100-class IVT ontology. No global IVT projection occurs.
    for task in _TASKS[:3]:
        for label in set(candidate[task]) - set(baseline[task]):
            if approved(task, "ADD", label):
                updated[task].add(label)
                decisions[f"{task}:ADD:{label}"]["applied_b"] = True
    for column, task in enumerate(_TASKS[:3]):
        referenced = {components[label][column] for label in updated["ivt"]}
        for label in set(baseline[task]) - set(candidate[task]):
            identity = f"{task}:REMOVE:{label}"
            if not decisions[identity]["approved"]:
                continue
            if label in referenced:
                decisions[identity]["reason_b"] = "REFERENCED_BY_REMAINING_IVT"
                continue
            updated[task].discard(label)
            decisions[identity]["applied_b"] = True

    partial = {task: sorted(updated[task]) for task in _TASKS}
    partial["phase"] = list(baseline["phase"])
    try:
        partial = _labels(partial)
    except (ApiSchemaError, OverflowError):
        # Do not rank subsets implicitly when their union exceeds wire limits.
        result["reason_b"] = "LOCAL_REPAIR_OUTSIDE_FINAL_ONLY_CONTRACT"
        for item in decisions.values():
            if item["applied_b"]:
                item.update(applied_b=False, reason_b="LOCAL_REPAIR_ROLLED_BACK")
        return result
    if _same(partial, baseline):
        result["reason_b"] = "NO_APPROVED_APPLICABLE_CHANGE"
    else:
        result.update(final_b=partial, decision_b="ACCEPT",
                      reason_b="APPROVED_LOCAL_CHANGES_APPLIED")
    return result
