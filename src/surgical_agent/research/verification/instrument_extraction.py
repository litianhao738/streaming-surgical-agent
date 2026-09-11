"""Instrument-anchored extraction instead of per-triplet rating.

Measured on closed archives, the five-seat panel cannot separate a correct IVT
relation from an incorrect one: correct candidates average 3.22 and incorrect
ones 3.14, with the same spread. The same reviewers do name the correct tissue
in their own prose far more often than the pipeline admits it, and more often
than the candidate pool even contains it.

This module asks a different question. For each instrument the perception head
already reports, a reviewer states which tissue that tool's tip acts on and what
action it performs, choosing from the ontology rather than scoring a supplied
list. Python assembles the triplets, so a relation can be produced without ever
having been proposed, which is what the rating contract structurally cannot do.

Nothing here is wired into a default. Assembly is bounded by the published
ontology: an instrument/verb/target combination that is not a real IVT class is
recorded and rejected, never coerced into a nearby legal triplet.
"""
from collections import Counter

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS

VERSION = "instrument_anchored_extraction_v1"
NULL_TARGET = 14
NULL_VERB = 9
FIELDS = ("instrument_id", "present", "verb_id", "target_id",
          "alternative_verb_id", "alternative_target_id", "image_indices", "observation")
TRIPLET_TO_IVT = {(c["instrument"], c["verb"], c["target"]): ivt
                  for ivt, c in COMPONENTS.items()}


def extraction_schema(instruments, image_count):
    """One row per queried instrument; the model cannot invent extra rows."""
    ids = sorted(instruments)
    if not ids or any(type(i) is not int or not 0 <= i < BOUNDS["instrument"] for i in ids):
        raise ValueError("a real instrument roster is required")
    row = {
        "instrument_id": {"type": "integer", "enum": ids},
        "present": {"type": "boolean"},
        "verb_id": {"type": ["integer", "null"], "enum": [*range(BOUNDS["verb"]), None]},
        "target_id": {"type": ["integer", "null"], "enum": [*range(BOUNDS["target"]), None]},
        "alternative_verb_id": {"type": ["integer", "null"], "enum": [*range(BOUNDS["verb"]), None]},
        "alternative_target_id": {"type": ["integer", "null"],
                                  "enum": [*range(BOUNDS["target"]), None]},
        "image_indices": {"type": "array", "items": {"type": "integer",
                                                     "enum": list(range(image_count))}},
        "observation": {"type": "string"},
    }
    return {"type": "object", "properties": {"instruments": {
        "type": "array", "minItems": len(ids), "maxItems": len(ids),
        "items": {"type": "object", "properties": row, "required": list(row),
                  "additionalProperties": False}}},
        "required": ["instruments"], "additionalProperties": False}


def row_error(row, instruments, image_count):
    """Validate one extracted instrument row; no field is coerced or invented."""
    if not isinstance(row, dict) or set(row) != set(FIELDS):
        return "INVALID_FIELDS"
    if type(row["instrument_id"]) is not int or row["instrument_id"] not in instruments:
        return "UNKNOWN_INSTRUMENT"
    if type(row["present"]) is not bool:
        return "INVALID_PRESENCE"
    for key, bound in (("verb_id", BOUNDS["verb"]), ("target_id", BOUNDS["target"]),
                       ("alternative_verb_id", BOUNDS["verb"]),
                       ("alternative_target_id", BOUNDS["target"])):
        value = row[key]
        if value is not None and (type(value) is not int or not 0 <= value < bound):
            return "INVALID_" + key.upper()
    refs = row["image_indices"]
    if (not isinstance(refs, list) or len(set(refs)) != len(refs)
            or any(type(i) is not int or not 0 <= i < image_count for i in refs)):
        return "INVALID_IMAGE_REFERENCE"
    observation = row["observation"]
    if not isinstance(observation, str) or not observation.strip() or len(observation) > 1000:
        return "INVALID_OBSERVATION"
    if not row["present"]:
        # An absent instrument cannot carry an action, a tissue or an alternative.
        if any(row[k] is not None for k in ("verb_id", "target_id",
                                            "alternative_verb_id", "alternative_target_id")):
            return "ABSENT_WITH_INTERACTION"
        return None
    if row["verb_id"] is None or row["target_id"] is None:
        return "PRESENT_WITHOUT_INTERACTION"
    if image_count - 1 not in refs:
        return "CURRENT_IMAGE_REQUIRED"
    if row["alternative_verb_id"] == row["verb_id"]:
        return "ALTERNATIVE_REPEATS_VERB"
    if row["alternative_target_id"] == row["target_id"]:
        return "ALTERNATIVE_REPEATS_TARGET"
    return None


def response_error(raw, instruments, image_count):
    if not isinstance(raw, dict) or set(raw) != {"instruments"}:
        return "INVALID_ENVELOPE"
    rows = raw["instruments"]
    if not isinstance(rows, list) or len(rows) != len(instruments):
        return "INCOMPLETE_ROSTER"
    seen = set()
    for row in rows:
        error = row_error(row, instruments, image_count)
        if error:
            return error
        if row["instrument_id"] in seen:
            return "DUPLICATE_INSTRUMENT"
        seen.add(row["instrument_id"])
    if seen != set(instruments):
        return "ROSTER_MISMATCH"
    return None


def triplets(raw):
    """Legal IVT ids implied by one valid response, with the rejected ones kept.

    The primary reading is `(instrument, verb, target)`. Each alternative varies
    exactly one component, so an ambiguous tool contributes at most three
    relations and never a free-floating pair.
    """
    accepted, rejected = {}, []
    for row in raw["instruments"]:
        if not row["present"]:
            continue
        i, v, t = row["instrument_id"], row["verb_id"], row["target_id"]
        options = [("primary", v, t)]
        if row["alternative_verb_id"] is not None:
            options.append(("alternative_verb", row["alternative_verb_id"], t))
        if row["alternative_target_id"] is not None:
            options.append(("alternative_target", v, row["alternative_target_id"]))
        for kind, verb, target in options:
            ivt = TRIPLET_TO_IVT.get((i, verb, target))
            if ivt is None:
                rejected.append({"instrument": i, "verb": verb, "target": target,
                                 "reading": kind, "reason": "NOT_AN_IVT_CLASS"})
                continue
            accepted.setdefault(ivt, []).append(kind)
    return {"ivt": accepted, "rejected": rejected}


def seat_labels(raw):
    """The five-head label sets one seat's response implies, primary reading only."""
    out = {"instrument": set(), "verb": set(), "target": set(), "ivt": set()}
    for row in raw["instruments"]:
        if not row["present"]:
            continue
        out["instrument"].add(row["instrument_id"])
        out["verb"].add(row["verb_id"])
        out["target"].add(row["target_id"])
        ivt = TRIPLET_TO_IVT.get((row["instrument_id"], row["verb_id"], row["target_id"]))
        if ivt is not None:
            out["ivt"].add(ivt)
    return out


def combine(responses, instruments, image_count, *, quorum=3, include_alternatives=False):
    """Majority extraction across seats; an invalid seat casts no vote.

    A relation is admitted when at least `quorum` structurally valid seats read
    it the same way. Alternatives are counted only when explicitly enabled, and
    a seat contributes at most one vote per relation regardless of how many of
    its readings produced it.
    """
    if type(quorum) is not int or not 1 <= quorum <= len(responses):
        raise ValueError("quorum must fit the number of seats")
    roster = sorted(instruments)
    errors, votes = {}, {"instrument": Counter(), "verb": Counter(),
                         "target": Counter(), "ivt": Counter()}
    valid, rejected = 0, []
    for seat, raw in sorted(responses.items()):
        error = response_error(raw, roster, image_count)
        if error:
            errors[seat] = error
            continue
        valid += 1
        labels = seat_labels(raw)
        assembled = triplets(raw)
        rejected += [{**row, "seat": seat} for row in assembled["rejected"]]
        if include_alternatives:
            labels["ivt"] |= set(assembled["ivt"])
            for ivt in assembled["ivt"]:
                for task, value in COMPONENTS[ivt].items():
                    labels[task].add(value)
        for task, values in labels.items():
            votes[task].update(values)
    prediction = {task: sorted(c for c, n in counts.items() if n >= quorum)
                  for task, counts in votes.items()}
    return {"prediction": prediction, "votes": {t: dict(c) for t, c in votes.items()},
            "valid_seats": valid, "errors": errors, "rejected_triplets": rejected,
            "quorum": quorum, "include_alternatives": include_alternatives,
            "status": "EXTRACTED" if valid >= quorum else "INSUFFICIENT_VALID_SEATS"}


def merge_with_h0(h0, extraction, *, mode):
    """Bounded ways to read an extraction against the frozen H0 five heads.

    `union` only ever adds, `replace` trusts extraction for the four interaction
    heads. Phase is never touched here: this contract does not ask about it.
    """
    if mode not in ("union", "replace"):
        raise ValueError("mode must be union or replace")
    out = {task: sorted(set(h0.get(task) or [])) for task in
           ("instrument", "verb", "target", "ivt")}
    out["phase"] = list(h0.get("phase") or [])
    if extraction["status"] != "EXTRACTED":
        return out
    for task in ("instrument", "verb", "target", "ivt"):
        found = set(extraction["prediction"][task])
        out[task] = sorted(found | set(out[task])) if mode == "union" else sorted(found)
    return out


def describe(ivt):
    return "/".join(_TASK_NAMES[t][v] for t, v in COMPONENTS[ivt].items())
