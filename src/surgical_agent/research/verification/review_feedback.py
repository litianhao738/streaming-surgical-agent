"""Evidence feedback for a proposer; no transport, GT, or admission decisions.

Only already supplied, semantically valid reviewer observations are forwarded.
The numeric issue summary is checked against the five normalized reviews before
feedback is built; missing observations are never reconstructed from a score.
"""

import math
from copy import deepcopy

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS, TASKS
from surgical_agent.research.verification.semantic_coordinator import item_error

SCHEMA_VERSION = "candidate_review_feedback_v1"
INSTRUCTION = (
    "Model observations are fallible, not ground truth. Verify every claim against "
    "the supplied causal images and the current target frame before proposing new candidates. "
    "These are separate reviewers' observations, not independent proof or calibrated confidence. "
    "Preserve disagreement; an invalid or missing observation provides no visual evidence. "
    "This feedback grants no authority to accept, delete, or change the final prediction."
)
_PROPOSITION_FIELDS = frozenset({"id", "task", "label_id", "name", "components"})
_ISSUE_FIELDS = frozenset({"candidate_id", "currently_selected", "mean", "scores", "invalid"})


def _bound_propositions(pool):
    if not isinstance(pool, dict) or set(pool) != {"propositions"} or not isinstance(pool["propositions"], list):
        raise ValueError("pool must contain only a proposition list")
    bound = {}
    for item in pool["propositions"]:
        if not isinstance(item, dict) or set(item) != _PROPOSITION_FIELDS:
            raise ValueError("incomplete or extra proposition fields")
        task, label = item["task"], item["label_id"]
        if (not isinstance(task, str) or task not in TASKS or type(label) is not int
                or not 0 <= label < BOUNDS[task]):
            raise ValueError("proposition outside ontology")
        pid = f"{task}_{label}"
        if item["id"] != pid or pid in bound:
            raise ValueError("unknown, duplicate, or inconsistent candidate binding")
        components = COMPONENTS[label] if task == "ivt" else None
        name = ("/".join(_TASK_NAMES[t][value] for t, value in components.items())
                if components else _TASK_NAMES[task][label])
        if item["name"] != name or item["components"] != components:
            raise ValueError("candidate name or components do not match its ontology ID")
        bound[pid] = {"id": pid, "task": task, "label_id": label, "name": name,
                      "components": deepcopy(components)}
    return bound


def build_review_feedback(pool, reviews, issues, *, image_count=3):
    """Return deterministic, fallible observations for unresolved candidates.

    ``reviews`` must have the complete five named seats. Each seat contains a
    normalized ``judgments`` mapping. ``issues`` is the unmodified output of
    ``recent_mean_panel.unresolved``. Its scores, invalid reasons, and mean must
    agree with fresh per-item semantic validation. Invalid items appear only in
    ``invalid_reviewers`` with their actual validation reason; no observation is
    copied from them. No input object is modified.
    """
    if type(image_count) is not int or not 1 <= image_count <= 3:
        raise ValueError("one to three real causal images required")
    propositions = _bound_propositions(pool)
    if not isinstance(reviews, dict) or set(reviews) != set(SEATS):
        raise ValueError("complete five named reviewer seats required")
    for seat in SEATS:
        raw = reviews[seat]
        if (not isinstance(raw, dict) or set(raw) != {"judgments"}
                or not isinstance(raw["judgments"], dict)):
            raise ValueError("each normalized reviewer must have one judgments mapping")
        if set(raw["judgments"]) - propositions.keys():
            raise ValueError("unknown candidate in normalized reviewer")
    if not isinstance(issues, list):
        raise TypeError("issues must be a list")
    bound_issues = {}
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != _ISSUE_FIELDS:
            raise ValueError("incomplete or extra issue fields")
        pid = issue["candidate_id"]
        if not isinstance(pid, str) or pid not in propositions or pid in bound_issues:
            raise ValueError("unknown or duplicate issue candidate binding")
        scores, invalid = issue["scores"], issue["invalid"]
        if (type(issue["currently_selected"]) is not bool or not isinstance(scores, list)
                or len(scores) != len(SEATS)
                or any(value is not None and (type(value) is not int or not 1 <= value <= 5) for value in scores)
                or not isinstance(invalid, dict) or set(invalid) - set(SEATS)
                or any(not isinstance(reason, str) or not reason for reason in invalid.values())):
            raise ValueError("invalid issue score or reviewer binding")
        mean = issue["mean"]
        if mean is not None and (type(mean) not in (int, float) or not math.isfinite(mean)):
            raise ValueError("invalid issue mean")
        bound_issues[pid] = issue
    candidates = []
    for pid in sorted(bound_issues):
        issue, proposition = bound_issues[pid], propositions[pid]
        valid, invalid_items, expected_scores, expected_invalid = [], [], [], {}
        for seat in SEATS:
            item = reviews[seat]["judgments"].get(pid)
            reason = item_error(item, proposition["task"], image_count)
            if reason:
                expected_invalid[seat] = reason
                expected_scores.append(None)
                invalid_items.append({"reviewer_seat": seat, "reason": reason})
            else:
                expected_scores.append(item["rating"])
                valid.append({"reviewer_seat": seat, **deepcopy(item)})
        expected_mean = None if expected_invalid else sum(expected_scores) / len(SEATS)
        if (issue["scores"] != expected_scores or issue["invalid"] != expected_invalid
                or issue["mean"] != expected_mean):
            raise ValueError("issue disagrees with actual reviewer evidence; no fabricated missing or valid votes")
        candidates.append({"candidate_id": pid, "proposition": deepcopy(proposition),
                           "valid_observations": valid, "invalid_reviewers": invalid_items})
    return {"schema_version": SCHEMA_VERSION, "instruction": INSTRUCTION, "candidates": candidates}
