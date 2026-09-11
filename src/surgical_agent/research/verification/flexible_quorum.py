"""Bounded four/five-valid-seat experiment; no transport, GT or new evidence.

Invalid judgments abstain and never become neutral votes. A smaller validity
floor changes the evidence requirement, not the meaning of a model's score.
The arithmetic average is not a calibrated probability of correctness.
"""

import math
from copy import deepcopy

from surgical_agent.research.verification.candidate_coordinator import MAX_POOL, SEATS
from surgical_agent.research.verification.prior_panel import TASKS, labels
from surgical_agent.research.verification.recent_mean_panel import select
from surgical_agent.research.verification.review_feedback import _bound_propositions
from surgical_agent.research.verification.semantic_coordinator import (
    aggregate as semantic_aggregate,
)

__all__ = ["aggregate", "recheck_queue", "select"]


def _propositions(pool):
    bound = _bound_propositions(pool)
    if not bound:
        raise ValueError("an empty candidate pool cannot establish a model pass")
    if len(bound) > MAX_POOL:
        raise ValueError("candidate pool cap exceeded")
    return bound


def _conflict(values):
    return any(value <= 2 for value in values) and any(value >= 4 for value in values)


def aggregate(reviews, pool, *, minimum_valid=4, image_count=3):
    """Average only valid judgments once the predeclared floor is satisfied.

    All five named seats must be supplied; a failed seat may supply ``None``.
    The original semantic validator still rejects malformed judgments, local
    absence claims, unsupported image citations and score/finding conflicts.
    ``minimum_valid=5`` reproduces the original all-five-valid means.
    """
    if type(minimum_valid) is not int or minimum_valid not in (4, 5):
        raise ValueError("minimum_valid must be the predeclared integer 4 or 5")
    if type(image_count) is not int or not 1 <= image_count <= 3:
        raise ValueError("one to three real causal images required")
    bound = _propositions(pool)
    _, clean = semantic_aggregate(reviews, pool, image_count=image_count)
    means, diagnostics = {}, {}
    for pid in bound:
        invalid = {seat: clean[seat]["errors"][pid] for seat in SEATS if pid in clean[seat]["errors"]}
        valid_seats = [seat for seat in SEATS if seat not in invalid]
        values = [clean[seat]["scores"][pid] for seat in valid_seats]
        means[pid] = sum(values) / len(values) if len(values) >= minimum_valid else None
        diagnostics[pid] = {
            "scores": [None if seat in invalid else clean[seat]["scores"][pid] for seat in SEATS],
            "invalid": invalid,
            "valid_count": len(values),
            "valid_seats": valid_seats,
            "explicit_conflict": _conflict(values),
        }
    return means, diagnostics


def recheck_queue(current, pool, means, diagnostics):
    """Identify existing weak claims and valid positive/negative conflicts.

    Accept either validity floor's aggregate result and its own current state.
    Unselected, unsupported candidates without a real reviewer conflict do not
    demand another review. An empty queue is a stopping decision, not a claim
    that the whole frame is correct or that all absent candidates are refuted.
    Returned observations are copied; no label or input object is changed.
    """
    current = labels(current)
    bound = _propositions(pool)
    if not isinstance(means, dict) or not isinstance(diagnostics, dict) or set(means) != set(bound) or set(diagnostics) != set(bound):
        raise ValueError("means and diagnostics must cover the exact candidate pool")
    if any(f"{task}_{label}" not in bound for task in TASKS for label in current[task]):
        raise ValueError("current labels must be in the candidate pool")
    queue = []
    for pid in sorted(bound):
        proposition, mean, diagnostic = bound[pid], means[pid], diagnostics[pid]
        if not isinstance(diagnostic, dict):
            raise TypeError("invalid per-candidate diagnostics")
        scores, invalid = diagnostic.get("scores"), diagnostic.get("invalid")
        if (not isinstance(scores, list) or len(scores) != len(SEATS)
                or any(value is not None and (type(value) is not int or not 1 <= value <= 5) for value in scores)
                or not isinstance(invalid, dict) or set(invalid) - set(SEATS)
                or any(not isinstance(reason, str) or not reason for reason in invalid.values())
                or any((score is None) != (seat in invalid) for seat, score in zip(SEATS, scores, strict=True))):
            raise ValueError("scores and invalid seats are inconsistent")
        valid_seats = [seat for seat, score in zip(SEATS, scores, strict=True) if score is not None]
        values = [value for value in scores if value is not None]
        if (mean is not None and (type(mean) not in (int, float) or not math.isfinite(mean)
                or len(values) < 4 or mean != sum(values) / len(values))):
            raise ValueError("mean must come from at least four actual valid scores")
        if (("valid_count" in diagnostic and diagnostic["valid_count"] != len(values))
                or ("valid_seats" in diagnostic and diagnostic["valid_seats"] != valid_seats)):
            raise ValueError("valid-seat metadata disagrees with actual scores")
        present = proposition["label_id"] in current[proposition["task"]]
        reasons = []
        if present and mean is None:
            reasons.append("SELECTED_MISSING_OR_INVALID")
        elif present and mean < 4:
            reasons.append("SELECTED_BELOW_SUPPORT_THRESHOLD")
        if _conflict(values):
            reasons.append("VALID_REVIEWER_CONFLICT")
        if reasons:
            queue.append({"candidate_id": pid, "currently_selected": present, "mean": mean,
                          "scores": deepcopy(scores), "invalid": deepcopy(invalid),
                          "valid_count": len(values), "valid_seats": valid_seats, "reasons": reasons})
    return queue
