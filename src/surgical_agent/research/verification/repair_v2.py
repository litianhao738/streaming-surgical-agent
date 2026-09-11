"""Ontology hypothesis completion and evidence-quorum review, without GT.

Completion proposes alternatives, never labels an image. Invalid evidence is
an abstention; at least four valid seats and no explicit contradiction remain
necessary for edits. Ratings are not calibrated correctness probabilities.
"""
from copy import deepcopy

from surgical_agent.research.verification import candidate_coordinator as candidates
from surgical_agent.research.verification import semantic_coordinator as evidence


def complete_target_hypotheses(h0, pool):
    """Enumerate legal targets for I/V already hypothesized, keeping the cap.

    On overflow return the original pool, rather than silently dropping or
    ranking alternatives. No instruments/verbs are inferred from a phase/GT.
    """
    instruments = {p["label_id"] for p in pool["propositions"] if p["task"] == "instrument"}
    verbs = {p["label_id"] for p in pool["propositions"] if p["task"] == "verb"}
    legal = [c for c, parts in sorted(candidates.COMPONENTS.items())
             if parts["instrument"] in instruments and parts["verb"] in verbs]
    expanded = deepcopy(pool)
    try:
        for c in legal:
            proposals = {q: [c] if q == "ivt" else [] for q in candidates.TASKS}
            expanded = candidates.make_pool(h0, proposals, expanded)
    except ValueError:
        return deepcopy(pool), {"status": "CAP_KEEP_ORIGINAL", "legal_alternatives": len(legal)}
    return expanded, {"status": "EXPANDED" if expanded != pool else "UNCHANGED",
                      "before": len(pool["propositions"]), "after": len(expanded["propositions"]),
                      "added_ids": sorted({p["id"] for p in expanded["propositions"]}
                                          - {p["id"] for p in pool["propositions"]})}


def aggregate_quorum(reviews, pool, *, image_count=3):
    _, clean = evidence.aggregate(reviews, pool, image_count=image_count)
    means = {}
    for p in pool["propositions"]:
        pid = p["id"]
        valid = [s for s in candidates.SEATS if pid not in clean[s]["errors"]]
        values = [clean[s]["scores"][pid] for s in valid]
        positive, negative = sum(x >= 4 for x in values), sum(x <= 2 for x in values)
        average = sum(values) / len(values) if values else None
        reason = ("INSUFFICIENT_VALID_SEATS" if len(valid) < 4 else
                  "EXPLICIT_CONFLICT" if positive and negative else
                  "INSUFFICIENT_DIRECTIONAL_SUPPORT" if
                  ((average >= 4 and positive < 3) or (average <= 2 and negative < 3)) else None)
        means[pid] = 3 if reason else average
        diagnostic = {"valid_seats": valid, "abstained_seats": [s for s in candidates.SEATS if s not in valid],
                      "valid_mean": average, "support_count": positive, "refute_count": negative,
                      "blocked_reason": reason, "effective_score": means[pid]}
        for seat in candidates.SEATS:
            clean[seat]["blocked"].pop(pid, None)
            if reason:
                clean[seat]["blocked"][pid] = {"reason": reason, "effective_score": 3}
            clean[seat].setdefault("quorum", {})[pid] = diagnostic
    return means, clean
