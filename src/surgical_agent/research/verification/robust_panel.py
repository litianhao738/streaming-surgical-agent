"""Valid-subset panel aggregation with a configurable reviewer roster.

The published selector `recent_mean_panel.select` is reused unchanged; only the
per-candidate statistic handed to it is produced here, so admission semantics,
component dependency protection and deletion protection stay exactly as
released. `recent_mean_panel.aggregate` blocks a candidate unless every seat
produced a valid item; measured on closed archives, 26 of 27 blocked candidates
had lost only one seat, so four intact opinions were discarded with the fifth.

This module keeps that guard but makes its strictness explicit: a candidate is
scored when at least `min_valid` seats are valid, using either the mean or the
median of the valid scores. It also accepts a smaller roster, so a three-seat
panel is expressed as a roster rather than as a special rule.

Nothing here changes a default. Callers opt in.
"""
from statistics import mean, median

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.semantic_coordinator import (
    aggregate as strict_aggregate,
)

STATISTICS = {"mean": mean, "median": median}


def resolve_roster(seats):
    """A roster is an ordered subset of the published seats, at least three."""
    roster = tuple(seats)
    if len(set(roster)) != len(roster) or not set(roster) <= set(SEATS):
        raise ValueError("roster must be distinct published reviewer seats")
    if len(roster) < 3:
        raise ValueError("a panel needs at least three seats")
    return roster


def aggregate(reviews, pool, *, image_count=3, seats=SEATS, min_valid=3, statistic="mean"):
    """Return ``(means, diagnostics)`` shaped like the published aggregator.

    ``means[pid]`` is ``None`` when fewer than ``min_valid`` seats produced a
    valid item for that candidate, which the frozen selector reads as "keep the
    current label". Invalid items never become a fabricated score or a vote.
    """
    roster = resolve_roster(seats)
    if statistic not in STATISTICS:
        raise ValueError("statistic must be mean or median")
    if type(min_valid) is not int or not 3 <= min_valid <= len(roster):
        raise ValueError("min_valid must be an integer between three and the roster size")
    if not pool["propositions"]:
        raise ValueError("an empty candidate pool cannot establish a model pass")
    if not isinstance(reviews, dict) or set(reviews) != set(roster):
        raise ValueError("reviews must cover exactly the configured roster")
    # The published item validator owns per-item semantics; only the seats in
    # this roster are presented to it, so an absent seat is never scored.
    _, clean = strict_aggregate({**{s: reviews.get(s) for s in roster},
                                 **{s: None for s in SEATS if s not in roster}},
                                pool, image_count=image_count)
    combine = STATISTICS[statistic]
    means, diagnostics = {}, {}
    for p in pool["propositions"]:
        pid = p["id"]
        invalid = {s: clean[s]["errors"][pid] for s in roster if pid in clean[s]["errors"]}
        scores = [clean[s]["scores"][pid] if s not in invalid else None for s in roster]
        valid = [s for s in scores if s is not None]
        means[pid] = combine(valid) if len(valid) >= min_valid else None
        diagnostics[pid] = {"scores": scores, "invalid": invalid,
                            "valid_seats": len(valid), "roster": list(roster),
                            "statistic": statistic,
                            "explicit_conflict": bool(valid) and min(valid) <= 2 and max(valid) >= 4}
    return means, diagnostics


def unresolved(state, pool, means, diagnostics, *, threshold=4.0):
    """Same reporting contract as the published panel, over this aggregation."""
    from surgical_agent.research.verification import recent_mean_panel as panel

    return panel.unresolved(state, pool, means, diagnostics, threshold=threshold)
