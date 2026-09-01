"""Bounded factorized candidate pools derived from perception top-k only."""

from __future__ import annotations

from surgical_agent.perception.contracts import (
    TASK_NAMES,
    JointPerceptionResult,
    RankedCandidate,
)
from surgical_agent.research.verification.contracts import CandidateSet


class FactorizedCandidateGenerator:
    """Keep current labels plus the first K ranked IDs for each task."""

    def __init__(self, *, max_candidates_per_task: int = 5) -> None:
        if (
            not isinstance(max_candidates_per_task, int)
            or isinstance(max_candidates_per_task, bool)
            or max_candidates_per_task <= 0
        ):
            raise ValueError("max_candidates_per_task must be a positive integer")
        self.max_candidates_per_task = max_candidates_per_task

    def build(self, result: JointPerceptionResult) -> CandidateSet:
        if not isinstance(result, JointPerceptionResult):
            raise TypeError("candidate generation requires a JointPerceptionResult")
        prediction = result.prediction
        current = {
            "instrument": prediction.instrument_ids,
            "verb": prediction.verb_ids,
            "target": prediction.target_ids,
            "ivt": prediction.triplet_ids,
            "phase": (prediction.phase_id,),
        }
        allowed: dict[str, tuple[int, ...]] = {}
        records: dict[str, tuple[RankedCandidate, ...]] = {}
        for task in TASK_NAMES:
            ranked = result.raw_evidence.ranked_candidates[task]
            selected_ranked = ranked[: self.max_candidates_per_task]
            ranked_ids = tuple(candidate.class_id for candidate in selected_ranked)
            allowed_ids = tuple(sorted(set(current[task]) | set(ranked_ids)))
            by_id = {candidate.class_id: candidate for candidate in ranked}
            records[task] = tuple(
                sorted(
                    (by_id[candidate_id] for candidate_id in allowed_ids),
                    key=lambda candidate: (-candidate.confidence, candidate.class_id),
                )
            )
            allowed[task] = allowed_ids
        return CandidateSet(
            initial_prediction=prediction,
            allowed_ids=allowed,
            source_frame_id=result.raw_evidence.source_max_frame_id,
            candidate_records=records,
        )
