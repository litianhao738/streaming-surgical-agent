"""Bounded factorized candidate pools for targeted verification."""

from __future__ import annotations

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.perception.contracts import (
    TASK_NAMES,
    JointPerceptionResult,
    RankedCandidate,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import CandidateSet


class FactorizedCandidateGenerator:
    """Keep top-k semantic candidates and complete the small Phase ontology.

    The four multi-label heads remain bounded by perception recall.  Phase has
    only seven legal classes, so withholding classes that fell outside the
    perception top-k would make an otherwise valid workflow repair impossible.
    Missing Phase ranks receive a neutral floor score only for deterministic
    ordering; the verifier is not shown these upstream scores.
    """

    def __init__(self, *, max_candidates_per_task: int = 8) -> None:
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
        required_components: dict[str, set[int]] = {
            "instrument": set(),
            "verb": set(),
            "target": set(),
        }
        ivt_components = load_ivt_components()
        for triplet_id in prediction.triplet_ids:
            instrument_id, verb_id, target_id = ivt_components[triplet_id]
            required_components["instrument"].add(instrument_id)
            required_components["verb"].add(verb_id)
            required_components["target"].add(target_id)

        allowed: dict[str, tuple[int, ...]] = {}
        records: dict[str, tuple[RankedCandidate, ...]] = {}
        for task in TASK_NAMES:
            ranked = result.raw_evidence.ranked_candidates[task]
            if task == "phase":
                ranked_by_id = {candidate.class_id: candidate for candidate in ranked}
                selected_ids = tuple(range(TASK_CLASS_COUNTS["phase"]))
            else:
                required = set(current[task]) | required_components.get(task, set())
                if len(required) > self.max_candidates_per_task:
                    raise ValueError(
                        f"{task} frozen closure exceeds the candidate bound"
                    )
                selected = set(required)
                for candidate in ranked:
                    if len(selected) >= self.max_candidates_per_task:
                        break
                    selected.add(candidate.class_id)
                selected_ids = tuple(sorted(selected))
                ranked_by_id = {candidate.class_id: candidate for candidate in ranked}
            allowed_ids = tuple(sorted(selected_ids))
            records[task] = tuple(
                sorted(
                    (
                        ranked_by_id.get(
                            candidate_id, RankedCandidate(candidate_id, 0.0)
                        )
                        for candidate_id in allowed_ids
                    ),
                    key=lambda candidate: (-candidate.confidence, candidate.class_id),
                )
            )
            allowed[task] = allowed_ids
        return CandidateSet(
            initial_prediction=prediction,
            allowed_ids=allowed,
            source_frame_id=result.raw_evidence.source_max_frame_id,
            candidate_records=records,
            candidate_version="factorized_closure_topk_phase_complete_v3",
        )
