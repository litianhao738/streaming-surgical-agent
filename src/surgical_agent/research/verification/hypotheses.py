"""Bounded factorized candidate pools for targeted verification."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.perception.contracts import (
    TASK_NAMES,
    JointPerceptionResult,
    RankedCandidate,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import CandidateSet


class FactorizedCandidateGenerator:
    """Build a bounded visual pool with an optional masked Training prior.

    Without a prior this preserves the legacy visual-top-k behavior.  With a
    prior it retains every visual IVT candidate, then adds frequent IVTs for the
    predicted phase and candidate instruments.  Component pools are closed over
    all expanded IVTs.  Phase remains complete because it has only seven legal
    classes.  The verifier receives IDs, not these upstream ranking scores.
    """

    def __init__(
        self,
        *,
        max_candidates_per_task: int = 8,
        phase_instrument_ivt_prior: Mapping[tuple[int, int], tuple[int, ...]]
        | None = None,
        complete_ontology: bool = False,
    ) -> None:
        if (
            not isinstance(max_candidates_per_task, int)
            or isinstance(max_candidates_per_task, bool)
            or max_candidates_per_task <= 0
        ):
            raise ValueError("max_candidates_per_task must be a positive integer")
        self.max_candidates_per_task = max_candidates_per_task
        self.complete_ontology = complete_ontology
        self.phase_instrument_ivt_prior = self._normalize_prior(
            phase_instrument_ivt_prior
        )

    @staticmethod
    def _normalize_prior(
        prior: Mapping[tuple[int, int], tuple[int, ...]] | None,
    ) -> Mapping[tuple[int, int], tuple[int, ...]]:
        if prior is None:
            return MappingProxyType({})
        if not isinstance(prior, Mapping):
            raise TypeError("phase_instrument_ivt_prior must be a mapping")
        components = load_ivt_components()
        normalized: dict[tuple[int, int], tuple[int, ...]] = {}
        for key, values in prior.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in key
                )
            ):
                raise ValueError(
                    "candidate-prior keys must be integer phase/instrument pairs"
                )
            phase_id, instrument_id = key
            if not 0 <= phase_id < TASK_CLASS_COUNTS["phase"]:
                raise ValueError("candidate-prior phase ID is outside the ontology")
            if not 0 <= instrument_id < TASK_CLASS_COUNTS["instrument"]:
                raise ValueError(
                    "candidate-prior instrument ID is outside the ontology"
                )
            ranked = tuple(values)
            if not ranked or len(set(ranked)) != len(ranked):
                raise ValueError(
                    "candidate-prior IVT rankings must be non-empty and unique"
                )
            if any(
                not isinstance(ivt_id, int)
                or isinstance(ivt_id, bool)
                or ivt_id not in components
                or components[ivt_id][0] != instrument_id
                for ivt_id in ranked
            ):
                raise ValueError(
                    "candidate-prior IVT does not match its instrument key"
                )
            normalized[(phase_id, instrument_id)] = ranked
        return MappingProxyType(normalized)

    def _expanded_ivt_ids(self, result: JointPerceptionResult) -> tuple[int, ...]:
        prediction = result.prediction
        ranked_ivt = result.raw_evidence.ranked_candidates["ivt"]
        selected: list[int] = []

        def add(ivt_id: int) -> None:
            if ivt_id not in selected and len(selected) < self.max_candidates_per_task:
                selected.append(ivt_id)

        for ivt_id in prediction.triplet_ids:
            add(ivt_id)
        # Preserve the formal H0 top-8 before adding the train-only recall prior.
        # Reserving the remaining capacity prevents a wider upstream ranking from
        # silently crowding every prior candidate out of the verifier pool.
        for candidate in ranked_ivt[:8]:
            add(candidate.class_id)
        if self.phase_instrument_ivt_prior:
            instrument_ids = list(prediction.instrument_ids)
            for candidate in result.raw_evidence.ranked_candidates["instrument"]:
                if candidate.class_id not in instrument_ids:
                    instrument_ids.append(candidate.class_id)
            for instrument_id in instrument_ids:
                for ivt_id in self.phase_instrument_ivt_prior.get(
                    (prediction.phase_id, instrument_id), ()
                ):
                    add(ivt_id)
        for candidate in ranked_ivt[8:]:
            add(candidate.class_id)
        return tuple(selected)

    def build(self, result: JointPerceptionResult) -> CandidateSet:
        if not isinstance(result, JointPerceptionResult):
            raise TypeError("candidate generation requires a JointPerceptionResult")
        prediction = result.prediction
        if self.complete_ontology:
            allowed = {
                task: tuple(range(TASK_CLASS_COUNTS[task])) for task in TASK_NAMES
            }
            records = {}
            for task in TASK_NAMES:
                scores = {
                    item.class_id: item.confidence
                    for item in result.raw_evidence.ranked_candidates[task]
                }
                records[task] = tuple(
                    sorted(
                        (
                            RankedCandidate(value, scores.get(value, 0.0))
                            for value in allowed[task]
                        ),
                        key=lambda item: (-item.confidence, item.class_id),
                    )
                )
            return CandidateSet(
                initial_prediction=prediction,
                allowed_ids=allowed,
                candidate_records=records,
                source_frame_id=result.raw_evidence.source_max_frame_id,
                candidate_version="complete_ontology_v5",
            )
        current = {
            "instrument": prediction.instrument_ids,
            "verb": prediction.verb_ids,
            "target": prediction.target_ids,
            "ivt": prediction.triplet_ids,
            "phase": (prediction.phase_id,),
        }
        expanded_ivt_ids = self._expanded_ivt_ids(result)
        required_components: dict[str, set[int]] = {
            "instrument": set(),
            "verb": set(),
            "target": set(),
        }
        ivt_components = load_ivt_components()
        closure_ivt_ids = (
            expanded_ivt_ids
            if self.phase_instrument_ivt_prior
            else prediction.triplet_ids
        )
        for triplet_id in closure_ivt_ids:
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
                required = (
                    set(expanded_ivt_ids)
                    if task == "ivt"
                    else set(current[task]) | required_components.get(task, set())
                )
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
            candidate_version=(
                "factorized_visual_train_prior_closure_phase_complete_v4"
                if self.phase_instrument_ivt_prior
                else "factorized_closure_topk_phase_complete_v3"
            ),
        )
