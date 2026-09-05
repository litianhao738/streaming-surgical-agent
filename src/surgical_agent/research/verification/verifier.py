"""Adapters that expose a strict KEEP/REPAIR Specialist contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from surgical_agent.api.errors import ApiCallFailure, ApiError
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.research.gate.contracts import SCOPE_TASKS, RepairScope
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    PhaseTransitionGraph,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import (
    CandidateSet,
    VerificationResult,
    _selected_ids,
    changed_tasks,
)
from surgical_agent.research.verification.repair import RepairEvidence, SpecialistResult


@dataclass
class BoundTargetedSpecialist:
    """Bind observation-local inputs around the existing targeted API verifier."""

    verifier: object
    context: PerceptionContext
    candidates: CandidateSet
    evidence: EvidenceProfile
    phase_transition_graph: PhaseTransitionGraph | None = None

    def _verify_visual_agreement(
        self,
        hypothesis: InitialPrediction,
        scope: RepairScope,
    ) -> SpecialistResult:
        """Two blind, differently focused reviews; neither sees the other answer.

        This is a corroboration rule, not a proof of semantic correctness. In
        particular it never turns a component mismatch into a null annotation.
        """
        requested = tuple(
            task
            for task in ("instrument", "verb", "target", "ivt", "phase")
            if task in SCOPE_TASKS[scope]
        )
        reviews = []
        for focus in ("scene_association", "motion_and_counterevidence"):
            result = self.verifier.verify(
                scope,
                self.context,
                hypothesis,
                self.candidates,
                self.evidence,
                requested_fields=requested,
                review_focus=focus,
            )
            if (
                not isinstance(result, VerificationResult)
                or tuple(field.task for field in result.field_outcomes) != requested
            ):
                return SpecialistResult(
                    "INVALID_RESPONSE", reason="REVIEW_FIELDS_MISMATCH"
                )
            if any(
                field.status != "Verified" or field.uncertainty is not None
                for field in result.field_outcomes
            ):
                return SpecialistResult("UNRESOLVED", reason="VISUAL_REVIEW_UNCERTAIN")
            reviews.append(result)
        first, second = reviews
        if any(
            _selected_ids(first.prediction, task)
            != _selected_ids(second.prediction, task)
            for task in requested
        ):
            return SpecialistResult("UNRESOLVED", reason="INDEPENDENT_REVIEWS_DISAGREE")
        hashes = (first.provenance.request_hash, second.provenance.request_hash)
        if None in hashes or len(set(hashes)) != 2:
            return SpecialistResult(
                "INVALID_RESPONSE", reason="REVIEW_PROVENANCE_NOT_DISTINCT"
            )
        proposed = first.prediction
        if scope == "interaction":
            components = load_ivt_components()
            projected = tuple(
                {components[value][index] for value in proposed.triplet_ids}
                for index in range(3)
            )
            # Tools without a representable IVT (e.g. specimen bag) remain in I.
            representable_tools = {values[0] for values in components.values()}
            if (
                projected[0] != set(proposed.instrument_ids) & representable_tools
                or projected[1] != set(proposed.verb_ids)
                or projected[2] != set(proposed.target_ids)
            ):
                return SpecialistResult(
                    "UNRESOLVED", reason="VISUAL_ASSOCIATION_NOT_CLOSED"
                )
        if scope == "instrument_presence":
            proposed = self._prune_ivts_for_instruments(proposed)
        if not self.candidates.admits(proposed):
            return SpecialistResult(
                "INVALID_RESPONSE", reason="REVIEW_OUTSIDE_ONTOLOGY"
            )
        kind = {
            "instrument_presence": "VISUAL_INSTRUMENT_AGREEMENT",
            "interaction": "VISUAL_INTERACTION_AGREEMENT",
            "workflow": "VISUAL_PHASE_AGREEMENT",
        }[scope]
        certificate = RepairEvidence(
            kind, tuple(f"api:review:{value}" for value in hashes)
        )
        if not changed_tasks(hypothesis, proposed):
            return SpecialistResult(
                "KEEP", reason="VISUAL_AGREEMENT_KEEP", repair_evidence=certificate
            )
        return SpecialistResult(
            "REPAIR",
            hypothesis=proposed,
            reason="VISUAL_AGREEMENT_REPAIR",
            repair_evidence=certificate,
        )

    @property
    def _evidence_first(self) -> bool:
        builder = getattr(self.verifier, "request_builder", None)
        return (
            getattr(builder, "prompt_version", None)
            == "targeted_verification_prompt_v8"
        )

    def _tracker_instrument_ids(self) -> tuple[int, ...]:
        """Return current predicted instruments when a Tracker cell supplies them."""

        snapshot = self.context.track_snapshot
        frames = snapshot.get("frames", ())
        if not isinstance(frames, (tuple, list)) or not frames:
            return ()
        current = frames[-1]
        if not isinstance(current, Mapping):
            return ()
        if current.get("frame_id") != self.context.sample.target_frame_id:
            return ()
        tracks = current.get("tracks", ())
        if not isinstance(tracks, (tuple, list)):
            return ()
        values = {
            track.get("instrument_id")
            for track in tracks
            if isinstance(track, Mapping)
            and isinstance(track.get("instrument_id"), int)
            and not isinstance(track.get("instrument_id"), bool)
        }
        return tuple(
            instrument_id
            for instrument_id in sorted(values)
            if instrument_id in self.candidates.allowed_ids["instrument"]
        )

    def _tracker_removal_evidence(
        self,
        *,
        current: InitialPrediction,
        proposed: InitialPrediction,
    ) -> RepairEvidence | None:
        """Certify only a deletion supported by two consecutive Tracker frames."""

        if not set(proposed.instrument_ids) < set(current.instrument_ids):
            return None
        snapshot = self.context.track_snapshot
        if snapshot.get("status") != "AVAILABLE":
            return None
        frames = snapshot.get("frames", ())
        if not isinstance(frames, (tuple, list)):
            return None
        observations: list[tuple[int, tuple[int, ...]]] = []
        required_frame_ids = tuple(self.context.sample.causal_frame_ids[-2:])
        causal_ids = set(required_frame_ids)
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            frame_id = frame.get("frame_id")
            tracks = frame.get("tracks", ())
            if (
                not isinstance(frame_id, int)
                or isinstance(frame_id, bool)
                or frame_id not in causal_ids
                or not isinstance(tracks, (tuple, list))
            ):
                continue
            raw_instrument_ids = {
                track.get("instrument_id")
                for track in tracks
                if isinstance(track, Mapping)
                and isinstance(track.get("instrument_id"), int)
                and not isinstance(track.get("instrument_id"), bool)
            }
            if not raw_instrument_ids.issubset(
                self.candidates.allowed_ids["instrument"]
            ):
                return None
            instrument_ids = tuple(sorted(raw_instrument_ids))
            observations.append((frame_id, instrument_ids))
        observations.sort(key=lambda item: item[0])
        if len(observations) < 2:
            return None
        recent = observations[-2:]
        if recent[-1][0] != self.context.sample.target_frame_id:
            return None
        if tuple(frame_id for frame_id, _ in recent) != required_frame_ids:
            return None
        if any(values != proposed.instrument_ids for _, values in recent):
            return None
        return RepairEvidence(
            "TRACKER_TEMPORAL_CONSENSUS",
            tuple(f"tracker:frame:{frame_id}" for frame_id, _ in recent),
        )

    def _verify_one_component(
        self,
        *,
        hypothesis: InitialPrediction,
        task: str,
    ) -> tuple[str, tuple[int, ...], dict[int, float]] | None:
        result = self.verifier.verify(
            "interaction",
            self.context,
            hypothesis,
            self.candidates,
            self.evidence,
            requested_fields=(task,),
        )
        if (
            not isinstance(result, VerificationResult)
            or len(result.field_outcomes) != 1
        ):
            return None
        outcome = result.field_outcomes[0]
        if outcome.task != task or outcome.status not in {"Verified", "Pending"}:
            return None
        if (outcome.status == "Verified") != (outcome.uncertainty is None):
            return None
        return (
            outcome.status,
            outcome.selected_ids,
            {
                record.class_id: record.confidence
                for record in outcome.candidate_records
            },
        )

    def _verify_evidence_first_interaction(
        self,
        hypothesis: InitialPrediction,
    ) -> SpecialistResult:
        """Resolve IVT by independent component experts plus exact tuple matching."""

        tracker_instruments = self._tracker_instrument_ids()
        instrument_result = (
            ("Verified", tracker_instruments, {})
            if tracker_instruments
            else self._verify_one_component(hypothesis=hypothesis, task="instrument")
        )
        verb_result = self._verify_one_component(hypothesis=hypothesis, task="verb")
        target_result = self._verify_one_component(hypothesis=hypothesis, task="target")
        if instrument_result is None or verb_result is None or target_result is None:
            return SpecialistResult(
                "UNRESOLVED",
                reason="COMPONENT_EXPERTS_NOT_DECISIVE",
            )
        instrument_status, instrument_ids, instrument_scores = instrument_result
        verb_status, verb_ids, verb_scores = verb_result
        target_status, target_ids, target_scores = target_result
        if instrument_status != "Verified":
            return SpecialistResult(
                "UNRESOLVED",
                reason="INSTRUMENT_EXPERT_NOT_DECISIVE",
            )
        components = load_ivt_components()
        selected_ivts: tuple[int, ...] = ()
        exact_component_consensus = (
            instrument_status == "Verified"
            and verb_status == "Verified"
            and target_status == "Verified"
        )
        if exact_component_consensus:
            selected_ivts = tuple(
                ivt_id
                for ivt_id in self.candidates.allowed_ids["ivt"]
                if components[ivt_id][0] in instrument_ids
                and components[ivt_id][1] in verb_ids
                and components[ivt_id][2] in target_ids
            )
        if not selected_ivts:
            # An active IVT needs positive evidence from both semantic experts.
            # Pending or factorized mismatch can resolve only to an explicitly
            # supplied null interaction for the visible instrument.
            selected_ivts = tuple(
                ivt_id
                for ivt_id in self.candidates.allowed_ids["ivt"]
                if components[ivt_id][0] in instrument_ids
                and components[ivt_id][1:] == (9, 14)
            )
        if not selected_ivts:
            return SpecialistResult(
                "UNRESOLVED",
                reason="NO_EXACT_COMPONENT_IVT_MATCH",
            )

        represented = {"instrument": set(), "verb": set(), "target": set()}
        for ivt_id in selected_ivts:
            instrument_id, verb_id, target_id = components[ivt_id]
            represented["instrument"].add(instrument_id)
            represented["verb"].add(verb_id)
            represented["target"].add(target_id)
        probabilities = dict(hypothesis.probabilities)
        for task, scores in (
            ("instrument", instrument_scores),
            ("verb", verb_scores),
            ("target", target_scores),
        ):
            if scores:
                dense = [0.0] * len(probabilities[task])
                for class_id, confidence in scores.items():
                    dense[class_id] = confidence
                probabilities[task] = tuple(dense)
        dense_ivt = [0.0] * len(probabilities["ivt"])
        for ivt_id in selected_ivts:
            instrument_id, verb_id, target_id = components[ivt_id]
            component_confidences = (
                instrument_scores.get(instrument_id, 1.0),
                verb_scores.get(verb_id, 0.5),
                target_scores.get(target_id, 0.5),
            )
            dense_ivt[ivt_id] = min(component_confidences)
        probabilities["ivt"] = tuple(dense_ivt)
        proposed = replace(
            hypothesis,
            instrument_ids=tuple(sorted(represented["instrument"])),
            verb_ids=tuple(sorted(represented["verb"])),
            target_ids=tuple(sorted(represented["target"])),
            triplet_ids=selected_ivts,
            probabilities=probabilities,
        )
        if not self.candidates.admits(proposed):
            return SpecialistResult(
                "INVALID_RESPONSE",
                reason="COMPONENT_CONSENSUS_OUTSIDE_POOL",
            )
        if proposed == hypothesis:
            return SpecialistResult("KEEP", reason="COMPONENT_EXPERTS_KEEP")
        return SpecialistResult(
            "REPAIR",
            hypothesis=proposed,
            reason="COMPONENT_EXPERTS_REPAIR",
            repair_evidence=(
                RepairEvidence(
                    "COMPONENT_EXPERT_CONSENSUS",
                    (
                        (
                            "tracker:target_frame"
                            if tracker_instruments
                            else "api:instrument_component"
                        ),
                        "api:verb_component",
                        "api:target_component",
                        "ontology:exact_ivt",
                    ),
                )
                if exact_component_consensus
                and set(represented["instrument"]) == set(instrument_ids)
                and set(represented["verb"]) == set(verb_ids)
                and set(represented["target"]) == set(target_ids)
                and all(
                    components[ivt_id][0] in instrument_ids
                    and components[ivt_id][1] in verb_ids
                    and components[ivt_id][2] in target_ids
                    for ivt_id in selected_ivts
                )
                else None
            ),
        )

    @staticmethod
    def _prune_ivts_for_instruments(
        proposed: InitialPrediction,
    ) -> InitialPrediction:
        """Apply the sole allowed instrument-scope structural consequence."""

        components = load_ivt_components()
        retained_ivts = tuple(
            ivt_id
            for ivt_id in proposed.triplet_ids
            if components[ivt_id][0] in proposed.instrument_ids
        )
        if retained_ivts == proposed.triplet_ids:
            return proposed
        probabilities = dict(proposed.probabilities)
        ivt_probabilities = list(probabilities["ivt"])
        for ivt_id in set(proposed.triplet_ids) - set(retained_ivts):
            ivt_probabilities[ivt_id] = 0.0
        probabilities["ivt"] = tuple(ivt_probabilities)
        return replace(
            proposed,
            triplet_ids=retained_ivts,
            probabilities=probabilities,
        )

    def _stabilize_workflow(
        self,
        proposed: InitialPrediction,
    ) -> InitialPrediction:
        """Reject a VLM phase jump forbidden by the frozen Training graph."""

        graph = self.phase_transition_graph
        if graph is None:
            return proposed
        summary = self.context.workflow_snapshot
        source_frame_id = summary.get("source_max_frame_id")
        if source_frame_id not in self.context.sample.causal_frame_ids[:-1]:
            return proposed
        phases = summary.get("recent_finalized_phases", ())
        if not isinstance(phases, (tuple, list)) or not phases:
            return proposed
        phase_states = summary.get("recent_phase_states", ())
        if phase_states and (
            not isinstance(phase_states, (tuple, list))
            or len(phase_states) != len(phases)
            or phase_states[-1] not in {"Accepted", "Verified"}
        ):
            return proposed
        try:
            previous_phase = int(phases[-1])
        except (TypeError, ValueError):
            return proposed
        if graph.allows(previous_phase, proposed.phase_id):
            return proposed
        if previous_phase not in self.candidates.allowed_ids["phase"]:
            return proposed
        probabilities = dict(proposed.probabilities)
        phase_probabilities = list(probabilities["phase"])
        phase_probabilities[previous_phase] = max(
            phase_probabilities[previous_phase], 0.5
        )
        probabilities["phase"] = tuple(phase_probabilities)
        return replace(
            proposed,
            phase_id=previous_phase,
            probabilities=probabilities,
        )

    def _workflow_transition_evidence(
        self,
        *,
        current: InitialPrediction,
        proposed: InitialPrediction,
    ) -> RepairEvidence | None:
        """Certify H1 only when it fixes a forbidden causal phase transition."""

        graph = self.phase_transition_graph
        if graph is None or current.phase_id == proposed.phase_id:
            return None
        summary = self.context.workflow_snapshot
        source_frame_id = summary.get("source_max_frame_id")
        phases = summary.get("recent_finalized_phases", ())
        phase_states = summary.get("recent_phase_states", ())
        if (
            source_frame_id not in self.context.sample.causal_frame_ids[:-1]
            or not isinstance(phases, (tuple, list))
            or not phases
            or (
                phase_states
                and (
                    not isinstance(phase_states, (tuple, list))
                    or len(phase_states) != len(phases)
                    or phase_states[-1] not in {"Accepted", "Verified"}
                )
            )
        ):
            return None
        try:
            previous_phase = int(phases[-1])
        except (TypeError, ValueError):
            return None
        if graph.allows(previous_phase, current.phase_id) or not graph.allows(
            previous_phase, proposed.phase_id
        ):
            return None
        return RepairEvidence(
            "WORKFLOW_TRANSITION_DOMINANCE",
            (
                f"workflow:frame:{source_frame_id}",
                f"phase_graph:{graph.sha256}",
            ),
        )

    def __post_init__(self) -> None:
        if not hasattr(self.verifier, "verify") or not callable(self.verifier.verify):
            raise TypeError("verifier must implement verify")
        if not isinstance(self.context, PerceptionContext):
            raise TypeError("context must be PerceptionContext")
        if not isinstance(self.candidates, CandidateSet):
            raise TypeError("candidates must be CandidateSet")
        if not isinstance(self.evidence, EvidenceProfile):
            raise TypeError("evidence must be EvidenceProfile")
        if self.phase_transition_graph is not None and not isinstance(
            self.phase_transition_graph, PhaseTransitionGraph
        ):
            raise TypeError(
                "phase_transition_graph must be PhaseTransitionGraph or None"
            )

    def verify(
        self,
        *,
        hypothesis: InitialPrediction,
        scope: RepairScope,
        attempt: int,
    ) -> SpecialistResult:
        del (
            attempt
        )  # attempt identity is logged by the bounded loop, not sent as semantics
        if not self.candidates.admits(hypothesis):
            return SpecialistResult("INVALID_RESPONSE", reason="INPUT_OUTSIDE_POOL")
        if (
            getattr(
                getattr(self.verifier, "request_builder", None), "prompt_version", None
            )
            == "targeted_verification_prompt_v9"
        ):
            try:
                return self._verify_visual_agreement(hypothesis, scope)
            except ApiCallFailure as exc:
                return SpecialistResult("NO_RESPONSE", reason=exc.cause.code)
            except ApiError as exc:
                return SpecialistResult("NO_RESPONSE", reason=exc.code)
            except Exception as exc:  # noqa: BLE001 - provider errors are not semantic evidence
                return SpecialistResult("NO_RESPONSE", reason=type(exc).__name__)
        if scope == "instrument_presence" and self._evidence_first:
            tracker_instruments = self._tracker_instrument_ids()
            if tracker_instruments:
                proposed = self._prune_ivts_for_instruments(
                    replace(hypothesis, instrument_ids=tracker_instruments)
                )
                if proposed == hypothesis:
                    return SpecialistResult("KEEP", reason="TRACKER_EXPERT_KEEP")
                return SpecialistResult(
                    "REPAIR",
                    hypothesis=proposed,
                    reason="TRACKER_EXPERT_REPAIR",
                    repair_evidence=self._tracker_removal_evidence(
                        current=hypothesis,
                        proposed=proposed,
                    ),
                )
        if scope == "interaction" and self._evidence_first:
            try:
                return self._verify_evidence_first_interaction(hypothesis)
            except Exception as exc:  # noqa: BLE001 - provider failure is an execution status
                return SpecialistResult("NO_RESPONSE", reason=type(exc).__name__)
        # Interaction is selected IVT-first.  Instrument, verb and target are
        # then derived from the selected IVTs, so a factorized model response
        # cannot construct a non-closed joint hypothesis.
        requested = (
            ("ivt",)
            if scope == "interaction"
            else tuple(task for task in SCOPE_TASKS[scope])
        )
        # Preserve the canonical five-head order expected by the parser.
        requested = tuple(
            task
            for task in ("instrument", "verb", "target", "ivt", "phase")
            if task in requested
        )
        try:
            result = self.verifier.verify(
                scope,
                self.context,
                hypothesis,
                self.candidates,
                self.evidence,
                requested_fields=requested,
            )
        except Exception as exc:  # noqa: BLE001 - provider failure is not semantic rejection
            return SpecialistResult("NO_RESPONSE", reason=type(exc).__name__)
        if not isinstance(result, VerificationResult):
            return SpecialistResult("INVALID_RESPONSE", reason="WRONG_RESULT_TYPE")
        statuses = {item.status for item in result.field_outcomes}
        if any(item.uncertainty is not None for item in result.field_outcomes):
            return SpecialistResult("UNRESOLVED", reason="SPECIALIST_UNCERTAIN")
        if "Rejected" in statuses:
            return SpecialistResult("REJECT", reason="SPECIALIST_REJECTED")
        if "Pending" in statuses:
            return SpecialistResult("UNRESOLVED", reason="SPECIALIST_PENDING")
        if statuses and statuses != {"Verified"}:
            return SpecialistResult("INVALID_RESPONSE", reason="INVALID_FIELD_STATUS")
        proposed = result.prediction
        repair_evidence = None
        if scope == "workflow" and self._evidence_first:
            proposed = self._stabilize_workflow(proposed)
            repair_evidence = self._workflow_transition_evidence(
                current=hypothesis,
                proposed=proposed,
            )
        if scope == "instrument_presence":
            # An instrument decision must not be forced to preserve an H0 IVT
            # whose instrument was just rejected.  V8 is allowed one tightly
            # bounded structural consequence: remove, never add, those IVTs.
            proposed = self._prune_ivts_for_instruments(proposed)
        if scope == "interaction":
            represented = {"instrument": set(), "verb": set(), "target": set()}
            components = load_ivt_components()
            for ivt_id in proposed.triplet_ids:
                instrument_id, verb_id, target_id = components[ivt_id]
                represented["instrument"].add(instrument_id)
                represented["verb"].add(verb_id)
                represented["target"].add(target_id)
            proposed = replace(
                proposed,
                instrument_ids=tuple(sorted(represented["instrument"])),
                verb_ids=tuple(sorted(represented["verb"])),
                target_ids=tuple(sorted(represented["target"])),
            )
            if not self.candidates.admits(proposed):
                return SpecialistResult(
                    "INVALID_RESPONSE",
                    reason="DERIVED_INTERACTION_OUTSIDE_POOL",
                )
        if proposed == hypothesis:
            return SpecialistResult("KEEP", reason="SPECIALIST_KEEP")
        return SpecialistResult(
            "REPAIR",
            hypothesis=proposed,
            reason="SPECIALIST_REPAIR",
            repair_evidence=repair_evidence,
        )


__all__ = ["BoundTargetedSpecialist"]
