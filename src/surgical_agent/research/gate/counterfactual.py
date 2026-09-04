"""Training-only forced-scope collection for the formal Benefit Gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from torch import Tensor

from surgical_agent.data.schemas import (
    DatasetSplit,
    FrameSupervisionTarget,
    InferenceSample,
)
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.gate.budget import VerificationBudgetManager
from surgical_agent.research.gate.contracts import (
    REPAIR_SCOPE_ORDER,
    RepairScope,
    SafetySupport,
)
from surgical_agent.research.gate.learned import prediction_utility
from surgical_agent.research.gate.oof_dataset import CounterfactualGateRecord
from surgical_agent.research.verification.repair import (
    BoundedVerifyRepairLoop,
    RepairProposal,
)
from surgical_agent.systems.final_pipeline import FinalStreamingPipeline


@dataclass(frozen=True)
class ScopeCounterfactual:
    scope: RepairScope
    proposal: RepairProposal
    initial_utility: float
    verified_utility: float | None
    benefit_label: Literal[0, 1] | None


@dataclass(frozen=True)
class FormalCounterfactualResult:
    record: CounterfactualGateRecord | None
    safety_class: Literal["HARD_VALID", "HARD_INVALID"]
    scope_results: tuple[ScopeCounterfactual, ...]


@dataclass(frozen=True)
class UnlabeledCounterfactualResult:
    sample: InferenceSample
    h0: InitialPrediction
    support: SafetySupport
    proposals: tuple[tuple[RepairScope, RepairProposal], ...]
    tracker_artifact_sha256: str


class FormalGateCounterfactualCollector:
    """Observe every legal scope independently without changing runtime Memory."""

    def __init__(
        self,
        *,
        pipeline: FinalStreamingPipeline,
        tracker_artifact_sha256: str,
        max_provider_attempts: int,
    ) -> None:
        if not isinstance(pipeline, FinalStreamingPipeline):
            raise TypeError("pipeline must be FinalStreamingPipeline")
        if pipeline.components.track_provider is None:
            raise ValueError("formal Gate collection requires video-OOF Tracker evidence")
        if (
            len(tracker_artifact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in tracker_artifact_sha256)
        ):
            raise ValueError("tracker_artifact_sha256 must be lowercase SHA-256")
        if max_provider_attempts <= 0:
            raise ValueError("max_provider_attempts must be positive")
        self.pipeline = pipeline
        self.tracker_artifact_sha256 = tracker_artifact_sha256
        self.collection_budget = VerificationBudgetManager(
            safety_reserve=0,
            optional_capacity=max_provider_attempts,
        )
        self._video_id: str | None = None
        self._previous: InitialPrediction | None = None
        self._previous_frame_id: int | None = None
        self._phases: list[str] = []

    def reset(self, video_id: str) -> None:
        self.pipeline.reset(video_id, recover=False)
        self.collection_budget.reset(video_id)
        self._video_id = video_id
        self._previous = None
        self._previous_frame_id = None
        self._phases = []

    def collect(
        self,
        sample: InferenceSample,
        frames: Tensor,
    ) -> UnlabeledCounterfactualResult:
        if sample.source_split is not DatasetSplit.TRAINING:
            raise ValueError("formal Gate counterfactual collection is Training-only")
        if self._video_id != sample.video_id:
            self.reset(sample.video_id)
        tracker = self.pipeline.tracker_snapshot(sample)
        if tracker.get("runtime_status") not in {"OK", "AVAILABLE_EMPTY"}:
            raise RuntimeError("OOF Tracker evidence is unavailable for a Training sample")
        workflow = {
            "status": "COUNTERFACTUAL_H0_ONLY",
            "source_max_frame_id": self._previous_frame_id,
            "recent_finalized_phases": tuple(self._phases[-16:]),
            "phase_stability": (
                None
                if not self._phases
                else self._phases.count(self._phases[-1]) / len(self._phases)
            ),
            "observed_transitions": tuple(pairwise(self._phases[-16:])),
        }
        context = self.pipeline.components.context_builder.build(
            sample,
            frames,
            workflow_snapshot=workflow,
            memory_snapshot={
                "component": "counterfactual_h0_history",
                "status": "TRAINING_ONLY",
                "source_max_frame_id": workflow["source_max_frame_id"],
            },
            prior_finalized_prediction=None,
            track_snapshot=tracker,
        )
        perception = self.pipeline.components.perception.predict(context)
        candidates = self.pipeline.components.candidate_generator.build(perception)
        evidence = self.pipeline.components.signal_extractor.extract(context, perception)
        violations = self.pipeline.components.safety_validator.validate(
            perception.prediction
        )
        support = self.pipeline.components.support_builder.build(
            perception=perception,
            candidates=candidates,
            violations=violations,
            tracker_snapshot=tracker,
            previous=self._previous,
        )
        self._previous = perception.prediction
        self._previous_frame_id = sample.target_frame_id
        self._phases.append(str(perception.prediction.phase_id))
        if support.safety_class == "HARD_INVALID":
            return UnlabeledCounterfactualResult(
                sample,
                perception.prediction,
                support,
                (),
                self.tracker_artifact_sha256,
            )

        loops: list[tuple[RepairScope, BoundedVerifyRepairLoop]] = []
        for scope in REPAIR_SCOPE_ORDER:
            if scope not in support.legal_scopes:
                continue
            specialist = self.pipeline.components.specialist_factory(
                context,
                candidates,
                evidence,
            )
            loops.append(
                (
                    scope,
                    BoundedVerifyRepairLoop(
                        specialist=specialist,
                        validator=self.pipeline.components.safety_validator,
                        budget=self.collection_budget,
                        max_attempts=self.pipeline.components.max_verify_attempts,
                    ),
                )
            )

        # These are mutually exclusive training counterfactuals over the same
        # immutable pre-decision state.  They have no causal dependency, so run
        # them concurrently while materializing results in the canonical scope
        # order.  Runtime inference still executes at most the one scope chosen
        # by Gate.
        def run_scope(
            scope: RepairScope,
            loop: BoundedVerifyRepairLoop,
        ) -> RepairProposal:
            return loop.run(
                h0=perception.prediction,
                candidates=candidates,
                scope=scope,
                priority="OPTIONAL",
                fallback_h0_allowed=True,
            )

        proposals: list[tuple[RepairScope, RepairProposal]] = []
        if loops:
            with ThreadPoolExecutor(
                max_workers=len(loops),
                thread_name_prefix="gate-counterfactual",
            ) as executor:
                futures = {
                    scope: executor.submit(run_scope, scope, loop)
                    for scope, loop in loops
                }
                proposals = [
                    (scope, futures[scope].result()) for scope, _loop in loops
                ]
        return UnlabeledCounterfactualResult(
            sample,
            perception.prediction,
            support,
            tuple(proposals),
            self.tracker_artifact_sha256,
        )


def label_counterfactual(
    collected: UnlabeledCounterfactualResult,
    target: FrameSupervisionTarget,
) -> FormalCounterfactualResult:
    """Apply Training GT only after every model/provider call is complete."""

    if not isinstance(collected, UnlabeledCounterfactualResult):
        raise TypeError("collected must be an unlabeled counterfactual")
    if not isinstance(target, FrameSupervisionTarget):
        raise TypeError("counterfactual labeling requires a Training target")
    sample = collected.sample
    if target.video_id != sample.video_id or target.frame_id != sample.target_frame_id:
        raise ValueError("counterfactual target identity mismatch")
    if collected.support.safety_class == "HARD_INVALID":
        return FormalCounterfactualResult(None, "HARD_INVALID", ())

    initial_utility, _ = prediction_utility(collected.h0, target)
    results: list[ScopeCounterfactual] = []
    labels: dict[RepairScope, int | None] = {
        scope: None for scope in REPAIR_SCOPE_ORDER
    }
    for scope, proposal in collected.proposals:
        verified_utility: float | None = None
        label: Literal[0, 1] | None = None
        observed_protected_keep = (
            proposal.status == "FALLBACK_KEEP"
            and proposal.reason == "HARD_VALID_H0_PROTECTED"
        )
        if proposal.status in {"VERIFIED_KEEP", "VERIFIED_REPAIR"} or (
            observed_protected_keep
        ):
            assert proposal.hypothesis is not None
            verified_utility, _ = prediction_utility(proposal.hypothesis, target)
            label = int(verified_utility > initial_utility)
            labels[scope] = label
        results.append(
            ScopeCounterfactual(
                scope,
                proposal,
                initial_utility,
                verified_utility,
                label,
            )
        )
    record = CounterfactualGateRecord(
        sample_id=f"{sample.video_id}:{sample.target_frame_id}",
        video_id=sample.video_id,
        frame_id=sample.target_frame_id,
        features=collected.support.gate_features,
        benefit_by_scope=labels,
        tracker_artifact_sha256=collected.tracker_artifact_sha256,
    )
    return FormalCounterfactualResult(record, "HARD_VALID", tuple(results))


__all__ = [
    "FormalCounterfactualResult",
    "FormalGateCounterfactualCollector",
    "ScopeCounterfactual",
    "UnlabeledCounterfactualResult",
    "label_counterfactual",
]
