from __future__ import annotations

import json
from dataclasses import replace
from threading import Barrier

import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import (
    DatasetSplit,
    FrameSupervisionTarget,
    FrameTaskMask,
    InferenceSample,
)
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.gate.budget import VerificationBudgetManager
from surgical_agent.research.gate.contracts import (
    FORMAL_GATE_FEATURE_ORDER,
    RouteDecision,
    SafetySupport,
    SoftRisk,
)
from surgical_agent.research.gate.counterfactual import (
    FormalGateCounterfactualCollector,
    UnlabeledCounterfactualResult,
    label_counterfactual,
)
from surgical_agent.research.gate.formal_policy import RuleBenefitGate
from surgical_agent.research.memory.pending import (
    BoundedPendingResolver,
    PendingResolutionResult,
)
from surgical_agent.research.outcome import FinalOutcome, OutcomeFinalizer
from surgical_agent.research.safety import (
    DecisionSupportBuilder,
    MandatorySafetyGuard,
    SafetyValidator,
    legal_repair_scopes,
)
from surgical_agent.research.signals.frame_evidence import FrameEvidenceSignalExtractor
from surgical_agent.research.temporal_events import (
    ReliabilityAwareTemplateReporter,
    TemporalEventAggregator,
)
from surgical_agent.research.verification.contracts import CandidateSet
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.research.verification.repair import (
    BoundedVerifyRepairLoop,
    RepairProposal,
    SpecialistResult,
)
from surgical_agent.runtime.final_artifacts import FinalPipelineArtifactWriter
from surgical_agent.runtime.finalization import (
    AtomicFinalizationStore,
    FinalizationRecord,
)
from surgical_agent.runtime.state import ObservationIdentity
from surgical_agent.systems.final_pipeline import (
    FinalPipelineComponents,
    FinalStreamingPipeline,
)


def _probabilities() -> dict[str, tuple[float, ...]]:
    return {
        task: tuple(0.5 for _ in range(count))
        for task, count in TASK_CLASS_COUNTS.items()
    }


def _valid_h0() -> InitialPrediction:
    # IVT 0 maps to instrument 0, verb 2 and target 1.
    return InitialPrediction(
        instrument_ids=(0,),
        verb_ids=(2,),
        target_ids=(1,),
        triplet_ids=(0,),
        phase_id=0,
        probabilities=_probabilities(),
        score_semantics="uncalibrated_rank_v1",
    )


def _pool(h0: InitialPrediction) -> CandidateSet:
    return CandidateSet(
        initial_prediction=h0,
        allowed_ids={
            "instrument": (0, 1),
            "verb": (2, 3),
            "target": (1, 2),
            "ivt": (0, 1),
            "phase": (0, 1),
        },
        source_frame_id=7,
    )


def test_hard_invalid_is_routed_by_guard_and_never_gate() -> None:
    validator = SafetyValidator()
    invalid = replace(_valid_h0(), instrument_ids=(1,))
    candidates = _pool(invalid)
    violations = validator.validate(invalid)
    support = SafetySupport(
        hard_violations=violations,
        soft_risks=(),
        gate_features={},
        legal_scopes=legal_repair_scopes(violations, candidates),
    )

    route = MandatorySafetyGuard().route(support)
    assert support.safety_class == "HARD_INVALID"
    assert route is not None
    assert route.kind == "VERIFY"
    assert route.scope == "interaction"
    assert route.priority == "MANDATORY"
    assert route.fallback_h0_allowed is False

    try:
        RuleBenefitGate().decide(support)
    except ValueError as exc:
        assert "HARD_VALID" in str(exc)
    else:  # pragma: no cover - explicit invariant failure branch
        raise AssertionError("Gate accepted hard-invalid support")


def test_tracker_style_risk_remains_soft_and_gate_owned() -> None:
    support = SafetySupport(
        hard_violations=(),
        soft_risks=(SoftRisk("TRACKER_CONFLICT", ("instrument",), 0.9),),
        gate_features={"tracker_available": 1.0, "tracker_conflict": 1.0},
        legal_scopes=("instrument_presence",),
    )
    assert support.safety_class == "HARD_VALID"
    action = RuleBenefitGate(risk_threshold=0.5).decide(support)
    assert action.kind == "REQUEST_VERIFY"
    assert action.scope == "instrument_presence"


def test_rule_gate_routes_to_scope_with_highest_legal_risk() -> None:
    support = SafetySupport(
        hard_violations=(),
        soft_risks=(
            SoftRisk("LOW_INSTRUMENT_RISK", ("instrument",), 0.55),
            SoftRisk("HIGH_PHASE_RISK", ("phase",), 0.95),
        ),
        gate_features={},
        legal_scopes=("instrument_presence", "workflow"),
    )

    action = RuleBenefitGate(risk_threshold=0.5).decide(support)

    assert action.kind == "REQUEST_VERIFY"
    assert action.scope == "workflow"


def test_budget_preserves_mandatory_priority() -> None:
    budget = VerificationBudgetManager(
        safety_reserve=1,
        optional_capacity=1,
        shared_capacity=1,
    )
    budget.reset("VID02")
    assert budget.request("OPTIONAL").bucket == "OPTIONAL"
    assert not budget.request("OPTIONAL").granted
    assert budget.request("MANDATORY").bucket == "SAFETY_RESERVE"
    assert budget.request("MANDATORY").bucket == "SHARED"
    assert not budget.request("MANDATORY").granted
    assert budget.snapshot().charged_attempts == 3


def test_budget_can_recover_exact_committed_buckets() -> None:
    budget = VerificationBudgetManager(
        safety_reserve=2,
        optional_capacity=3,
        shared_capacity=1,
    )

    budget.restore_charged_buckets(
        "VID02", ("OPTIONAL", "SAFETY_RESERVE", "OPTIONAL")
    )

    snapshot = budget.snapshot()
    assert snapshot.safety_reserve_remaining == 1
    assert snapshot.optional_remaining == 1
    assert snapshot.shared_remaining == 1
    assert snapshot.charged_attempts == 3

    budget.restore_charged_buckets("VID03", ())
    assert budget.snapshot().video_id == "VID03"
    assert budget.snapshot().charged_attempts == 0


class _AlwaysMissing:
    def verify(self, **_: object) -> SpecialistResult:
        return SpecialistResult("NO_RESPONSE", reason="TIMEOUT")


class _RepairInstrument:
    def __init__(self, repaired: InitialPrediction) -> None:
        self.repaired = repaired

    def verify(self, **_: object) -> SpecialistResult:
        return SpecialistResult(
            "REPAIR",
            hypothesis=self.repaired,
            reason="SELECTED_IN_POOL_CANDIDATE",
        )


def test_bounded_loop_uses_different_failure_semantics_by_route_source() -> None:
    h0 = _valid_h0()
    candidates = _pool(h0)
    optional_budget = VerificationBudgetManager(
        safety_reserve=0, optional_capacity=2
    )
    optional_budget.reset("VID02")
    optional = BoundedVerifyRepairLoop(
        specialist=_AlwaysMissing(),
        validator=SafetyValidator(),
        budget=optional_budget,
        max_attempts=2,
    ).run(
        h0=h0,
        candidates=candidates,
        scope="interaction",
        priority="OPTIONAL",
        fallback_h0_allowed=True,
    )
    assert optional.status == "FALLBACK_KEEP"
    assert len(optional.attempts) == 2

    mandatory_budget = VerificationBudgetManager(
        safety_reserve=1, optional_capacity=0
    )
    mandatory_budget.reset("VID02")
    mandatory = BoundedVerifyRepairLoop(
        specialist=_AlwaysMissing(),
        validator=SafetyValidator(),
        budget=mandatory_budget,
        max_attempts=1,
    ).run(
        h0=h0,
        candidates=candidates,
        scope="interaction",
        priority="MANDATORY",
        fallback_h0_allowed=False,
    )
    assert mandatory.status == "PENDING_UNRESOLVED"


def test_verification_attempt_budget_fields_are_serialized_from_contract() -> None:
    h0 = _valid_h0()
    budget = VerificationBudgetManager(safety_reserve=0, optional_capacity=1)
    budget.reset("VID02")
    proposal = BoundedVerifyRepairLoop(
        specialist=_AlwaysMissing(),
        validator=SafetyValidator(),
        budget=budget,
        max_attempts=1,
    ).run(
        h0=h0,
        candidates=_pool(h0),
        scope="interaction",
        priority="OPTIONAL",
        fallback_h0_allowed=True,
    )
    sample = InferenceSample(
        video_id="VID02",
        target_frame_id=5,
        causal_frame_ids=(0, 1, 2, 3, 4, 5),
        media_refs=tuple(f"synthetic:{index}" for index in range(6)),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit_test_v1",
    )
    context = CausalPerceptionContextBuilder(
        max_frames=6,
        max_images=6,
        selection_strategy="fixed_all",
        history_image_detail="low",
        target_image_detail="auto",
    ).build(
        sample,
        torch.zeros((6, 3, 8, 8)),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )
    support = SafetySupport(
        hard_violations=(),
        soft_risks=(SoftRisk("VISUAL_AMBIGUITY", ("ivt",), 1.0),),
        gate_features={},
        legal_scopes=("interaction",),
    )
    route = RouteDecision(
        "VERIFY",
        "BENEFIT_GATE",
        "UNIT_VERIFY",
        scope="interaction",
        priority="OPTIONAL",
        fallback_h0_allowed=True,
    )

    audit = FinalStreamingPipeline._audit_payload(
        tracker_snapshot={"runtime_status": "DISABLED"},
        context=context,
        support=support,
        route=route,
        proposal=proposal,
    )

    attempt = audit["proposal"]["attempts"][0]
    assert attempt["budget_bucket"] == "OPTIONAL"
    assert attempt["budget_reason"] == "OPTIONAL_CAPACITY"


def test_repair_is_candidate_bounded_and_postchecked() -> None:
    invalid = replace(_valid_h0(), instrument_ids=(1,))
    repaired = replace(invalid, instrument_ids=(0,))
    budget = VerificationBudgetManager(safety_reserve=1, optional_capacity=0)
    budget.reset("VID02")
    proposal = BoundedVerifyRepairLoop(
        specialist=_RepairInstrument(repaired),
        validator=SafetyValidator(),
        budget=budget,
        max_attempts=1,
    ).run(
        h0=invalid,
        candidates=_pool(invalid),
        scope="interaction",
        priority="MANDATORY",
        fallback_h0_allowed=False,
    )
    assert proposal.status == "VERIFIED_REPAIR"
    assert proposal.hypothesis == repaired
    assert proposal.attempts[0].postcheck_hard_valid is True


def test_optional_repair_cannot_replace_a_hard_valid_h0() -> None:
    h0 = _valid_h0()
    proposed = replace(h0, instrument_ids=(0, 1))
    budget = VerificationBudgetManager(safety_reserve=0, optional_capacity=1)
    budget.reset("VID02")

    proposal = BoundedVerifyRepairLoop(
        specialist=_RepairInstrument(proposed),
        validator=SafetyValidator(),
        budget=budget,
        max_attempts=1,
    ).run(
        h0=h0,
        candidates=_pool(h0),
        scope="instrument_presence",
        priority="OPTIONAL",
        fallback_h0_allowed=True,
    )

    assert proposal.status == "FALLBACK_KEEP"
    assert proposal.hypothesis is h0
    assert proposal.reason == "HARD_VALID_H0_PROTECTED"
    assert proposal.attempts[0].specialist_status == "UNPROVEN_REPAIR"


def test_protected_keep_is_observed_as_zero_counterfactual_benefit() -> None:
    h0 = _valid_h0()
    sample = InferenceSample(
        video_id="VID02",
        target_frame_id=7,
        causal_frame_ids=(7,),
        media_refs=("synthetic:7",),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit_test_v1",
    )
    collected = UnlabeledCounterfactualResult(
        sample=sample,
        h0=h0,
        support=SafetySupport(
            hard_violations=(),
            soft_risks=(),
            gate_features={name: 0.0 for name in FORMAL_GATE_FEATURE_ORDER},
            legal_scopes=("workflow",),
        ),
        proposals=(
            (
                "workflow",
                RepairProposal(
                    "FALLBACK_KEEP",
                    h0,
                    "HARD_VALID_H0_PROTECTED",
                ),
            ),
        ),
        tracker_artifact_sha256="a" * 64,
    )
    target = FrameSupervisionTarget(
        video_id="VID02",
        frame_id=7,
        instrument_ids=(),
        verb_ids=(),
        target_ids=(),
        triplet_ids=(),
        phase_id=1,
        mask=FrameTaskMask(False, False, False, False, True),
        source_granularity="phase_only",
        source="unit_test",
    )

    result = label_counterfactual(collected, target)

    assert result.scope_results[0].verified_utility == 0.0
    assert result.scope_results[0].benefit_label == 0
    assert result.record is not None
    assert result.record.benefit_by_scope["workflow"] == 0


def test_atomic_finalization_keeps_pending_out_of_trusted_memory(tmp_path) -> None:
    store_path = tmp_path / "VID02.finalization.json"
    store = AtomicFinalizationStore(state_path=store_path)
    store.reset("VID02")
    h0 = _valid_h0()
    finalizer = OutcomeFinalizer()
    pending = finalizer.finalize(
        h0=h0,
        proposal=RepairProposal("PENDING_UNRESOLVED", h0, "NO_BUDGET"),
    )
    store.commit(
        observation=ObservationIdentity("VID02", "seg-0", 1, 1.0),
        outcome=pending,
        pending_scope="interaction",
    )
    snapshot = store.snapshot()
    assert snapshot.trusted == ()
    assert len(snapshot.pending) == 1

    accepted = finalizer.finalize(
        h0=h0,
        proposal=RepairProposal("GATE_ACCEPTED", h0, "GATE_SELECTED_USE_H0"),
    )
    store.commit(
        observation=ObservationIdentity("VID02", "seg-0", 2, 2.0),
        outcome=accepted,
    )
    assert len(store.snapshot().accepted) == 1

    recovered = AtomicFinalizationStore(state_path=store_path)
    recovered.reset("VID02", recover=True)
    assert len(recovered.snapshot().pending) == 1
    assert len(recovered.snapshot().accepted) == 1
    assert recovered.prior_trusted_hypothesis == h0


def test_atomic_store_uses_one_resume_file_per_video(tmp_path) -> None:
    store = AtomicFinalizationStore(state_dir=tmp_path)
    first = ObservationIdentity("VID01", "segment-0", 1, 1.0)
    second = ObservationIdentity("VID02", "segment-0", 1, 1.0)

    store.reset("VID01")
    store.commit(
        observation=first,
        outcome=FinalOutcome("Pending", None, "TEST_PENDING"),
    )
    store.reset("VID02")
    store.commit(
        observation=second,
        outcome=FinalOutcome("Pending", None, "TEST_PENDING"),
    )

    assert (tmp_path / "VID01.json").is_file()
    assert (tmp_path / "VID02.json").is_file()
    store.reset("VID01")
    assert store.snapshot().pending[0].observation == first


def test_final_artifact_materializes_committed_audit(tmp_path) -> None:
    store = AtomicFinalizationStore()
    observation = ObservationIdentity("VID01", "segment-0", 1, 1.0)
    store.reset("VID01")
    record = store.commit(
        observation=observation,
        outcome=FinalOutcome("Pending", _valid_h0(), "TEST_PENDING"),
        audit={"route": {"kind": "PENDING"}},
    )
    writer = FinalPipelineArtifactWriter(
        tmp_path / "run",
        run_id="formal_test",
        cell="A_base",
        initial_model_requested="openai/gpt-5.6-sol",
        verification_model_requested="google/gemini-3.1-pro-preview",
    )
    writer.begin()
    writer.write(record)

    manifest_path = writer.finalize()

    assert manifest_path.is_file()
    content = (tmp_path / "run" / "final_records" / "VID01.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"route":{"kind":"PENDING"}' in content
    assert '"state":"Pending"' in content
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["initial_model_requested"] == "openai/gpt-5.6-sol"
    assert (
        manifest["verification_model_requested"]
        == "google/gemini-3.1-pro-preview"
    )


def test_final_artifact_replaces_pending_with_its_atomic_resolution(tmp_path) -> None:
    store = AtomicFinalizationStore(max_resolution_attempts=1)
    pending_observation = ObservationIdentity("VID01", "segment-0", 1, 1.0)
    store.reset("VID01")
    pending = store.commit(
        observation=pending_observation,
        outcome=FinalOutcome("Pending", _valid_h0(), "TEST_PENDING"),
        pending_scope="interaction",
    )
    writer = FinalPipelineArtifactWriter(
        tmp_path / "run",
        run_id="pending_resolution_test",
        cell="A_base",
    )
    writer.begin()
    writer.write(pending)
    store.commit(
        observation=ObservationIdentity("VID01", "segment-0", 2, 2.0),
        outcome=FinalOutcome("Accepted", _valid_h0(), "TEST_ACCEPTED"),
    )
    resolved = store.resolve_pending(
        pending_observation.key,
        checked_t=2.0,
        outcome=FinalOutcome("Verified", _valid_h0(), "NEW_CAUSAL_EVIDENCE"),
    )

    writer.write(resolved)
    writer.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "final_records" / "VID01.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["outcome"]["state"] == "Verified"
    assert rows[0]["audit"]["pending_resolution"]["original_state"] == "Pending"


class _FixedPerception:
    def __init__(self, prediction: InitialPrediction) -> None:
        rankings = {
            "instrument": (RankedCandidate(0, 0.9), RankedCandidate(1, 0.1)),
            "verb": (RankedCandidate(2, 0.9), RankedCandidate(3, 0.1)),
            "target": (RankedCandidate(1, 0.9), RankedCandidate(2, 0.1)),
            "ivt": (RankedCandidate(0, 0.9), RankedCandidate(1, 0.1)),
            "phase": (RankedCandidate(0, 0.9), RankedCandidate(1, 0.1)),
        }
        self.result = JointPerceptionResult(
            prediction=prediction,
            raw_evidence=PerceptionEvidence(
                source="unit_test",
                ranked_candidates=rankings,
                self_reported_confidence={task: 0.9 for task in TASK_CLASS_COUNTS},
                evidence_refs=(),
                source_max_frame_id=5,
            ),
            api_provenance=ApiCallProvenance.local(),
        )

    def predict(self, _: object) -> JointPerceptionResult:
        return self.result


class _CurrentTrackProvider:
    def reset(self, video_id: str) -> None:
        self.video_id = video_id

    def snapshot(self, sample: InferenceSample):
        assert sample.video_id == self.video_id
        return {
            "status": "AVAILABLE",
            "source_max_frame_id": sample.target_frame_id,
            "frames": (
                {
                    "frame_id": sample.target_frame_id,
                    "tracks": (
                        {
                            "track_id": "track-1",
                            "instrument_id": 0,
                            "bbox_tlwh": (0.1, 0.1, 0.2, 0.2),
                            "score": 0.9,
                            "age": 1,
                        },
                    ),
                },
            ),
        }


class _AlwaysKeep:
    def verify(self, **_: object) -> SpecialistResult:
        return SpecialistResult("KEEP", reason="UNIT_KEEP")


class _ConcurrentKeep:
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier

    def verify(self, **_: object) -> SpecialistResult:
        self.barrier.wait(timeout=2.0)
        return SpecialistResult("KEEP", reason="UNIT_CONCURRENT_KEEP")


class _ResolveVerified:
    def __init__(self, hypothesis: InitialPrediction) -> None:
        self.hypothesis = hypothesis

    def resolve(self, **_: object) -> PendingResolutionResult:
        return PendingResolutionResult(
            "VERIFIED",
            self.hypothesis,
            "NEW_CAUSAL_EVIDENCE",
        )


def test_executable_pipeline_uses_fixed_images_and_commits_accepted(tmp_path) -> None:
    h0 = _valid_h0()
    budget = VerificationBudgetManager(safety_reserve=1, optional_capacity=1)
    store = AtomicFinalizationStore(state_path=tmp_path / "state.json")
    components = FinalPipelineComponents(
        context_builder=CausalPerceptionContextBuilder(
            max_frames=6,
            max_images=6,
            selection_strategy="fixed_all",
            history_image_detail="low",
            target_image_detail="auto",
        ),
        perception=_FixedPerception(h0),
        candidate_generator=FactorizedCandidateGenerator(max_candidates_per_task=2),
        signal_extractor=FrameEvidenceSignalExtractor(),
        safety_validator=SafetyValidator(),
        support_builder=DecisionSupportBuilder(),
        mandatory_guard=MandatorySafetyGuard(),
        benefit_gate=RuleBenefitGate(risk_threshold=0.5),
        budget=budget,
        specialist_factory=lambda *_: _AlwaysMissing(),
        outcome_finalizer=OutcomeFinalizer(),
        finalization_store=store,
    )
    pipeline = FinalStreamingPipeline(components)
    sample = InferenceSample(
        video_id="VID02",
        target_frame_id=5,
        causal_frame_ids=(0, 1, 2, 3, 4, 5),
        media_refs=tuple(f"synthetic:{index}" for index in range(6)),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit_test_v1",
    )
    result = pipeline.run(sample, torch.zeros((6, 3, 8, 8)))

    assert getattr(result.outcome, "state", None) == "Accepted", result.outcome
    assert result.context is not None
    assert result.context.selected_image_frame_ids == (0, 1, 2, 3, 4, 5)
    assert result.route is not None and result.route.kind == "USE_H0"
    assert len(store.snapshot().accepted) == 1


def test_formal_counterfactual_observes_each_scope_without_committing_gt() -> None:
    h0 = _valid_h0()
    components = FinalPipelineComponents(
        context_builder=CausalPerceptionContextBuilder(
            max_frames=6,
            max_images=6,
            selection_strategy="fixed_all",
            history_image_detail="low",
            target_image_detail="auto",
        ),
        perception=_FixedPerception(h0),
        candidate_generator=FactorizedCandidateGenerator(max_candidates_per_task=2),
        signal_extractor=FrameEvidenceSignalExtractor(),
        safety_validator=SafetyValidator(),
        support_builder=DecisionSupportBuilder(),
        mandatory_guard=MandatorySafetyGuard(),
        benefit_gate=RuleBenefitGate(risk_threshold=0.5),
        budget=VerificationBudgetManager(safety_reserve=1, optional_capacity=1),
        specialist_factory=lambda *_: _AlwaysKeep(),
        outcome_finalizer=OutcomeFinalizer(),
        finalization_store=AtomicFinalizationStore(),
        track_provider=_CurrentTrackProvider(),
    )
    sample = InferenceSample(
        video_id="VID02",
        target_frame_id=5,
        causal_frame_ids=(0, 1, 2, 3, 4, 5),
        media_refs=tuple(f"synthetic:{index}" for index in range(6)),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit_test_v1",
    )
    target = FrameSupervisionTarget(
        video_id="VID02",
        frame_id=5,
        instrument_ids=h0.instrument_ids,
        verb_ids=h0.verb_ids,
        target_ids=h0.target_ids,
        triplet_ids=h0.triplet_ids,
        phase_id=h0.phase_id,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit_test",
    )
    collector = FormalGateCounterfactualCollector(
        pipeline=FinalStreamingPipeline(components),
        tracker_artifact_sha256="a" * 64,
        max_provider_attempts=3,
    )

    collected = collector.collect(sample, torch.zeros((6, 3, 8, 8)))
    result = label_counterfactual(collected, target)

    assert result.record is not None
    assert dict(result.record.benefit_by_scope) == {
        "instrument_presence": 0,
        "interaction": 0,
        "workflow": 0,
    }
    assert len(result.scope_results) == 3
    assert components.finalization_store.snapshot().trusted == ()


def test_formal_counterfactual_runs_independent_scopes_concurrently() -> None:
    h0 = _valid_h0()
    barrier = Barrier(3)
    components = FinalPipelineComponents(
        context_builder=CausalPerceptionContextBuilder(
            max_frames=6,
            max_images=6,
            selection_strategy="fixed_all",
        ),
        perception=_FixedPerception(h0),
        candidate_generator=FactorizedCandidateGenerator(max_candidates_per_task=2),
        signal_extractor=FrameEvidenceSignalExtractor(),
        safety_validator=SafetyValidator(),
        support_builder=DecisionSupportBuilder(),
        mandatory_guard=MandatorySafetyGuard(),
        benefit_gate=RuleBenefitGate(risk_threshold=0.5),
        budget=VerificationBudgetManager(safety_reserve=1, optional_capacity=1),
        specialist_factory=lambda *_: _ConcurrentKeep(barrier),
        outcome_finalizer=OutcomeFinalizer(),
        finalization_store=AtomicFinalizationStore(),
        track_provider=_CurrentTrackProvider(),
    )
    sample = InferenceSample(
        video_id="VID02",
        target_frame_id=5,
        causal_frame_ids=(0, 1, 2, 3, 4, 5),
        media_refs=tuple(f"synthetic:{index}" for index in range(6)),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit_test_v1",
    )
    collector = FormalGateCounterfactualCollector(
        pipeline=FinalStreamingPipeline(components),
        tracker_artifact_sha256="a" * 64,
        max_provider_attempts=3,
    )

    collected = collector.collect(sample, torch.zeros((6, 3, 8, 8)))

    assert tuple(scope for scope, _proposal in collected.proposals) == (
        "instrument_presence",
        "interaction",
        "workflow",
    )
    assert all(
        proposal.status == "VERIFIED_KEEP"
        for _scope, proposal in collected.proposals
    )


def test_pending_resolution_is_bounded_postchecked_and_promoted_atomically() -> None:
    store = AtomicFinalizationStore(max_resolution_attempts=1)
    h0 = _valid_h0()
    store.reset("VID02")
    pending_observation = ObservationIdentity("VID02", "segment-0", 1, 1.0)
    store.commit(
        observation=pending_observation,
        outcome=FinalOutcome("Pending", h0, "TEST_PENDING"),
        pending_scope="interaction",
    )
    now = ObservationIdentity("VID02", "segment-0", 3, 3.0)
    store.commit(
        observation=now,
        outcome=FinalOutcome("Accepted", h0, "GATE_ACCEPTED"),
    )
    resolver = BoundedPendingResolver(
        backend=_ResolveVerified(h0),
        validator=SafetyValidator(),
    )

    resolved = resolver.resolve_eligible(now=now, store=store)

    assert len(resolved) == 1
    assert resolved[0].outcome.state == "Verified"
    assert store.snapshot().pending == ()
    assert store.snapshot().verified[0].observation == pending_observation


def test_temporal_events_and_report_preserve_reliability_wording() -> None:
    h0 = _valid_h0()
    records = (
        FinalizationRecord(
            ObservationIdentity("VID01", "segment-0", 1, 1.0),
            FinalOutcome("Accepted", h0, "GATE_ACCEPTED"),
        ),
        FinalizationRecord(
            ObservationIdentity("VID01", "segment-0", 2, 2.0),
            FinalOutcome("Verified", h0, "VERIFIED_KEEP"),
        ),
        FinalizationRecord(
            ObservationIdentity("VID02", "segment-0", 1, 1.0),
            FinalOutcome("Pending", h0, "PENDING_UNRESOLVED"),
        ),
    )

    events = TemporalEventAggregator().aggregate(records)
    report = ReliabilityAwareTemplateReporter().render(events)

    assert len(events) == 2
    assert events[0].reliability == "DEFINITE"
    assert events[1].reliability == "UNCERTAIN"
    assert [item["reliability"] for item in report["events"]] == [
        "DEFINITE",
        "UNCERTAIN",
    ]
