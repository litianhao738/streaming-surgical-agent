import json
from dataclasses import replace

from surgical_agent.api.errors import ApiCallFailure, ApiTransportError
from surgical_agent.api.request_hash import canonical_request_hash
from surgical_agent.config.final_experiment import load_tracker_gate_cell
from surgical_agent.config.loader import load_api_config
from surgical_agent.research.gate.budget import VerificationBudgetManager
from surgical_agent.research.gate.contracts import SafetySupport
from surgical_agent.research.outcome import OutcomeFinalizer
from surgical_agent.research.safety import (
    DecisionSupportBuilder,
    MandatorySafetyGuard,
    SafetyValidator,
    legal_repair_scopes,
)
from surgical_agent.research.verification.coverage import run_scope_coverage
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.research.verification.repair import (
    BoundedVerifyRepairLoop,
    RepairEvidence,
    SpecialistResult,
)
from surgical_agent.research.verification.targeted_api import (
    TargetedApiVerifier,
    TargetedVerificationRequestBuilder,
)
from surgical_agent.research.verification.verifier import BoundTargetedSpecialist
from surgical_agent.systems.final_pipeline_factory import (
    build_final_api_pipeline,
    validate_verifier_pair,
)
from tests.unit.test_final_pipeline_contracts import _pool, _RepairInstrument, _valid_h0
from tests.unit.test_targeted_api_verifier import (
    _context,
    _evidence,
    _perception,
    _prediction,
    _response,
)


class ReviewClient:
    def __init__(self, selections):
        self.selections = iter(selections)
        self.requests = []

    def call(self, request):
        self.requests.append(request)
        selections = next(self.selections)
        payload = json.loads(request.payload["input_text"])
        fields = []
        for path in payload["flagged_fields"]:
            task = path.split("/")[1]
            values = selections[task]
            fields.append(
                {
                    "path": path,
                    "selected_ids": list(values),
                    "topk": [
                        {"id": value, "confidence": 0.9 - i * 0.1}
                        for i, value in enumerate(values or (0,))
                    ],
                    "status": "Verified",
                    "uncertainty": None,
                }
            )
        return replace(
            _response({"schema_version": "targeted_verification_v1", "fields": fields}),
            request_hash=canonical_request_hash(request),
        )


def _specialist(h0, selections):
    perception = replace(_perception(), prediction=h0)
    pool = FactorizedCandidateGenerator(complete_ontology=True).build(perception)
    client = ReviewClient(selections)
    config = load_api_config(
        "configs/perception/targeted_openrouter_astra_v9_fixed3.yaml"
    )
    bound = BoundTargetedSpecialist(
        verifier=TargetedApiVerifier(
            client=client,
            request_builder=TargetedVerificationRequestBuilder(config=config),
        ),
        context=_context(),
        candidates=pool,
        evidence=_evidence(),
    )
    return bound, pool, client


def test_complete_pool_is_independent_of_wrong_h0_phase_and_topk():
    pool = FactorizedCandidateGenerator(complete_ontology=True).build(_perception())
    assert pool.allowed_ids["ivt"] == tuple(range(100))
    assert pool.allowed_ids["phase"] == tuple(range(7))
    assert pool.admits(
        _prediction(
            instrument_ids=(0,),
            verb_ids=(9,),
            target_ids=(14,),
            triplet_ids=(94,),
            phase_id=6,
        )
    )


def test_astra_v9_configs_construct_full_coverage_pipeline(tmp_path):
    initial = load_api_config("configs/perception/joint_openrouter_astra_fixed3.yaml")
    verification = load_api_config(
        "configs/perception/targeted_openrouter_astra_v9_fixed3.yaml"
    )
    validate_verifier_pair(initial, verification)
    cell = load_tracker_gate_cell("configs/ablations/a_base.yaml")
    pipeline = build_final_api_pipeline(
        client=ReviewClient([]),
        api_config=initial,
        verification_api_config=verification,
        cell=cell,
        state_dir=tmp_path,
        phase_ivt_support={0: (94,)},
    )
    assert pipeline.components.max_coverage_scopes == 3
    assert pipeline.components.candidate_generator.complete_ontology
    assert pipeline.components.support_builder.phase_ivt_support == {0: (94,)}


def test_visual_reviews_are_blind_distinct_and_can_repair_hard_valid_phase():
    h0 = replace(_valid_h0(), phase_id=4)
    bound, pool, client = _specialist(h0, [{"phase": (0,)}, {"phase": (0,)}])
    budget = VerificationBudgetManager(optional_capacity=1, safety_reserve=0)
    budget.reset("VID01")
    proposal = BoundedVerifyRepairLoop(
        specialist=bound,
        validator=SafetyValidator(),
        budget=budget,
        require_repair_evidence=True,
    ).run(
        h0=h0,
        candidates=pool,
        scope="workflow",
        priority="OPTIONAL",
        fallback_h0_allowed=True,
    )
    assert proposal.hypothesis.phase_id == 0
    assert proposal.attempts[0].accepted
    assert len(client.requests) == 2
    assert canonical_request_hash(client.requests[0]) != canonical_request_hash(
        client.requests[1]
    )
    for request in client.requests:
        data = json.loads(request.payload["input_text"])
        assert "current_fields" not in data and "workflow_summary" not in data
    outcome = OutcomeFinalizer().finalize(h0=h0, proposal=proposal)
    assert outcome.verified_tasks == ("phase",) and outcome.state == "Accepted"


def test_visual_disagreement_cannot_generate_repair_certificate():
    bound, _, _ = _specialist(_valid_h0(), [{"phase": (0,)}, {"phase": (1,)}])
    result = bound.verify(hypothesis=_valid_h0(), scope="workflow", attempt=1)
    assert result.status == "UNRESOLVED"
    assert result.reason == "INDEPENDENT_REVIEWS_DISAGREE"
    assert result.repair_evidence is None


def test_unmatched_association_is_not_silently_replaced_by_null():
    wrong = {"instrument": (0,), "verb": (1,), "target": (13,), "ivt": (94,)}
    bound, _, _ = _specialist(_valid_h0(), [wrong, wrong])
    result = bound.verify(hypothesis=_valid_h0(), scope="interaction", attempt=1)
    assert result.status == "UNRESOLVED"
    assert result.reason == "VISUAL_ASSOCIATION_NOT_CLOSED"


def test_agreed_explicit_null_interaction_is_admissible():
    null = {"instrument": (0,), "verb": (9,), "target": (14,), "ivt": (94,)}
    bound, _, _ = _specialist(_valid_h0(), [null, null])
    result = bound.verify(hypothesis=_valid_h0(), scope="interaction", attempt=1)
    assert result.status == "REPAIR"
    assert result.hypothesis.triplet_ids == (94,)
    assert result.repair_evidence.kind == "VISUAL_INTERACTION_AGREEMENT"


def test_hard_invalid_h0_does_not_waive_v9_evidence_requirement():
    invalid = replace(_valid_h0(), instrument_ids=(1,))
    budget = VerificationBudgetManager(safety_reserve=1, optional_capacity=0)
    budget.reset("SYNTHETIC")
    proposal = BoundedVerifyRepairLoop(
        specialist=_RepairInstrument(_valid_h0()),
        validator=SafetyValidator(),
        budget=budget,
        require_repair_evidence=True,
    ).run(
        h0=invalid,
        candidates=_pool(invalid),
        scope="interaction",
        priority="MANDATORY",
        fallback_h0_allowed=False,
    )
    assert proposal.status == "PENDING_UNRESOLVED"
    assert not proposal.attempts[0].accepted


def test_coverage_continues_after_success_and_scoped_certificates_bind_final_labels():
    h0 = replace(_valid_h0(), instrument_ids=(0, 1), phase_id=1)

    class Specialist:
        def verify(self, *, hypothesis, scope, attempt):
            if scope == "instrument_presence":
                proposed, kind = (
                    replace(hypothesis, instrument_ids=(0,)),
                    "VISUAL_INSTRUMENT_AGREEMENT",
                )
            elif scope == "interaction":
                proposed, kind = (
                    replace(
                        hypothesis,
                        instrument_ids=(0,),
                        verb_ids=(9,),
                        target_ids=(14,),
                        triplet_ids=(94,),
                    ),
                    "VISUAL_INTERACTION_AGREEMENT",
                )
            else:
                proposed, kind = (
                    replace(hypothesis, phase_id=0),
                    "VISUAL_PHASE_AGREEMENT",
                )
            return SpecialistResult(
                "REPAIR",
                proposed,
                repair_evidence=RepairEvidence(kind, ("api:review:a", "api:review:b")),
            )

    pool = FactorizedCandidateGenerator(complete_ontology=True).build(
        replace(_perception(), prediction=h0)
    )
    budget = VerificationBudgetManager(safety_reserve=0, optional_capacity=3)
    budget.reset("SYNTHETIC")
    result = run_scope_coverage(
        h0=h0,
        candidates=pool,
        initial_scope="instrument_presence",
        specialist=Specialist(),
        validator=SafetyValidator(),
        budget=budget,
    )
    assert [item.scope for item in result.attempts] == [
        "instrument_presence",
        "interaction",
        "workflow",
    ]
    assert result.hypothesis.phase_id == 0 and result.hypothesis.triplet_ids == (94,)
    outcome = OutcomeFinalizer().finalize(h0=h0, proposal=result)
    assert outcome.state == "Verified"
    # An old certificate for a different final value cannot verify that value.
    altered = replace(result, hypothesis=replace(result.hypothesis, phase_id=1))
    assert (
        "phase"
        not in OutcomeFinalizer().finalize(h0=h0, proposal=altered).verified_tasks
    )


def test_phase_ivt_relation_routes_to_either_repairable_endpoint():
    h0 = _valid_h0()
    validator = SafetyValidator(strict_phase_allowed_ivt={0: (), 1: (0,)})
    violations = validator.validate(h0)
    legal = legal_repair_scopes(violations, _pool(h0))
    assert "workflow" in legal and "interaction" in legal
    support = SafetySupport(
        hard_violations=violations, soft_risks=(), gate_features={}, legal_scopes=legal
    )
    assert MandatorySafetyGuard().route(support).kind == "VERIFY"
    assert not validator.validate(replace(h0, phase_id=1))


def test_unobserved_training_pair_is_soft_not_a_hard_prohibition():
    h0 = _valid_h0()
    perception = replace(_perception(), prediction=h0)
    pool = FactorizedCandidateGenerator(complete_ontology=True).build(perception)
    support = DecisionSupportBuilder(phase_ivt_support={0: (94,)}).build(
        perception=perception,
        candidates=pool,
        violations=(),
        tracker_snapshot={},
        previous=None,
    )
    assert support.safety_class == "HARD_VALID"
    assert any(risk.code == "PHASE_IVT_UNOBSERVED" for risk in support.soft_risks)


def test_content_moderation_stops_scope_coverage_and_preserves_safe_error():
    h0 = _valid_h0()
    bound, pool, client = _specialist(h0, [])

    def refused(request):
        client.requests.append(request)
        raise ApiCallFailure(
            ApiTransportError(
                "", code="content_moderation", retryable=False, status_code=403
            ),
            attempt_count=1,
            retry_count=0,
        )

    client.call = refused
    budget = VerificationBudgetManager(safety_reserve=0, optional_capacity=3)
    budget.reset("SYNTHETIC")
    result = run_scope_coverage(
        h0=h0,
        candidates=pool,
        initial_scope="instrument_presence",
        specialist=bound,
        validator=SafetyValidator(),
        budget=budget,
    )
    assert result.status == "FALLBACK_KEEP"
    assert len(result.attempts) == len(client.requests) == 1
    assert result.attempts[0].specialist_reason == "content_moderation"


def test_partial_review_failure_is_visible_in_frame_reliability():
    h0 = _valid_h0()

    class Specialist:
        def verify(self, *, hypothesis, scope, attempt):
            if scope == "instrument_presence":
                return SpecialistResult(
                    "KEEP",
                    repair_evidence=RepairEvidence(
                        "VISUAL_INSTRUMENT_AGREEMENT", ("api:review:a", "api:review:b")
                    ),
                )
            return SpecialistResult("UNRESOLVED", reason="VISUAL_REVIEW_UNCERTAIN")

    pool = FactorizedCandidateGenerator(complete_ontology=True).build(
        replace(_perception(), prediction=h0)
    )
    budget = VerificationBudgetManager(safety_reserve=0, optional_capacity=3)
    budget.reset("SYNTHETIC")
    result = run_scope_coverage(
        h0=h0,
        candidates=pool,
        initial_scope="instrument_presence",
        specialist=Specialist(),
        validator=SafetyValidator(),
        budget=budget,
    )
    outcome = OutcomeFinalizer().finalize(h0=h0, proposal=result)
    assert outcome.state == "Accepted" and outcome.lower_reliability
    assert outcome.verified_tasks == ("instrument",)
