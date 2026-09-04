# Streaming Surgical Agent — Pipeline Architecture, Pseudocode, and Implementation Notes

> Scope: current rigorous pipeline only.
> This document intentionally excludes experiment tables, historical discussion, and non-essential engineering details.

---

# 1. Overall Architecture

```text
Causal 6-Frame Stream ≤ t
        │
        ▼
Tracker Evidence
        │
        ▼
Joint Prediction H0
        │
        ▼
Deterministic Safety Coordinator
   ┌───────────────┴───────────────┐
   │                               │
Hard Invalid                    Hard Valid
   │                               │
   ▼                               ▼
Mandatory Verify              Benefit Gate
   │                         ┌──────┴──────┐
   │                         │             │
   │                      Accept         Verify
   │                         │             │
   │                         │             ▼
   └─────────────────────────┼──────► Targeted Verify / Repair
                             │             │
                             │             ▼
                             │   Post-Verification Safety Check
                             │         │              ▲
                             │         │ fail         │
                             │         └──────────────┘
                             │           bounded loop
                             │
                             ▼
                      Outcome Finalizer
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
          Accepted        Verified         Pending
             │               │                │
             ▼               ▼                ▼
        Short Memory    Reliable Memory   Pending Buffer
                                              │
                                              ▼
                                      Future causal evidence
                                              │
                                              └──────↺
                                         Pending Resolution

                             │
                             ▼
                  Temporal Event Aggregator
                             │
                             ▼
                Reliability-Aware Report
```

---

# 2. Architecture Description

The pipeline processes a surgical video causally at time `t`.

Each prediction uses only the current frame and historical frames:

```text
[t-5, t-4, t-3, t-2, t-1, t]
```

No future frame is allowed.

The main logic is:

```text
Joint Prediction
→ Deterministic Hard Safety Check
→ Mandatory Verification or Benefit Gate
→ Optional Targeted Verification
→ Post-Verification Safety Check
→ Outcome Finalization
→ Reliability-Aware Memory
→ Deferred Pending Resolution
```

The main responsibilities are deliberately separated:

```text
Safety Coordinator
→ determines whether H0 contains hard structural/semantic violations.

Benefit Gate
→ determines whether an already hard-valid H0 is worth additional verification cost.

Specialist
→ verifies only the flagged task/field.

Post-Verification Safety Check
→ ensures KEEP/REPAIR results remain structurally valid.

Outcome Finalizer
→ assigns the single authoritative final semantic state.

Memory
→ stores outputs according to reliability.

Pending Resolver
→ revisits unresolved findings only when new causal evidence arrives.
```

---

# 3. Semantic States

Only the following semantic states should be used:

```text
Candidate
Accepted
Verified
Pending
Rejected
```

Definitions:

```text
Candidate
= raw JointPerception result H0 before final routing.

Accepted
= hard-valid result used without successful Specialist verification.

Verified
= Specialist verification succeeded and the resulting hypothesis passed postcheck.

Pending
= unresolved semantic result that must be stored separately for future resolution.

Rejected
= a valid semantic candidate was explicitly disproved by verification.
```

Execution errors such as API timeout or invalid JSON should not be called `Rejected`.

Recommended execution-only statuses:

```text
EXECUTION_ERROR
INVALID_RESPONSE
PENDING_NO_H0
POLICY_FAILURE
```

---

# 4. Causal Context Builder

## 4.1 Purpose

Construct a fixed causal window for time `t`.

The visual input must not depend on Tracker ON/OFF.

No adaptive keyframe selection is used.

## 4.2 Pseudocode

```python
def build_causal_context(observation, segment_buffer):

    if segment_buffer.is_new_segment(observation):
        segment_buffer.reset()

    frames = segment_buffer.get_latest_contiguous_frames(
        current_frame=observation.frame,
        max_frames=6,
    )

    assert all(frame.time <= observation.time for frame in frames)
    assert not crosses_segment_boundary(frames)

    return CausalContext(
        frames=frames,
        current_frame=observation.frame,
    )
```

Recommended API detail:

```text
Historical 5 frames → detail=low
Current frame t      → detail=auto
```

---

# 5. Tracker Module

## 5.1 Purpose

Tracker provides only local reliability evidence.

It does not:

```text
select frames
predict Verb
predict Target
predict IVT
predict Phase
```

It can provide:

```text
instrument presence
track continuity
new/disappeared instruments
GPT-vs-Tracker support
GPT-vs-Tracker conflict
```

## 5.2 Pseudocode

```python
def run_tracker(observation, tracker_enabled):

    if not tracker_enabled:
        return TrackSnapshot(
            available=False,
            status="DISABLED_BY_ABLATION",
            tracks=[],
            mask=0,
        )

    try:
        result = tracker.predict(
            frames_up_to_t=observation.frames
        )

        return TrackSnapshot(
            available=True,
            status="OK",
            tracks=result.tracks,
        )

    except Exception as error:
        # Never silently reuse an older snapshot.
        return TrackSnapshot(
            available=False,
            status="ERROR",
            tracks=[],
            error_type=type(error).__name__,
        )
```

---

# 6. Joint Perception

## 6.1 Purpose

Perform one structured multimodal prediction for the current time `t`.

Output:

```text
Instrument
Verb
Target
IVT
Phase
Uncertainty
```

Initial state:

```text
Candidate
```

## 6.2 Pseudocode

```python
def joint_perception(context):

    response = call_vlm(
        images=context.frames,
        tasks=[
            "instrument",
            "verb",
            "target",
            "ivt",
            "phase",
            "uncertainty",
        ],
    )

    if response.timeout:
        return PerceptionResult(
            status="NO_RESPONSE"
        )

    if not valid_schema(response):
        return PerceptionResult(
            status="INVALID_SCHEMA"
        )

    H0 = parse_hypothesis(response)

    return PerceptionResult(
        status="SUCCESS",
        hypothesis=H0,
        ranked_candidates=response.candidates,
    )
```

## 6.3 Frozen Candidate and blind Verify contract

Candidate construction is deterministic and finishes before Gate routing:

```text
Instrument / Verb / Target / IVT
    = H0 labels + H0-IVT closure components + remaining perception ranks
Phase
    = all seven ontology IDs
maximum candidates per requested field
    = 8
```

The small Phase ontology is complete so a correct workflow repair cannot become
impossible merely because the initial Phase fell outside perception top-3.
Instrument, Verb and Target pools retain any component required by a selected
IVT; IVT retains up to all eight compact perception candidates.

Formal targeted Verify uses `blind_candidate_selection_v1` and prompt version
`targeted_verification_prompt_v6`. It receives the fixed candidate IDs but not
H0 selected values or upstream rank scores. Its ontology appendix is scope-local:
Instrument and Workflow omit the 100-row IVT table, while Interaction receives
only the IVT rows in its frozen candidate pool. All returned semantic labels describe
`target_frame_id`; earlier images provide causal context and must never be unioned
into the target-frame label set. If an unrequested IVT remains frozen, the request
exposes only its required component IDs as deterministic closure constraints.
Tracker, Workflow and Memory remain absent from the formal fixed-visual Verify
payload, preserving the Tracker x Gate ablation boundary.

During Training-only Gate counterfactual collection, all legal scopes branch from
the same immutable pre-decision snapshot and execute concurrently; results are
serialized in canonical scope order. This is a data-generation optimization only.
Online inference still executes at most the single scope selected by Gate.

---

# 7. Deterministic Safety Coordinator

This module merges:

```text
DecisionSupportBuilder
+
SafetyValidator.precheck
```

## 7.1 Purpose

Separate hard invalidity from soft uncertainty.

### Hard checks

Use only deterministic constraints that truly indicate structural invalidity:

```text
schema/ontology validity
invalid label IDs
invalid IVT closure
explicitly illegal triplet combinations
strict impossible phase constraints
```

### Soft risks

These should normally become Gate features rather than hard invalidity:

```text
low confidence
explicit uncertainty
temporal jump
rare phase-transition pattern
Tracker disagreement
Tracker missing-tool evidence
```

## 7.2 Pseudocode

```python
def deterministic_safety_coordinator(
    H0,
    candidates,
    tracker,
    previous_state,
    ontology,
):

    hard_violations = []
    soft_risks = []

    # Hard constraints
    hard_violations += check_ontology(
        H0,
        ontology,
    )

    hard_violations += check_ivt_closure(
        H0,
        ontology,
    )

    hard_violations += check_triplet_compatibility(
        H0,
        ontology,
    )

    hard_violations += check_strict_phase_constraints(
        H0,
        ontology,
    )

    # Soft reliability evidence
    soft_risks += confidence_features(H0)
    soft_risks += uncertainty_features(H0)
    soft_risks += temporal_jump_features(
        previous_state,
        H0,
    )

    if tracker.available:
        soft_risks += tracker_conflict_features(
            H0,
            tracker,
        )

    legal_scopes = build_legal_repair_scopes(
        H0,
        candidates,
        hard_violations,
    )

    safety_class = (
        "HARD_INVALID"
        if hard_violations
        else "HARD_VALID"
    )

    gate_features = build_gate_features(
        H0,
        soft_risks,
        tracker,
    )

    return SafetySupport(
        safety_class=safety_class,
        hard_violations=hard_violations,
        soft_risks=soft_risks,
        gate_features=gate_features,
        legal_scopes=legal_scopes,
    )
```

---

# 8. Mandatory Safety Guard

## 8.1 Purpose

Hard-invalid H0 must never be directly accepted by the Gate.

The Benefit Gate only handles hard-valid H0.

## 8.2 Pseudocode

```python
def route_after_safety(H0, support):

    if support.safety_class == "HARD_INVALID":

        scope = find_one_scope_covering_all_violations(
            support.hard_violations,
            support.legal_scopes,
        )

        if scope is None:
            return Pending(
                reason="NO_LEGAL_REPAIR_SCOPE"
            )

        return MandatoryVerify(
            scope=scope
        )

    assert support.safety_class == "HARD_VALID"

    return SendToGate()
```

---

# 9. Benefit Gate

## 9.1 Purpose

The Gate does not decide whether the surgical prediction is correct.

It decides:

> Is additional verification likely to improve the final result enough to justify its cost?

The Gate only processes `HARD_VALID` hypotheses.

## 9.2 Pseudocode

```python
def gate_policy(
    support,
    gate_mode,
):

    assert support.safety_class == "HARD_VALID"

    if gate_mode == "RULE":

        return rule_gate(
            support.gate_features,
            support.legal_scopes,
        )

    if gate_mode == "LEARNED":

        p_benefit = learned_gate.predict_proba(
            support.gate_features
        )

        if p_benefit >= threshold:

            scope = choose_verification_scope(
                support
            )

            return RequestVerify(scope)

        return UseH0()
```

---

# 10. Gate Ablation Definition

Use the following four groups:

```text
A_base
Tracker OFF
Rule Gate

B_tracker
Tracker ON
Rule Gate

C_gate
Tracker OFF
Learned Benefit Gate

D_full
Tracker ON
Learned Benefit Gate
```

Mandatory Safety Guard remains identical in all four groups.

Only the optional hard-valid routing policy changes.

---

# 11. Unified Verification Budget Manager

The previous:

```text
Safety Verification Reserve
+
Optional Verification Budget
```

should be one module:

```text
VerificationBudgetManager
```

with two priorities:

```text
MANDATORY > OPTIONAL
```

## Pseudocode

```python
class VerificationBudgetManager:

    def request(self, priority):

        if priority == "MANDATORY":

            if self.safety_reserve_available():
                self.consume_safety_reserve()
                return GRANT

            if self.shared_capacity_available():
                self.consume_shared_capacity()
                return GRANT

            return DENY

        if priority == "OPTIONAL":

            if self.optional_capacity_available():
                self.consume_optional_capacity()
                return GRANT

            return DENY
```

Routing rule:

```text
HARD_INVALID + mandatory budget denied
→ Pending

HARD_VALID + optional verification denied
→ FALLBACK_KEEP(H0)
→ Accepted
```

---

# 12. Targeted Verification / Repair

## 12.1 Purpose

Only inspect the flagged task/field.

Avoid rerunning all tasks.

Examples:

```text
Target suspicious
→ verify Target only

IVT closure broken
→ verify the smallest legal affected scope

Phase suspicious
→ verify Phase only
```

The loop must be bounded.

Recommended first implementation:

```text
max_verify_attempts = 1
```

At most:

```text
max_verify_attempts = 2
```

Do not implement an open-ended recursive agent.

---

# 13. Verification Inner Loop

```text
Targeted Verify
      │
      ▼
KEEP / REPAIR
      │
      ▼
Post-Verification Safety Check
      │
  ┌───┴────┐
  │        │
PASS      FAIL
  │        │
  ▼        ▼
Done    attempts left?
           │
       ┌───┴───┐
       │       │
      YES      NO
       │       │
       └──↺    ├─ Optional → FALLBACK_KEEP
               └─ Mandatory → Pending
```

## Pseudocode

```python
def bounded_verify_repair_loop(
    H0,
    scope,
    source,
    fallback_H0_allowed,
    max_attempts,
):

    current = H0
    current_scope = scope

    for attempt in range(max_attempts):

        result = specialist_verify(
            hypothesis=current,
            scope=current_scope,
        )

        if result.status in {
            "NO_RESPONSE",
            "INVALID_RESPONSE",
            "UNRESOLVED",
        }:
            continue

        if result.status == "REJECT":
            return Proposal(
                status="REJECTED_BY_VERIFY"
            )

        if result.status == "KEEP":
            candidate = current
            reason = "VERIFIED_KEEP"

        elif result.status == "REPAIR":
            candidate = apply_bounded_repair(
                current,
                scope=current_scope,
                result=result,
            )
            reason = "VERIFIED_REPAIR"

        else:
            continue

        # Critical rule:
        # KEEP/REPAIR cannot directly become Verified.
        postcheck = post_verification_safety_check(
            candidate
        )

        if postcheck.hard_valid:
            return Proposal(
                status=reason,
                hypothesis=candidate,
            )

        next_scope = choose_next_legal_scope(
            candidate,
            postcheck.violations,
        )

        if next_scope is None:
            break

        current = candidate
        current_scope = next_scope
        # bounded inner loop ↺

    if fallback_H0_allowed:
        return Proposal(
            status="FALLBACK_KEEP",
            hypothesis=H0,
        )

    return Proposal(
        status="PENDING_UNRESOLVED",
    )
```

---

# 14. Post-Verification Safety Check

This is a required stage.

Repair must not directly produce `Verified`.

## Pseudocode

```python
def post_verification_safety_check(H):

    violations = []

    violations += check_ontology(H)
    violations += check_ivt_closure(H)
    violations += check_triplet_compatibility(H)
    violations += check_strict_phase_constraints(H)

    return PostCheck(
        hard_valid=(
            len(violations) == 0
        ),
        violations=violations,
    )
```

This should be:

```text
local
deterministic
cheap
```

Do not call the Gate again after repair.

---

# 15. Outcome Finalizer

The old final `DeterministicCoordinator` should be renamed:

```text
OutcomeFinalizer
```

It does not perform another reasoning stage.

It is the only module allowed to assign the final semantic state.

## Pseudocode

```python
def outcome_finalizer(H0, proposal):

    if proposal.status == "GATE_ACCEPTED":

        return FinalOutcome(
            state="Accepted",
            hypothesis=H0,
            provenance="GATE_ACCEPTED",
        )

    if proposal.status == "FALLBACK_KEEP":

        return FinalOutcome(
            state="Accepted",
            hypothesis=proposal.hypothesis,
            provenance="FALLBACK_KEEP",
            lower_reliability=True,
        )

    if proposal.status == "VERIFIED_KEEP":

        return FinalOutcome(
            state="Verified",
            hypothesis=proposal.hypothesis,
            provenance="VERIFIED_KEEP",
        )

    if proposal.status == "VERIFIED_REPAIR":

        return FinalOutcome(
            state="Verified",
            hypothesis=proposal.hypothesis,
            provenance="VERIFIED_REPAIR",
        )

    if proposal.status == "PENDING_UNRESOLVED":

        return FinalOutcome(
            state="Pending",
            hypothesis=proposal.hypothesis_if_any,
        )

    if proposal.status == "REJECTED_BY_VERIFY":

        return FinalOutcome(
            state="Rejected",
            hypothesis=None,
        )
```

---

# 16. Reliability-Aware Memory

Use three separate logical stores:

```text
Reliable Long-Term Memory
→ Verified

Short-Term Memory
→ Accepted

Pending Buffer
→ Pending
```

Do not mix Pending into trusted memory.

A simple bounded structure is enough:

```python
memory = {
    "verified": deque(maxlen=N_verified),
    "accepted": deque(maxlen=N_accepted),
    "pending": dict(),
}
```

No Vector DB or learned memory encoder is required for the current version.

---

# 17. Finalization Transaction

Every terminal observation must be auditable.

Trusted and Pending states must be committed atomically.

## Pseudocode

```python
def finalization_transaction(
    observation,
    final,
):

    tx = storage.begin_transaction(
        observation.id
    )

    try:

        # Audit every terminal observation
        tx.write_durable_frame_record(
            observation=observation,
            state=final.state,
            provenance=final.provenance,
        )

        if final.state == "Verified":

            tx.write_trusted_semantic_delta(
                state="Verified",
                hypothesis=final.hypothesis,
            )

            tx.update_memory(
                destination="RELIABLE_LONG_TERM",
                hypothesis=final.hypothesis,
            )

        elif final.state == "Accepted":

            tx.write_trusted_semantic_delta(
                state="Accepted",
                hypothesis=final.hypothesis,
            )

            tx.update_memory(
                destination="SHORT_TERM",
                hypothesis=final.hypothesis,
            )

        elif final.state == "Pending":

            # Critical rule:
            # Pending is not trusted memory,
            # but it must still be persisted.
            tx.write_pending_state_delta(
                hypothesis=final.hypothesis,
                created_at=observation.time,
            )

            tx.insert_pending_buffer(
                source_id=observation.id,
                hypothesis=final.hypothesis,
            )

        elif final.state == "Rejected":
            # audit only
            pass

        tx.commit()

    except Exception:
        tx.rollback()
        raise
```

Critical distinction:

```text
Pending ≠ Trusted Memory
Pending ≠ Nothing
```

---

# 18. Pending Resolution

Pending Resolution is the most likely component to become unnecessarily expensive.

Therefore it must be bounded.

Recommended fields:

```text
created_t
last_checked_t
scope
resolution_attempts
max_resolution_attempts
expiry
```

Recommended first implementation:

```text
max_resolution_attempts = 1 or 2
```

Do not re-call the Specialist on every subsequent frame.

---

# 19. Pending Resolution Inner Loop

```text
t0
↓
Pending
↓
Pending Buffer
    │
    ├─ t1 arrives
    │    └─ no useful new evidence
    │         → keep Pending
    │
    ├─ t2 arrives
    │    └─ new causal evidence
    │         → resolution attempt
    │
    └─ resolved
         ├─ Verified
         └─ Rejected
```

This remains causal because later frames are only used after they actually arrive.

## Pseudocode

```python
def resolve_pending_items(
    now_t,
    memory,
    pending_buffer,
):

    items = pending_buffer.get_eligible(now_t)

    for item in items:

        if item.resolution_attempts >= item.max_resolution_attempts:
            continue

        if not has_meaningful_new_evidence(
            item,
            now_t,
            memory,
        ):
            continue

        context = build_resolution_context(
            pending=item,
            causal_evidence_up_to=now_t,
            reliable_memory=memory,
        )

        assert context.max_time <= now_t

        result = resolve_pending(
            item,
            context,
        )

        if result.status == "STILL_PENDING":
            pending_buffer.mark_checked(
                item.id,
                now_t,
            )
            continue

        if result.status == "VERIFIED":

            postcheck = (
                post_verification_safety_check(
                    result.hypothesis
                )
            )

            if not postcheck.hard_valid:
                continue

            atomic_pending_resolution(
                item=item,
                new_state="Verified",
                hypothesis=result.hypothesis,
            )

        elif result.status == "REJECTED":

            atomic_pending_resolution(
                item=item,
                new_state="Rejected",
            )
```

---

# 20. Temporal Event Aggregator

Keep this module simple.

It should not become another trainable event-boundary model.

A first implementation can use:

```text
stable Phase
+
stable major IVT
```

to group frames.

## Pseudocode

```python
def update_event_stream(frame):

    if frame.state not in {
        "Accepted",
        "Verified",
        "Pending",
    }:
        return

    if should_start_new_event(
        current_event,
        frame,
    ):
        close_current_event()
        open_new_event(frame)

    else:
        extend_current_event(frame)
```

A simple boundary rule is enough:

```text
stable Phase change
or
stable major-IVT change
```

---

# 21. Reliability-Aware Report

Use the finalized event-level records.

```text
Verified → definite wording
Accepted → high-confidence observation wording
Pending  → uncertainty wording
Rejected → excluded
Candidate → excluded
```

## Pseudocode

```python
def generate_report(event):

    return report_template(
        verified=render_definite(
            event.verified
        ),

        accepted=render_observed(
            event.accepted
        ),

        pending=render_uncertain(
            event.pending
        ),
    )
```

A template reporter is preferred for the current implementation.

---

# 22. Main Stream Pseudocode

```python
def run_video(video):

    runtime.reset(video.id)

    for packet_t in video.stream():

        recover_previous_transaction()

        observation = stream_accept(packet_t)

        if observation.invalid:
            audit_execution_error()
            continue

        tracker_snapshot = run_tracker(
            observation,
            tracker_enabled=config.tracker_enabled,
        )

        previous_state = freeze_state_t_minus_1()

        context = build_causal_context(
            observation,
            segment_buffer,
        )

        perception = joint_perception(
            context
        )

        if perception.no_h0:
            finalize_execution_status()
            continue

        H0 = perception.hypothesis

        support = deterministic_safety_coordinator(
            H0=H0,
            candidates=perception.candidates,
            tracker=tracker_snapshot,
            previous_state=previous_state,
            ontology=ontology,
        )

        if support.safety_class == "HARD_INVALID":

            proposal = handle_mandatory_verify(
                H0,
                support,
            )

        else:

            action = gate_policy(
                support,
                config.gate_mode,
            )

            if action == USE_H0:

                proposal = GateAccepted(H0)

            else:

                proposal = handle_optional_verify(
                    H0,
                    action,
                )

        final = outcome_finalizer(
            H0,
            proposal,
        )

        finalization_transaction(
            observation,
            final,
        )

        # temporal inner loop
        resolve_pending_items(
            now_t=observation.time,
            memory=runtime.memory,
            pending_buffer=runtime.pending,
        )

        freeze_state_t()

        # next t ↺
```

---

# 23. Important Implementation Notes

## 23.1 Do not restore adaptive keyframe selection

Do not implement:

```text
frame difference
optical flow
instrument displacement selection
motion-peak selection
Top-K visual frame filtering
```

Reasons:

```text
extra computation
possible information loss
Tracker would affect model input
harder ablation
more engineering complexity without guaranteed gain
```

Use fixed causal 6-frame input instead.

---

## 23.2 Tracker should remain soft evidence

Do not use:

```text
Tracker says absent
→ force GPT prediction invalid
```

because Tracker can miss detections.

Use:

```text
Tracker disagreement
→ risk feature
→ Benefit Gate input
```

unless the condition is a genuinely deterministic structural constraint.

---

## 23.3 Keep hard safety rules conservative

Do not classify all clinical rarity or temporal inconsistency as hard invalidity.

Otherwise the pipeline may degenerate into:

```text
almost every frame
→ Mandatory Verify
```

Hard constraints should be limited to genuinely deterministic impossibilities.

Prefer sending the following to Gate features:

```text
low confidence
rare phase transition
temporal jump
Tracker conflict
uncertainty
```

---

## 23.4 Verification must be bounded

Do not implement:

```text
Verify
→ Repair
→ Verify
→ Repair
→ ...
until correct
```

At inference time there is no GT to know when the answer is truly correct.

Recommended:

```text
max_verify_attempts = 1
```

or at most:

```text
max_verify_attempts = 2
```

---

## 23.5 Pending Resolution must also be bounded

Do not resolve every Pending on every new frame.

Recommended:

```text
max_resolution_attempts = 1 or 2
```

Only recheck when meaningful new causal evidence appears.

Otherwise Pending may create uncontrolled API growth.

---

## 23.6 Repair must always be revalidated

Mandatory invariant:

```text
REPAIR
→ Post-Verification Safety Check
→ PASS
→ Verified
```

Never:

```text
REPAIR
→ Verified directly
```

---

## 23.7 Pending must be persisted separately

Mandatory invariant:

```text
Pending
→ PendingStateDelta
→ Pending Buffer
```

Pending cannot enter trusted semantic memory, but it also cannot disappear.

---

## 23.8 Do not make Event Aggregation another research problem

Use deterministic Phase / major-IVT grouping first.

Do not add:

```text
event-boundary neural network
learned event encoder
complex temporal clustering
```

unless later experiments show it is necessary.

---

## 23.9 Keep Memory simple

Current version does not need:

```text
Vector DB
embedding retrieval
memory graph
learned compression
learned decay network
```

A bounded structured memory is sufficient.

---

## 23.10 Segment boundaries must reset the causal buffer

Do not define continuity only by exact frame-ID increments.

Check:

```text
same video
same segment
monotonic time
expected sampling interval
```

At a true discontinuity:

```text
RESET causal visual buffer
```

Do not carry frames from the previous segment into the next causal window.

---

## 23.11 Validation/Test GT must remain evaluation-only

GT must not influence:

```text
Gate decision
Mandatory routing
Verification decision
Memory update
Pending resolution
Report generation
Tracker fallback
```

GT is allowed only in offline evaluation and Training-only Gate label generation.

---

# 24. Recommended Invariants / Unit Tests

```text
INV-01
No frame later than t can enter context at time t.

INV-02
Tracker ON/OFF cannot change visual-frame selection.

INV-03
Tracker failure cannot silently reuse an old snapshot.

INV-04
HARD_INVALID H0 cannot become Accepted directly.

INV-05
Benefit Gate can only process HARD_VALID H0.

INV-06
Mandatory Verification failure cannot fallback to HARD_INVALID H0.

INV-07
Every KEEP/REPAIR verification result must pass Post-Verification Safety Check.

INV-08
Pending cannot enter trusted semantic memory.

INV-09
Pending must produce a PendingStateDelta.

INV-10
Validation/Test GT cannot affect inference decisions.

INV-11
OutcomeFinalizer is the only authority assigning final semantic state.

INV-12
FinalizationTransaction must be atomic.

INV-13
Segment boundaries reset the causal window.

INV-14
Verification attempts are bounded.

INV-15
Pending-resolution attempts are bounded.

INV-16
Optional verification denied by budget becomes FALLBACK_KEEP, not Verified.

INV-17
Malformed API output is an execution failure, not semantic Rejected.
```

---

# 25. Final Recommended Minimal Research Pipeline

The implementation may contain recovery, logging, cache, budget, and transaction details, but the research method itself should remain:

```text
Fixed Causal 6-Frame Input
        ↓
Tracker Reliability Evidence
        ↓
Joint Structured Prediction
        ↓
Deterministic Hard Safety Check
        ↓
Mandatory Verify / Benefit Gate
        ↓
Bounded Targeted Verification
        ↓
Post-Verification Safety Check ↺
        ↓
Outcome Finalizer
        ↓
Reliable Memory / Pending Buffer
        ↓
Bounded Pending Resolution ↺
        ↓
Temporal Event Aggregation
        ↓
Reliability-Aware Report
```

The design principle is:

> **Do not add a module if it increases computation and complexity while potentially discarding information or creating an uncontrolled recursive loop.**
