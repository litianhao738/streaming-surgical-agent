# Documentation Authority and Live Checkpoint

## Sources of truth

1. [Streaming Surgical Final Pipeline Architecture and Pseudocode](architecture/Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md)
   is the sole authority for the online module order, Guard/Gate branches,
   bounded Repair, OutcomeFinalizer, states, Memory and pseudocode. It is the
   user-supplied complete Pipeline truth source.
2. [Canonical Tracker x Gate Ablation and Academic Protocol](architecture/CANONICAL_PIPELINE_TRACKER_GATE_SPEC_2026-09-03.md)
   is the authority for the `A_base/B_tracker/C_gate/D_full` experiment,
   labels, clock, statistics and research progression. It imports the runtime
   graph above and may not redefine it.
3. [V3.1-API academic architecture revision](architecture/Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md)
   provides non-conflicting research rationale and claim boundaries.
4. [V3.1-API Codex implementation specification](architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md)
   provides non-conflicting historical engineering detail.
5. [Session handoff 2026-09-04](SESSION_HANDOFF_2026-09-04.md) is the latest
   verified implementation and pilot checkpoint. It records dated facts only
   and is not a design authority.
6. [Session handoff 2026-09-02](SESSION_HANDOFF_2026-09-02.md) is historical
   context only.

The user's latest explicit instruction has precedence over every document.
Current code and generated artifacts are implementation evidence; a target
document alone is never evidence that its code or experiment has run.

## Live implementation checkpoint

### Implemented and locally verified

- The executable final path is
  `StreamController -> Tracker -> fixed six-frame context -> JointPerception ->
  DecisionSupport/SafetyValidator -> MandatorySafetyGuard or Gate -> bounded
  Verify/Repair -> OutcomeFinalizer -> atomic finalization -> Output/Memory`.
- The Rule Gate and learned-Gate slot share the same safety, Repair, budget,
  outcome and state path. Gate is evaluated only for hard-valid H0.
- Repair is candidate-bounded, scope-bounded, postchecked after every round and
  capped at one or two attempts. Optional failure becomes lower-reliability
  fallback acceptance; mandatory failure becomes Pending.
- Accepted, Verified and Pending are committed atomically. Pending never enters
  trusted Memory; a later bounded resolution replaces the original Pending
  output row instead of creating a contradictory duplicate.
- All final API profiles use the same ordered causal image tuple for
  JointPerception and Verify (up to six frames at stream boundaries, always
  including the target). Formal Verify input omits Tracker, workflow and Memory, so its visual
  evidence is identical across Tracker cells. The active V6 verifier is blind to
  H0 selections and upstream rank scores, predicts the target frame only, uses
  all seven Phase candidates, receives explicit frozen-IVT closure constraints,
  and includes only the ontology subset required by its scope.
- Provider ceilings count exact transport attempts, including retries. Cache
  hits consume no provider-attempt budget.
- The five-fold Tracker OOF index covers all ten Training videos and passed
  artifact-hash validation. Its held-out instrument evaluation scored 14,231
  frames from the nine instance-supervised videos; VID31 has exact prediction
  coverage but is excluded from box metrics because it has no instance boxes.
- Formal D0 collection intersects runtime and label-bearing identities, uses
  timeline-spread sampling for engineering subsets, derives conservative
  all-instance task masks for ordinary Training videos, retains VID31 frame
  labels, joins GT only after all provider calls for a video, and writes an
  atomic per-observation resume ledger.
- Video-cross-fitted bootstrap G0 training is implemented. Bootstrap artifacts
  are explicitly non-deployable; a Training-held-out threshold run is labeled
  diagnostic and cannot masquerade as final Validation calibration.

### Deliberately not claimed complete

- A 20-frame, ten-video Training-only V6 capability pilot attempted all three
  same-snapshot scopes. Only 17 frames had H0 outputs because three were rejected
  by provider moderation. Instrument and Interaction produced no positive repairs;
  Workflow produced two positive and one harmful repair. This does not establish
  general Repair capability or Test performance.
- The Repair capability decision and `N_max=1` versus `N_max=2` require real
  Training-only API evidence.
- D1 must be one causal policy-matched Training rollout using cross-fitted G0
  artifacts. Final G1 fitting and Validation operating-point selection remain
  downstream of that evidence. No current Gate artifact is paper-final.
- The optional Pending resolver has storage and bounded transition support but
  no default external resolution backend; it is disabled unless configured.
- Testing remains sealed from training, routing and model selection.

## Current next action

Do not retrain Tracker and do not train Gate yet. Inspect the 2026-09-04 handoff
and the 20-frame pilot, then design a predeclared paired Training-only test of a
fixed stronger Verifier while preserving the Tracker × Gate factorial boundary.
Resolve provider coverage and demonstrate positive Repair value with controlled
harm before freezing `N_max` or beginning formal D0.
