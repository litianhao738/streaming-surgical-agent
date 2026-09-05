# Canonical Tracker x Gate Ablation and Academic Protocol

Date: 2026-09-03 (Asia/Shanghai)

Status: **user-approved design; core runtime and D0/bootstrap scaffolding
implemented, real API evidence and final Gate still incomplete**.

This document is the sole normative authority for the formal Tracker x Gate
factorial, label and frame-clock contracts, budget matching, statistical
analysis, and canonical Phase 0/A/B/C/D gates. The runtime graph, branch
semantics, field contracts, bounded Repair pseudocode, OutcomeFinalizer,
and state transaction are defined in
[`Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md`](Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md).

The two documents have deliberately disjoint authority. If pipeline prose in an
older document conflicts with the runtime specification, the runtime
specification wins. If an experiment name, factor, label, split, clock, budget,
or stage gate conflicts with this protocol, this protocol wins. Neither document
is evidence that the corresponding code or experiment has been completed.

## 1. Scope and non-claims

The primary module ablation changes exactly two binary factors:

- `T`: predicted Tracker evidence unavailable/available;
- `G`: deterministic control/one frozen learned Benefit Gate.

Repair, candidate construction, mandatory safety, OutcomeFinalizer, Memory,
Workflow, Perception, budget management, finalization, evaluator, and reporting
are fixed support infrastructure. They are not additional main ablation factors.

This protocol does not claim that:

- the present repository already implements the target runtime;
- existing demo Gate artifacts are paper-final models;
- verification is beneficial before the Training-only Repair probe passes;
- an uncalibrated VLM rank score is a correctness probability;
- frame count can replace independent-surgery count;
- Validation/Test labels may enter online routing, prompts, candidates, Memory,
  Pending resolution, or state updates.

## 2. Imported runtime contract

The formal experiment imports the following runtime contract without redefining
it here:

```text
StreamController
-> ObservationExecutionJournal BEGIN + pinned pre-t semantic snapshot
-> idempotent causal-frame ingest
-> TrackerModule [T0/T1]
-> fixed CausalContextBuilder
-> JointPerception
-> DecisionSupportBuilder + shared SafetyValidator
-> MandatorySafetyGuard
   |-- hard-invalid -> no Gate; PENDING or mandatory VERIFY
   `-- hard-valid   -> explicit GatePolicy [G0/G1]
                      |-- USE_H0
                      `-- REQUEST_VERIFY(scope)
-> one unified VerificationBudgetManager
-> 0/1 bounded TargetedVerifyAndRepair episode
   (fixed H0, scope, candidate pool; at most N_max charged rounds)
-> deterministic Repair postcheck (inside the bounded episode)
-> one OutcomeFinalizer
-> one FinalizationTransaction
-> Output_t; admitted Event_t is visible no earlier than t+1
```

Frozen runtime meanings:

- Gate runs only on hard-valid, policy-eligible `H0`; it never commits output.
- A hard-invalid `H0` never reaches Gate or direct admission.
- Each accepted observation has at most one Repair episode. Multiple bounded
  rounds are the episode's internal check-and-repair loop, not multiple episodes
  and not crash recovery.
- The original `H0`, candidate pool, images, snapshots, and base episode identity
  remain fixed throughout the episode. A hard-invalid in-pool proposal may become
  the current candidate for the next bounded round, while authority never expands
  beyond the original pool. Each charged round has its own derived
  `round_request_id` and `provider_request_id`; those IDs do not redefine the
  episode or widen its authority.
- Candidate construction uses version
  `factorized_closure_topk_phase_complete_v3`: Phase exposes all seven ontology
  IDs, IVT retains up to eight perception candidates, and I/V/T retain components
  required by frozen selected IVTs. Formal Verify uses the versioned blind
  target-frame prompt `targeted_verification_prompt_v6`; H0 selections and
  upstream rank scores are not provider inputs. The ontology appendix is scoped:
  Instrument and Workflow omit the full IVT mapping, while Interaction includes
  only frozen candidate IVT rows. Fixed unrequested-IVT closure requirements are
  explicit constraints, not Tracker evidence.
- Training-only Gate counterfactual branches execute concurrently from the same
  immutable pre-decision snapshot and are serialized in canonical scope order.
  This does not alter online inference, where Gate selects at most one scope.
- The shared SafetyValidator decides whether the episode may terminate or
  continue; the non-reasoning OutcomeFinalizer alone assigns the final state.
- Optional Repair failure falls back only to an already hard-valid `H0` as
  `FALLBACK_KEEP`; mandatory Repair failure is `PENDING_UNRESOLVED`.
- The exact primary outcomes are `GATE_ACCEPTED`, `VERIFIED_KEEP`,
  `VERIFIED_REPAIR`, `FALLBACK_KEEP`, `PENDING_UNRESOLVED`, and `REJECTED`.
- Only the first three outcomes may update TrustedPrior/Workflow, and ordinary
  EventMemory admission still requires an independently frozen reliability and
  provenance policy.
- Deferred Pending resolution is disabled in the primary factorial unless a
  separate protocol is frozen before Gate training. It never rewrites an old
  output or retrospectively improves an old frame metric.

## 3. Experiment-facing field projection imported from Pipeline

The complete Pipeline owns runtime objects and field meanings. This section
defines their experiment-artifact serialization only; it cannot redefine the
runtime contract. In particular:

```text
TrackSnapshot.status    -> artifact tracker_status
TrackSnapshot.available -> artifact tracker_available
```

### 3.1 Observation time

`frame_id` is a dataset/media identifier, not a duration. Runtime and artifacts
carry:

```text
frame_id: original dataset/media identifier
segment_id: frozen reset-boundary identifier from the clock manifest
observation_index: contiguous position in the approved runtime clock
timestamp_seconds: optional physical time
```

State causality is expressed with `observation_index` and explicit source
watermarks, not by subtracting frame IDs.
The Pipeline packages these clock/media fields and their dataset/clock hashes as
a cell-independent `CanonicalObservationKey`. Paired four-cell joins and Gate
decision audits use that key. Each `ObservationToken` additionally carries a
cell-execution-unique `run_id` and explicit `cell_id`, so transaction, budget,
round, and provider-request identities cannot collide across cells even when a
semantic response is content-cacheable.

### 3.2 Tracker fields

```text
predicted_local_track_id: local predicted identity, never a GT track ID
instrument_id: predicted instrument class
bbox_tlwh_normalized: current normalized box
current_detection_score: detector score at this observation
matched_observation_count: number of successful associations
missed_observation_count: current unmatched streak
first_seen_observation_index
last_seen_observation_index
is_observed_now
tracker_status: OK | AVAILABLE_EMPTY | DISABLED_BY_ABLATION | ERROR
tracker_available: bool
```

The ambiguous names `age`, `max_age`, bare `track_id`, and bare `score` are not
part of the target contract. Use `max_missed_observations` plus a frozen maximum
frame/timestamp link gap. Memory decay uses `event_age_steps` or
`event_age_seconds`.

### 3.3 Score fields

```text
rank_score: uncalibrated VLM candidate-ranking signal
detection_score: detector output
gate_benefit_score: Gate routing score
gate_benefit_probability: allowed only after documented calibration
reliability_score: bounded evidence weight, not correctness probability
retrieval_score: similarity x capped reliability x temporal weight
```

### 3.4 Tracker-derived Gate features

Feature names must reflect their actual aggregation level:

```text
supported_instrument_class_count
new_instrument_class_count
disappeared_instrument_class_count
current_track_survival_ratio
track_id_jaccard
tracker_available
```

Class-set features must not be described as instance or track counts. Every
Tracker-derived value has an explicit availability mask, so disabled, truly
empty, and failed Tracker states remain distinguishable.

### 3.5 State watermarks and bounds

```text
processed_through_observation_index
included_source_max_observation_index
max_events_total_per_video
```

All terminal outcomes advance the processed watermark exactly once. Only
admitted semantic evidence advances the included-source watermark. EventMemory
uses one genuinely total-bounded canonical event collection; reliability views
are derived indexes rather than independently full-capacity stores.

## 4. Formal label and Gate-utility contract

Runtime, Gate-data construction, and offline evaluation share one versioned
frame-supervision adapter. Each `(video_id, frame_id, task)` carries:

```text
label_available
label_source
annotation_semantics
exhaustive_presence
```

Missing or non-exhaustive labels are excluded from losses and metrics; they are
never converted to negative labels. Instance annotations may be unioned into
frame-level I/V/T/IVT presence only after exhaustive frame semantics and the
ontology mapping are documented. If that requirement fails for a scope, the
scope may run as an engineering path but cannot train or support a formal claim.

Ground truth is unavailable to the online pipeline. A separate offline process
may join labels only after `H0`, candidates, routes, Repair traces, and committed
outputs are frozen.

Gate supervision uses a predeclared, task-mask-aware per-observation routing
surrogate rather than final video-wise mAP:

```text
routing_utility_delta(scope)
  = utility(final_candidate, valid_tasks_in_scope)
  - utility(H0, valid_tasks_in_scope)
  - declared_verification_cost
```

The exact task utility, weights, cost unit, harm threshold, and missing-route
policy are frozen before data collection. An unobserved route carries
`route_observed=false`; it is never assigned zero benefit. Final claims remain
based on predeclared video-level recognition and cost endpoints.

Official splits and cross-fitting remain grouped by video. Frame-random splits
are forbidden. Every label-derived prior, normalizer, calibrator, Gate, or
Tracker artifact records its source video IDs and split role.

## 5. Formal frame clock

The audited VID110 media directory contains three PNG IDs absent from its
annotation target IDs. Two pairs are byte-identical despite different IDs:
`039175 == 039176` and `040600 == 040601`. A media-driven clock would therefore
change Tracker counts and causal windows.

Before regenerating formal Tracker artifacts, freeze a versioned clock policy:

- formal evaluation targets come from canonical annotation/sidecar target IDs;
- the first formal pipeline uses those same IDs as the observation clock;
- media-only observations require a separately versioned extension;
- byte-identical distinct-ID observations follow one predeclared rule shared by
  every factorial cell;
- frame gaps and segment resets are recorded in every Tracker artifact;
- the clock-manifest hash is embedded in every downstream artifact.

The clock manifest itself stores the frozen `dataset_contract_hash`, target-set
hash, split role, and ordered canonical keys. Runtime verifies that binding
against the loaded dataset/target-contract artifact before constructing any
ObservationToken or evaluating a Gate policy.

## 6. Tracker x Gate factorial

Only the two named policy factors change:

| Cell | Tracker factor | Gate policy slot |
| --- | --- | --- |
| `A_base` | disabled with explicit masks | frozen deterministic Rule Gate |
| `B_tracker` | predicted Tracker | the same frozen deterministic Rule Gate |
| `C_gate` | disabled with explicit masks | frozen learned Gate |
| `D_full` | predicted Tracker | the same frozen learned Gate artifact |

Executable configs, manifests, artifacts, tables, and tests use only
`A_base/B_tracker/C_gate/D_full`.

### 6.1 What is frozen in all four cells

- split, versioned dataset/target contract and its hash, target observations,
  the clock manifest cross-bound to that contract, causal-window size (`3`),
  segment boundaries, selected image IDs, and provider image-detail profile;
- one runtime code artifact, canonical serializer, and semantic-hash algorithm;
  Tracker state/error-transition policy, base model, provider call-policy and
  durable-record schemas, provider policy/metadata schema,
  ProviderCallAccounting schema/authority/signature policy,
  execution-accounting ledger schema, timing/currency/quantization policy,
  and instance-free-answer
  source/provenance binding policy,
  prompt/schema, decoding, candidate generator, DecisionSupport schema,
  bounded-retrieval trace schema/policy, EvidenceBundle schema/builder, and
  Gate-feature schema, except for the
  explicit Tracker payload and availability masks;
- SafetyValidator, MandatorySafetyGuard and scope priority, routing controller,
  scope registry, unified budget manager, Repair executor/round validator and
  progress rule, `N_max`, Repair postcheck, finalization, and outcome rules;
- Workflow, EventMemory, reliability policy, optional Pending policy, reporter,
  evaluator, seeds, failure policy, and cost/call accounting.

Tracker must not select images in the primary factorial. A supplementary
Tracker-aware image-selection experiment may be reported separately, but it
cannot be used to attribute the primary Tracker effect.

Natural downstream changes in `H0`, Gate inputs, routes, Repair, and committed
history are treatment effects, not additional configuration factors.

### 6.2 Tracker factor

T0 and T1 use the same code path and schema:

- T0 emits `DISABLED_BY_ABLATION`, zeros/masks Tracker-derived values, and never
  substitutes GT or stale predicted tracks;
- T1 emits the predicted snapshot pinned to the current ObservationToken;
- true empty, disabled, and failure states have distinct masks/status values;
- formal Tracker predictions must follow the frozen clock and fold-safe
  provenance required by the Gate-data split.

### 6.3 Gate factor

G0 is the one deterministic Rule Gate defined by the complete Pipeline source
of truth. It reads only hard-valid DecisionSupport, chooses the legal scope with
the greatest soft risk, and requests verification exactly when that scoped risk
meets the frozen threshold. A fixed scope-order tie break is used. A/B share the
same threshold and code; Tracker evidence is zero/masked in A and predicted in B.
G0 is not `single_pass`, does not bypass MandatorySafetyGuard, and is not
post-hoc matched to Test call counts.

G1 is one frozen learned Gate with one feature schema. T0 and T1 load the same
artifact; paired T0/T1 availability views must appear during training so T0 is
not an unseen deployment condition.

The explicit action set is:

```text
USE_H0
REQUEST_VERIFY(instrument_presence)
REQUEST_VERIFY(interaction)
REQUEST_VERIFY(workflow)
```

Forced safety routes have `gate_evaluated=false`, follow the identical mandatory
rule in G0/G1, are excluded from Gate loss/routing metrics, and remain included
in total system cost and failure reporting.

### 6.4 Budget and bounded Repair fairness

One `VerificationBudgetManager` owns mandatory reserve and optional allocation
under a common provider ceiling. Optional episodes cannot consume the mandatory
reserve. The same total allowance, allocation rule, episode cap, and frozen
`N_max` apply to all cells.

The primary budget scope is per video and resets only at the audited video
boundary. The normalized manifest freezes:

- total, mandatory-reserve, and optional-allocation starting balances, chosen
  without Test outcomes;
- strictly increasing observation order within a video and no concurrent
  reservation arbitration in the primary run;
- deterministic partial grants with strict mandatory headroom: every state has
  `shared_available >= mandatory_available`; a mandatory request receives
  `min(N_max, mandatory_available, shared_available)`, while an optional request
  receives `min(N_max, optional_available,
  max(0, shared_available - mandatory_available))`; a zero result is denied;
- video order, cell order, reservation/settlement policy, and failure policy;
- semantic request-hash algorithm, cache namespace, warm/cold policy,
  provider replay-idempotency policy, sampling seed/sentinel inside the decoding
  policy, routing configuration, expected returned model identity,
  normalized provider-response metadata schema/hash policy, authenticated
  ProviderCallAccounting schema/authority/signature policy, and the separate
  execution-accounting ledger schema.

Every budget state/ref and its version-0 genesis record are namespaced by the
cell-execution identity and video. The first canonical target must descend from
the frozen genesis balances, each later target from its exact committed
predecessor, and the handle closes only after end-of-video coverage checks.
Mid-video reset or cross-video/cross-run state reuse invalidates the cell.

Byte-identical semantic requests may share an immutable content-addressed
provider response to control model nondeterminism across cells. Such a replay is
still charged as a logical round. Report replay latency separately from the
stored origin latency/cost, so execution order cannot create a false efficiency
gain.
For both Base Perception and Repair, only the normalized instance-free semantic
answer is shareable. The trusted runtime gateway rebinds Base content to the
current ObservationToken/context/request and Repair content to the current
episode/round/request IDs. Repair provider payloads use a content-stable
`candidate_wire_key`; the gateway maps it one-to-one to the current logical
candidate ID. An origin `source_stamp`, bound provenance, logical candidate ID,
or cached bound `RepairProposal` from another logical call is never reused.
The cache entry retains the origin call's returned model identity and normalized
provider-metadata hash, both of which are revalidated against the current
manifest. Identity, metadata, receipt, request-binding, or content-hash drift
stops and invalidates the cell; it is not counted as semantic Rejected,
Fallback, or Pending.

A later paid Repair round must strictly shrink the remaining candidate-ID set
or add a new member of a frozen finite failure-code set. Free-form rationale is
not progress. This strict-progress rule is additional to the common hard
`N_max` cap and is identical in every cell.

The Rule Gate and learned Gate cells receive the same allowed budget; budgets
are never post-hoc matched to realized successful Test calls. Report separately:

- routed episodes and charged logical rounds;
- provider requests, cache hits/misses/bypasses, transport retries, and failures,
  separated for base Perception and Repair and also totaled;
- base/Repair/total input-output tokens and monetary cost, origin latency, and
  end-to-end latency;
- unused reservations and deterministic settlement outcomes.

Every experiment manifest is normalized and compared with an allowlist. Only
Tracker enablement/provenance and the Gate policy slot may differ at the
configuration level. This is demonstrated by one content-addressed comparison
artifact covering exactly `A_base/B_tracker/C_gate/D_full`, their canonical manifest
projection hashes, the frozen difference-allowlist hash, all observed difference
paths, and a derived PASS/FAIL status; a self-declared `passed=true` flag inside
one cell is not evidence.

### 6.5 Estimands and reporting

The independent statistical unit is the surgery/video, not the frame. Report
each cell separately and predeclare the Tracker main effect, Gate main effect,
and Tracker x Gate interaction as paired video-level contrasts. Do not select a
contrast after viewing Test results.

With eight Test videos, report video-cluster uncertainty and an exact paired
sign-flip/randomization analysis where its assumptions apply. Correct the
declared confirmatory family for multiplicity. A non-significant result is not
evidence of equivalence without a predeclared equivalence margin.

Recognition, Repair, routing, abstention/fallback, provider failure, token,
cost, latency, and artifact-lineage results are reported together. Any missing
cell, leakage, identity drift, budget violation, or manifest difference outside
the allowlist blocks the main factorial claim.
Every required clock key remains in the evaluation denominator. A
`FALLBACK_KEEP` evaluates its presented H0 while retaining the fallback flag;
`PENDING_UNRESOLVED` and `REJECTED` remain explicit no-semantic-prediction
outcomes and are never dropped as failed calls. Phase 0 must freeze their exact
contribution to each non-selective and selective endpoint, including coverage
and abstention, before any formal rollout.

## 7. Implementation order and academic stage gates

### Phase 0 — freeze the research and data contract

Before new Gate-data collection, Validation tuning, or Test access:

1. unify runtime and evaluator frame-supervision semantics;
2. prove or reject exhaustive frame-level presence for every claimed task;
3. freeze the routing utility, primary endpoint, harm definition, split map,
   video-group folds, six-outcome evaluator/denominator mapping, budget rule,
   and Repair promotion criteria;
4. freeze the frame clock, duplicate-media rule, causal image plan, scope
   registry, and the Training-only protocol that chooses `N_max`;
5. isolate legacy demo artifacts/caches from a new formal namespace;
6. record code/spec/data/environment hashes and keep Test sealed.

Failure to establish multi-video valid supervision for a scope blocks formal
training and claims for that scope; frame count cannot replace independent video
count.

### Phase A — complete the shared engineering contract

Do not train the final Gate in this phase.

1. Extract shared runtime, Gate, verification, state, and finalization types so
   dependency direction is acyclic.
2. Implement and test the exact-predecessor clock, gaps, video reset, and cold
   restart recovery from a chained causal-frame-buffer/Tracker physical
   checkpoint, including the all-targets-committed `END_OF_VIDEO` case;
   implement ObservationToken, the fixed Tracker-independent image plan, and
   immutable pre-`t` semantic snapshots.
3. Rename Tracker fields, add availability/status masks and idempotent
   isolated transition/pin/install semantics, including the frozen
   `ADVANCE_AS_UNOBSERVED` error post-state; then regenerate clock-compatible
   prediction artifacts from the existing detector checkpoint before deciding
   whether detector retraining is necessary.
4. Implement one reusable SafetyValidator, stable complete
   CandidateHypothesis IDs, bounded scoped candidate sets, snapshot retrieval,
   immutable RetrievalTrace/EvidenceBundle/MaskedGateFeatures, and the enclosing
   DecisionSupportBundle.
5. Implement the explicit MandatorySafetyGuard, Gate policy slot, and unified
   idempotent budget reservation/settlement path: open reservations are current
   transaction journal state, while the published semantic bundle is the sole
   owner of committed BudgetState.
6. Implement one shared Base/Repair provider gateway with immutable call-policy
   binding, instance-free cache/rebind semantics, returned-model/metadata checks,
   and authenticated call accounting.
7. Implement one bounded TargetedVerifyAndRepair episode with fixed `H0`, scope,
   pool, and at most `N_max` charged rounds. Responses may only KEEP or select a
   candidate wire key from the complete pool sent before the call; the gateway
   maps it to its pre-bound current logical candidate ID, and every continuation
   must satisfy the frozen strict-progress measure.
8. Implement the bounded Repair postcheck with the shared SafetyValidator and
   the non-reasoning OutcomeFinalizer, including exact fallback or Pending
   behavior.
9. Replace duplicate Memory buckets with one total-bounded event collection and
   replace sequential writes with an atomic or journaled, idempotent
   FinalizationTransaction. Treat envelopes referenced by authenticated
   `CommittedFinalizationRecord`s—not in-memory runner yields—as the
   output-of-record, and verify an exact
   clock-ordered per-video export after cold recovery.
10. Assemble all four cells from one shared config and prove allowed differences
   using normalized manifests.
11. Pass unit, integration, call-order, causality, leakage, failure-injection,
    artifact-schema, local-data, portability, lint, and full regression checks.

### Phase B — establish Repair capability and freeze `N_max`

Run a Training-only, video-group cross-fitted capability probe from identical
immutable pre-decision snapshots. Candidate construction and routing inputs stay
gold-free; labels are joined offline.

Measure candidate oracle gain/Recall@K, KEEP/REPAIR precision, rescue rate, harm
rate, mask-aware utility, scope/candidate violations, and per-round calls,
tokens, cost, and latency. Compare the marginal rescue, harm, and cost of round
2 against round 1. Choose and freeze the smallest justified `N_max` before Gate
data generation; `N_max` is not a third factorial factor.

Promotion thresholds are frozen before provider calls. Structural violations
must be zero. If Repair has no reproducible positive value across independent
Training folds, do not train a Gate to route to it; fix, simplify, or remove the
unsupported scope first.

### Phase C — train and freeze one Gate

Only after Phases 0, A, and B pass:

1. construct Training-only D0 examples from immutable snapshots and observed
   counterfactual routes, preserving supervision and `route_observed` masks;
2. fit video-cross-fitted `bootstrap_gate` models;
3. run exactly one causal policy-matched D1 rollout, routing each held-out video
   with a Gate not trained on that video's labels;
4. fit one shared final Gate using the predeclared D0/D1 recipe and paired T0/T1
   availability views;
5. use Validation once under the frozen algorithm to choose the operating point,
   calibration if claimed, allowed budget, and G0 Rule Gate threshold;
6. freeze artifact, feature schema, masks, thresholds, lineage, and all
   compatibility hashes without Test refresh.

Demo/bootstrap artifacts are not deployable in the formal runner. A formal Gate
artifact records separate provenance for Training fit, Training policy rollout,
Validation operating-point selection, and `test_data_seen=false`.

### Phase D — run the frozen four-cell experiment

Run every target observation of every Test video in all four cells without
retraining, threshold changes, post-hoc budget matching, or successful-frame-only
analysis. Release normalized manifests, per-video outputs, provider/accounting
traces, metrics, and checksum indexes.

This Phase 0/A/B/C/D dependency order is normative. The live checkpoint and
next action are maintained only in [`docs/README.md`](../README.md).
