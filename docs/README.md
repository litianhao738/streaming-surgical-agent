# Documentation Authority and Live Checkpoint

## Main API H0 checkpoint — 2026-09-06

[Main API Pipeline](MAIN_API_PIPELINE.md) is the current run guide for the
user-adopted initial predictor: three causal images, one Qwen API call, original
baseline prompt and final-label five-head JSON. It is now the default of
`run_dataset_api_pipeline.py`, with production-packaged templates and a
`--preflight` mode that sends no API request. `joint_openrouter_h0.yaml` is the
one main real configuration; `joint_mock_h0.yaml` exercises the same protocol offline.

The [collaborator quickstart](API_EXPERIMENT_QUICKSTART_2026-09-06.md) documents
the published synchronous baseline, installation, three-target smoke, full-split
selection and independent GT scoring. Record the Git commit with every run.

The separate [Gemini grounded end-to-end smoke](GEMINI_GROUNDED_E2E_2026-09-06.md)
uses `run_grounded_api_pipeline.py`: the same current H0 prompt and final-only
contract, followed by blind localization, a visual proposal, contrast review and
checked acceptance. It uses synchronous OpenRouter calls, not Batch. The report
separates engineering targets without interaction GT from the fully annotated
Training supplement; the default Qwen-only entry remains unchanged.

The [diff-review diagnostic, 2026-09-07](DIFF_REVIEW_TRIAL_2026-09-07.md) adds
per-change visual assessments and compares whole-candidate acceptance with
bounded local edits on frozen historical candidates. It is a separate experiment;
the report distinguishes synthetic mock checks from actual paid observations.
The [subsequent small test](DIFF_REVIEW_SMALL_TEST_2026-09-07.md) records another
first-call timeout, unknown cost and no new semantic-review result.
The [one-off Qwen verifier test](DIFF_REVIEW_QWEN_DIRECT_TEST_2026-09-07.md)
records two successful reviews after a provider Schema compatibility correction;
both repair policies harmed predictions on the reviewed subset. It identifies a
label-presence versus edit-verdict ambiguity and does not promote either policy.

The [24-target neutral-presence A/B trial](PRESENCE_REVIEW_RESULTS_2026-09-07.md)
completed 104 synchronous Qwen calls on fresh Training targets, with current H0
connected to candidate generation and paired old/new review. Neither A nor B
improved IVT over H0. A separate length-only offline replay recovered all 13
overlong new reviews and still showed no IVT gain; strict and supplementary
results remain separate. The default H0 is unchanged. See the
[pre-inference protocol](PRESENCE_REVIEW_PROTOCOL_2026-09-07.md) for the frozen design.

The [1000-character review update and SurgReflect/RAG audit](SURGREFLECT_VERIFICATION_AND_RAG_2026-09-07.md)
versions the relaxed review contracts while preserving the 300-character history.
New runs of `run_presence_review_trial.py` default to 1000; full-text offline
revalidation made all 16 old and 16 new saved reviews valid without API calls.
It did not change the negative semantic conclusion. The same note distinguishes
the parent repository's actual verification/statistics mechanism from unverified
paper claims and explains why RAG is not yet the preferred next change.

The [new verifier development and confirmation record](VERIFIER_DEVELOPMENT_AND_CONFIRMATION_2026-09-07.md)
reports the frozen 16-target Training comparison of numeric and decoded-name
propositions, separate-question verification, and conditional supplementation
with two real earlier images versus an identical-input repeat. No repair variant
passed the predeclared development screen. The completed development total is
136 synchronous calls / $2.063434, with no unknown charges. H0 is retained;
Validation confirmation was not run because no repair policy qualified.
The [mechanism and optional memory/reflection note](VERIFIER_REPAIR_MECHANISM_2026-09-07.md)
distinguishes the default H0, historical whole-candidate review, and experimental
local A/B repair, and explains the remaining visual-judgment and shared-omission limits.
These experiments preserve the default H0 and the sealed Testing split.

The [full-paper review and single feedback-reflection proposal](VERIFIER_PAPER_METHODS_AND_REFLECTION_2026-09-07.md)
checks the newly supplied SurgReflect PDF against its public code and reviews
StreamingVLM, Flash-VStream, M3-Agent, Vgent and TDC-Video. It distinguishes the
completed blind repeat from untested feedback reflection and documents a
zero-call structural-consistency diagnostic. This is a research proposal, with
no new paid experiment or change to the retained H0 decision.

The [H0-first prior and multi-model panel preparation](H0_PRIOR_PANEL_PROTOCOL_2026-09-07.md)
records the user's revised design: Training-only video-excluded priors, three
independent visual judges, and at most three review rounds with two local
patches. It includes design-only schemas/prompts, eight image-only proposed
targets and public model/price snapshots. The subsequent
[execution report](H0_PRIOR_PANEL_EXECUTION_2026-09-07.md) records the completed
eight-target trial, which did not improve IVT. Historical single-judge proposals
remain as history.

The following authority and implementation notes describe the separate research
Tracker/Gate/Verifier Pipeline. Its top-k-dependent Gate is not compatible with
main H0's hard labels and is explicitly rejected rather than supplied invented
confidence. Historical presets and frozen experiments remain for reproduction.

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
5. [Session handoff 2026-09-04](SESSION_HANDOFF_2026-09-04.md) is a historical
   verified implementation and pilot checkpoint. It records dated facts only
   and is not a design authority.
6. [Session handoff 2026-09-02](SESSION_HANDOFF_2026-09-02.md) is historical
   context only.

The user's latest explicit instruction has precedence over every document.
Current code and generated artifacts are implementation evidence; a target
document alone is never evidence that its code or experiment has run.

The current visual-input decision is documented in
`architecture/CAUSAL_THREE_FRAME_DECISION_2026-09-04.md`: primary H0 and Verify
use the same fixed latest-three causal images; older six-frame artifacts are
historical and cannot be mixed into the new protocol.

## Live implementation checkpoint

Latest Tracker engineering evidence: [Tracker / OOF remediation, 2026-09-05](TRACKER_REMEDIATION_2026-09-05.md).
Box supervision is versioned, resume restores the original detector structure,
gap-aware associations have been regenerated, and the final CLI validates OOF
routing and coverage before creating API transports. Existing weights remain
legacy baselines; corrected-supervision full/OOF training has not run.

Latest H0 evidence: [strict 40-target rescore](H0_STRICT_RESCORE_2026-09-06.md),
[error diagnosis](H0_LOW_ACCURACY_DIAGNOSIS_2026-09-06.md), and
[paper synthesis](H0_RECOMMENDED_METHOD_PAPER_SYNTHESIS_2026-09-06.md).
The original prompt remains the adopted baseline; adding training examples is untested.

Earlier Verifier evidence: [Astra six-issue smoke and fixes, 2026-09-05](ASTRA_SIX_ISSUE_SMOKE_AND_FIXES_2026-09-05.md).
The V9 dual-review coverage experiment did not improve semantic results and is
not promoted over V8. That report also records the Training instance-supervision
prior fix, provider refusal handling, and the remaining semantic limitations.

### Implemented and locally verified

- The executable final path is
  `StreamController -> Tracker -> fixed three-frame context -> JointPerception ->
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
  JointPerception and Verify (up to three frames at stream boundaries, always
  including the target). Formal Verify input omits Tracker, workflow and Memory, so its visual
  evidence is identical across Tracker cells. The active V6 verifier is blind to
  H0 selections and upstream rank scores, predicts the target frame only, uses
  all seven Phase candidates, receives explicit frozen-IVT closure constraints,
  and includes only the ontology subset required by its scope.
- The CholecTrack20 visual clock advances by 25 source-frame IDs per annotated
  second. Causal windows and immediate temporal comparison reset at larger
  gaps; the first post-gap request is intentionally shorter than three images.
- Provider ceilings count exact transport attempts, including retries. Cache
  hits consume no provider-attempt budget.
- The historical five-fold Tracker OOF index covers all ten Training videos and passed
  artifact-hash validation. Its held-out instrument evaluation scored 14,231
  frames from the nine instance-supervised videos; VID31 has exact prediction
  coverage but is excluded from box metrics because it has no instance boxes.
  Detector checkpoints remain reusable as legacy baselines. Gap-aware association
  artifacts now exist under `artifacts/preflight/tracker_remediation_20260905/reassociated/`.
  Formal Training runs use that tree's `oof/index.json` via `--tracker-oof-index`;
  Validation/Testing use `full/predicted_tracks.json` via `--tracker-artifact`.
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

Keep H0 frozen and assess actual semantic gains and harms in the
[bounded verifier trials](VERIFIER_DEVELOPMENT_AND_CONFIRMATION_2026-09-07.md)
before promoting any repair policy or beginning formal D0. The older Gate and
Tracker factorial workflow remains separate; no final-only confidence is fabricated
to enter its top-k-dependent Gate.

The [H0 prior/panel execution report](H0_PRIOR_PANEL_EXECUTION_2026-09-07.md)
records the completed eight-target Training experiment: one instrument-label
correction in the no-prior arm, no IVT improvement in either arm, and three
failed review arms retained as H0 fallbacks. Keep the pure API H0 as the default;
the new direct OpenRouter runner remains an isolated research entry point.

The [five-model mean-score experiment](H0_FIVE_MEAN_EXECUTION_2026-09-07.md)
implements actual ordinal 1–5 predictions, arithmetic means, and up to three
review rounds with local repair. It records academic surgical-video context in
the verifier input and preserves provider moderation failures and model changes.
This is a separate developmental replay; consult its execution status before
claiming semantic benefit or a completed three-round experiment.
