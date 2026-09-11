# Documentation Authority and Live Checkpoint

Six-target Training backbone comparison completed: [results and cost](MAINLINE_BACKBONE_COMPARISON_2026-09-11.md).
Recommend Gemini 3.8 Flash for the next four-video preparation; no full collection started.

Current mainline (2026-09-11): **`prior-gated-joint-mainline-v1.0.0`**, explicitly selected by the user. Original prior gate plus joint Phase, 13 calls/target, no blind Phase panel. [Current flow](../LATEST_PIPELINE.md) · [Frozen evidence](PRIOR_GATED_JOINT_MAINLINE_FREEZE_2026-09-11.md). The frozen Gemini base and five reviewers remain unchanged; the six-target Training model comparison is a separate experiment. No Gate collection has started. Statements below about unchanged defaults describe their historical experiment dates.

Historical user-selected default (2026-09-10): `parallel-phase-repair-v1.3.0-glm-low`. Both branches use GLM-5.3-Flash via Together in place of Ministral, plus the same Qwen, GPT, Gemini and DeepSeek seats. GLM reasoning is mandatory and explicitly set to low; excluding reasoning text does not disable it. Four live interface checks passed. [Default instructions and limitations](DEFAULT_GLM_REPAIR_2026-09-10.md). Existing selectors and historical versions retained; no accuracy improvement claim.

Prior gate plus jointly verified Phase (2026-09-11, negative result, default unchanged): [report](PRIOR_GATED_JOINT_PHASE_2026-09-11.md). Phase enters the same propose -> five-seat verify -> Python admit flow as the four heads (one visual recommendation without prior hints, seven Phases rated jointly with the four-head pool, unique best >= 4 replaces). Leakage guards are enforced on every request; the gate keeps the H0 Phase bucket because even the ground-truth Phase does not lower four-head errors in replay. On 16 fresh VID110 targets the joint panel changed no Phase, so the candidate equals the prior-gated candidate (mean F1 54.76 H0 / 61.01 default / 62.21 both gated arms; errors 95 / 88 / 84); predeclared standard failed. 288 calls, one file-lock interruption recovered without repeating any attempt.

Prior-gated IVT admission (2026-09-11, candidate, default unchanged): [report](PRIOR_GATED_IVT_ADMISSION_2026-09-11.md). A Python step after the v1.3.0 four heads vetoes selected IVTs whose leave-video-out, H0-phase-conditioned Training rate is below 0.01 and admits pool IVTs at or above 0.70; Phase is frozen to H0. Zero extra calls. Thresholds fixed on 136 Training targets by offline replay; one paid confirmation on 16 fresh VID110 targets passed the predeclared standard (mean F1 55.46 H0 / 54.33 default / 58.97 candidate; errors 91 / 98 / 87). Single video, small sample.

Lightweight Phase seat trial (2026-09-10): [paired eight-target experiment](LIGHT_PHASE_SEATS_TRIAL_2026-09-10.md). Grok replaced by Ministral 8B and Qwen by 35B-A3B; 32 valid requests. Pair latency decreased 49.72%, but Phase stays 75% versus H0 87.5%. Three other seats cached; this is not full-pipeline timing. Default unchanged.

Temporal Phase ratings (2026-09-10): [same-32 short versus longer causal history](PHASE_TEMPORAL_RATINGS_TRIAL_2026-09-10.md). All 320 responses valid and replayed. Both arms have Phase 75.00% versus H0 78.13%, zero corrections and one harmful switch. Phase repair remains enabled; no default promotion or permanent H0 freeze.

Simple Phase choice versus mean ratings (2026-09-10): [same-32 experiment](PHASE_SIMPLE_RATINGS_TRIAL_2026-09-10.md). All 320 calls completed and replayed. H0 Phase 78.13%; both simplified arms 75%. Ratings produced zero correct and one harmful Phase switch; default unchanged.

Same-32 Phase comparison (2026-09-10): [original prompt, legacy five-head repair and revised template](PHASE_MECHANISM_COMPARISON_2026-09-10.md). Phase F1: H0 78.13%; fresh compact/original 71.88%; legacy/revised 75.00%. All 832 calls completed. The format-corrected revision has 160 valid responses but one correction versus two harmful Phase switches. Default unchanged.

32-target expansion (2026-09-10): [H0/default/split results](EXPANDED_SPLIT_REVIEW_2026-09-10.md). Mean F1 58.03% / 59.63% / 58.85%; split fails the predeclared aggregate criteria. Shared Phase drops from 78.13% to 65.63% (3 corrections, 7 harmful switches). All 704 attempts archived; 29-target no-error sensitivity keeps the same conclusion. Default unchanged.

New-target paired confirmation: [fresh eight-target H0/default/split comparison](FRESH_SPLIT_REVIEW_2026-09-09.md). Mean F1 54.59% / 64.96% / 65.55%; split slightly improves the aggregate but still loses Target quality. All 176 calls parsed. The old cohort had the opposite aggregate direction; default unchanged.

Latest paired component/relation review experiment: [split review report](SPLIT_REVIEW_TRIAL_2026-09-09.md). Same-eight development did not improve quality; transport interruption, recovery, costs and timing limits are recorded. Default unchanged.

Latest local audit: [default improvement trials](DEFAULT_IMPROVEMENT_TRIAL_2026-09-09.md). The current compact default now emits precise rejection diagnostics after successful execution/scoring. ROI and learned aggregation remain isolated experiments; their original-prompt results are not compact-default results.

## Previous user-selected default repair (2026-09-09)

The user selected graph R1 plus independent short-history Phase review with the measured compact verifier prompts as the default repair: `parallel-phase-repair-v1.1.0-compact-prompt`. [Default manifest](../DEFAULT_PIPELINE_VERSION.json) is consumed by `scripts/run_pipeline.py`; see [execution instructions](DEFAULT_REPAIR_PIPELINE_2026-09-09.md). The compact adapter reuses the original parallel scheduler, JSON contracts and selectors. Its supported scope remains eight archived Training targets with cached H0. Existing original-profile plans retain their original prompts. This is a local workspace update, not a new Git release. Historical manifests below retain their original results.

## Latest final-only repair release (2026-09-09)

Read [LATEST_PIPELINE.md](../LATEST_PIPELINE.md) first for the current default flow, parallel branches, JSON contracts and model roles. The [release guide](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/docs/releases/PARALLEL_PHASE_REPAIR_V1.md) and [offline replay](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/scripts/replay_parallel_phase_repair.py) describe the historical base v1.0.0 release; the [current version pointer](../LATEST_PIPELINE_VERSION.json) also records the local compact-prompt default.

Cached H0 -> graph proposal/five-family four-head review in parallel with independent five-family Phase choices -> Python merge. Same-eight Training results: latest IVT/Phase F1 38.71%/75%; retained observed result 40%/75%. Tracker/Gate and full Testing integration remain incomplete. The default pure H0 entry is separate.

The records below retain earlier checkpoints; they do not override this release's measured scope.

Historical four-head repair snapshot: [best-version manifest](../BEST_REPAIR_VERSION.json) and
[concise same-eight-target experiment record](../EXPERIMENT_RESULTS.md), updated 2026-09-09.
The frozen [prior-graph one-round release](https://github.com/litianhao738/streaming-surgical-agent/tree/best-repair-2026-09-09)
keeps commit `a0406baa7e5aeb8313403e792a8e609e2fceccef` and its portable offline replay.
Second-round feedback, relaxed quorum, and four/five-head LLM repairs did not
outperform it on this development cohort. These are not full Tracker/Gate or Testing results.
The default pure API H0 is unchanged. Earlier repair reports and GraphRAG design
notes below are dated history; use the current record for implemented, measured status.

The [Gate training readiness check](GATE_TRAINING_READINESS_2026-09-08.md)
adds final-only features, verified OOF joins, grouped CPU pilot fitting and a
frozen 40-target collection plan. The current seed has only one beneficial
episode; real Gate training remains data-blocked. No new paid collection or
real-model fitting was performed in that preparation.

Start with the [Verifier / Repair mechanisms and measured results summary](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)
for the tested alternatives, paired gains and harms, task masks, actual rounds,
costs and publication boundaries. Its [machine-readable metric excerpts](experiments/verifier_repair_results_20260908.json)
retain source hashes; that earlier summary was documentation-only. The current code release and its limits are linked above.

The [recent five-family panel, 12-target three-round trial](RECENT_FIVE_PANEL_THREE_ROUNDS_2026-09-08.md)
compares mean thresholds 4 and 3.5 on shared candidates. IVT F1 improved slightly,
but errors increased and third-round IVT exact-set accuracy fell to zero. The
Grok seat is explicitly a flagship exception; an empty-pool false-pass bug was
fixed after the paid run, with original outputs preserved. This remains research-only.

The [second-round continuation and sparse LLM Repair test](SECOND_ROUND_LLM_DELTA_REPAIR_2026-09-08.md)
reuse cached Gemini H0/R1. Eighteen new calls yielded one correct Target addition
in the original continuation, with no IVT gain. The sparse LLM arm returned four
empty patches; its new nonempty patch-review path has offline tests only.

The [OpenRouter Gemini H0 panel test](OPENROUTER_GEMINI_H0_PANEL_2026-09-08.md)
records the new research entrypoint with Gemini 3.8 Flash for both H0 and candidate
generation through the Google route. Twenty-eight calls on four Training targets
completed; the panel kept every new H0. Target F1 increased versus cached Qwen,
while Verb, IVT and Phase decreased. The historical published H0 remains available.

The [follow-up cause audit and repair ablations](VERIFIER_REPAIR_CAUSE_AND_ABLATIONS_2026-09-08.md)
test ontology candidate completion, valid-evidence quorum, one-call joint reflection
and cross-video labeled references. Twenty-six additional calls improved candidate
coverage but did not improve aggregate IVT; harmful variants remain research-only.

The [2026-09-08 semantic Verifier/Repair fixes and API comparison](VERIFIER_REPAIR_SEMANTIC_FIX_2026-09-08.md)
record candidate-bound evidence checks, the Gemini reviewer row-format fix,
mask-checked Training evaluation and blind-proposal ablation. The four fully
annotated targets show a small Verb gain but no Target/IVT gain; the frozen H0
remains the default. Failed contracts, one 403 fallback and all costs are retained.

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

Latest Tracker evidence: [corrected full / OOF import, 2026-09-07](TRACKER_CLIP_V2_IMPORT_2026-09-07.md).
The supplied AutoDL full and five-fold outputs use `clip_to_frame_v2`; checkpoint
metadata, hashes, held-out routing and local metric reproduction have been checked.
The active local tree is `artifacts/training/tracker_clip_v2_oof5_20260906`.
Earlier weights remain available through a verified legacy archive. This local
installation has not been published to GitHub and does not enable Tracker in H0.

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

Corrected full/five-fold Tracker outputs have now been imported and checked; use
the [2026-09-07 import record](TRACKER_CLIP_V2_IMPORT_2026-09-07.md), not the earlier
legacy association report, for their status. This does not establish action or
target recognition quality and does not enable Tracker in the pure API baseline.
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
