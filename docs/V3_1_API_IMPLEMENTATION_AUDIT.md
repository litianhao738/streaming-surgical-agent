# V3.1-API Repository Implementation Audit

## Audit Scope

This document records the repository state after P2 Local Smoke completion. It
does not claim completion of the API perception baseline, tracking, learned Gate,
Specialist verification, workflow model, EventMemory, or paper evaluation.

## Current Stage State

| Stage or contract | Status | Meaning |
| --- | --- | --- |
| P0 repository scaffold | PASS | Repository, configuration, typed boundaries, and phase gate exist |
| P1 data qualification | PASS WITH CONSTRAINTS | Explicit VID30/VID31 sidecars and masks are materialized |
| Data structure | PASS | 20 JSON files parse to one canonical schema |
| Split safety | PASS | Official 10/2/8 partition is explicit and guarded |
| Official media metadata | PARTIAL | Synapse confirms 1 FPS; some extraction wording is not explicit |
| Media alignment | OPERATIONAL | VID30/VID31 use explicit sidecars; test MP4 uses the verified `frame_id - 1` rule |
| Ontology | PASS WITH PROVENANCE | CholecT50/Cholec80 mappings are recorded and locally checked |
| Partial-label semantics | PASS FOR TRAINING | Task-wise masks are frozen; official negative-sentinel wording remains unknown |
| Gold-free boundary | PASS | Inference and evaluation target types are structurally separate |
| P2 Local Smoke | PASS | Real masked step, checkpoint reload, canonical rollout, evaluation, and atomic artifacts passed |
| P3 API infrastructure | PARTIAL | Mock and real OpenRouter `openai/gpt-5.6-sol` multimodal structured smoke pass; exact immutable backend identity is unavailable |
| P4-P12 | NOT STARTED / PHASE-GATED | No later research stage has passed |

P2 is an engineering baseline, not a scientific-performance baseline. VID30 remains a documented
candidate reconstruction. VID31 contributes CholecT50 frame-level Instrument/Verb/Target/Triplet
presence plus Cholec80 phase; instance, bounding-box, operator, and tracking supervision stay
masked for VID31.

Later chronological notes in this document preserve earlier audit states. Where they describe
VID30/VID31 or MP4 indexing as fully blocked, the current table and the 2026-08-23 local manifest
supersede those statements for runtime use; the raw files themselves remain unchanged evidence.

## Implemented P1 Contracts

- `src/surgical_agent/data/parser.py`: strict read-only canonical JSON parser.
- `src/surgical_agent/data/schemas.py`: provenance-bearing canonical records and separate
  Gold-free/evaluation types.
- `src/surgical_agent/data/media_backend.py`: exact numeric PNG resolution and fail-closed
  MP4 resolution; the active Track20 profile freezes `decoder_index = frame_id - 1`.
- `src/surgical_agent/data/derived_supervision.py`: strict VID30/VID31 sidecar and frame-level
  IVT contracts.
- `src/surgical_agent/data/masks.py`: one independent per-task mask implementation.
- `src/surgical_agent/data/splits.py`: official split discovery and train-only guard.
- `src/surgical_agent/data/causal_window.py`: deterministic no-future frame windows.
- `tools/audit/audit_cholectrack20.py`: reproducible qualification and artifact generation.

No parser path clips raw bounding boxes, picks a nearest frame, moves a video across splits,
silently substitutes sidecars, or invents an ontology name. Repaired supervision is reachable
only through the explicit local manifest.

## Verified Local Evidence

- Dataset root: `D:/cholec_dataset`; runtime access is read only and original release assets are
  unchanged, while reviewed sidecars are explicitly materialized beside VID30/VID31.
- Official release partition: 10 training, 2 validation, and 8 testing videos.
- Parsed records: 35,009 annotated frames and 65,247 tool instances.
- Annotation cadence: 1 FPS, supported by official dataset documentation and the supplied
  official email.
- Raw test video rate: 25 FPS, supported by the email and confirmed for all eight local MP4s
  using `ffprobe`.
- Embedded category mapping: identical across all 20 JSONs; seven instrument and four operator
  terms are verified.
- Raw bbox audit: 64,305 boxes inside the unit frame, 941 positive-extent boundary-crossing
  boxes, and one `[-1,-1,-1,-1]` tuple. Values are preserved without interpretation.
- A read-only audit of official Synapse project `syn53182642` found the same split tree and the
  same VID30/VID31 annotation/media mismatch as the local release. No dedicated ontology mapping
  or frame-extraction/indexing document was present in the inspected tree, entity annotations,
  or Wiki pages.
- `D:/cholec_dataset/repair_manifest.json` materializes reviewed sidecars without changing the
  20-video directory partition or original JSON/media hashes.
- VID30 is a candidate reconstructed validation source. VID31 contributes CholecT50 frame-level
  Instrument/Verb/Target/Triplet presence and Cholec80 phase; instance, bounding-box, operator,
  and track supervision are disabled for VID31.

The supplied email contains a sensitive download credential. It was used only as documentary
evidence, was not copied into the repository, and is represented as `REDACTED_NOT_STORED`.

## Current Evidence Boundaries

1. The original VID30/VID31 JSON-media pairs remain unsuitable for direct use. This raw defect is
   retained as evidence; runtime code must load the explicit sidecars.
2. VID30 reconstruction is evidence-backed but not an official maintainer correction. Paper
   experiments must disclose it and include a sensitivity result that excludes VID30.
3. VID31 cannot provide instance detection, bbox-to-IVT association, operator, or tracking
   supervision. Those tasks remain masked; its frame-level Instrument/Verb/Target/Triplet
   presence must never be attached to boxes or interpreted as instance supervision.
4. Official Synapse text does not spell out every PNG extraction/index convention or negative
   sentinel meaning. Exact local/cross-dataset rules and task-wise masks make these nonblocking
   provenance caveats, not permission to invent medical semantics.
5. The declared `GPT-5.6 Terra` name is not an auditable API identity. Exact provider, endpoint
   provenance, returned model identifier, and capabilities remain blocked until P3 real smoke.
6. The first recognition slice has now frozen I/V/T/IVT as frame-level multi-label and Phase as
   frame-level single-label under `joint_perception_frame_v1`. Cross-granularity conversion remains
   forbidden; any future instance-level schema, prediction matching, and metrics still require a
   separate approval based on the P2 eligibility evidence.
7. Canonical per-sample Gate error, task normalization scales, and weights remain blocked until
   they are aligned with the single EvaluationEngine before P9.

## Reproducibility Evidence

- `reports/P1_REPORT.md`: stage decision, commands, test results, and stop condition.
- `dataset_reports/cholectrack20_usability_report.md`: human-readable data qualification.
- `dataset_reports/cholectrack20_audit_stats.json`: consolidated machine-readable audit.
- `artifacts/p1/`: split, alignment, ontology, label-mask, and canonical-schema manifests.
- `artifacts/p1/synapse_metadata_audit.json`: sanitized entity/version/Wiki/hash provenance from
  the official project; it contains no authentication token.
- `tools/audit/audit_cholectrack20_synapse.py`: optional read-only Synapse metadata audit using
  runtime environment authentication.
- `tests/unit/test_p1_data_contracts.py`: malformed-schema, masking, alignment, causality,
  split, and Gold-free failure tests.
- `tests/unit/test_p1_synapse_audit.py`: runtime-secret, semantic-hash, deterministic parsing,
  and sanitized-artifact tests.
- `tests/integration/test_p1_cholectrack20_local.py`: read-only local release checks.
- `reports/P2_REPORT.md`: P2 decision, implemented boundaries, commands, and residual gates.
- `scripts/run_local_smoke.py`: portable P2 entry point with sanitized provenance.
- `tests/unit/test_p2_*.py`: model, masked loss, checkpoint, portability, runner, Gold-free,
  reset, and artifact tests.
- `tests/integration/test_p2_local_pipeline.py`: real VID02/VID31/VID30 route and pipeline checks.

## Phase Decision

P1 is closed as `PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION`; P2 Local Smoke is closed as `PASS`.
P3 mock infrastructure and a real OpenRouter `openai/gpt-5.6-sol` multimodal
structured-response smoke are implemented. P3 remains `PARTIAL` only because
the response did not expose an immutable exact-backend identity and independent
identity evidence. The real transport, response schema, first-call accounting,
and second-call cache replay are operational. The former P4 granularity gate is closed only for
the approved frame-recognition contract: I/V/T/IVT are multi-label and Phase is single-label.
Future instance-level output and matching remain deferred and cannot reuse that approval. P9
cannot start until the canonical per-sample task error contract is approved and implemented in
the single EvaluationEngine.

## P2 Implementation Result

P2 implements `data/dataset.py`, `models/baseline.py`, `training/losses.py`,
`training/trainer.py`, `training/checkpoint.py`, `systems/pipeline.py`,
`systems/baseline_system.py`, `evaluation/evaluator.py`, `inference/writer.py`, and
`scripts/run_local_smoke.py`. The adapter verifies the official 10/2/8 split, repair-manifest
layout, original VID30/VID31 hashes, and all three materialized sidecar hashes before sampling.

The single runner uses frames-only context, local smoke perception, NeverVerify,
DisabledSpecialistRegistry, NoOpCoordinator, No-op Workflow, and No-op Memory. It preserves the
frozen order and explicit reset/commit trace. Model inputs are Gold-free; only EvaluationEngine
can compare PredictionRecord with targets. P2 outputs fixed frame-level multi-label logits plus
phase and never converts them into fabricated instances.

The 2026-08-24 full smoke audited all 20 videos and 37,675 operational samples. This count includes
the explicit VID30/VID31 sidecar frame sets and is not a replacement for the raw-release count of
35,009 annotated frames. One VID02 + VID31 training batch produced a finite five-task masked loss,
finite gradients, one optimizer update, and an exact checkpoint reload. Canonical inference then
ran VID02, VID31, and VID30 and produced hash-verified JSONL/manifest artifacts. Reported scores are
explicitly `ENGINEERING_SMOKE_NOT_PAPER_METRICS`.

### Local identity-resolution continuation (2026-08-22, historical intermediate state)

Because maintainer correction is unavailable, the repository now has a separate read-only
VID30/VID31 resolution tool and field-level supervision manifest. Local contact-sheet evidence
supports only a candidate VID31-JSON-core to VID30-media relation; no whole-JSON swap is
authorized. Cholec80 video30/video31 pixel evidence is still required before any derived field is
enabled for supervision. Negative/out-of-range labels and IVT consistency are now governed by
separate task-validity and consistency policies. This was an intermediate P1 state superseded by
the materialized sidecar decision below.

### Cholec80 frame-evidence update (2026-08-22, historical intermediate state)

Complete Cholec80 frame exports and official phase files were subsequently supplied and audited
read-only. Pixel evidence supports Track20 VID30 media -> Cholec80 video30 and Track20 VID31
media -> Cholec80 video31. This does not authorize a whole-JSON swap: VID30 JSON remains disabled
because it duplicates VID17 core annotations; VID31 bbox/IVT/tracking fields remain disabled. A
derived image-aligned VID31 phase source is available from the Cholec80 video31 phase file. The
auditable policy is recorded under `artifacts/data_resolution/cholectrack20_vid30_vid31/`.

### Materialized runtime decision (2026-08-23)

CholecT50 VID31 frame-level labels and Cholec80 phase evidence were combined with the reviewed
Track20 identity findings into three explicit sidecars and `D:/cholec_dataset/repair_manifest.json`.
The official split and class ranges remain unchanged. Repository state is now P1 `PASS`; P2 is
`PASS`.

## Pipeline-Document Synchronization Self-Check (2026-08-24)

- Research and implementation documents both freeze one canonical pipeline and
  `src/surgical_agent/` as the sole implementation package.
- At that checkpoint, the design kept future instance predictions separate from optional
  frame-level multi-label supervision and frame-level phase; cross-granularity conversion was
  required to fail closed.
- The implementation specification contains P2 file ownership, routing, target/mask rules,
  minimal training, runner ordering, artifacts, commands, tests, and PASS criteria, plus a P2-P12
  file/artifact matrix.
- `configs/base.yaml`, `configs/data/cholectrack20.yaml`, README files, and repository audit agreed
  that P1 and P2 were PASS, while P3/P4/P9 retained their stage-specific hard gates at that
  historical checkpoint.
- The default dataset config is portable (`root: null`); CLI and `CHOLECTRACK20_ROOT` resolve the
  external read-only dataset without embedding a workstation path.
- Historical P2 evidence from 2026-08-24 includes Ruff, compileall, full Pytest (`79 passed`), a
  real CPU optimizer step, checkpoint round-trip, canonical rollout, full granularity audit, and
  atomic output hashes. The independent 2026-08-25 run in the current `.venv-p2` environment,
  with `CHOLECTRACK20_ROOT` configured but the optional raw Cholec80 provenance directory absent,
  reports `78 passed, 1 skipped`; the skip is the optional upstream-hash check and does not bypass
  materialized VID30/VID31 runtime sidecar/hash tests. Both results are retained with their runtime
  conditions rather than overwriting the historical result.

## Frame-Level Joint-Perception Contract Synchronization (2026-08-28)

The approved and implemented first joint-perception slice now closes the former P4 granularity
ambiguity for frame recognition only:

- The active wire schema is `joint_perception_frame_v1`. Instrument, Verb, Target, and IVT use
  multi-label `selected_ids`; Phase alone uses the single-label `selected_id`. Every task also
  carries ranked `{id, score}` candidates with fixed counts of 7/10/15/20/7 respectively.
- The active wire schema does not accept `instances`, bbox, track identity, or per-instance
  `choice` fields. The strict parser converts it to a frame-level `InitialPrediction`; the
  `PredictionFinalizer` then creates the durable `PredictionRecord` with `instrument_ids`,
  `verb_ids`, `target_ids`, `triplet_ids`, `phase_id`, dense task score vectors,
  `granularity="frame_multilabel"`, and explicit score semantics. The writer persists that record,
  and the evaluator pairs it with supervision only at the isolated offline boundary.
- `FrameMetricAccumulator` consumes that durable contract. For I/V/T/IVT it computes AP only for
  task-valid video/class units with positive GT support, averages each class over eligible videos,
  and averages defined classes into video-wise mAP; IVT null classes 94–99 retain support but are
  excluded from AP/mAP. For Phase it computes per-video Accuracy and macro-F1 over GT-present
  classes, then averages eligible videos equally. The separate CLI in `scripts/evaluate.py` remains
  a blocked P12 entry point; implementing that entry point must reuse these
  `frame_recognition_metrics_v1` semantics rather than introduce another prediction shape or
  aggregate definition.
- Any future instance-level detection/tracking output remains deferred. It requires a separately
  approved, versioned schema, matching rule, and evaluator; a per-instance `choice` must never be
  interpreted as a frame-level unique label or mixed into `joint_perception_frame_v1`.
- This synchronization changes no dataset split, task ontology, annotation semantics, masks,
  runtime API behavior, or paper metric definition. It only makes the architecture documents match
  the already implemented frame-level contract.

This check reduces known consistency and implementation risks but is not a proof that future
algorithmic code will be defect-free. Each later phase must still pass its own unit, integration,
causality, leakage, provenance, and experiment-attribution gates before advancement.
