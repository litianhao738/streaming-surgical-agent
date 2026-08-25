# P1 Report: CholecTrack20 Data Qualification and Canonical Parser

## Stage

`P1 - Data Audit + Canonical Parser + Engineering Hard Gates`

## Status

`P1_STATUS: PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION`

P1 implementation and tests were executed. The official raw VID30/VID31 JSON files remain
misaligned, but their reviewed sidecars are materialized under `D:/cholec_dataset` and are guarded
by task-specific masks. The data layer is ready for the next implementation stage under this
explicit protocol. No P2 or later stage was executed.

| Contract | Status |
| --- | --- |
| Data structure | PASS |
| Official media metadata | PARTIAL |
| Canonical parser | PASS |
| Split safety | PASS |
| Causality | PASS |
| Gold-free runtime boundary | PASS |
| Media alignment | OPERATIONAL via explicit sidecars; raw VID30/VID31 remain invalid |
| Ontology provenance | PASS with recorded CholecT50/Cholec80 provenance |
| Partial-label semantics | PASS for training masks; official negative-sentinel wording unresolved |
| Data ready for a future isolated P2 smoke | PASS |
| Full local numeric training | PASS with explicit task masks and sidecars |
| Semantic API data input | READY; API implementation remains P3 work |
| Formal training/evaluation data layer | READY_WITH_CONSTRAINTS |
| VID30 | CANDIDATE_REPAIRED_VALIDATION |
| VID31 | PHASE_AND_FRAME_LEVEL_IVT_TRAINING; instance tasks masked |
| P3 API engineering stage | NOT STARTED; P1 data blocker removed |
| P9 prerequisite | BLOCKED/TBD |

## Repository/Data Examined

- The two current V3.1-API architecture/implementation documents, README, P0 audit/report,
  data configuration, existing data code, schemas, tests, and audit artifacts.
- Read-only local root `D:/cholec_dataset`, including official split directories, 20 JSONs,
  22,681 PNGs, 8 MP4s, local release manifests, license, and DUA metadata.
- Official CholecTrack20 repository/paper, official IVT numeric map, parent CholecT50 format,
  and the supplied official dataset email. The email credential was not retained.
- Official Synapse project `syn53182642`, dataset folder `syn60059476`, Wiki pages `628401` and
  `628453`, all non-frame entity annotations, file-handle metadata, and VID30/VID31 remote frame
  filename sets. Synapse operations were read only.

## Files Created

- `src/surgical_agent/data/{parser,masks,media_backend,splits,causal_window,targets}.py`
- `tools/audit/audit_cholectrack20.py`
- `tools/audit/audit_cholectrack20_synapse.py`
- `tests/unit/test_p1_data_contracts.py`
- `tests/unit/test_p1_synapse_audit.py`
- `tests/integration/test_p1_cholectrack20_local.py`
- `reports/P1_REPORT.md`
- Five machine-readable `artifacts/p1/*.json` manifests.
- `artifacts/p1/synapse_metadata_audit.json` and one necessary 2.23 MB official VID31 JSON copy.

## Files Modified

- `src/surgical_agent/data/schemas.py`
- `src/surgical_agent/inference/engine.py`
- `tests/unit/test_p0_repository.py`
- `configs/data/cholectrack20.yaml`
- `README.md`, `reports/P0_REPORT.md`, and `docs/V3_1_API_IMPLEMENTATION_AUDIT.md`
- `dataset_reports/cholectrack20_audit_stats.json`
- `dataset_reports/cholectrack20_usability_report.md`

The two V3.1 research-design documents were not modified, and no source dataset file changed.

## Commands Actually Executed

```text
.\.venv\Scripts\python.exe tools\audit\audit_cholectrack20.py --dataset-root D:\cholec_dataset --ivt-map tmp\pdfs\cholectrack20email\ivt-maps.txt
.\.venv\Scripts\python.exe tools\audit\audit_cholectrack20_synapse.py --project-id syn53182642 --dataset-root D:\cholec_dataset --output artifacts\p1\synapse_metadata_audit.json
.\.venv\Scripts\python.exe -m ruff check src tests tools scripts
.\.venv\Scripts\python.exe -m compileall -q src scripts tools tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\repository_audit.py
.\.venv\Scripts\python.exe -m pytest -q
```

The Synapse command received authentication only through its process environment. The temporary
official IVT reference and PDF render were removed after evidence generation.

## Tests Actually Executed

- Canonical valid/malformed parsing and raw-ID preservation.
- Independent partial-label masking and conservative `-1` handling.
- Exact PNG resolution, no nearest fallback, and fail-closed unresolved MP4 offset.
- Official split preservation and train-only held-out-data rejection.
- Deterministic causal ordering and no-future-frame invariant.
- Structural Gold-free inference/evaluation separation and runtime type rejection.
- Full local split/parser totals plus explicit `VID30`/`VID31` blocker checks.
- Runtime-only Synapse authentication, deterministic entity/annotation parsing, semantic JSON
  equivalence, and generated-artifact no-JWT checks.

## Actual Test Outputs

- Pytest final: `41 passed`.
- Ruff: `All checks passed!`
- Compileall: PASS.
- Pip check: `No broken requirements found.`
- P0 repository scaffold audit: PASS; it remains a P0-scope check, not P1 evidence.

## Dataset Split Audit

- Training: 10 videos; Validation: 2 videos; Testing: 8 videos.
- Every video appears exactly once and its JSON `video.split` matches its official directory.
- No random re-split, merge, silent exclusion, or held-out fit was performed.

## Annotation Schema Audit

- All 20 JSONs expose `info`, `annotations`, `categories`, and `video` and parse strictly.
- Canonical totals: 35,009 annotated frames and 65,247 tool instances.
- Parsed fields cover instrument, verb, target, phase, triplet, bbox, operator, three track IDs,
  score/area/crowd, and ten visual-condition flags. Missing required fields: zero.
- Annotation object keys are preserved as integer canonical frame IDs; media meaning is handled
  only by the alignment layer.
- Raw bbox values are not clipped: 64,305 are inside the unit frame, 941 positive-extent boxes
  cross its boundary, and one raw tuple is `[-1,-1,-1,-1]`.
- Exact ranges and observed categorical value sets are in
  `artifacts/p1/canonical_schema_summary.json`.

## Ontology Audit

- All 20 JSON `categories` objects are identical (SHA-256 recorded in the artifact).
- Instrument IDs 0-6 and operator IDs 0-3 have verified dataset-embedded names.
- Verb, target, triplet names and phase numeric ID-to-name mapping remain unresolved.
- Synapse Wiki `628401` confirms the instrument names, but neither inspected Wiki, the complete
  non-frame entity annotation set, nor the only Wiki attachment (`ct20-img.png`) contains the
  unresolved numeric ontology tables.
- The official 100-row IVT numeric relation agrees on 22,960 of 22,994 comparable instances.
  The 34 mismatches are provenance-recorded; no class or component was corrected.

## Official Sources Found

| Source | Claim supported | Local verification | Accessed |
| --- | --- | --- | --- |
| https://github.com/CAMMA-public/cholectrack20 | 20 videos, 10/2/8 split, 1 FPS annotation/PNG and 25 FPS raw test video | Split, files, FPS metadata | 2026-08-20 |
| https://openaccess.thecvf.com/content/CVPR2025/papers/Nwoye_CholecTrack20_A_Multi-Perspective_Tracking_Dataset_for_Surgical_Tools_CVPR_2025_paper.pdf | Dataset design and annotation provenance | Compared with local schema | 2026-08-20 |
| https://github.com/CAMMA-public/ivtmetrics/blob/main/ivtmetrics/maps.txt | Candidate IVT-to-I/V/T numeric relation | Compared with 22,994 local instances | 2026-08-20 |
| https://github.com/CAMMA-public/cholectrack20/issues/10 | Public VID30/VID31 mismatch report | Matches local ID-set diagnostics | 2026-08-20 |
| https://doi.org/10.7303/syn53182642 | Official tree, Wiki, annotations, versions, and file handles | Authenticated read-only audit; no access errors | 2026-08-21 |
| Supplied official dataset email PDF | 1 FPS annotations, 25 FPS raw test video, license/access context | FPS confirmed by local files | 2026-08-20 |

## Raw-Release Alignment Audit (Historical Evidence)

- PNG videos: 8 `EXACT`, 2 `EXTRA_MEDIA_ONLY`, and 2 `NONTRIVIAL_ALIGNMENT`.
- Test MP4s: all 8 probe at 25 FPS; this original audit had not yet frozen the decoder offset.
  The later cross-dataset audit resolved the runtime rule as `decoder_index = frame_id - 1`.
- Synapse Wiki `628401` independently confirms 1 FPS annotations, but does not document source
  frame extraction, PNG naming semantics, decoder index base, or an offset rule.
- Synapse remote frame filename sets exactly equal the local sets for VID30 (2,717) and VID31
  (3,903). Four dispersed PNG files per video also have exact remote/local MD5 matches.
- Resolution is exact numeric PNG-stem equality only. There is no nearest-frame, default-offset,
  or cross-video fallback.

## Raw VID30 Result (Before Sidecar Reconstruction)

`BLOCKED_ALIGNMENT`. Its Validation annotation has 1,066 IDs; local PNGs have 2,717 IDs, with
748 in common, 318 annotated IDs missing media, and 1,969 extra PNG IDs. The PNG ID set exactly
equals the `VID31` JSON annotation set, but using that as a repair would cross official splits and
is forbidden without a corrected official release.

The current official Synapse release reproduces this state. `VID30.json` is entity
`syn60059640`, version 1, under `syn60059639`; its remote and local MD5 are both
`ddc8775b343ff4bb36b59cc4b22b1ea8`. The official Frames folder `syn60059641` contains the
same 2,717 numeric PNG names as the local copy, and four dispersed remote/local PNG samples have
identical MD5 values. This rules out an incomplete or altered local download as the explanation.

## Raw VID31 Result (Before Sidecar Reconstruction)

`BLOCKED_ALIGNMENT`. Its Training annotation has 2,717 IDs; local PNGs have 3,903 IDs, with
2,698 in common, 19 annotated IDs missing media, and 1,205 extra PNG IDs. The official issue and
visual spot checks indicate that this annotation likely describes VID30, but no valid replacement
VID31 annotation is available.

The current official Synapse release also reproduces this state. `VID31.json` is entity
`syn60064538`, version 1, under `syn60064532`; the official Frames folder is `syn60064541`.
The remote and local JSON files differ in raw MD5 because their serialization differs, but their
parsed JSON values have the same canonical SHA-256
`ea248dd3b33b2df29b70b0d4a9f01c3844879b6fab404dc07935d61351a2180b`. All 3,903 remote PNG
names equal the local set, and four dispersed remote/local PNG samples have identical MD5 values.

A corrected official entity, an extraction/alignment manifest, or a maintainer explanation is
required. A complete dataset re-download cannot resolve mismatches already present in the current
official release.

## Partial-Label Semantics

- One canonical implementation treats each task independently as valid only when its raw ID is
  a non-boolean integer greater than or equal to zero.
- Raw `-1` is named only `UNRESOLVED_MISSING_SENTINEL`, never none/background/negative.
  Additionally, `triplet=-2` occurs 1,129 times across five videos; it is independently recorded
  as unresolved and receives no semantic class name.
- V/T are valid for 24,122 instances; IVT is valid for 22,994. Four validity combinations occur,
  proving that IVT validity cannot be inferred from V/T validity.
- This P1 mask determines supervision availability only. It does not define a Gate error metric.

## Canonical Parser Status

`PASS`. It validates observed raw schema, preserves numeric IDs and raw bbox/track values, records
source provenance, and emits no unverified medical names. Invalid schemas fail explicitly.

## Gold-Free Status

`PASS`. `InferenceSample` has only video/frame/media/split/alignment information. GT instances,
label masks, phases, IVT, tracks, and correctness flags exist only in `EvaluationTarget`; runtime
rejects the evaluation type.

## Split Safety

`PASS`. Official directories plus JSON metadata are the split source. `require_train_only()`
rejects Validation/Test entries, and local integration tests assert one split per video.

## Causality Status

`PASS` for the P1 foundation. Causal frame IDs are deterministic, increasing, video-local through
their resolver, end at the target, and cannot include a frame ID greater than the target.

## Historical Blockers Before Cross-Dataset Resolution

- Corrected official `VID30`/`VID31` entities, extraction/alignment manifest, or maintainer
  explanation. Current official Synapse entities and local files already agree.
- Explicit annotation ID to MP4 decoder-index rule.
- Release-specific verb/target/triplet names and phase numeric mapping.
- Authoritative semantic meaning of categorical `-1`, triplet `-2`, and the all-`-1` bbox tuple.

## P2 Readiness

`DATA_READY_FOR_P2_SMOKE: PASS`. After the 2026-08-23 materialization, training-oriented P2 work
may also use the explicit sidecar protocol; P2 itself has not yet been executed.

## P3 Readiness

The Gold-free data payload boundary is ready for later API-client engineering, but P3 remains
`NOT STARTED` and blocked by phase order. Provider, endpoint, credentials, exact model identifier,
generation parameters, cache, usage, and retry are intentionally not frozen here.

## 2026-08-22 VID30/VID31 Resolution Continuation (Historical Intermediate State)

P1 remains `PARTIAL`; no later research stage was started. A dedicated read-only workflow was
added under `tools/data_resolution/cholectrack20_vid30_vid31/` rather than embedding release
exceptions in the runtime parser.

- Local execution generated all four VID30/VID31 JSON-to-PNG contact sheets, a local identity
  audit, and a field-level supervision manifest under
  `artifacts/data_resolution/cholectrack20_vid30_vid31/`.
- A dispersed 12-frame visual review supports the VID31 JSON geometry as a candidate for VID30
  media and rejects both nominal JSON/media pairings as complete pairings. This is supporting
  evidence only; phase and scene-condition fields are not transferred.
- All raw files remain present and unchanged. VID30/VID31 are retained for unsupervised/manual
  review use, while unresolved supervision fields remain disabled in the derived manifest.
- Task IDs now require verified task-specific ranges. Negative/out-of-range values disable only
  their own task. IVT component consistency is separate and records exact relations,
  specimen-bag scope differences, partial-label non-applicability, and unresolved conflicts
  without rewriting raw labels.
- Cholec80 `video30` and `video31` MP4 files are still required for the 2x2 pixel-identity audit.
  The whole Cholec80 release is not required. Cholec80 cannot reconstruct missing Track20 bbox or
  three-perspective track IDs.

Verification after this continuation: Ruff PASS, compileall PASS, pip check PASS, repository
audit PASS, and Pytest `46 passed`. The upstream MP4 branch was additionally exercised end to end
on two synthetic videos and correctly recovered the known 2x2 identity assignment.

## 2026-08-22 Cholec80 Frame Evidence Update (Historical Intermediate State)

Complete Cholec80 frame exports for videos 30 and 31, plus their official phase files, were
audited read-only. This resolves media identity but does not make the malformed Track20 JSON files
ordinary training pairs.

- Pixel retrieval over 20 dispersed samples identifies Track20 VID30 media as Cholec80 `video30`
  (median perceptual-hash distance `8`, grayscale correlation `0.882`), rather than `video31`
  (distance `16`, correlation `0.447`).
- It identifies Track20 VID31 media as Cholec80 `video31` (distance `6`, correlation `0.935`),
  rather than `video30` (distance `16`, correlation `0.503`).
- VID30 JSON remains disabled for formal labels: its core annotation duplicates VID17 and its
  local phase agrees with Cholec80 video30 on only `226/1,066` annotation frames.
- VID31 JSON core fields remain disabled for formal tracking and IVT supervision. Its local phase
  agrees with Cholec80 video31 on `2,498/2,717` frames after the observed rule
  `Cholec80 phase frame = Track20 annotation frame - 2`; a separate derived override now provides
  all 2,717 official phase values without modifying Track20 JSON.
- New evidence: `upstream_identity_audit.json`, `upstream_phase_alignment.json`,
  `derived/VID31/vid31_image_phase_supervision.json`, `derived_supervision_manifest.json`, and four 2x2 comparison
  contact sheets under `artifacts/data_resolution/cholectrack20_vid30_vid31/`.

The conservative formal-training policy is therefore: retain VID30 as visual-only/manual-QA data;
retain VID31 core labels as disabled; allow only a future per-sample phase loader to consume the
image-aligned VID31 phase supervision. P1 remains `PARTIAL` because partial-label semantics,
formal test-MP4 decoding alignment, and the raw VID30/VID31 core-label provenance remain open.

## Current Formal-Training Readiness

`LOCAL_NUMERIC_TRAINING_READY: PASS_WITH_EXPLICIT_MASKS_AND_SIDECARS` and
`FORMAL_TRAINING_READY: READY_WITH_DOCUMENTED_CONSTRAINTS`. The raw VID31 JSON is never paired
with VID31 images; VID31 contributes CholecT50 frame-level Instrument/Verb/Target/Triplet
presence plus Cholec80 phase, with instance/bbox/operator/track supervision disabled. VID30 uses the documented
candidate validation reconstruction. API implementation remains a later project stage rather than
a data blocker.

## 2026-08-23 Materialized Runtime Decision

- Active manifest: `D:\cholec_dataset\repair_manifest.json`.
- VID30: `Validation\VID30\vid30_repaired.json`, candidate reconstructed validation labels.
- VID31: `Training\VID31\vid31_phase_repaired.json` plus
  `vid31_frame_ivt_repaired.json`; all instance-level tasks are masked.
- Official 10/2/8 split, 20 video directories, and all original class ranges remain unchanged.
- Original JSON and media files remain present and hash-identical to the audited inputs.
- Test MP4 decoding is frozen to `decoder_frame_index = annotation_frame_id - 1`.
- Current repository verification after pipeline-document synchronization: Ruff PASS, compileall
  PASS, pip check PASS, repository audit PASS, Pytest `63 passed`.

## P9 Prerequisites Still Unresolved

`P9_PREREQUISITE: BLOCKED/TBD`. Canonical per-sample task errors, task normalization scales,
normalized verification benefit, Gate targets, and weights were not defined in P1.

## Generated Evidence

- `artifacts/p1/dataset_split_manifest.json`
- `artifacts/p1/media_alignment_manifest.json`
- `artifacts/p1/ontology_provenance.json`
- `artifacts/p1/partial_label_statistics.json`
- `artifacts/p1/canonical_schema_summary.json`
- `artifacts/p1/synapse_metadata_audit.json`
- `artifacts/p1/synapse_metadata/vid31.json` (small official metadata copy; semantic comparison)
- `dataset_reports/cholectrack20_audit_stats.json`
- `dataset_reports/cholectrack20_usability_report.md`

## Stop Condition

`NEXT_RECOMMENDED_STAGE: START_P2_LOCAL_SMOKE_WITH_EXPLICIT_SIDECARS`

```text
P1 DATA CONTRACT READY WITH DOCUMENTED CONSTRAINTS; P2/P3/TRAINING/GATE NOT YET EXECUTED
```
