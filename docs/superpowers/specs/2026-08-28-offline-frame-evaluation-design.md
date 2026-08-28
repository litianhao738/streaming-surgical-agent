# Offline Prediction–GT Alignment and Frame Evaluation Design

**Date:** 2026-08-28

**Status:** Approved for implementation

**Scope:** CholecTrack20 Validation/Test frame-recognition evaluation only

## 1. Purpose

Implement the missing offline boundary that turns one completed dataset API
rollout into a deterministic frame-recognition report:

```text
completed prediction run
  -> artifact-only integrity validation
  -> split and Test-authorization gate
  -> split-local canonical selection reconstruction
  -> evaluation-only frame GT aggregation
  -> explicit scored/unscored identity alignment
  -> existing frame_recognition_metrics_v1 computation
  -> immutable evaluation report and provenance manifest
```

This work makes `scripts/evaluate.py` usable. It does not claim that all P12
experiments are implemented and does not add Gate, Specialist, Memory,
tracking, instance-detection, paired-system, or bootstrap experiment logic.

The evaluator is offline-only. It must not read an API key, create an API
client, decode surgical media, issue provider calls, update model state, or
write into the input prediction run.

## 2. Accepted architecture

Use one canonical CLI, `scripts/evaluate.py`, backed by four package-level
components with separate responsibilities:

1. **CompletedRunReader** validates only run artifacts and reconstructs strict
   `PredictionRecord` values. It never receives a dataset path.
2. **EvaluationSelectionResolver** reconstructs the selected split without
   scanning unrelated splits.
3. **FrameGroundTruthLoader** loads authorized labels and constructs
   `FrameSupervisionTarget` values.
4. **OfflineFrameEvaluator** aligns identities, invokes the existing
   `FrameMetricAccumulator`, and writes evaluation artifacts.

This is preferred over placing all logic in the script or creating a second
competing evaluation command. The CLI remains a thin stable boundary while
artifact, selection, GT, and report behavior stay independently testable.

The ordered call sequence is normative:

```text
parse CLI without credentials
  -> validate fresh/non-overlapping paths
  -> read and validate completed run artifacts only
  -> resolve effective split from artifact + predictions
  -> if Testing: require explicit authorization and paper-mode shape
  -> reconstruct only that split's canonical runtime selection
  -> load only that split's GT
  -> align, compute, persist
```

No implementation may construct `CholecTrack20DatasetAdapter` or call the
current all-split `discover_official_split_manifest()` inside this flow.
Those paths deserialize every split JSON during discovery and would violate
the split-local evaluation boundary.

## 3. Command contract

The normal Validation command is:

```powershell
.venv-p2\Scripts\python.exe scripts\evaluate.py `
  --run-dir artifacts\api_dataset\<run-id> `
  --dataset-root D:\cholec_dataset
```

Optional `--output-dir` overrides the output directory. If omitted, the
evaluator writes beside the input run as `<run-id>__evaluation`; it never
writes beneath `--run-dir`.

Test GT remains locked unless the user deliberately adds:

```text
--authorize-test-gt-evaluation
```

There is no CLI `--split`, `--video-id`, prediction-file selector,
credential, endpoint, provider, model, upload, or cache flag. Mode, split,
videos, and record locations come only from the frozen run artifacts.

Resolved `--run-dir`, `--dataset-root`, and output paths must be pairwise
non-overlapping: no path may equal, contain, or be contained by another. The
output path must be fresh. No existing directory is deleted or reused.

## 4. Completed run input contract

### 4.1 Required files

The input run must contain:

- `run_status.json`;
- `manifest.json`;
- `dataset_rollout_artifact.json`;
- every prediction/evidence JSONL named by the frame manifest.

`run_status.json` must contain exactly:

- `schema_version = "frame_result_run_status_v1"`;
- `status = "COMPLETE"`;
- a nonempty `run_id`;
- `manifest_file = "manifest.json"`.

`manifest.json` must use
`frame_result_artifact_manifest_v1`, have status `COMPLETE`, the same
run ID, a positive record count, exact `videos` entries, and metadata
`{"paper_metric_eligible": <exact bool>}`. Each video entry must contain
exactly `record_count`, `predictions_file`, `predictions_sha256`,
`evidence_file`, and `evidence_sha256`.

`dataset_rollout_artifact.json` must use
`cholectrack20_api_rollout_v1` and the producer's exact current top-level
field set:

```text
schema_version, status, run_id, mode, split, video_ids,
expected_frame_counts, completed_frame_counts, provider, model_requested,
models_returned, prompt_version, response_schema_version,
repair_manifest_sha256, alignment_versions, usage, cache_entry_count,
track20_image_uploaded, paper_metric_eligible, manifest_file
```

Its terminal status must be `MOCK_COMPLETE` for provider `mock` or
`REAL_RESPONSE_RECEIVED` for provider `openrouter`. Its manifest name and
run ID must match the other files. Expected and completed counts must be
identical. The rollout and frame-manifest `paper_metric_eligible` values must
both be exact booleans and equal; disagreement is an input error.

The strict raw type/shape contract is:

| Field | Required raw JSON contract |
|---|---|
| `schema_version`, `status`, `run_id`, `provider`, `model_requested`, `prompt_version`, `response_schema_version` | nonempty strings; schema/status/provider additionally match the constants above |
| `mode` | exact string `engineering` or `paper` |
| `split` | JSON null for engineering; exact `validation` or `testing` for paper |
| `video_ids` | nonempty canonical-order array of unique safe video-ID strings; one for engineering, two for paper Validation, eight for paper Testing |
| `expected_frame_counts`, `completed_frame_counts` | objects whose keys equal `video_ids` exactly and whose values are positive non-boolean integers; the two objects are equal |
| `models_returned` | sorted unique array of nonempty strings |
| `repair_manifest_sha256` | lowercase 64-character SHA-256 string |
| `alignment_versions` | nonempty sorted unique array of nonempty strings |
| `usage` | JSON object; opaque to frame evaluation and never copied into its report |
| `cache_entry_count` | non-negative non-boolean integer |
| `track20_image_uploaded`, `paper_metric_eligible` | exact booleans |
| `manifest_file` | exact string `manifest.json` |

The frame manifest's `record_count` and every per-video `record_count` are
positive non-boolean integers, the top-level count equals their sum, and its
video keys equal `video_ids`. Prediction/evidence filenames are nonempty safe
relative strings owned by their named video. All manifest digests are
lowercase 64-character SHA-256 strings. Metadata contains exactly the one
boolean eligibility field; no opaque metadata is accepted.

The current v1 status/manifest does not commit a producer-authored hash of
`dataset_rollout_artifact.json`. The evaluator therefore:

- verifies every producer-declared prediction/evidence file hash and every
  evidence-to-prediction link;
- validates all three top-level documents for internal consistency;
- computes hashes of the status, manifest, and rollout artifact as observed;
- records those hashes in the evaluation manifest;
- never describes the v1 rollout artifact as externally authenticated or
  promotes it to a paper-eligible artifact.

This is an explicit current-schema limitation, not a hidden integrity claim.

### 4.2 JSON parsing

All JSON documents and JSONL rows must reject duplicate object keys,
`NaN`, `Infinity`, and `-Infinity`. JSONL must contain exactly one object
per nonempty line and no blank records. Relative artifact paths must be safe,
must not escape `--run-dir` after resolution, and must match the video entry
that names them. Input files are discovered only through the manifests, never
through a glob.

### 4.3 Strict prediction reconstruction

Every prediction row must contain exactly:

```text
run_id, video_id, frame_id, source_split, causal_frame_ids,
instrument_ids, verb_ids, target_ids, triplet_ids, phase_id,
granularity, backend, gate_action, verification_status, alignment_version,
probabilities, trace, failure_reason, schema_version, score_semantics
```

The reader applies stricter raw-JSON validation before constructing the
dataclass:

- `frame_id`, `phase_id`, every causal ID, and every task ID are exact
  non-boolean integers;
- `run_id` matches the producer's safe run-ID grammar and `video_id` is one
  safe filename component matching its manifest video key;
- causal IDs and task-ID lists become tuples; their ordering, uniqueness,
  bounds, and causal tail are validated;
- `source_split` is an exact canonical split string and becomes
  `DatasetSplit`;
- `probabilities` contains exactly the five heads with lengths
  7/10/15/100/7;
- every score is a finite non-boolean number in `[0, 1]`;
- `trace` is a list of strings and becomes a tuple;
- `failure_reason` must be null for an evaluable completed run;
- all remaining textual fields are nonempty strings;
- `schema_version = "prediction_record_v1"`;
- `granularity = "frame_multilabel"`;
- `score_semantics` is `probability_v1` or
  `uncalibrated_rank_v1`.

All records in one run must use one score semantics. Mixed semantics are
rejected because their dense scores are not a comparable ranking scale within
one AP computation.

Every evidence row must contain exactly:

```text
run_id, video_id, frame_id, prediction_sha256, task_values, global_values,
evidence_version, schema_version
```

The reader validates its exact non-boolean identity types, run/video pairing,
lowercase SHA-256, `evidence_version = "evidence_frame_v1"`, and
`schema_version = "evidence_record_v1"`. Frame recognition does not
semantically decode or copy nested evidence signals; file integrity and the
prediction-hash link are sufficient for this evaluator.

Duplicate identities, mismatched prediction/evidence identity sets, record
count mismatches, wrong file hashes, mixed run IDs, mixed source splits, and
artifact/prediction alignment-version disagreement are rejected before any
dataset access.

## 5. Split-local selection and Test authorization

### 5.1 Engineering Validation

An engineering artifact has `mode="engineering"`, `split=null`, exactly one
video, and a positive bounded count. The evaluator derives one effective
split from all prediction records and accepts only Validation. Training,
Testing, or mixed engineering predictions are rejected.

The split-local resolver verifies only the named Validation directory,
annotation metadata, repair manifest, and exact PNG stems. It reconstructs
the same first-N canonical runtime selection represented by the artifact.

### 5.2 Paper-mode Validation

A paper-mode Validation artifact has `mode="paper"`,
`split="validation"`, the canonical sorted two-video Validation membership,
and no truncation. Predictions must equal the complete split-local runtime
selection in exact video-major order.

Validation evaluation may be repeated for development. It must not open,
deserialize, enumerate, or stat any path below the Testing directory. A test
uses an injected filesystem sentinel to enforce this behavior.

The phrase `paper-mode` describes rollout completeness only. It does not
mean the report is paper-metric eligible.

### 5.3 Explicit Test evaluation

Before any path below the Testing directory is accessed, all of these
artifact-only gates must pass:

1. `--authorize-test-gt-evaluation` is present;
2. run status, frame manifest, declared hashes, JSONL reconstruction, and
   internal cross-file checks pass;
3. `mode="paper"` and `split="testing"`;
4. the artifact lists eight unique videos with positive matching expected,
   completed, manifest, and reconstructed prediction counts;
5. all predictions declare source split Testing.

Only then may the split-local resolver open Testing metadata/annotations. It
reconstructs the canonical sorted eight-video runtime identity sequence and
requires the predictions to equal it exactly.

Without authorization, the evaluator fails before any dataset discovery or
GT parser is called. Engineering/truncated Test artifacts remain
non-evaluable even with authorization.

This flag creates an explicit, provenance-recorded Test evaluation action. It
does not enforce a cryptographic one-time lock or prove researcher behavior
outside the repository. The recommended protocol remains: develop on
Validation, freeze every variant, run all variants blindly on Test, seal the
artifacts, then authorize local Test evaluation without changing or rerunning
methods in response to scores.

Training evaluation is out of scope.

### 5.4 Selection proof

The evaluator computes SHA-256 over the compact canonical JSON encoding of
the ordered `[[video_id, frame_id], ...]` list:

```python
json.dumps(
    identities,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("utf-8")
```

It records hashes for the canonical runtime identities, reconstructed
prediction identities, scored identities, and unscored prediction
identities. Canonical runtime and prediction identity hashes must be equal for
paper mode. For engineering mode, prediction identities must equal the
canonical first-N sequence.

Because the v1 producer did not persist a selection-policy version or
producer-authored identity digest, this recomputation proves agreement with
the current verified dataset and resolver, not immutable historical selector
code. The report records that limitation and remains paper-ineligible.

## 6. Evaluation-only GT loader

The GT loader is not imported by the API pipeline and never produces an
`InferenceSample`.

For Validation:

- official raw annotations are used for ordinary videos;
- VID30 uses the exact repaired annotation source named by the verified
  repair manifest;
- the local repair-manifest SHA-256 must equal the rollout artifact value;
- repair-manifest paths must remain inside the resolved dataset root;
- derived `field_supervision` is applied before per-instance masks, so a
  disabled task becomes an empty tuple with a false frame mask.

VID30 provenance is explicitly
`CANDIDATE_REPAIRED_VALIDATION`. Its core labels were candidate-reconstructed
from the Track20 VID31 route, whose official split is Training. Any evaluation
containing VID30 is therefore candidate/engineering evidence and must remain
`paper_metric_eligible=false` until an independently approved GT provenance
decision replaces this contract.

For Testing, after the authorization gates, the loader uses only the official
local Testing JSON and records relative source names and SHA-256 digests.

For an instance-annotated frame, one `FrameSupervisionTarget` is built:

- Instrument IDs are the sorted union of valid instance instrument IDs.
- Verb IDs are the sorted union of valid instance verb IDs.
- Target IDs are the sorted union of valid instance target IDs.
- IVT IDs are the sorted union of valid instance triplet IDs.
- A non-phase task mask is true only when the source permits that task and
  every annotated instance has a valid label for it.
- If any instance label is unavailable for a task, that frame task is masked
  and its emitted ID tuple is empty; unavailable is never treated as negative.
- A frame with no instances receives empty tuples and false masks for all five
  tasks unless a future versioned source contract explicitly authorizes an
  exhaustive all-negative interpretation.
- Phase follows the repository's existing rule: use the single valid phase
  observed among permitted instances, reject conflicting valid phases, and
  use `None` plus a false mask when no valid phase exists.

Negative/out-of-range raw IDs remain unavailable labels. They are not
background, null, or negative classes. IVT IDs 94–99 remain valid stored
labels but are metric-excluded by the existing implementation.

## 7. Prediction/GT alignment

Rollout completeness and metric coverage are separate proofs:

1. **Runtime proof:** prediction identities must match the canonical runtime
   selection as defined in Section 5.
2. **GT proof:** every authoritative target identity inside that evaluated
   runtime selection must have exactly one prediction.
3. **Scored set:** only identities with authoritative targets are passed to
   `FrameMetricAccumulator`.
4. **Unscored runtime frames:** a canonical prediction with no authoritative
   annotation is permitted only as an explicitly classified
   `no_authoritative_gt` frame. Its identity count and digest are persisted.

This is not an arbitrary set intersection. Missing predictions for GT-bearing
frames, GT identities outside a paper runtime selection, predictions outside
the canonical runtime selection, duplicates, or replaced frames are errors.
The evaluator must reject an empty scored set.

The current local Validation audit demonstrates why this distinction is
required:

- VID110: 1716 exact PNG identities, 1713 annotation identities, with three
  media-only identities (39175, 40600, 40601);
- VID30 repaired route: 2717 PNG identities and 2717 annotation identities.

The implementation must derive these relationships from the selected data,
not hard-code those frame IDs. A paper-mode complete Validation rollout may
therefore contain 4433 valid predictions while the metric consumes 4430
authoritative targets, with the three exclusions reported rather than hidden.

## 8. Formal metrics

Aligned scored pairs are passed unchanged to the existing
`FrameMetricAccumulator`. No second metric implementation is introduced.

The nested report remains `frame_recognition_metrics_v1`:

- Instrument, Verb, Target, and IVT use mask-aware video-wise mAP;
- AP is computed only for class/video units with positive GT support;
- class AP is averaged over eligible videos, then defined classes are
  averaged;
- IVT classes 94–99 are excluded from AP/mAP;
- Phase uses per-video Accuracy and Macro-F1 over GT-present classes, then an
  equal-weight mean across eligible videos.

Dense `probability_v1` values are probabilities. Dense
`uncalibrated_rank_v1` values are bounded ranking scores for AP and must not
be described as calibrated probabilities. Phase uses the durable single
`phase_id`.

## 9. Evaluation scope and paper eligibility

`evaluation_scope` has exactly two values:

- `engineering_partial`: a valid bounded engineering Validation run;
- `paper_mode_complete`: a complete paper-mode Validation or Testing
  runtime selection.

The latter name describes selection completeness, not scientific
eligibility.

The output eligibility is true only if both equal input flags are true and no
disqualifier applies. It may be demoted but never promoted. Stable
`eligibility_reasons` include:

- `input_not_paper_eligible`;
- `engineering_partial`;
- `vid30_candidate_repair`;
- `rollout_v1_uncommitted_selection`.

With the current `cholectrack20_api_rollout_v1` producer, every evaluation
remains false because the producer writes false and v1 lacks an immutable
selection commitment. Metrics are still valid engineering evidence.

## 10. Output contracts

### 10.1 Canonical JSON

All three output documents use UTF-8 bytes from:

```python
json.dumps(
    payload,
    sort_keys=True,
    indent=2,
    ensure_ascii=True,
    allow_nan=False,
) + "\n"
```

SHA-256 always covers the exact persisted file bytes. Integer metric class IDs
are converted to canonical decimal-string object keys before serialization.

### 10.2 Status

`evaluation_status.json` uses
`offline_frame_evaluation_status_v1`.

The initial document contains exactly:

```text
schema_version, status="INCOMPLETE", run_id
```

A sanitized failure replaces it with the same fields plus one
`failure_category` from:

```text
input_contract, input_integrity, selection_mismatch, test_gt_locked,
gt_contract, alignment_mismatch, metric_failure, output_failure
```

The terminal document contains exactly:

```text
schema_version, status="COMPLETE", run_id,
report_file, report_sha256, manifest_file, manifest_sha256
```

### 10.3 Report

`evaluation_report.json` uses `offline_frame_evaluation_v1` and contains
exactly:

```text
schema_version
run_id
source                    # fixed rollout identity fields
evaluation_scope
paper_metric_eligible
eligibility_reasons       # sorted unique stable codes
test_gt_authorized
video_ids                 # canonical order
rollout_frame_counts      # video -> int
scored_frame_counts       # video -> int
unscored_prediction_counts # video -> int
unscored_prediction_identity_sha256
gt_task_valid_frame_counts# task -> video -> int
gt_provenance             # video -> official_raw/candidate status
metrics                   # exact frame_recognition_metrics_v1 payload
```

The exact `source` keys are `mode`, `split`, `provider`, `model_requested`,
`models_returned`, `prompt_version`, `response_schema_version`, and
`alignment_versions`. `gt_provenance` values are exactly `official_raw` or
`candidate_repaired_validation`. The unscored identity digest is the Section
5.4 hash of the canonical empty list when no runtime prediction is unscored.

`test_gt_authorized` is false for Validation and true only for an authorized
Testing evaluation.

### 10.4 Manifest

`evaluation_manifest.json` uses
`offline_frame_evaluation_manifest_v1`, status `COMPLETE`, and contains
exactly:

```text
schema_version, status, run_id, report_file, report_sha256
input_hashes
  run_status_sha256, frame_manifest_sha256, rollout_artifact_sha256
  prediction_files, evidence_files
dataset
  repair_manifest_sha256, gt_sources
selection
  identity_schema_version="ordered_video_frame_identity_v1"
  canonical_runtime_identity_sha256
  prediction_identity_sha256
  scored_identity_sha256
  unscored_prediction_identity_sha256
prediction_schema_version
metric_schema_version
score_semantics
paper_metric_eligible
```

`prediction_files` and `evidence_files` are video-keyed mappings with exact
`relative_path` and `sha256` fields. `gt_sources` is a video-keyed mapping with
exact `relative_path`, `sha256`, and `provenance_status` fields. All paths are
safe relative paths. No label values, absolute user paths, provider response
text, free-form model output, or credentials are copied.

The writer persists `INCOMPLETE` first, then the report, then the manifest,
re-reads and verifies their hashes, and finally publishes `COMPLETE`.
Failures never publish a complete report status.

## 11. Failure behavior

All integrity, selection, GT, alignment, metric, and serialization violations
fail closed. The CLI exits nonzero and prints only a concise sanitized error.

Especially:

- an incomplete or tampered prediction/evidence run is rejected;
- a Test run without authorization fails before dataset access;
- a Test engineering subset is rejected even with authorization;
- a missing/replaced GT-bearing frame is rejected;
- a canonical media-only frame is explicitly unscored, counted, and hashed;
- partial task labels are masked, not converted into negatives;
- current paper ineligibility and VID30 candidate provenance are preserved.

## 12. Implementation boundaries

Expected production changes are limited to:

- strict artifact reader(s) under `src/surgical_agent/evaluation/`;
- split-local selection and frame-GT loaders under that package;
- offline coordinator/report writer under that package;
- replacement of the blocked `scripts/evaluate.py` with a thin CLI;
- focused exports, tests, and a short usage-document update.

The API request path, runtime prediction wire schema, `PredictionRecord`,
formal metric formulas, official split membership, and model configuration do
not change.

## 13. Verification plan

Tests must cover at least:

1. exact input document/row fields, duplicate JSON keys, primitive bool/type
   rejection, tuple/enum normalization, and uniform score semantics;
2. incomplete status, unsafe path, bad hash/count, duplicate identity,
   prediction/evidence mismatch, and top-level cross-file disagreement;
3. engineering `split=null` resolving only Validation;
4. paper Validation selection without touching any Testing path;
5. Test rejection without authorization before any dataset function/path is
   accessed;
6. Test rejection for engineering/truncated artifacts even with
   authorization;
7. authorized complete Test evaluation on synthetic fixtures only;
8. repair-manifest digest/path containment, VID30 field qualification, and
   candidate provenance/eligibility;
9. instance-to-frame unions, partial-label masking, empty-frame fail-closed
   behavior, and phase conflict/no-phase behavior;
10. exact runtime selection proof, missing/replaced frame rejection, and
    equal-count wrong-frame rejection;
11. a media-only prediction fixture that is explicitly unscored and reported,
    while every GT-bearing frame remains required;
12. metric reuse and formal report serialization;
13. output non-overlap, fresh lifecycle, canonical bytes, hashes, sanitized
    failure categories, and no writes beneath input/dataset roots;
14. operational sentinels proving no API client, provider call, credential
    read, or real Test scoring occurs.

After focused tests, run the full repository suite. A local read-only
Validation integration check must confirm the actual media/GT count behavior.
Automated tests must not unlock or compute metrics from the user's real Test
GT by default.

## 14. Explicit non-goals

- No training, threshold tuning, calibration, or model update.
- No API calls or API-key handling.
- No instance detection/tracking metric or matching rule.
- No Gate/Specialist/Memory metric implementation.
- No paired baseline/full comparison or bootstrap CLI in this increment.
- No cryptographic one-use Test lock or claim about behavior outside the repo.
- No external leaderboard or submission integration.
- No new paper-eligibility producer schema in this increment.
