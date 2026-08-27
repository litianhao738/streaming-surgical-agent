# Offline Prediction–GT Alignment and Frame Evaluation Design

**Date:** 2026-08-28

**Status:** Approved in chat; written specification pending user review

**Scope:** CholecTrack20 Validation/Test frame-recognition evaluation only

## 1. Purpose

Implement the missing offline boundary that turns one completed dataset API
rollout into a deterministic frame-recognition report:

```text
completed prediction run
  -> verified PredictionRecord stream
  -> independently loaded and aggregated frame GT
  -> exact (video_id, frame_id) join
  -> existing frame_recognition_metrics_v1 computation
  -> immutable evaluation report and provenance manifest
```

This work makes `scripts/evaluate.py` usable. It does not claim that all P12
experiments are implemented and does not add Gate, Specialist, Memory,
tracking, instance-detection, paired-system, or bootstrap experiment logic.

The evaluator is offline-only. It must not read an API key, create an API
client, decode surgical media, issue provider calls, update model state, or
write into the input prediction run.

## 2. Accepted approach

Use one canonical CLI, `scripts/evaluate.py`, backed by small package-level
components with separate responsibilities:

1. a completed-run reader verifies and reconstructs durable predictions;
2. an evaluation-only GT loader aggregates authorized CholecTrack20 labels;
3. an alignment coordinator proves the prediction/GT identity sets match;
4. the existing `FrameMetricAccumulator` computes the formal metrics;
5. an evaluation writer stores a report and its provenance atomically.

This is preferred over either placing all logic in the script or creating a
second competing evaluation command. The script stays a thin, stable entry
point while the integrity, GT, and report logic remains independently
testable.

## 3. Command contract

The normal Validation command is:

```powershell
.venv-p2\Scripts\python.exe scripts\evaluate.py `
  --run-dir artifacts\api_dataset\<run-id> `
  --dataset-root D:\cholec_dataset
```

Optional `--output-dir` overrides the fresh evaluation output directory. If
omitted, the evaluator writes to the input run's parent as
`<run-id>__evaluation`; it never writes beneath `--run-dir`.

Test GT is locked unless the user deliberately adds:

```text
--authorize-test-gt-evaluation
```

There is no CLI `--split`, `--video-id`, or prediction-file selector. Mode,
split, videos, and record locations come only from the run artifacts so a
second set of user arguments cannot disagree with the frozen rollout.

There are no credential, endpoint, provider, model, upload, or cache flags.

## 4. Accepted input run

The evaluator accepts only a run directory containing:

- `run_status.json` with schema `frame_result_run_status_v1`, status
  `COMPLETE`, the same run ID, and the expected manifest reference;
- `manifest.json` with schema `frame_result_artifact_manifest_v1`, status
  `COMPLETE`, nonzero records, and the same run ID;
- `dataset_rollout_artifact.json` with schema
  `cholectrack20_api_rollout_v1`, terminal status `MOCK_COMPLETE` or
  `REAL_RESPONSE_RECEIVED` consistent with its provider, and rollout selection
  metadata consistent with the frame manifest;
- every prediction and evidence JSONL file named by the manifest.

The reader must not discover input JSONL files by glob and must reject unsafe
or escaping relative paths. It verifies every manifest file SHA-256, record
count, prediction/evidence identity set, evidence-to-prediction SHA-256 link,
duplicate identity, run ID, video ID, source split, schema version,
granularity, dense-head length, ID range, finite score, score semantics, and
rollout frame count before returning any `PredictionRecord`.

JSON values are normalized explicitly before dataclass construction:

- `source_split` becomes `DatasetSplit`;
- JSON arrays become tuples;
- probability heads become immutable task mappings;
- unknown or missing fields are rejected rather than ignored.

The reader recomputes the canonical rollout selection from `--dataset-root`
and requires exact ordered `(video_id, frame_id)` equality. This closes the
current artifact gap in which equal counts alone cannot prove that the same
frames were processed.

For an engineering run, the rollout artifact has no split selector. The
evaluator therefore resolves one effective split from all reconstructed
`PredictionRecord.source_split` values and accepts only Validation. Mixed,
Training, or engineering Testing predictions are rejected.

## 5. Split and leakage policy

### 5.1 Validation

Completed engineering and paper Validation runs may be evaluated repeatedly.
An engineering/truncated run produces a valid engineering report but can
never be promoted to a paper result. A complete paper Validation run may be
evaluated, but its paper eligibility still inherits the input artifact's
eligibility flag.

### 5.2 Testing

Testing is a deliberate one-way evaluation action. Before any Test annotation
JSON is opened, all of the following must already be true:

1. `--authorize-test-gt-evaluation` is present;
2. the run lifecycle and all artifact hashes pass;
3. the rollout artifact declares `mode=paper` and `split=testing`;
4. the artifact internally declares a non-truncated eight-video selection and
   its manifest identities match its declared video/frame counts.

Only after those pre-unlock gates pass may the evaluator parse the Test GT.
It then recomputes the canonical dataset selection and requires the run's
prediction identities to equal every canonical Testing identity exactly. A
failure in either the canonical selection or the GT join still leaves the
evaluation output incomplete.

Engineering or truncated Test artifacts are never GT-evaluable. Without the
authorization flag, the evaluator fails before calling the GT parser. The
recommended research protocol remains: develop and select on Validation,
freeze every variant, run all variants blindly on Test, seal their artifacts,
then unlock the local Test GT once and evaluate them without changing or
rerunning the methods in response to scores.

The evaluator cannot prove researcher behavior outside the repository, so it
records the explicit Test-unlock event and all input hashes instead of making
a stronger claim.

Training evaluation is out of scope for this entry point.

## 6. Evaluation-only GT loader

Runtime inference continues to receive only `InferenceSample`. The new GT
loader lives under the evaluation package and is never imported by the API
pipeline.

For Validation, it uses the verified dataset repair manifest, including the
repaired VID30 annotation source. For Testing, after the unlock gates pass, it
uses the official local Testing JSON. It records relative source names and
SHA-256 digests in evaluation provenance.

For an instance-annotated frame, the loader constructs one
`FrameSupervisionTarget` as follows:

- Instrument IDs are the sorted union of valid instance instrument IDs.
- Verb IDs are the sorted union of valid instance verb IDs.
- Target IDs are the sorted union of valid instance target IDs.
- IVT IDs are the sorted union of valid instance triplet IDs.
- A non-phase task mask is true only when every annotated instance has a valid
  label for that task. If any instance is masked for that task, the frame task
  is masked and its emitted ID tuple is empty; the missing label is never
  treated as a negative.
- An annotation frame with no instances is not assigned an inferred semantic:
  all five task masks are false unless a future versioned source contract
  explicitly declares that an empty list is exhaustive all-negative
  supervision. The currently observed Validation and Testing sources contain
  no empty annotation frames.
- Phase follows the repository's existing phase-only rule: use the single
  valid phase observed among instances; reject conflicting valid phases; use
  `phase_id=None` and a false phase mask when no valid phase exists.

Negative or out-of-range raw IDs remain unavailable labels. They are not
background, null, or negative classes. IVT IDs 94–99 remain valid stored
labels but are excluded from formal IVT AP/mAP by the existing metric code.

The GT loader returns only the selected frame identities. Any selected frame
without a constructible target is an error; it is never silently dropped.

## 7. Alignment and formal metrics

Predictions and targets are indexed by the exact pair
`(video_id, frame_id)`. The evaluator rejects:

- duplicate predictions or targets;
- missing or extra identities on either side;
- mixed splits or run IDs;
- unexpected video ordering or non-increasing frame ordering;
- a paper run whose selection differs from the canonical full split.

After alignment, the pairs are passed unchanged to the existing
`FrameMetricAccumulator`. No second metric definition is introduced.

The formal report remains `frame_recognition_metrics_v1`:

- Instrument, Verb, Target, and IVT: mask-aware video-wise mAP;
- AP is computed only for class/video units with positive GT support;
- class AP is averaged over eligible videos, then defined classes are
  averaged;
- IVT classes 94–99 are excluded from AP/mAP;
- Phase: per-video Accuracy and Macro-F1 over GT-present classes, then an
  equal-weight mean across eligible videos.

Dense `probabilities` may use `probability_v1` or
`uncalibrated_rank_v1`. The latter is a ranking score for AP, not a calibrated
probability. Phase uses the durable single `phase_id` prediction.

## 8. Output artifacts

The output directory must be fresh. It contains:

```text
<run-id>__evaluation/
  evaluation_status.json
  evaluation_report.json
  evaluation_manifest.json
```

`evaluation_report.json` uses wrapper schema
`offline_frame_evaluation_v1` and contains:

- input run ID, mode, split, evaluated videos, and evaluated frame counts;
- evaluation scope: `engineering_partial` or `paper_complete`;
- inherited `paper_metric_eligible` without promotion;
- `test_gt_authorized`;
- GT task-valid frame counts;
- the exact nested `frame_recognition_metrics_v1` report.

Integer class IDs are serialized as canonical decimal-string JSON object
keys. Non-finite numbers are forbidden.

`evaluation_manifest.json` records:

- SHA-256 of the report, input manifest, and rollout artifact;
- referenced prediction/evidence file hashes;
- dataset repair-manifest and GT-source hashes;
- input prediction and metric schema versions;
- score semantics and selection-completeness evidence;
- report status and paper eligibility.

`evaluation_status.json` is written `INCOMPLETE` first and becomes `COMPLETE`
only after report and manifest hashes verify. Failures leave a sanitized
failure category and never expose annotation values, model output text,
provider errors, paths containing credentials, or credentials themselves.

The evaluator may compute metrics from current runs whose
`paper_metric_eligible` is false, but the output remains false. Evaluation is
useful engineering evidence; it is not allowed to upgrade scientific
eligibility.

## 9. Failure behavior

All integrity, selection, GT, alignment, and serialization violations fail
closed before a COMPLETE evaluation status is published. No partial metric
report is presented as valid. The CLI exits nonzero and prints only a concise,
sanitized error.

Especially:

- an INCOMPLETE or tampered prediction run is rejected;
- a Test run without explicit authorization is rejected before GT parsing;
- a Test engineering subset is rejected even with authorization;
- a missing/extra/replaced frame is rejected, not evaluated on an
  intersection;
- partial task labels are masked, not converted into negative labels;
- current `paper_metric_eligible=false` is preserved.

## 10. Implementation boundaries

Expected production changes are limited to:

- a strict completed-run reader under `src/surgical_agent/evaluation/`;
- a frame-GT aggregation loader under `src/surgical_agent/evaluation/`;
- an offline evaluation coordinator/report writer under
  `src/surgical_agent/evaluation/`;
- replacement of the blocked `scripts/evaluate.py` with a thin CLI;
- exports, focused unit/integration tests, and a short usage document update.

The runtime dataset iterator, API request path, prediction wire schema,
`PredictionRecord`, and formal metric formulas do not change.

## 11. Verification plan

Tests must cover at least:

1. completed-run reconstruction and explicit JSON-to-dataclass normalization;
2. incomplete status, unsafe path, bad hash, bad count, duplicate identity,
   prediction/evidence mismatch, and tampered prediction rejection;
3. exact canonical selection comparison, including equal-count wrong-frame
   rejection;
4. instance-to-frame unions, partial-label masking, fail-closed empty-instance
   frames, and phase conflict/no-phase behavior;
5. exact prediction/GT joining with missing and extra frames rejected;
6. Validation engineering and complete-paper reports with inherited paper
   eligibility;
7. Test rejection without authorization before the GT parser is called;
8. Test rejection for engineering/truncated artifacts even with
   authorization;
9. authorized complete Test evaluation on synthetic fixtures;
10. canonical JSON output, hashes, lifecycle, and no writes beneath the input
    run;
11. CLI help/argument behavior and confirmation that evaluation cannot create
    an API client or consume a credential;
12. regression tests proving the existing formal metric semantics are reused.

After focused tests, run the full repository suite. Local real-dataset
Validation can be used as a read-only integration check. Automated tests must
not unlock or score the user's real Test GT by default.

## 12. Explicit non-goals

- No training, threshold tuning, calibration, or model update.
- No API calls or API-key handling.
- No instance detection/tracking metric or matching rule.
- No Gate/Specialist/Memory metric implementation.
- No paired baseline/full comparison or bootstrap CLI in this increment.
- No hidden leaderboard or external submission integration.
- No change to official split membership, ontology, masks, or prediction
  schema.
