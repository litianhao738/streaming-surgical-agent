# CholecTrack20 Real API Rollout Design

**Status:** Approved in chat on 2026-08-27; written specification pending final user review.

**Purpose:** Add a real-data, OpenRouter-backed CholecTrack20 rollout entry point
that reuses the existing canonical streaming pipeline and produces a
paper-aligned single-pass baseline without allowing labels, hidden sampling, or
unbounded paid calls into runtime inference.

## 1. Research Contract

The paper-facing endpoint is a complete, deterministic, per-video rollout. The
implementation may use bounded engineering subsets while it is being tested,
but truncated or strided runs are never paper results.

The following constraints remain unchanged:

- inference is strict causal and Gold-free;
- the target frame is the greatest frame ID in every causal request;
- at most three causal visual frames are sent per state;
- video boundaries reset all streaming state;
- labels and evaluation targets never enter API requests, prompts, caches, or
  causal state;
- single-pass baseline and future full-agent runs use the same exact selected
  videos, causal windows, prompt/schema, ontology, model identifier, and
  evaluation implementation;
- validation is used to freeze operating choices; test is run only after those
  choices are frozen;
- formal statistics treat complete videos, not frames, as the paired bootstrap
  unit;
- real data upload is an explicit experiment decision and is never inferred
  from the presence of a credential.

The existing exact immutable backend identity status remains `PARTIAL`. A
completed rollout is therefore a reproducible single-pass baseline artifact,
not by itself a paper-performance or exact-backend claim.

## 2. Scope

### 2.1 In scope

- A dataset-scale CLI for the existing `CanonicalStreamingPipeline`.
- Explicit engineering and paper execution modes.
- Native-resolution PNG loading for training/validation sources.
- Exact MP4 decoding for official test sources using the frozen
  `decoder_index = frame_id - 1` alignment rule.
- OpenRouter `openai/gpt-5.6-sol` Joint Perception, deterministic Evidence
  Signals, `NeverVerify`, and KEEP coordination.
- Persistent parsed-response cache, usage ledger, paired prediction/evidence
  files, and a dataset rollout manifest.
- A hard provider-call limit and cache reuse across reruns.
- Mock and local-data verification that make no paid API calls.

### 2.2 Out of scope

- Benefit Gate training or inference.
- Specialist verification calls or repairs.
- Learned WorkflowState or EventMemory.
- Instance detection, bounding-box, operator, or tracking predictions.
- Automatic prompt tuning, threshold selection, or policy changes on test.
- An automatic full paid validation/test run during implementation.
- Paper claims from truncated runs, smoke runs, or an identity status that is
  still partial.

## 3. Execution Modes

### 3.1 Engineering mode

Engineering mode exists only to verify infrastructure and inspect cost before a
complete rollout. It requires:

- exactly one explicit `--video-id`;
- a positive `--max-frames`;
- deterministic earliest-frame selection from the adapter;
- the same prompt, schema, causal-window, media geometry, and runtime pipeline
  used by paper mode;
- `paper_metric_eligible: false` in every artifact.

No random sampling or frame stride is supported. This avoids accidentally
turning an engineering choice into an undocumented paper variable.

### 3.2 Paper mode

Paper mode requires an explicit `--split validation` or `--split testing` and:

- selects every official video in the named split in canonical sorted order;
- selects every adapter-eligible frame in each video in canonical order;
- rejects `--video-id`, `--max-frames`, stride, or random-sampling options;
- resets streaming state exactly once at each video boundary;
- records the complete expected and completed video/frame counts;
- fails the run if any selected state is missing rather than silently emitting a
  partial paper artifact.

Paper execution follows this sequence:

1. bounded engineering run;
2. one complete video;
3. complete Validation;
4. freeze image handling, prompt/schema, ontology, model/config, and later
   operating choices;
5. complete frozen Test.

## 4. Runtime Data Flow

```text
CholecTrack20DatasetAdapter.iter_inference_video
-> ordered Gold-free InferenceSample stream
-> exact causal PNG/MP4 frame loader
-> path-free canonical API image identifiers
-> CausalPerceptionContextBuilder
-> JointApiVlm through CachedMultimodalApiClient
-> strict JointPerception parser and ontology validation
-> FrameEvidenceSignalExtractor
-> NeverVerify / disabled Specialists / NoOpCoordinator KEEP
-> PredictionFinalizer
-> FrameResultWriter
-> workflow/event commit and prior-finalized state advance
-> dataset rollout manifest and usage summary
```

The rollout never constructs `ResolvedSample`, `EvaluationTarget`, or
`FrameSupervisionTarget`. A new adapter iterator produces `InferenceSample`
directly from media/frame identity. The API runtime assembly has no parameter
through which evaluation objects can enter.

## 5. Components and Ownership

### 5.1 Exact causal media loader

Create a focused media loader under `src/surgical_agent/data/` that consumes an
`InferenceSample` and returns one tensor shaped `[T, 3, H, W]`.

For PNG-backed samples it:

- requires one existing regular file per causal frame;
- opens each exact `media_ref` with Pillow;
- converts to RGB without resizing;
- verifies all frames in the causal window share one native geometry;
- normalizes pixels to finite `float32` values in `[0, 1]`.

For test MP4 samples it:

- requires `ct20_test_mp4_annotation_id_minus_1_v1` alignment;
- requires all refs in the sample to name the same regular MP4 file;
- maps each canonical frame ID to decoder index `frame_id - 1`;
- rejects zero/negative canonical frame IDs;
- uses OpenCV headless to read the exact requested decoder indices;
- converts BGR to RGB;
- fails closed when a requested frame cannot be decoded or frame geometry is
  inconsistent.

`opencv-python-headless` is an explicit compatible dependency in both the Python
project metadata and the AutoDL requirements. It is used only for MP4 decoding;
PNG behavior remains Pillow-based.

### 5.2 Dataset selection

Create immutable selection contracts that resolve CLI arguments into an ordered
tuple of video IDs and an expected frame-selection policy before any provider
call. Selection validates the official 10/2/8 manifest through the existing
adapter.

Add `CholecTrack20DatasetAdapter.iter_inference_video`, which never parses or
constructs task labels:

- PNG-backed splits enumerate the exact media resolver's canonical frame IDs;
- test uses the existing identity-only test-frame-key reader and never parses
  test task values;
- derived VID30/VID31 routing may select a verified media source but never opens
  their annotation, phase, or frame-IVT sidecars.

Engineering selection calls
`iter_inference_video(video_id, max_samples=max_frames)`. Paper selection calls
`iter_inference_video(video_id)` for every video in the frozen split. The
selected video list and counts are persisted before rollout begins.

After the loader consumes local media paths, the application creates an
otherwise identical runtime `InferenceSample` whose `media_refs` are stable,
path-free identifiers of the form
`cholectrack20:<video_id>:frame:<canonical_frame_id>`. Only this sanitized sample
enters `CanonicalStreamingPipeline`. Request hashes and caches therefore remain
portable across machines and cannot disclose absolute dataset paths.

### 5.3 API pipeline assembly

The dataset runner reuses the existing components:

- `CausalPerceptionContextBuilder(max_frames=3)`;
- `JointPerceptionRequestBuilder`;
- `JointApiVlm`;
- `FrameEvidenceSignalExtractor`;
- `DisabledCandidateGenerator`;
- `NeverVerify`;
- `DisabledSpecialistRegistry`;
- `NoOpCoordinator`;
- `PredictionFinalizer`;
- `NoOpCausalStore("workflow")` and `NoOpCausalStore("memory")`;
- `FrameResultWriter`.

The named real-data config differs from the synthetic smoke config only where
the experiment contract requires it:

```yaml
synthetic_input_required: false
data_upload_authorized: true
max_causal_frames: 3
requested_model_identifier: openai/gpt-5.6-sol
```

The CLI additionally requires the literal flag `--authorize-data-upload`.
Config authorization without the CLI flag, or the flag without the exact named
config, fails before reading the API key or opening any image.

### 5.4 Provider accounting and budget

The real dataset runner keeps one logical provider attempt per uncached state
and no automatic retry. A budgeted transport wrapper sits below the cache, so
cache hits do not consume the budget.

Every invocation requires `--max-provider-calls` with either a positive integer
or the literal value `exact-selection`. `exact-selection` resolves to the
selected frame count. The wrapper rejects the next network send before it would
exceed the resolved limit. Token counts, latency, and returned provider cost are
recorded by the existing usage ledger; a second dollar-limit mechanism is
deferred unless real operation shows it is needed.

Provider errors, schema errors, invalid accounting, cache corruption, and
budget exhaustion are typed, sanitized failures. There is no local-model
fallback inside the same experiment.

### 5.5 Cache reuse

The CLI accepts a persistent `--cache-root` and a fresh result `--run-id`. If a
run stops, the user starts a new run ID with the same cache root. The runner
replays the video from its first state in canonical order:

- completed requests are served from parsed-response cache with zero provider
  calls and zero current provider cost;
- cached predictions reconstruct the same prior-finalized state;
- the first missing request resumes paid execution;
- the fresh `FrameResultWriter` builds one internally consistent result set.

The existing canonical request metadata detects incompatible cache entries, so
this feature does not add a second experiment-fingerprint subsystem. In-place
result append is not supported.

## 6. CLI Contract

Create `scripts/run_dataset_api_pipeline.py`.

Engineering example:

```bash
python scripts/run_dataset_api_pipeline.py \
  --mode engineering \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --video-id VID30 \
  --max-frames 3 \
  --config configs/perception/joint_openrouter_dataset.yaml \
  --api-key-file docs/API.txt \
  --output-root artifacts/api_dataset \
  --cache-root artifacts/api_dataset_cache/baseline_v1 \
  --run-id engineering_vid30_001 \
  --max-provider-calls exact-selection \
  --authorize-data-upload
```

Paper validation example:

```bash
python scripts/run_dataset_api_pipeline.py \
  --mode paper \
  --split validation \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/perception/joint_openrouter_dataset.yaml \
  --api-key-file docs/API.txt \
  --output-root artifacts/api_dataset \
  --cache-root artifacts/api_dataset_cache/single_pass_validation_v1 \
  --run-id validation_full_001 \
  --max-provider-calls exact-selection \
  --authorize-data-upload
```

`--run-id` must be a safe new identifier and its result directory must be
fresh. API keys are accepted through the existing secret wrapper and ignored
`docs/API.txt` file and are never serialized.

## 7. Persistence

The result and cache roots contain:

```text
<output-root>/<run-id>/
  run_status.json
  api_usage.jsonl
  predictions/<video_id>.jsonl
  evidence/<video_id>.jsonl
  manifest.json
  dataset_rollout_artifact.json

<cache-root>/
  <request_hash>.json
```

The final rollout artifact records:

- execution mode and split/video selection;
- expected and completed frame counts per video;
- ordered video IDs and video-boundary resets;
- dataset repair-manifest SHA-256 and alignment versions;
- requested and returned model identity evidence;
- prompt/schema/config versions;
- first-call/cache-hit/retry/provider-call/token/cost totals;
- paired-output, usage, manifest, and cache locations;
- `track20_image_uploaded: true`;
- `paper_metric_eligible: false` until all paper gates are independently met;
- explicit failure category and incomplete status when applicable.

No artifact contains the API key, HTTP headers, raw provider body, free-form
reasoning, prompt text, image bytes, base64 images, evaluation targets, or
absolute local dataset paths.

## 8. Failure Semantics

- Invalid mode/selection/config/auth/budget: fail before provider access.
- Dataset manifest/hash/alignment failure: fail before provider access.
- PNG/MP4 decode or geometry failure: fail before the affected request.
- Cache corruption or request mismatch: stop without overwriting the entry.
- Provider/accounting/schema failure: write one sanitized usage failure record
  with exact provider-call accounting and stop the run.
- Budget exhaustion: write `provider_call_budget_exhausted` with no network
  call for the rejected state.
- Any incomplete paper run is excluded from formal aggregation.

## 9. Evaluation Boundary

This feature produces the complete single-pass prediction/evidence baseline.
It does not put GT into the rollout process and does not choose thresholds.

Offline evaluation may consume a completed Validation run through the
existing formal frame-metric implementation. It must retain task masks and
source granularity, exclude unsupported classes, compute video-wise I/V/T/IVT
mAP and Phase macro-F1/Accuracy, and keep test frozen. Test predictions with no
local permitted GT remain inference artifacts for the official or separately
authorized evaluation path.

## 10. Testing Strategy

All automated tests use mock transports or local media and make zero paid API
calls.

Required tests cover:

- engineering and paper argument compatibility;
- paper mode rejecting truncation and partial video completion;
- double authorization failing before a provider call;
- official split ordering and frame ordering;
- labels/evaluation objects having no route into the pipeline request;
- native PNG RGB loading and geometry validation;
- test MP4 `frame_id - 1` mapping and video reset;
- three-frame causal ordering and target-frame maximality;
- provider-call hard budget;
- first mock pass writing paired outputs, usage, cache, and manifests;
- a fresh run replaying cached prefix states with zero provider calls before
  the first uncached state;
- sanitized failures and exact-secret scans;
- incomplete runs never marked complete or paper eligible;
- complete mock multi-video rollout with one reset per video;
- a local-data integration run over a bounded PNG sample and a bounded official
  test MP4 sample when `CHOLECTRACK20_ROOT` is available;
- all existing tests remaining green.

No real provider call is part of implementation verification. The first real
CholecTrack20 engineering call is a separate user-invoked operational step
after code review and local mock/data checks pass.

## 11. Acceptance Criteria

1. Engineering mode processes an explicit bounded real-data video selection
   through the canonical API pipeline using mock transport.
2. Paper mode resolves complete Validation or Test selections and refuses every
   truncation mechanism.
3. PNG and test MP4 causal frames are loaded under their frozen alignment
   contracts without label access.
4. Real-data execution cannot start without both the exact real-data config and
   `--authorize-data-upload`.
5. Provider calls cannot exceed the declared hard call budget.
6. Interrupted execution can be restarted with a fresh run ID; cached prefix
   states reconstruct causal history with zero repeated provider calls.
7. Completed runs contain paired predictions/evidence, usage, cache,
   selection, provenance, and a complete manifest with no credential or raw
   image/provider payload.
8. Multi-video runs preserve frame order and reset state at every video
   boundary.
9. Full tests, Ruff, compilation, secret scans, and an AutoDL mock command pass.
10. The implementation does not make or claim a paid CholecTrack20 API run.
