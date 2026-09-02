# Streaming SurgicalAgent V3.1

CholecTrack20-only research code for strict-causal surgical video inference,
predicted tracking, selective structured verification, and reliability-aware
event memory.

> **Current status:** data qualification and the P2 engineering smoke pass. The
> OpenRouter `openai/gpt-5.6-sol` joint API path, dataset rollout, structured
> five-head prediction, offline evaluation, Always Verify, rule Gate,
> candidate-bounded joint verification, deterministic coordination, and
> finalized-only workflow/event memory and executable Track–Workflow context
> profiles are implemented. The learned Gate runtime accepts only an explicit
> train-derived `final_g1` artifact. The predicted tracker now has real
> `smoke|full|oof` training/export modes and four independent context ablations;
> trained checkpoints and Gate artifacts are still runtime outputs, so the full
> paper system is not yet claimed as complete.

## Current documentation

- [V3.1-API academic architecture revision](docs/architecture/Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md): research-design source of truth.
- [V3.1-API Codex implementation specification](docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md): code implementation contract.

The academic document controls research questions, claims, pipeline semantics,
and module responsibilities. The Codex specification controls repository
changes, interfaces, phase gates, tests, commands, and acceptance criteria.
The user's latest explicit instruction takes precedence over both documents.

Implementation is phase-gated from P0 through P12. A phase may advance only
after its actual tests pass. Unknown annotation semantics, FPS, ontology, or
experimental conditions are recorded as `BLOCKED`; they are never inferred.
All stages must use one canonical streaming pipeline under `src/surgical_agent/`;
experiment systems assemble components rather than copying the rollout loop.

## Repository layout

- `configs/`: data, experiment, and ablation configuration.
- `src/surgical_agent/`: canonical V3.1 package.
- `src/streaming_surgical_agent/`: compatibility namespace retained during migration; canonical implementation lives in `surgical_agent`.
- `scripts/`: executable phase entry points.
- `tests/`: unit, integration, and synthetic fixtures.
- `tools/audit/`: read-only dataset audit tooling.
- `tools/dataset/`: dataset acquisition utility; not part of runtime.
- `dataset_reports/`: dataset audit evidence.
- `docs/`: implementation contracts, audits, and reports.
- `artifacts/`, `outputs/`, `logs/`: generated and ignored runtime state.

The raw dataset is external and read-only at runtime. Its path must come from a
CLI option or resolved configuration; it must not be embedded in source code.

## Data profile

- CholecTrack20 is external and read only. Resolve it with `--dataset-root`, then
  `CHOLECTRACK20_ROOT`, then an optional local config value; source code and the
  committed default config contain no workstation path.
- Main requested backbone: OpenRouter `openai/gpt-5.6-sol` for joint perception
  and same-backbone joint verification. Requested and provider-returned model
  identifiers are persisted separately for audit.
- Dataset API rollouts use the versioned
  `joint_perception_gate_owned_compact_v1` wire
  contract (I3/V4/T5/IVT8/P3). The parser places returned IDs into the original
  7/10/15/100/7 task vectors and zero-fills omitted classes, so evaluator shapes
  and frame-level multi-label semantics remain unchanged. These values are
  uncalibrated top-k ranking scores, not full model logits.
- Credentials remain external and must never be committed.
- The P3 mock uses a generated non-sensitive image. CholecTrack20 images are not
  sent to an external API before data-use authorization is confirmed.

### Testing ground truth scope

The audited Synapse bundle contains one official raw JSON annotation file for
each of the eight Testing videos. Instrument, Phase, bounding-box, and track-ID
supervision is available across all 15,282 annotated Testing frames (29,994
instances). Verb and Target have 6,631 fully supervised frames, while IVT has
5,953 fully supervised frames, concentrated in VID06, VID25, VID92, and VID111.

Evaluation is task-masked: a task is scored only when every annotated instance
in that frame has a valid label. A `-1` label or a partially labelled frame is
ignored for that task; it is never converted into a negative class and never
enters the metric denominator. Testing GT remains offline-evaluation-only and
must never enter training, model selection, prompts, causal state, or API
requests.

## Phase order

`P0 Audit/Docs/Scaffold -> P1 Engineering Hard Gates -> P2 Local Smoke ->
P3 API Client/Cache -> P4 API Single-pass -> P5 Track/Workflow Context ->
P6 Train-only Priors -> P7 EventMemory/Candidates ->
P8 Verification Probe -> P9 D0 -> P10-A G0 -> P10-B Policy-matched Rollout ->
P10-C D1 -> P10-D G1 -> P10-E Validation Operating Point ->
P11 Frozen G1 Runtime -> P12 Ablation/Cost/Statistics`

## P2 local smoke

P2 validates engineering only: one real masked optimizer step, exact checkpoint
reload, the canonical NeverVerify pipeline, Gold-free PredictionRecord output,
offline smoke evaluation, and atomic artifacts. Its random lightweight model
scores are not paper results.

```powershell
.\.venv-p2\Scripts\python.exe scripts\run_local_smoke.py `
  --config configs\experiments\local_smoke.yaml `
  --dataset-root D:\cholec_dataset `
  --device cpu
```

See [AutoDL quickstart](docs/AUTODL_QUICKSTART.md) before moving the code and
external dataset to a GPU instance.

## Predicted tracker

The first innovation module uses a trainable Faster R-CNN MobileNetV3-FPN
instrument detector followed by a strictly causal Hungarian associator. Verify
the optimizer/checkpoint/prediction/export path locally with:

```powershell
.\.venv-p2\Scripts\python.exe scripts\train_tracker.py `
  --mode smoke `
  --dataset-root D:\cholec_dataset `
  --device cpu `
  --no-pretrained `
  --max-train-batches 1 `
  --max-prediction-frames 2
```

Formal `full` and fold-safe `oof` commands are in the AutoDL quickstart. The
generated checkpoint and `predicted_tracks.json` stay under ignored
`artifacts/training/tracker/` and are not source-controlled.

### Adaptive six-frame causal input

Dataset API inference consumes a maximum six-frame causal buffer ending at the
target frame. The tracker and deterministic evidence selector inspect all six
frames, while the API upload budget remains three images: the current target is
always included and up to two historical frames are selected by visual change,
predicted-track displacement, track-set change, and recency. Each rollout writes
`causal_window_audit.jsonl` with the full window, selected image IDs, omitted
image IDs, and bounded temporal evidence. No future frame or annotation label is
used by the selector.

For a one-call full-window engineering probe, combine
`--target-frame-id <FRAME_ID>` with `--max-frames 1`; the selected target must
exist in the requested video.

## P3 API infrastructure

The deterministic mock exercises multimodal transport boundaries, structured
response validation, one injected retry, canonical request hashing, cache replay,
and logical/provider-call accounting without network access:

```powershell
.\.venv-p2\Scripts\python.exe scripts\smoke_api.py `
  --config configs\api\mock.yaml
```

The real OpenRouter adapter is implemented, but credentials and data-upload
authorization remain explicit runtime inputs and are never committed.

Two OpenRouter routing profiles are intentionally kept separate:

- `configs/perception/joint_openrouter_dataset.yaml` is the paper profile. It
  pins the OpenAI upstream provider and disables fallback for repeatability.
- `configs/perception/joint_openrouter_latency_dataset.yaml` is the efficiency
  profile. It allows OpenAI, Azure, and Amazon Bedrock, sorts them by latency,
  and permits fallback. The routing profile is included in the canonical
  request payload hash, so it cannot accidentally reuse paper-profile cache.

Paper-mode rollouts reject the efficiency profile; use it only with
`--mode engineering` for latency and robustness measurements.

The current causal V3 entry point uses the no-training rule Gate by default;
passing a valid frozen Gate artifact switches it to learned-Gate inference:

```powershell
.\.venv-p2\Scripts\python.exe scripts\infer_v3.py --help
.\.venv-p2\Scripts\python.exe scripts\run_always_verify.py --help
```

## Dataset tooling

P1 provides a strict read-only parser, split manifest, exact PNG resolver,
the frozen test rule `decoder_index = annotation_frame_id - 1`, independent task
masks, explicit VID30/VID31 sidecars, and separate Gold-free runtime/evaluation
types. Reproduce the raw-release qualification artifacts with:

```powershell
.\.venv\Scripts\python.exe tools\audit\audit_cholectrack20.py `
  --dataset-root D:\cholec_dataset
```

For the optional IVT consistency comparison, also pass `--ivt-map` with a
separately downloaded official `ivtmetrics/maps.txt` path. The artifact records
the source URL and SHA-256; the external reference file is not vendored.

The official Synapse metadata can be audited independently without recursively
downloading the dataset. Authentication is accepted only from the process
environment, and the generated artifact is sanitized:

```powershell
$env:SYNAPSE_AUTH_TOKEN = '<runtime-only token>'
.\.venv\Scripts\python.exe tools\audit\audit_cholectrack20_synapse.py `
  --dataset-root D:\cholec_dataset `
  --project-id syn53182642
Remove-Item Env:SYNAPSE_AUTH_TOKEN
```

The checked-in reports contain no dataset download credential. Generated P1
artifacts live under `artifacts/p1/`; the consolidated evidence is under
`dataset_reports/` and `reports/P1_REPORT.md`.

The downloader uses environment variables and never stores credentials in
source code:

```powershell
$env:SYNAPSE_EMAIL = '...'
$env:SYNAPSE_AUTH_TOKEN = '...'
$env:CHOLECTRACK20_ACCESS_KEY = '...'
$env:CHOLECTRACK20_DATASET_ROOT = 'D:\cholec_dataset'
python tools/dataset/download_cholectrack20.py
```
