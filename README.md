# Streaming SurgicalAgent V3.1

CholecTrack20-only research code for strict-causal surgical video inference,
predicted tracking, selective structured verification, and reliability-aware
event memory.

> **Current status:** P0 PASS; P1 data qualification is
> `PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION`; P2 Local Smoke is PASS. VID30/VID31
> use explicit sidecars and task masks while original assets remain unchanged.
> The canonical component pipeline is implemented; API, Gate, Specialist,
> tracking, workflow, and memory research modules remain phase-gated.

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
- Declared main backbone: `GPT-5.6 Terra` for joint perception and
  same-backbone verification.
- Exact API provider and provider-returned `model_identifier`: still BLOCKED
  until verified during P3; the declared display name is not treated as an
  auditable API identifier.
- Credentials remain external and must never be committed.

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
