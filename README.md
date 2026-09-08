# Streaming SurgicalAgent V3.1

> **2026-09-08 修复研究更新：[当前保留方案与离线重放](docs/VERIFIED_REPAIR_CANDIDATE_2026-09-08.md)。**
> 已补齐开发 8／确认 8／反馈 4 目标的真实结果；开发收益未在确认批复现，不能称为稳定最优。
> [完整评分](docs/REPAIR_REVISION_RESULTS_2026-09-08.md) · [GraphRAG 位置、输入输出与结构设计（尚未实现）](docs/GRAPH_RAG_VERIFIER_DESIGN_2026-09-08.md)

> **先看实验结果：[当前实验数据与可量化指标](EXPERIMENT_RESULTS.md)（2026-09-08）**
>
> 按 H0、Tracker、Verifier/Repair、Gate 汇总已完成数据，包含 Precision、Recall、F1、集合 Accuracy、修复收益和费用。
> **目前尚无最新完整 Pipeline 的四组消融结果；模块小测试不等于完整系统成绩。**
> [可下载指标快照](docs/experiments/current_quantitative_results_20260908.json) · [历史修复实验汇总](docs/VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)

CholecTrack20-only research code for strict-causal surgical video inference,
predicted tracking, selective structured verification, and reliability-aware
event memory.

## Main API Pipeline — 2026-09-06

The main runnable API entry point is `scripts/run_dataset_api_pipeline.py`.
It defaults to `configs/perception/joint_openrouter_h0.yaml`: OpenRouter
`qwen/qwen3.8-max-0902`, strict Alibaba routing, three causal images
`[t-50,t-25,t]`, and **one joint call for the target's I/V/T/IVT/Phase**.
It uses the evaluated original prompt, the full 100-class IVT ontology, a strict
final-label JSON schema, temperature 0 and low reasoning. The schema-only and
tuned prompt candidates were not adopted. Runtime prompts are packaged with
the code and do not depend on ignored experiment artifacts.

Start with [the main API run guide](docs/MAIN_API_PIPELINE.md), including
zero-cost request preflight, offline smoke and real API commands.
New collaborators should follow the [clone-to-results quickstart](docs/API_EXPERIMENT_QUICKSTART_2026-09-06.md)
for installation, external dataset requirements, a three-target run and offline scoring.
Historical Tracker/Gate/Verifier experiments remain explicit separate profiles;
the main H0 does not return ranking/confidence scores and is not silently
connected to a Gate trained on the old top-k contract.

> **Research Pipeline status; the authoritative live checkpoint is
> [docs/README.md](docs/README.md):** the core final Pipeline is executable with
> fixed three-frame input, predicted Tracker evidence, explicit hard Guard and
> Rule/learned Gate slot, bounded targeted Verify/Repair, deterministic
> postcheck, OutcomeFinalizer, atomic Memory/Pending state, exact provider-attempt
> budgets and resumable D0 collection. Tracker OOF training/evaluation is
> complete. Small real API Repair experiments have run without establishing
> stable net semantic benefit. The latest final-only H0 and five-family repair
> remain separate from this legacy ranked Pipeline. Current-policy formal Gate
> training, Validation operating-point selection and the four-cell experiment
> remain incomplete; no current Gate artifact is final.

## Current documentation

- [Complete target Pipeline architecture and pseudocode](docs/architecture/Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md): the sole source of truth for the full online graph, every Guard/Gate/Repair path, fields, outcomes, causal Memory, and pseudocode; implementation is incomplete.
- [Canonical Tracker × Gate ablation and academic protocol](docs/architecture/CANONICAL_PIPELINE_TRACKER_GATE_SPEC_2026-09-03.md): the sole source of truth for the `A_base/B_tracker/C_gate/D_full` experiment, labels, frame clock, statistics, and research gates.
- [Three-frame causal input decision](docs/architecture/CAUSAL_THREE_FRAME_DECISION_2026-09-04.md): records the current fixed three-frame visual contract and keeps older six-frame artifacts historical.
- [V3.1-API academic architecture revision](docs/architecture/Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md): historical research rationale and claim framing where it does not conflict with the two sources above.
- [V3.1-API Codex implementation specification](docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md): legacy detailed engineering guidance where it does not conflict with the two sources above.

Precedence is: the user's latest explicit instruction; the complete Pipeline
document for runtime semantics; the canonical Tracker × Gate protocol for
experiment semantics and progression; then the academic and legacy engineering
documents only where non-conflicting. The dated session handoff records verified
facts but is never normative. See [the docs index](docs/README.md) for the live
checkpoint and next action.

Unknown annotation semantics, FPS, ontology, or experimental conditions are
recorded as `BLOCKED`; they are never inferred. All stages must use one canonical
streaming pipeline under `src/surgical_agent/`; experiment systems assemble
components rather than copying the rollout loop.

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
- Main requested backbone: OpenRouter `qwen/qwen3.8-max-0902` for one-call joint
  initial perception. Requested and returned identifiers are both persisted.
- Main H0 uses `joint_perception_final_only_v1`: selected label IDs only,
  without top-k, generated confidence, or reasoning prose. Internal vectors
  carry `hard_label_v1` semantics; their 0/1 values are label indicators,
  not probabilities or suitable calibrated scores for mAP.
- Historical research rollouts can explicitly select the versioned
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

## Research progression

The normative dependency order is:

```text
Phase 0 research/data-contract freeze
-> Phase A shared implementation and tests
-> Phase B Training-only Repair capability probe and N_max freeze
-> Phase C cross-fitted Gate data, training, Validation choice, and freeze
-> Phase D sealed four-cell Test experiment
```

The older `P0-P12` labels elsewhere in the repository are legacy engineering
milestones, not the live research gate. Their historical order was
`P0 Audit/Docs/Scaffold -> P1 Engineering Hard Gates -> P2 Local Smoke -> P3 API
Client/Cache -> ... -> P12 Ablation/Cost/Statistics`.

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
completed full and five-fold OOF tracker checkpoints, resumable training states,
prediction artifacts, manifests, and metrics are versioned under
`artifacts/training/tracker/` and `artifacts/training/tracker_oof5/`. Tensor
files use Git LFS, so install Git LFS and run `git lfs pull` after cloning.
Other generated training and runtime artifacts remain ignored.

Downloaded OOF predictions must pass the local leakage/provenance/coverage
audit before they are used to construct Gate training examples:

```powershell
.\.venv-p2\Scripts\python.exe scripts\evaluate_tracker_oof.py `
  --dataset-root D:\cholec_dataset `
  --tracker-oof-index artifacts\training\tracker_oof5\oof\index.json `
  --tracker-config configs\tracker\fasterrcnn_mobilenet_v3_5090_oof5.yaml
```

This report evaluates instrument AP50/F1 on the nine instance-supervised
Training videos. VID31 is checked for exact prediction coverage but excluded
from bounding-box metrics because its repaired route has no valid instance box
supervision.

### Formal input contract

The final Pipeline profiles use a fixed causal window of up to three ordered
images and never let Tracker select images. JointPerception and targeted Verify
receive the same selected frame IDs in all four cells. Verify omits Tracker,
workflow and Memory fields, preserving the Tracker ablation boundary. Verify is
also blind to H0 selections and upstream rank scores, labels the target frame
rather than the union of its history, and receives all seven Phase IDs. Frozen
unrequested-IVT components are passed only as deterministic closure constraints. Older
six-frame/adaptive dataset profiles remain legacy engineering or supplementary
profiles and are not valid for the primary factorial.

For a one-call full-window engineering probe, combine
`--target-frame-id <FRAME_ID>` with `--max-frames 1`; the selected target must
exist in the requested video.

## Historical API / verification experiments

The commands and top-k profiles below are retained for research reproduction.
For the default final-label H0, use [the main API guide](docs/MAIN_API_PIPELINE.md).

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
