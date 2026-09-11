# Streaming SurgicalAgent V3.1

> **当前主线（2026-09-11）：`prior-gated-joint-mainline-v1.0.0`。** 按用户要求启用，每目标 13 次调用，原先验门控加联合 Phase，取消盲投 Phase 面板。冻结基座仍为 Gemini 3.8 Flash，五席审核不变。[当前流程](LATEST_PIPELINE.md) · [冻结证据](docs/PRIOR_GATED_JOINT_MAINLINE_FREEZE_2026-09-11.md)。已有确认未通过完整性标准，默认选择不表示全面最优。三基座六目标 Training 对照走独立入口；没有启动 Gate 数据采集。下列带“默认未改”的条目保留其历史实验时点含义。

> **先验门控 + 联合 Phase 审核修复（2026-09-11，负结果，默认未改）：** [报告](docs/PRIOR_GATED_JOINT_PHASE_2026-09-11.md)。把 Phase 接入与四头相同的"提案 → 五席审核 → Python 接纳"流程，审核请求只含 H0 与原始候选池、Phase 推荐不带先验提示、门控仍以 H0 Phase 为桶（重放显示即使真值 Phase 也不能降低四头错漏）。16 个新 VID110 目标上联合面板一次也没改 Phase，候选与先验门控候选逐字节相同：五头平均 F1 **54.76 / 61.01 / 62.21**（H0 / 默认 / 两个门控臂），总错漏 95 / 88 / 84；预注册标准未通过。先验门控在第二批新帧上仍正向。
>
> **先验门控 IVT 接纳（2026-09-11，候选，默认未改）：** [报告](docs/PRIOR_GATED_IVT_ADMISSION_2026-09-11.md)。在 v1.3.0 四头之后增加纯 Python 步骤：按排除当前视频的 Training 先验（以 H0 Phase 为条件）否决罕见 IVT、接纳高频 IVT，Phase 冻结为 H0，不增加模型调用。136 个 Training 目标离线重放选定阈值后，16 个新 VID110 目标一次付费确认通过预注册标准：五头平均 F1 **55.46 / 54.33 / 58.97**（H0 / 默认 / 候选），总错漏 91 / 98 / 87。收益主要来自否决 `hook/dissect/cystic_plate`；仍是单视频、小样本。

> **历史默认（2026-09-10）：`parallel-phase-repair-v1.3.0-glm-low`。** GLM-5.3-Flash替换Ministral席，与Qwen35B-A3B、GPT-luna、Gemini Flash-Lite、DeepSeek视觉Flash参与两路审核。GLM不能关闭推理，已设low；Together路由4次真实请求通过。[流程与实测](docs/DEFAULT_GLM_REPAIR_2026-09-10.md)。旧版本保留，未宣称准确率提升。

> **轻量Phase审核席（2026-09-10）：** [8目标双席替换对照](docs/LIGHT_PHASE_SEATS_TRIAL_2026-09-10.md)。Grok→Ministral 8B、Qwen→35B-A3B，32次请求通过；两席并行等待约减半，Phase仍75%，H0为87.5%。只换一席与同时换两席结果相同，未取得Phase净收益，默认未改。

> **Phase时序评分对照（2026-09-10）：** [短历史与较长因果历史，同32目标](docs/PHASE_TEMPORAL_RATINGS_TRIAL_2026-09-10.md)。320次请求全部有效；两组Phase均75.00%，各改对0、改坏1，仍低于H0 78.13%。默认Phase保持启用，未以固定H0替代修复；本轮未推广默认。

> **简短Phase＋均分对照（2026-09-10）：** [同32目标的提示简化与评分实验](docs/PHASE_SIMPLE_RATINGS_TRIAL_2026-09-10.md)。320次真实调用完成：H0 Phase **78.13%**，简短单选／评分均 **75.00%**；评分改对0、改坏1。均分更保守但未带来Phase净收益，默认未改。

> **同32目标Phase机制对照（2026-09-10）：** [旧提示／早期五头／修改模板的完整结果](docs/PHASE_MECHANISM_COMPARISON_2026-09-10.md)。Phase F1：H0 **78.13%**，新精简与旧提示均 **71.88%**，早期五头及修改模板均 **75.00%**。832次调用完成；格式修正版160份回答全部有效，但改对1、改坏2，未取得Phase净收益。默认未改。

> **32目标扩大对照（2026-09-10）：** [H0／默认／分组完整结果](docs/EXPANDED_SPLIT_REVIEW_2026-09-10.md)。平均F1 **58.03% / 59.63% / 58.85%**，分组未胜出；共享Phase从78.13%降至65.63%，改对3、改坏7。704次调用完成，异常敏感性核验通过，默认未改。

> **换样本复测：** [新8目标：H0、默认与分组审核](docs/FRESH_SPLIT_REVIEW_2026-09-09.md)。平均F1 **54.59% / 64.96% / 65.55%**；分组小幅改善但Target仍下降，与旧8目标总体方向不同。176次真实请求已核验，默认未改。

> **分组审核试验：** [I/V/T 与 IVT 分开审核的结果](docs/SPLIT_REVIEW_TRIAL_2026-09-09.md)：同8目标未改善，审核费用增加约34%、等待约5%（含明确恢复计时限制），默认保持不变。

> **最新修复对照：** [多方案开发＋24新Training目标确认](docs/DEFAULT_IMPROVEMENT_TRIAL_2026-09-09.md)。未发现可稳定替换默认的语义方案；默认已补齐具体审核拒绝原因日志，预测与API次数不变。

> **上一默认方案（2026-09-09）：图谱一轮＋独立 Phase 短历史审核，版本 `parallel-phase-repair-v1.1.0-compact-prompt`，已保留。**
> 统一入口：`python scripts/run_pipeline.py`（显示当前版本）；[默认版本配置](DEFAULT_PIPELINE_VERSION.json) · [旧版运行说明](docs/DEFAULT_REPAIR_PIPELINE_2026-09-09.md)。
> 该入口采用精简审核 prompt，复用原均分接纳与独立 Phase 并行执行器，目前支持归档的 8 个 Training 目标及缓存 H0；transactional 接纳实验、完整数据集及 Tracker/Gate 修复尚未接入。

> **2026-09-09：[最新 Pipeline 流程图与输入输出说明](LATEST_PIPELINE.md)**
> 固定 H0 → **图谱四头修复**与**独立 Phase 审核**并行 → Python 合并五头。
> [下载与重放](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/docs/releases/PARALLEL_PHASE_REPAIR_V1.md) · [版本清单](LATEST_PIPELINE_VERSION.json) · [离线重放入口](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/scripts/replay_parallel_phase_repair.py) · [保留的 Phase 结果](BEST_PHASE_RESULT.json)
> 上述下载链接与成绩属于基础 v1.0.0：并行重跑 IVT F1 **38.71%**、Phase **75.00%**；此前保留结果为 **40.00% / 75.00%**。精简提示结果见[配对报告](docs/VERIFIER_COMPACT_PROMPT_TRIAL_2026-09-09.md)，新默认尚未发布 Git 标签。
> 原图谱标签、默认纯 API H0 与历史结果保留。[实验记录](EXPERIMENT_RESULTS.md)。

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
