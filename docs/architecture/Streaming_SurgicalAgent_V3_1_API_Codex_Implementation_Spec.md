# Streaming SurgicalAgent V3.1-API — Codex 代码构建任务书

**版本定位：V3.1-API / Academic Implementation Contract**  
**主数据集：CholecTrack20**  
**主论文方向：Benefit-Routed Sparse Specialist Verification for Reliable Causal Surgical Streaming**

> 本文件可直接交给 Codex。目标不是一次性生成完整项目，而是让 Codex 在已有 repository 上，严格按照 **Audit → Docs → Scaffold → Hard Gates → API Baseline → Context → Priors → Memory → Verification → Gate → Full Runtime → Paper Experiments** 的顺序增量构建、真实测试并报告。
>
> **任何阶段未真实通过测试，不得进入下一阶段；任何字段、annotation semantics、ontology、FPS、tracker 训练来源或 API 能力不明确时，必须标记 `BLOCKED`，禁止自行假设。**

> **当前执行检查点（2026-08-24，2026-08-25 独立复核与 P3 mock 实现）**：P0 `PASS`；P1 `PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION`；P2 Local Smoke `PASS`，运行约束由 `<resolved dataset root>/repair_manifest.json` 冻结。真实 masked optimizer step、checkpoint round-trip、VID02/VID31/VID30 canonical pipeline、Gold-free PredictionRecord、atomic artifact 和 20-video target-granularity audit 均已运行通过。VID30 仍是候选重建验证源；VID31 只启用 CholecT50 frame-level Instrument/Verb/Target/Triplet presence 与 Cholec80 phase，不启用 instance、bounding box、operator 或 track supervision。历史完整环境为 `79 passed`；2026-08-25 当前环境在未配置可选原始 Cholec80 provenance 目录时为 `78 passed, 1 skipped`，两次结果按各自运行环境保留。P3 provider-neutral client/hash/cache/retry/usage、mock adapter 与合成图像 smoke 已通过，但 exact API identity 和真实调用仍为 `BLOCKED`，故 P3 为 `PARTIAL`；P4 prediction/evaluation granularity 与 P9 canonical per-sample task error 继续保持各自阶段性 `BLOCKED`。不得因 mock 通过而进入 P4。

---

# 0. 任务目标与最终研究主线

你正在实现 **Streaming SurgicalAgent V3.1-API**。

论文主方法不应被实现成“更强 API + 一堆模块”，而应严格保持以下研究问题：

> **Given the same multimodal backbone and causal context, when and which task-scoped Specialist should the system invoke, what historical evidence should be trusted, and when is a scoped repair sufficiently supported?**

在线主链：

```text
Streaming Surgical Video
        ↓
Strict Causal Window W_t
        ↓
Predicted TrackProvider
        +
WorkflowState_(t-1) from WorkflowStateStore
        ↓
Structured Context Builder
        ↓
Joint API-VLM Perception
[1 base API call / streaming state]
        ↓
Initial Surgical State P_t
I / V / T / IVT / Phase
+ Top-K hypotheses
+ uncertainty proxies
        ↓
Surgical / Temporal / Track / Workflow Signal Extractor
[optional explicit EventMemory disagreement]
        ↓
Task-wise Verification Benefit Gate
        ↓
      ACCEPT  ←────────→  VERIFY(scope)
         │                       ↓
         │             Sparse Specialist Router
         │                       ↓
         │             Same-backbone Specialist Agent
         │             spatial_track | interaction | workflow
         │             + scope-specific candidate set H0...Hk
         │             + visual evidence
         │             + track/workflow evidence
         │             + reliability-weighted memory
         │             + train-only soft priors
         │                       ↓
         │              KEEP / scoped REPAIR
         │                       ↓
         │             Deterministic Coordinator
         │             validates scope/candidate/touched tasks
         └──────────────┬────────┘
                        ↓
                 Finalized State P'_t
                        ↓
              ReliabilityProfile
                        ↓
                  FinalizedEvent
                    ┌───┴────────────────┐
                    ↓                    ↓
       WorkflowStateStore.update()  EventMemory.commit()
                        ↓
                       t+1
```

论文证据链必须区分以下两个合同。

Backbone-level Framework Gain 主比较：

```text
same exact API model identifier
+ same test videos
+ same causal visual window
+ same task ontology / output schema / EvaluationEngine

Single-pass Baseline B  vs  Full V3.1-API Agent A
Framework Gain = A - B
```

这里的差值是完整 framework effect，不得单独归因给 Gate、Memory 或 backbone 能力。Verification policy 核心归因则必须保持：

```text
Same API backbone
+ matched causal context
+ same candidate generator
+ same Specialist pool and deterministic Coordinator
+ matched verification budget
```

从而只把 Layer 2 的性能差异归因给 **verification policy and routing**。Context design、Specialist role value 与 memory reliability 必须分别由 Layer 3 消融归因。禁止通过比较不同 backbone 的结果计算或宣称 Framework Gain。

当前项目特定输入事实：

```text
Local CholecTrack20 root: D:/cholec_dataset
Declared main backbone name: GPT-5.6 Terra
Backbone role: Joint Perception Agent + routed same-backbone Specialist Verification Agents
Exact provider/model_identifier: BLOCKED until P3 API provenance smoke
```

本机绝对数据路径只能存在于 resolved/local data profile，不得嵌入 Python 源码。声明模型名称不是 API 可审计标识；P3 必须记录 provider、endpoint/config、provider 返回的 exact model identifier、prompt/schema version 和 request hash，之后才能冻结正式实验 backbone。

---

# 1. 文档优先级与冲突处理

开始编码前必须完整阅读 workspace 中可用的项目文档与代码。

若存在以下文件，优先级如下：

1. **最新用户明确指令**
2. `Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md/.docx`
   - V3.1-API 研究方法、API 公平性、Gate、Verifier、Memory、实验协议的 source of truth。
3. `Streaming_SurgicalAgent_Codex_Ready_Implementation_Spec_v1.1.md/.docx`
   - Data / Schema / Split / Gold-free inference / strict causal / partial-label / evaluator separation / provenance 的 source of truth。
4. 旧 V3.1 Pipeline / Codex 文档
   - 仅用于理解演进历史；若与 V3.1-API 冲突，以 V3.1-API 为准。
5. SurgReflect paper/repository
   - 只作为 API-LVLM、reflection、structured output、orchestration 的参考。
   - **禁止复制其 benchmark-specific gold-derived statistics 到本项目 runtime。**

任何冲突不得弱化以下硬约束：

- strict causal；
- Gold-free inference；
- partial-label mask；
- train-only priors；
- validation-only operating-point selection；
- test frozen；
- predicted track 为主实验；
- video boundary reset；
- API provenance/cache；
- current event 不得检索到自己。

---

# 2. 首要执行顺序：P0 内必须先审计、再文档、再框架

**禁止一打开 repository 就直接实现算法。**

P0 必须按以下顺序执行。

## P0-A Repository Audit

先检查：

- repository 当前目录树；
- 当前 git 状态；
- Python/package 环境；
- 已有 configs；
- 已有 parser / dataset / schemas；
- train / infer / eval 脚本；
- tests；
- CholecTrack20 路径入口；
- tracker 相关代码；
- API client 是否已存在；
- 旧 V1/V3 代码是否存在；
- README 与实际实现是否一致。

真实运行已有 tests/smoke（若存在）。

生成：

```text
docs/V3_1_API_IMPLEMENTATION_AUDIT.md
```

至少包含：

```text
Repository state
Existing reusable modules
Missing modules
Conflicting interfaces
Observed data fields
Dependency state
Existing tests and actual results
Planned file changes
BLOCKED items
P0-P12 implementation map
```

## P0-B Documentation Synchronization

只有 Audit 完成后，才能根据真实 repository 状态更新：

- `README.md`
- `docs/README.md`
- `docs/architecture/Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md`
- `docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md`

不再维护内容重叠的 `ARCHITECTURE.md / PIPELINE.md / IMPLEMENTATION_PLAN.md`；研究设计与实现合同分别由上述两份 source-of-truth 文档承担，避免多份文档漂移。

要求：

- 文档不能描述尚不存在的实现为“已完成”；
- 尚未实现的模块必须标记 `PLANNED`；
- 无法确认的信息标记 `BLOCKED/TBD`；
- 不允许静默修改研究假设。

## P0-C Architecture Scaffold

然后再搭代码框架，只创建：

- package directories；
- typed schemas/contracts；
- protocol / ABC；
- config schema；
- registry；
- CLI skeleton；
- test skeleton；
- minimal API abstraction；
- minimal import path。

P0 不允许提前实现 P4-P12 的完整算法。

P0 完成后真实执行：

```bash
python -c "import surgical_agent"
pytest --collect-only -q
```

以及 repository 已存在的最低风险 config-load smoke。

只有真实通过才进入 P1。

---

# 3. 不可违反的 Engineering Contracts

## 3.1 Split Contract

```text
Train:
- Gate training
- train-derived priors/statistics
- optional calibration training
- optional tracker training if specifically designed fold-safe

Validation:
- minimal operating-point selection
- Gate threshold tau_B
- optional probability calibration
- sensitivity analysis

Test:
- frozen config
- frozen prompts
- frozen model identifier
- frozen thresholds
- frozen priors
- inference + offline evaluation only
```

Test 不得反向更新：

- Gate；
- priors；
- prompt；
- memory weights；
- candidate K；
- thresholds；
- verifier margin。

Gate 的固定训练生命周期为：

```text
train D0 → bootstrap G0 → train-only G0 selective rollout
→ train D1 → final G1 → validation tau_B → frozen test
```

Test 禁止 Gate refresh、D1 重建、G1 重训或基于 test trajectory 调整 provenance cap。

跨数据集来源也服从同一 split contract。当前 CholecT50/Cholec80 仅允许作为
`repair_manifest.json` 已记录的 VID30/VID31 侧车证据；不得把与 Track20 Validation/Test
同源的视频当作额外训练样本。未来任何上游预训练必须先建立 video-identity 去重清单，
并在 artifact 中证明 held-out identity 未进入训练。

## 3.2 Gold-free Inference Contract

`InferenceSample` 类型中禁止出现：

```text
GT
EvaluationTarget
LabelMask
gold_phase
gold_ivt
gold_track_id
gold labels
future labels
```

只有 `EvaluationEngine` 可以读取 `EvaluationTarget`。

API prompt builder 必须只接受 Gold-free runtime schemas。

## 3.3 Strict Causal Contract

目标时刻 `t`：

```text
max(causal_frames) <= t
track evidence max_frame <= t
workflow source max_frame < current commit
memory event frame < current event commit
```

当前 event 不允许检索自身。

未验证 FPS 前：

- 只使用 `frame_id / annotation_step / index`；
- 禁止把固定 annotation spacing 解释为秒；
- 禁止在论文/日志中写“5 seconds”之类未验证时间长度。

## 3.4 Partial-label Contract

Instrument / Verb / Target / IVT / Phase 沿用统一 completeness mask。

缺失 V/T/IVT：

- 不是 negative；
- 不进入 loss；
- 不进入 metric denominator；
- 不进入 Gate `Delta_t` 对应任务项；
- 不进入 reliability correctness calibration target。

同一 mask semantics 必须复用，禁止各模块自己实现一套。
Mask 必须同时绑定 task 和 supervision granularity。instance mask 不能授权 frame-level
negative，frame-level mask 也不能授权 bbox/track/instance association；粒度转换必须由唯一
EvaluationEngine 中预注册的版本化规则完成，否则 fail closed。

## 3.5 Track Contract

论文主结果只能使用：

```text
PredictedTrackProvider
```

`OracleTrackProvider` 只能用于 upper-bound diagnostic，必须在 config / artifact / table 中显式标记 `oracle=true`。

如果 tracker/detector 在 CholecTrack20 上训练，则 P9 Gate Dataset 的 held-out train videos 必须对 tracker fold-safe。

## 3.6 Workflow Contract

当前时刻 Perception 只能读取：

```text
finalized history <= t-1
```

禁止当前 GT Phase feedback。

`WorkflowStateStore` 必须维护紧凑 causal state：

```text
recent finalized phase history
phase stability
observed transitions / transition tendency
compact temporal state
source_max_frame_id
```

它只由 `FinalizedEvent` 更新，消费者仅为 context builder、Perception 与 Gate temporal/workflow features。它不得执行 episodic similarity retrieval，也不得依赖 EventMemory 是否启用。

若实现 GT workflow teacher forcing，仅允许：

```text
oracle_workflow=true
```

且不能进入主结果。

## 3.7 Workflow / Memory State Machine

禁止：

```text
InitialPrediction -> Memory.commit()
```

必须：

```text
InitialPrediction
→ Gate ACCEPT
or
→ VERIFY → KEEP / REPAIR
→ FinalizedState
→ ReliabilityProfile
→ FinalizedEvent
├→ WorkflowStateStore.update()
└→ EventMemory.commit()
```

合法 status：

```text
gate_accepted
verified_accepted
repaired
```

每个 video 开始：

```python
system.reset(video_id)
```

Tracker online state、WorkflowStateStore、EventMemory 与 rolling budget 全部独立 reset。两个状态服务可共享底层 `FinalizedEvent` log，但接口、状态语义、配置与 ablation switch 必须分离；关闭 EventMemory retrieval 不得关闭 Workflow。

---

# 4. V3.1-API 三个论文贡献必须在代码中一一可消融

## Contribution A — Track–Workflow Structured Context

代码必须能够独立运行：

```text
A0 frames-only
A1 frames + predicted track
A2 frames + predicted track + historical workflow
A3 A2 + downstream surgical soft priors/signals
```

不要让 context ablation 隐式改变：

- API model；
- prompt task definition；
- image count；
- ontology；
- output schema。

## Contribution B — Task-wise Benefit-Gated Sparse Specialist Routing

V1 surgical knowledge 不删除，而是变成连续 soft features：

```text
IVT internal compatibility
Phase-IVT compatibility
phase transition anomaly
prediction change
temporal instability
track continuity/conflict
memory disagreement
optional uncertainty proxies
```

第一主线固定三个 scope：

```text
spatial_track
interaction
workflow
```

Gate 学习每个 scope 的额外验证预期收益：

```text
benefit_hat(scope) ≈ expected improvement after invoking that Specialist
```

而不是：

```text
P(prediction is wrong)
```

主运行策略默认每个 streaming state 最多调用一个 Specialist：

```python
r_star = argmax_r benefit_hat[r]
VERIFY(r_star) if benefit_hat[r_star] > tau_B else ACCEPT
```

`tau_B` 只能由 validation 固定。

全 test video 结束后 global top-k 选 verification 只能叫：

```text
oracle_offline_budget
```

不得作为 online 主方法。

## Contribution C — Reliability-Aware Evidence-Constrained Same-Backbone Specialist Verification

三个 Specialist 共享 baseline perception 的 exact API model identifier，但角色和修改权限不同：

|Scope|职责|允许修改|禁止修改|
|---|---|---|---|
|`spatial_track`|核验器械语义、已有 proposal/track 的空间连续性|授权 scope 内 instrument/相关 IVT 语义；P5 明确允许时才可调整关联|新增 bbox、实例或 track ID|
|`interaction`|核验 verb、target 与 IVT closure|授权 scope 内 V/T/IVT|phase、bbox、track identity|
|`workflow`|核验 phase 与因果阶段转移|phase|实例级 I/V/T/IVT、bbox、track identity|

每个 Specialist 必须：

1. 与 baseline perception 使用相同 API model identifier；
2. 角色不同：verification / falsification，而不是重复 perception；
3. 读取新增证据：
   - candidate alternatives；
   - predicted track；
   - historical workflow；
   - memory evidence；
   - soft priors；
4. 只能输出：
   - `KEEP(H0)`
   - `REPAIR(H0 -> Hk)`，且 `Hk` 属于对应 scope 的候选池；
5. 返回 `agent_role`、`scope_id`、`selected_candidate_id` 和 `touched_tasks`；
6. 禁止自由生成候选池外类别或修改 scope 外任务。

`DeterministicCoordinator` 不是额外 LLM Agent。它只校验 route、candidate 和 touched-task 权限，合法时提交更新，非法或 parse 失败时执行带 reason code 的 KEEP fallback；它不得新增语义或读取 GT。Agent 数量本身不是论文贡献，贡献是受预算约束的 task-wise benefit routing 与安全 scoped repair。

Memory retrieval：

```text
q_capped_i(task)
= min(q_base_i(task), provenance_cap(status_i, source_quality_i))

RetrievalScore_i(task)
= Similarity_i
× q_capped_i(task)
× TemporalWeight(age_i)
```

`provenance_cap` 不得提高 `q_base`，且 TemporalWeight 只能在 retrieval 施加一次。四项方法学修订均属于上述三个贡献内部的严谨性增强，不得包装成第四个贡献。

---

# 5. 推荐项目代码架构

在 P0 Audit 后根据现有 repository 最小修改；若当前无冲突，可向以下结构收敛。

```text
StreamingSurgicalAgent/
├── pyproject.toml
├── README.md
├── configs/
│   ├── base.yaml
│   ├── data/
│   │   └── cholectrack20.yaml
│   ├── api/
│   │   ├── default.yaml
│   │   └── provider_template.yaml
│   ├── tracking/
│   │   └── default.yaml
│   ├── memory/
│   │   └── default.yaml
│   ├── gate/
│   │   └── default.yaml
│   ├── verification/
│   │   └── default.yaml
│   ├── experiments/
│   │   ├── dry_run.yaml
│   │   ├── local_smoke.yaml
│   │   ├── api_a0_frames_only.yaml
│   │   ├── api_a1_track.yaml
│   │   ├── api_a2_track_workflow.yaml
│   │   ├── api_a3_common_context.yaml
│   │   ├── never_verify.yaml
│   │   ├── random_matched.yaml
│   │   ├── confidence_proxy.yaml
│   │   ├── v1_rule_gate.yaml
│   │   ├── binary_joint_gate.yaml
│   │   ├── v3_scoped_benefit_gate.yaml
│   │   ├── always_joint_verify.yaml
│   │   ├── always_spatial_track.yaml
│   │   ├── always_interaction.yaml
│   │   ├── always_workflow.yaml
│   │   ├── always_all_specialists.yaml
│   │   └── v3_1_api_full.yaml
│   └── ablations/
│       ├── no_memory.yaml
│       ├── raw_recent_memory.yaml
│       ├── finalized_memory.yaml
│       └── reliability_memory.yaml
│
├── src/surgical_agent/
│   ├── config/
│   │   ├── schemas.py
│   │   ├── loader.py
│   │   └── hashing.py
│   ├── schemas/
│   │   ├── data.py
│   │   ├── inference.py
│   │   ├── prediction.py
│   │   ├── candidates.py
│   │   ├── verification.py
│   │   ├── reliability.py
│   │   ├── memory.py
│   │   └── evaluation.py
│   ├── data/
│   │   ├── cholectrack20/
│   │   │   ├── parser.py
│   │   │   ├── split.py
│   │   │   ├── media.py
│   │   │   ├── ontology.py
│   │   │   └── audit.py
│   │   ├── targets/
│   │   │   ├── builder.py
│   │   │   └── masks.py
│   │   ├── windows.py
│   │   └── dataset.py
│   ├── api/
│   │   ├── contracts.py
│   │   ├── client.py
│   │   ├── registry.py
│   │   ├── request_hash.py
│   │   ├── cache.py
│   │   ├── retry.py
│   │   ├── usage.py
│   │   ├── errors.py
│   │   └── providers/
│   │       ├── base.py
│   │       └── <provider>.py
│   ├── perception/
│   │   ├── contracts.py
│   │   ├── joint_api_vlm.py
│   │   ├── local_smoke.py
│   │   ├── context_builder.py
│   │   ├── parser.py
│   │   └── prompts/
│   │       ├── perception_prompt.txt
│   │       └── perception_schema.json
│   ├── tracking/
│   │   ├── contracts.py
│   │   ├── predicted_provider.py
│   │   ├── oracle_provider.py
│   │   └── summaries.py
│   ├── workflow/
│   │   ├── contracts.py
│   │   ├── state_store.py
│   │   └── summary.py
│   ├── research/
│   │   ├── knowledge/
│   │   │   ├── contracts.py
│   │   │   ├── prior_builder.py
│   │   │   ├── prior_store.py
│   │   │   └── signals.py
│   │   ├── candidates/
│   │   │   ├── contracts.py
│   │   │   └── generator.py
│   │   ├── gate/
│   │   │   ├── contracts.py
│   │   │   ├── features.py
│   │   │   ├── policies.py
│   │   │   ├── benefit_model.py
│   │   │   ├── d0_dataset.py
│   │   │   ├── policy_matched_rollout.py
│   │   │   ├── d1_dataset.py
│   │   │   ├── training.py
│   │   │   └── calibration.py
│   │   ├── verification/
│   │   │   ├── contracts.py
│   │   │   ├── joint_verifier.py
│   │   │   ├── router.py
│   │   │   ├── coordinator.py
│   │   │   ├── registry.py
│   │   │   ├── specialists/
│   │   │   │   ├── spatial_track.py
│   │   │   │   ├── interaction.py
│   │   │   │   └── workflow.py
│   │   │   ├── response_parser.py
│   │   │   └── prompts/
│   │   │       ├── joint_verifier_prompt.txt
│   │   │       ├── spatial_track_prompt.txt
│   │   │       ├── interaction_prompt.txt
│   │   │       ├── workflow_prompt.txt
│   │   │       └── specialist_schema.json
│   │   ├── reliability/
│   │   │   ├── contracts.py
│   │   │   ├── estimator.py
│   │   │   └── provenance_cap.py
│   │   └── memory/
│   │       ├── contracts.py
│   │       ├── event_memory.py
│   │       ├── snapshot.py
│   │       └── retrieval.py
│   ├── systems/
│   │   ├── pipeline.py
│   │   ├── baseline_api.py
│   │   ├── baseline_local.py
│   │   └── v3_1_api_streaming.py
│   ├── engines/
│   │   ├── rollout.py
│   │   ├── gate_bootstrap.py
│   │   ├── gate_refresh.py
│   │   ├── gate_train.py
│   │   ├── inference.py
│   │   └── evaluation.py
│   ├── metrics/
│   │   ├── recognition.py
│   │   ├── gate.py
│   │   ├── verification.py
│   │   ├── streaming.py
│   │   ├── memory.py
│   │   └── api_cost.py
│   └── artifacts/
│       ├── manifest.py
│       └── provenance.py
│
├── scripts/
│   ├── audit_data.py
│   ├── run_local_smoke.py
│   ├── run_api_baseline.py
│   ├── build_priors.py
│   ├── probe_joint_verifier.py
│   ├── probe_specialists.py
│   ├── build_gate_dataset.py
│   ├── train_gate.py
│   ├── run_streaming.py
│   ├── evaluate.py
│   └── aggregate_runs.py
│
├── tests/
│   ├── test_causality.py
│   ├── test_gold_free.py
│   ├── test_partial_labels.py
│   ├── test_api_schema.py
│   ├── test_api_cache.py
│   ├── test_api_usage.py
│   ├── test_track_safety.py
│   ├── test_workflow_causality.py
│   ├── test_prior_split.py
│   ├── test_memory_contract.py
│   ├── test_memory_snapshot.py
│   ├── test_candidate_constraints.py
│   ├── test_specialist_constraints.py
│   ├── test_specialist_router.py
│   ├── test_coordinator_scope_safety.py
│   ├── test_gate_pair.py
│   ├── test_budget_causal.py
│   └── test_evaluation_isolation.py
│
├── docs/
└── outputs/
```

## 5.1 当前仓库落地规则

当前 repository 的唯一主实现包是 `src/surgical_agent/`。`src/streaming_surgical_agent/` 仅作为兼容命名空间保留版本导出；禁止在两个包中各实现一套 parser、runner、model 或 evaluation。现有 `src/surgical_agent/data/` 是已测试的 P1 基础，后续必须复用，不得为了贴合示意目录而搬移或复制。

当前大量文件只有 owner docstring，这表示模块尚未实现，不表示接口已经通过。每个占位文件只能在所属阶段转为真实代码；例如 P2 不得提前填充 Learned Gate、Specialist 或 EventMemory 算法，但可以建立它们未来要实现的 Protocol 和显式 No-op 实现。

## 5.2 单一 Canonical Pipeline

新增 `src/surgical_agent/systems/pipeline.py` 作为唯一逐视频/逐状态运行循环。`baseline_system.py`、`v1_streaming.py` 和 `v3_streaming.py` 只能负责组装不同 `PipelineComponents`，不得各自复制循环。主干顺序冻结为：

```text
1. detect video boundary and reset Tracker / Workflow / Memory / budget
2. resolve Gold-free InferenceSample and isolated EvaluationTarget
3. snapshot all prior causal state before processing t
4. build causal ContextBundle
5. call PerceptionBackend and validate InitialPrediction
6. build ontology-valid CandidateSet and GateFeatures
7. call GatePolicy: ACCEPT or VERIFY(scope)
8. if VERIFY(scope), SpecialistRouter selects exactly one enabled Specialist
9. Specialist returns KEEP or scoped REPAIR from its CandidateSet
10. DeterministicCoordinator validates role/scope/candidate/touched_tasks
11. Finalizer emits PredictionRecord and FinalizedEvent
12. PredictionWriter persists runtime trace
13. commit WorkflowStateStore and EventMemory after finalization
14. EvaluationEngine consumes PredictionRecord + EvaluationTarget off the runtime branch
```

`PipelineComponents` 至少包含 `context_builder`、`perception`、`candidate_generator`、`signal_extractor`、`gate_policy`、`specialist_router`、`specialist_registry`、`coordinator`、`finalizer`、`workflow_store`、`event_memory`、`prediction_writer`。P2 使用 frames-only context、local smoke perception、NeverVerify、DisabledSpecialistRegistry、NoOpCoordinator、No-op Workflow 和 No-op Memory，但仍经过同一主干并产生相同 trace 字段。后续阶段只替换组件，不改变步骤顺序。

运行失败必须显式分类：data contract error、schema error、provider error、candidate violation、state causality error 和 artifact write error。不得 catch-all 后继续生成正常 PredictionRecord；允许的 deterministic fallback 必须配置化、记录 reason code，并且不能读取 GT。

## 5.3 消融实现合同

每个实验配置必须由 `base + data + stage component config + one ablation override` 合成，并将 resolved config hash 写入 manifest。消融不得修改 split、frame window、ontology、prompt/schema、backbone、cached perception records、EvaluationEngine 或 seed policy，除非该变量正是当前层明确研究的对象。

|消融层|允许变化|必须冻结|
|---|---|---|
|Context A0/A1/A2|frames / predicted track / historical workflow|Perception model、window、output schema、evaluation|
|Policy/Router|Never/Random/Confidence/V1/Binary Joint/G1/Always Joint/Always Each/Always All|perception outputs、context、scope candidates、Specialist pool、Coordinator、memory/reliability、matched budget|
|Specialist|Joint / spatial_track / interaction / workflow / all|frozen snapshots、backbone、candidate generator、Coordinator、evaluation|
|Memory|none/raw/finalized/reliability|context、policy/router、Specialist pool、Coordinator、candidate K、evaluation|
|Reliability|weighting/cap/temporal policy|retrieved candidate events and all upstream predictions|

禁止为了“架构漂亮”而无必要复制已有正确模块。P0 后应优先复用而不是推倒重写。

---

# 6. 核心 Typed Contracts

第一版应优先建立以下边界。

## 6.1 API

```python
class MultimodalApiClient(Protocol):
    def call(self, request: ApiRequest) -> ApiResponse:
        ...
```

`ApiRequest` 必须可 canonicalize 和 hash。

`ApiResponse` 至少包含：

```text
provider
model_identifier
request_hash
raw_response
parsed_payload
input_tokens
output_tokens
image_count
latency_ms
retry_count
timestamp
cache_hit
estimated_cost
```

未知字段用 `None`，禁止伪造。

## 6.2 Perception

```python
class PerceptionBackend(Protocol):
    def predict(
        self,
        sample: InferenceSample,
        context: StructuredContext,
    ) -> InitialPrediction:
        ...
```

输出：

```text
instance_predictions[]: prediction_id + bbox + optional track_ref + I/V/T/IVT top-k
optional frame_multilabel: I/V/T/IVT presence sets with explicit granularity
frame-level Phase top-k
evidence_refs
optional self_reported_confidence
provenance
```

CholecTrack20 是多器械数据；禁止把所有实例压成一个未定义的“主 I/V/T/IVT”。`InitialPrediction` 和 `PredictionRecord` 必须携带 granularity。frame-level 与 instance-level 输出只能由各自兼容的 EvaluationTarget 评估。默认 Verifier 只能修正 scoped semantic candidate 或 phase，不能修改 bbox、实例数量和 track identity。

API self confidence 字段必须命名：

```text
self_reported_confidence
```

未经真实校准，禁止命名 `probability_of_correctness`。

## 6.3 Candidate Generator

```python
class CandidateGenerator(Protocol):
    def build(
        self,
        prediction: InitialPrediction,
        ontology: Ontology,
    ) -> CandidateSet:
        ...
```

Candidates 禁止由 test GT 构造。
每个 candidate 必须包含 `scope_id`（例如 `inst:0:verb` 或 `frame:phase`）、task、ontology ID 和来源；不同实例、不同粒度的候选不得混在一个无 scope 列表中。

## 6.4 Gate

```python
class VerificationPolicy(Protocol):
    def decide(
        self,
        features: GateFeatures,
        state: GateRuntimeState,
    ) -> GateDecision:
        ...
```

`GateDecision`：

```text
action: ACCEPT | VERIFY(scope)
benefit_hat_by_scope: {spatial_track, interaction, workflow}
selected_scope: null | spatial_track | interaction | workflow
reason_scores
policy_id
threshold
```

`selected_scope` 必须属于配置冻结的 role registry；主方法每个状态最多选择一个 scope。Gate 不得输出具体修复标签，也不得依据 realized benefit 或 future state 选 route。

Gate model artifact 必须区分：

```text
gate_stage: bootstrap_g0 | final_g1
training_dataset_ids
training_recipe_id
normalization_version
rollout_policy_id
source_split: train
generation_version
```

## 6.5 Sparse Specialist Verification

```python
class JointVerifier(Protocol):
    def verify(
        self,
        prediction: InitialPrediction,
        candidates: CandidateSet,
        evidence: VerificationEvidence,
        memory: MemorySnapshot,
    ) -> VerificationResult:
        ...

class SpecialistRouter(Protocol):
    def route(
        self,
        decision: GateDecision,
        registry: SpecialistRegistry,
    ) -> SpecialistVerifier:
        ...

class SpecialistVerifier(Protocol):
    role: Literal["spatial_track", "interaction", "workflow"]

    def verify(
        self,
        scope_id: str,
        prediction: InitialPrediction,
        candidates: CandidateSet,
        evidence: VerificationEvidence,
        memory: MemorySnapshot,
    ) -> SpecialistResult:
        ...

class DeterministicCoordinator(Protocol):
    def validate(
        self,
        gate_decision: GateDecision,
        candidates: CandidateSet,
        result: SpecialistResult,
    ) -> ValidatedUpdate:
        ...
```

输出只能：

```text
agent_role
scope_id
KEEP | REPAIR(candidate_id)
touched_tasks
evidence trace
```

P8-A 的 `JointVerifier` 只用于先证明“第二次证据化调用”本身有价值，并作为后续路由对照。论文主运行由 `SpecialistRouter + SpecialistVerifier + DeterministicCoordinator` 执行。invalid candidate id、role/scope 不一致、scope 外 touched task 或 schema 错误都必须 fail closed 到显式 KEEP，并记录 reason code；Coordinator 不得自行修复或访问 GT。

## 6.6 Reliability

```python
class ReliabilityEstimator(Protocol):
    def estimate(
        self,
        finalized_state: FinalizedState,
        evidence: FinalizationEvidence,
    ) -> ReliabilityProfile:
        ...
```

建议：

```text
global_q_base / global_q_capped
q_base_* / q_capped_* for Instrument, Verb, Target, IVT, Phase, Track
status: gate_accepted | verified_accepted | repaired
source / source_quality
applied_provenance_cap
reliability_version
```

语义始终是：

```text
reliability score
```

不是 correctness probability，除非单独做过 calibration 并记录。

`q_capped(task) = min(q_base(task), provenance_cap(status, source_quality))`。所有 q 必须位于 `[0,1]`；cap policy 只按预声明 train/validation protocol 决定，并在 test 前冻结。

## 6.7 Workflow State

```python
class WorkflowStateStore(Protocol):
    def reset(self, video_id: str) -> None: ...
    def snapshot(self) -> WorkflowState: ...
    def update(self, event: FinalizedEvent) -> None: ...
```

`WorkflowState` 至少包含 recent finalized phase history、stability、transitions/tendency、compact temporal state 和 `source_max_frame_id`。该接口不得提供 similarity retrieval。

## 6.8 Event Memory

```python
class EventMemory(Protocol):
    def reset(self, video_id: str) -> None: ...
    def snapshot(self) -> MemorySnapshot: ...
    def retrieve(self, query: RetrievalQuery) -> list[RetrievedEvent]: ...
    def commit(self, event: FinalizedEvent) -> None: ...
```

`EventMemory` 是 bounded per-video episodic finalized-event store，用于 candidate support、similar events、reliability/provenance evidence 和 retrieval trace。`MemorySnapshot` 一步内 immutable；不得承担基础 WorkflowContext 生成。

---

# 7. Joint API-VLM Perception 规范

主实验默认每个 streaming state 一次 joint perception API call。

不要默认复制母体论文“五个 task expert = 五次 API”。

## 输入

必须使用 compact structured context：

```json
{
  "causal_frames": ["..."],
  "track_context": {
    "tracks": []
  },
  "workflow_context": {
    "recent_finalized_phases": [],
    "transition_summary": null,
    "source_max_frame_id": null
  },
  "ontology_version": "..."
}
```

## 输出

推荐 schema：

```json
{
  "instances": [
    {
      "prediction_id": "inst:0",
      "bbox_xywh_norm": [0.0, 0.0, 0.0, 0.0],
      "track_ref": null,
      "instrument": {"choice": null, "topk": []},
      "verb": {"choice": null, "topk": []},
      "target": {"choice": null, "topk": []},
      "ivt": {"choice": null, "topk": []}
    }
  ],
  "frame_multilabel": null,
  "phase": {"choice": null, "topk": []},
  "evidence_refs": [],
  "self_reported_confidence": {},
  "schema_version": "perception_v1"
}
```

必须：

- JSON schema validate；
- ontology validate；
- unique `prediction_id`、bbox range、candidate scope 与 granularity validate；
- invalid choice fail explicitly；
- 不得将多个 instance 静默压成单一 label，也不得把 frame-level label 绑定到 bbox；
- malformed API response 不得静默修复成合法答案；
- retries 必须记录；
- prompt/version 进入 provenance。

---

# 8. API Client、Cache 与 Reproducibility

API 系统是论文可复现性的高风险点，必须作为一等公民。

## Request Hash

hash 至少覆盖：

```text
provider
model identifier
system prompt
user prompt
prompt version
structured context
image identifiers/content hashes
generation parameters
response schema version
```

同一 request hash：

- 默认优先 cache replay；
- cache hit 不重复计入实际 provider monetary call；
- usage 统计要区分 `logical_call` 和 `provider_call`。

## Artifact

```text
outputs/<run_id>/
├── resolved_config.yaml
├── run_manifest.json
├── environment.json
├── git_info.json
├── dataset_manifest.json
├── api_cache/
│   └── <request_sha256>.json
├── api_usage.jsonl
├── prompts/
│   ├── perception_prompt.txt
│   ├── joint_verifier_prompt.txt
│   ├── specialists/
│   │   ├── spatial_track_prompt.txt
│   │   ├── interaction_prompt.txt
│   │   └── workflow_prompt.txt
│   └── schema_versions.json
├── priors/
├── gate/
├── predictions/
├── traces/
├── metrics/
└── logs/
```

每个 API record 至少记录：

```text
provider
exact model identifier
timestamp
prompt version
temperature / generation settings if applicable
request hash
cache hit
input/output tokens if provider exposes
image count
latency
retry count
provider error
estimated or actual cost if determinable
```

主实验优先采用 provider 可用的最确定设置。若无法保证 deterministic，cache 必须固定最终正式结果。

---

# 9. Train-only Surgical Priors 与 Soft Signals

新增：

```text
research/knowledge/
```

统计只能来自 Train。

可包含：

```text
phase transition
phase-instrument
phase-IVT
I-V
I-T
V-T
```

前提是数据真实支持。

要求：

- additive/Laplace smoothing；
- support count；
- source split；
- ontology version；
- schema version；
- config hash；
- low support -> neutral / weak score；
- val/test 禁止 rebuild。

V1 rule knowledge 在 V3.1-API 中变成 `GateFeatures` 和 Specialist evidence，不直接硬决定 route 或 VERIFY。

---

# 10. Evidence-Constrained Sparse Specialist Verification

第一次 API 角色：

```text
PERCEPTION:
What is happening?
```

P8-A 先使用统一 Joint Verifier 回答基础能力问题：在相同 snapshot 上，第二次 evidence-conditioned call 是否具有可重复净收益。只有该闸门通过，才进入 P8-B 的 Specialist promotion probe。

论文主运行的第二次 API 角色由 Gate 选择其一：

```text
SPATIAL_TRACK SPECIALIST:
Falsify instrument/spatial/track hypotheses within the authorized scope.

INTERACTION SPECIALIST:
Falsify verb/target/IVT hypotheses within the authorized scope.

WORKFLOW SPECIALIST:
Falsify phase/transition hypotheses within the authorized scope.
```

所有 Specialist prompt 都必须遵循：`Try to falsify H0 using supplied evidence; KEEP unless another provided candidate is better supported.` 不允许只是“please reconsider”，也不得把角色名当作唯一差异。

必须输入：

```text
agent_role and authorized scope_id
H0
scope-specific H1...Hk
same causal visual evidence
scope-relevant track/workflow evidence
scope-relevant memory evidence
scope-relevant soft priors
trigger reasons
```

输出 schema：

```json
{
  "agent_role": "interaction",
  "scope_id": "inst:0:ivt",
  "decision": "KEEP",
  "selected_candidate_id": null,
  "touched_tasks": [],
  "evidence_for_current": [],
  "evidence_against_current": [],
  "evidence_for_selected": [],
  "reason_code": "...",
  "self_reported_confidence": null
}
```

若 `decision=REPAIR`：

- `selected_candidate_id` 必须属于 CandidateSet；
- candidate 必须属于当前 `scope_id`；
- `touched_tasks` 必须是该 role 的允许子集；
- 不得生成 pool 外标签；
- 所有 task update 必须可追溯到 candidate。

`DeterministicCoordinator` 在任何状态下都不得调用 LLM。它依次检查 `selected_scope == agent_role`、scope/candidate membership、touched-task allowlist 与 schema version；任一失败均输出带 reason code 的 KEEP fallback。主配置每个状态最多调用一个 Specialist；多 Specialist 并行只作为 `Always All Specialists` 成本上界，不作为主运行机制。

必须报告：

```text
per-Specialist KEEP / REPAIR count
per-Specialist repair success rate / repair precision / harm rate
scope-wise CandidateRecall@K
scope violation count (must be 0)
routing distribution and extra-call cost
```

---

# 11. WorkflowState 与 Provenance-Aware Event Memory

`WorkflowStateStore` 与 `EventMemory` 都是**单个 video 内**的持久服务，不是 benchmark-wide knowledge base。二者消费同一 finalized event stream，但必须独立配置、独立 reset、独立消融；禁止跨病例状态。

```text
FinalizedState
→ ReliabilityProfile
→ FinalizedEvent
├→ WorkflowStateStore.update(FinalizedEvent)
└→ EventMemory.commit(FinalizedEvent)
```

WorkflowStateStore 只维护 compact causal workflow state，供 context builder、Perception 和 Gate temporal/workflow features 使用。EventMemory 只维护 bounded episodic finalized events，供 candidate support、Verification evidence、reliability/provenance retrieval，以及可选的显式 memory-disagreement feature 使用。

`FinalizedEvent`：

```text
video_id
frame_id
instance_states[] with scoped I / V / T / IVT
optional frame_multilabel with explicit source granularity
frame-level Phase
predicted_track_state
verification_status
evidence_refs
ReliabilityProfile
provenance
```

第一版 ReliabilityProfile 不必再训练大网络。

建议由以下 deterministic evidence 组合：

```text
temporal agreement
track agreement
workflow compatibility
candidate stability
Specialist evidence margin
optional calibrated uncertainty
```

先计算确定性的 task-wise `q_base`，再施加 provenance-aware upper bound：

```text
q_capped(task) = min(q_base(task), provenance_cap(status, source_quality))
```

`status` 至少为 `gate_accepted | verified_accepted | repaired`。cap 必须 config-driven，按预声明 train/validation protocol 决定，并在 test 前冻结。它是保守证据上界，不是 correctness probability。所有 q clip 到 `[0,1]`，且 `q_capped <= q_base`。

检索分数唯一允许的时间衰减位置是 retrieval：

```text
RetrievalScore_i(task)
  = Similarity_i × q_capped_i(task) × TemporalWeight(age_i)
```

ReliabilityEstimator 不得预先把 TemporalWeight 乘进 q；retrieval 不得重复施加。`TemporalWeight(age)` 应随 age 单调不增。

Memory 必须 bounded：

```text
max_events_per_video
recent candidate limit
optional phase bucket
```

记录：

```text
memory size
retrieval latency
retrieval trace
reliability contribution
q_base / q_capped per task
verification status
source / source_quality
applied provenance cap
TemporalWeight(age)
final RetrievalScore
reliability_version
```

Graph Memory / rollback / quarantine 不进入第一主线。

---

# 12. Gate Dataset：Normalized Benefit 与一次 Policy-Matched Refresh

Gate supervision 是整个项目最容易出学术漏洞的地方。

每个 train-only paired sample 的两个分支必须共享 immutable pre-decision snapshot：

```text
same causal frames
same predicted track
same WorkflowState
same EventMemorySnapshot
same soft priors
same candidate set
same InitialPrediction
```

从同一状态分叉。主 scope 集合固定为 `R={spatial_track, interaction, workflow}`：

```text
Branch A:
P_t
→ no additional verification
→ loss_before

Branch B_r, for each enabled r in R:
P_t
→ same-backbone Specialist(r)
→ deterministic Coordinator
→ P'_(t,r)
→ loss_after_(r)
```

对 `k ∈ {I, V, T, IVT, Phase}` 和每个实际执行的 route `r` 逐任务计算：

```text
raw_delta_(t,r,k) = error_before_(t,k) - error_after_(t,r,k)
delta_(t,r,k) = raw_delta_(t,r,k) / max(s_k, epsilon)

Delta_(t,r) = sum_k(m_(t,k) * w_k * delta_(t,r,k))
              / (sum_k(m_(t,k) * w_k) + epsilon)
```

`m_(t,k)` 必须复用统一 completeness mask。缺失任务同时从分子和分母排除，不得编码为零收益、负样本或 error。`w_k` 必须预声明；`s_k` 只允许由 train 派生并冻结。若 canonical task error 已处于可比 `[0,1]` 范围，则 `s_k=1`。禁止使用 validation/test 统计归一化尺度。若某个 route 因 API、parse、budget 或配置原因未真实执行，必须保存 `route_observed_mask_(t,r)=false` 并从该 head 的 loss 排除；禁止把未观察 route 的 `Delta_(t,r)` 写成 0。

**TBD / BLOCKED: canonical per-sample task error semantics must be resolved before Gate dataset implementation.** `error_(t,k)` 必须调用唯一 canonical `EvaluationEngine` / task definition。在该语义被数据契约和评估协议确认前，不得自行选择 BCE、CE、F1 surrogate 等实现。

严禁：

- P_t / P'_t 来自不同 memory trajectory；
- 用 test GT 生成 Gate dataset；
- verifier 分支 commit 后再回头算 before branch；
- Gate Features 读取 realized Delta；
- current GT label 进入 prompt。

每条 Gate artifact 至少需要：

```text
Delta per observed route
raw_delta / normalized delta per route and task
task masks / task weights / task scales
route_observed_mask
normalization_version
pre_decision_snapshot_id
rollout_policy_id
source_split: train
generation_version
```

固定且只执行一次的 refresh lifecycle：

```text
1. D0: train-only scoped counterfactual rollout；保存所有 pre-decision states，
   从同一 snapshot 分支 ACCEPT vs each enabled same-backbone Specialist。
2. G0: 仅由 D0 训练 bootstrap Gate；不作为 final deployed Gate。
3. G0 rollout: 在 train videos 上运行真实 causal selective policy，
   实际推进 ACCEPT/VERIFY(scope)、KEEP/REPAIR、FinalizedState、
   WorkflowState 与 EventMemory transitions。
4. D1: 从 G0 actual trajectory 的全部 encountered states 构建，
   不限于 G0 选择 VERIFY 的 states；每个状态仍做 same-snapshot scoped branches，
   未执行 route 保留 observed-mask 语义。
5. G1: 由 D1 训练，或采用训练前声明并固定的 D0+D1 recipe。
6. Validation: 只选择 final tau_B / minimal calibration。
7. Test: frozen G1 + frozen tau_B；禁止 refresh。
```

禁止 runtime 任意混合 D0/D1。禁止将该流程扩展为无限 policy iteration、RL 或 DAgger；本合同只允许一个 `D0 → G0 → D1 → G1` refresh cycle。

如果 upstream tracker 是 Track20-trained，必须 fold-safe。

Frozen external API-VLM 不需要每 fold 重训。

---

# 13. Gate Model 第一版

不要使用复杂 Transformer。

第一版：

```text
GateFeatures
→ shared small MLP / lightweight encoder
→ three scoped regression heads
→ benefit_hat_by_scope
```

建议同时支持：

```text
regression target: observed Delta_(t,r)
route_observed_mask-aware loss
per-scope ranking diagnostics
binary any-positive-benefit diagnostic
```

G0 与 G1 都以连续 `benefit_hat` 为核心，但只有 G1 是最终部署与论文主方法；G0 仅用于 bootstrap rollout 和诊断。

Validation 主要只选：

```text
tau_B
predeclared route tie-break
```

避免同时搜索大量：

- prior thresholds；
- Specialist/Coordinator weights（第一版应避免可学习加权）；
- repair margin；
- memory weights；
- K；
- Gate architecture。

---

# 14. 阶段化实施 P0-P12

以下文件级矩阵是当前仓库的执行地图。路径不存在时才新建；已有同职责模块时直接扩展。每一行只在前一阶段 PASS 后开始。

|阶段|主要代码落点|必须产出的 artifact / report|
|---|---|---|
|P0-P1|`data/*`、`runtime/phase.py`、`artifacts/*`|repository/data audit、repair manifest、P0/P1 report|
|P2|`data/dataset.py`、`models/baseline.py`、`training/{losses,trainer}.py`、`systems/{pipeline,baseline_system}.py`、`evaluation/evaluator.py`、`inference/writer.py`、`scripts/run_local_smoke.py`|resolved config、target-granularity eligibility audit、sample/mask summary、checkpoint、PredictionRecord、smoke metrics、`P2_REPORT.md`|
|P3|`api/{client,request_hash,cache,retry,usage,errors}.py`、`api/providers/*`、`scripts/smoke_api.py`|sanitized API provenance、cache fixture、usage/latency log、exact model identifier、`P3_REPORT.md`|
|P4|`models/perception/api_joint_vlm.py`、prompt/schema resources、`systems/baseline_system.py`|Single-pass cached rollout、schema failures、recognition/cost metrics、`P4_REPORT.md`|
|P5|`tracking/*`、`workflow/state_store.py`、context builder、pipeline component assembly|A0/A1/A2 traces、context source-frame audit、`P5_REPORT.md`|
|P6|`research/knowledge/*`、Gate signal extraction|train-only prior store、support/smoothing manifest、`P6_REPORT.md`|
|P7|`research/memory/*`、`research/verification/hypotheses.py`|bounded memory trace、immutable snapshots、CandidateRecall@K、`P7_REPORT.md`|
|P8|`research/verification/{joint_verifier,router,coordinator,specialists}/*`、Always policy assembly|P8-A Joint probe；P8-B per-Specialist capability/harm/cost/promotion report；`P8_REPORT.md`|
|P9|`research/gate/oof_dataset.py`、canonical EvaluationEngine error adapter|scoped D0 branches、route masks、task masks/scales/weights/version provenance、`P9_REPORT.md`|
|P10|`training/{gate_trainer,rollout}.py`、multi-head Gate calibration/policy modules|G0 scoped benefits、G0 rollout、D1、G1、frozen `tau_B`/tie-break、P10 substage reports|
|P11|`systems/v3_streaming.py`、`scripts/infer_v3.py`|frozen Gold-free sparse Specialist rollout、scope/state/cost traces、`P11_REPORT.md`|
|P12|`evaluation/*`、`scripts/evaluate.py`、aggregation/statistics tooling|三层实验表、per-video bootstrap/CI、cost/latency、release manifest、`P12_REPORT.md`|

## P0 — Audit + Docs + Scaffold

实现：

- repository audit；
- docs sync；
- architecture scaffold；
- typed contracts；
- config skeleton；
- test skeleton。

PASS：

```text
import works
config load works
pytest collection works
audit report exists
BLOCKED items explicit
```

## P1 — Engineering Hard Gates

验证/复用：

- parser；
- canonical schemas；
- target masks；
- media backend；
- causal windows；
- Gold-free inference；
- evaluation isolation；
- checkpoint/artifact contracts。

必须 PASS：

```text
causal
mask
no-GT
split safety
video reset
```

## P2 — Local Smoke Baseline

P2 只验证基础工程，不承担论文性能主张。它必须使用第 5.2 节的 canonical pipeline，而不是临时 notebook 或另写一条训练脚本。

### P2-0 前置条件

```text
P0 = PASS
P1 = PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION
active data manifest = D:/cholec_dataset/repair_manifest.json
P2 = PASS (2026-08-24 engineering smoke evidence)
```

若 manifest、侧车哈希、10/2/8 split 或 canonical raw JSON 发现规则不一致，P2 立即 `BLOCKED`。

### P2-1 DatasetAdapter 与 TargetBuilder

实现 `src/surgical_agent/data/dataset.py`，复用现有 parser、split、media、causal window 和 derived-supervision loader：

1. 普通视频读取与 VID 目录同名的官方 JSON；不得 glob 后任选 JSON。
2. VID30 validation 必须通过 repair manifest 显式路由到 `vid30_repaired.json`。
3. VID31 training 必须读取 `vid31_phase_repaired.json` 和 `vid31_frame_ivt_repaired.json`；bbox、operator、instance association、track 和 visual-condition mask 全为 false。
4. Test 只允许构造 Gold-free `InferenceSample`，不得进入 optimizer、early stopping、threshold selection 或 P2 smoke metric 调参。
5. 在 `data/schemas.py` 增加 evaluation-only `FrameSupervisionTarget` 与 `FrameTaskMask`。只有来源明确声明完整 frame-level presence semantics 时，I/V/T/IVT 的 frame mask 才能为 true；Track20 raw partial instance annotations 不得自动聚合成完整 multi-hot negative space。VID31 CholecT50 sidecar可提供该语义。
6. Phase 必须来自唯一有效值或显式 sidecar；同帧有效 phase 冲突时 fail closed，不得多数投票。普通 Track20 instance targets 继续保留在 `EvaluationTarget`，不为迁就 P2 本地模型而丢失粒度。
7. 输出 `(InferenceSample, EvaluationTarget, optional FrameSupervisionTarget, sample_provenance)`；传给 model 的对象只能是 `InferenceSample` 和已解析媒体 tensor。
8. 生成 `p2_target_granularity_audit.json`，逐视频/任务统计 instance-eligible、frame-multilabel-eligible、phase-eligible、masked 和冲突帧数。P2 只记录证据，不替 P4 静默选择论文指标粒度。

### P2-2 本地 Smoke Model

实现 `src/surgical_agent/models/baseline.py`：

- 默认 `pretrained=false`，禁止运行时联网下载权重；
- 第一条通过路径使用 current frame，随后验证相同 DatasetAdapter 可构造 causal window；
- 输出固定 shape 的 frame-level instrument/verb/target/triplet logits 与 phase logits，并显式声明 `frame_multilabel` granularity；P2 不伪造 instance predictions；
- 模型输出必须转换成统一 `PredictionRecord`，不得携带 GT、`FrameTaskMask` 或 `LabelMask`；
- P2 可使用 BCE/CE 作为明确标注 `SMOKE_ONLY` 的优化损失，但这些 loss 不得自动成为 P9 canonical per-sample Gate error。

### P2-3 Masked Loss 与最小训练

实现 `training/losses.py` 和 `training/trainer.py`：

1. 每个任务独立归一化 `sum(valid loss) / max(valid count, 1)`；无有效标签的任务返回零贡献并记录 count=0。
2. 至少执行一个真实 forward、finite loss、backward、optimizer step 和 checkpoint round-trip。
3. 断言 VID31 instance-level target 根本不进入本地 smoke loss/evaluation；其 frame-level sidecar与 instance target 不发生配对。
4. seed、device、dtype、batch sample IDs、manifest/config hash 写入 artifact。

### P2-4 Canonical Runner、Writer 与 Evaluation

实现 `systems/pipeline.py`、`systems/baseline_system.py`、`inference/writer.py` 和 `evaluation/evaluator.py`：

- P2 组件固定为 FramesOnlyContext + LocalSmokePerception + NeverVerify + DisabledSpecialistRegistry + NoOpCoordinator + NoOpWorkflow + NoOpMemory；
- 每个视频开始时执行统一 reset，即使 No-op 组件也必须记录 reset trace；
- PredictionWriter 原子写入 JSONL/manifest，失败不能被当作成功运行；
- EvaluationEngine 是唯一可同时读取 PredictionRecord 与 EvaluationTarget 的组件；
- P2 指标只标为 engineering smoke metrics，论文 canonical task metrics 可在同一 engine 中后续扩展，不得另建第二个 evaluator。

### P2-5 测试样本与最小覆盖

```text
VID02 training: ordinary raw route + phase-only smoke when phase is uniquely valid
VID31 training: CholecT50 frame-level I/V/T/Triplet presence plus Cholec80 phase enabled; instance/bbox/operator/track supervision masked
VID30 validation: repaired sidecar route + granularity-safe inference/evaluation
Testing: discovery allowed, training/selection access forbidden
```

必须新增单元测试：sidecar routing、target aggregation、mask-to-loss、output shape、checkpoint、PredictionRecord Gold-free、video reset、artifact atomicity。必须新增 integration test：上述三个 VID 的小样本端到端运行，并在固定 seed/device 下对 sample order、mask counts 和 metric keys 做确定性检查。

### P2-6 运行命令与 PASS 门槛

```text
python -m ruff check .
python -m compileall -q src tools tests
python -m pytest -q
python scripts/run_local_smoke.py --config configs/experiments/local_smoke.yaml
```

P2 必须同时满足：

```text
Data → local masked train-smoke → checkpoint reload
Data → canonical pipeline inference → PredictionRecord → Evaluation
all outputs finite and schema-valid
zero test leakage
VID30/VID31 routes and masks match repair manifest
all prior tests remain PASS
P2 report and provenance manifest exist
```

P2 结果不得写入论文主性能表，也不得据此选择 Gate、Verifier、Memory 或 API prompt。它只证明基础 Pipeline 可运行、可审计、可在后续模块嵌入时保持同一数据和评估边界。

## P3 — API Client + Cache

实现：

- provider-agnostic API abstraction；
- config-based provider；
- request hash；
- cache；
- retry；
- schema validation；
- usage log；
- provenance。

先 mock tests，再做**最小真实 API smoke**。

若用户尚未提供 API key/provider：

```text
REAL_API_SMOKE: BLOCKED
```

其他 mock/cache/schema 测试仍继续。

**禁止把 API key 写入 source/config artifact。**

仅从环境变量/secret mechanism 读取。

## P4 — API Single-pass Baseline

前置硬门槛：根据 P2 target-granularity audit 冻结 `PredictionRecord` 与 EvaluationEngine 的
instance matching / frame-multilabel / phase 规则。若 API capability 与可用 GT 粒度无法形成
同粒度比较，P4 标记 `BLOCKED`；禁止通过选择“主器械”、把 frame label 绑定 bbox，或忽略
未匹配实例来制造可评估结果。

实现 B0：

```text
causal frames
→ Joint API-VLM
→ structured instance I/V/T/IVT + optional frame multilabel + frame Phase + scoped candidates
→ evaluation
```

完成完整 rollout。

## P5 — Track + Workflow Structured Context

开始 P5 前必须冻结 detector/tracker architecture、权重或训练来源、Track20 class mapping、
是否使用 Track20 训练以及 fold-safe 策略。任一项不明确时 `TRACKER_PROVENANCE: BLOCKED`；
OracleTrack 只能继续作为 diagnostic，不能代替主路径通过 P5。

实现：

```text
A0 frames-only
A1 + predicted track
A2 + historical workflow
```

确认：

- current GT Phase 不回流；
- OracleTrack 主配置禁用；
- context source frame audit。

## P6 — Train-only Priors + Soft Signals

实现：

- PriorBuilder；
- PriorStore；
- support/smoothing；
- Gate feature extraction；
- provenance。

## P7 — Memory + Candidate Generator

实现：

- candidate pool；
- CandidateRecall@K diagnostic；
- EventMemory；
- MemorySnapshot；
- retrieval；
- bounded memory。

必须 PASS：

```text
no raw prediction commit
no self retrieval
snapshot immutable
video reset
bounded memory
candidate no-GT
```

## P8 — API Verification Probe

P8 分成两个顺序不可交换的研究闸门。

### P8-A — Joint Verification Capability

先小规模 difficult-case probe，再运行 Always Joint Verify，回答：

```text
Does same-backbone evidence-conditioned verification create repeatable net benefit?
```

比较：

```text
Never Verify
vs
Always Joint Verify
```

同时报告：

```text
repair count
repair precision
harm rate
CandidateRecall@K
cost
```

如果 Joint Verification 没有可重复净收益：

```text
P8-A = FAIL
GATE TRAINING STATUS = BLOCKED
```

停止 P8-B 及 P9-P11，先修 Joint Verifier/candidate/context。

不能靠 Learned Gate 创造不存在的 Agent value。

### P8-B — Specialist Promotion Probe

只有 P8-A PASS 后，才从相同 immutable snapshot 分别调用 `spatial_track`、`interaction`、`workflow` Specialist，并与 Joint Verifier 比较。至少一个 Specialist 必须在预声明任务或困难子集上改善以下至少一项，且没有不可接受的 harm：

```text
normalized benefit
repair precision
harm rate
same-benefit token/call cost
```

同时报告 per-Specialist CandidateRecall@K、KEEP/REPAIR、scope violation 和实际成本。若三个 Specialist 都不优于或不补充 Joint Verifier，则保留“Joint Verifier + Binary Benefit Gate”作为可实现主线，标记 `SPECIALIST ROUTING CLAIM = NOT SUPPORTED`；不得仅凭角色提示数量宣称多 Agent 创新。

## P9 — D0 Bootstrap Gate Dataset

构建：

- train-only ACCEPT vs each enabled Specialist scoped counterfactual；
- same immutable pre-decision snapshot；
- normalized task-wise mask-aware `Delta_(t,r)`；
- `route_observed_mask`；
- fold-safe train components；
- GateFeatures；
- D0 artifact provenance。

若 canonical per-sample task error semantics 尚未与 EvaluationEngine 对齐，本阶段标记 `BLOCKED`。严禁 test。

## P10-A — Train Bootstrap G0

由 D0 训练轻量 regressor G0。Artifact 必须标记 `bootstrap_g0`，不得作为 final deployed Gate。

## P10-B — Policy-Matched Train Rollout

使用 G0 在 train videos 上运行真实 causal selective rollout。必须实际执行 `ACCEPT|VERIFY(scope)`、routed KEEP/REPAIR、Coordinator validation、FinalizedState、WorkflowStateStore 与 EventMemory transition，保存每个 encountered pre-decision state。

## P10-C — Build D1

从 G0 actual trajectory 的**全部 encountered states**构建 D1；不得只采样 G0 选择 VERIFY 的状态。每条样本从同一 immutable snapshot 分支 ACCEPT 与可执行的 same-backbone Specialists；未执行 route 必须标记 `route_observed_mask=false`，不得伪造零收益。

## P10-D — Train Final G1

由 D1 训练 G1，或采用训练前预声明的固定 D0+D1 recipe。禁止 runtime 任意混合数据策略。

## P10-E — Validation Operating Point

Validation 仅冻结 final `tau_B`、预声明 route tie-break 与最小 calibration，不重训 G1。

比较至少：

```text
Never Verify
Random @ matched rate
Confidence Proxy
V1 Rule Gate
G0 Bootstrap Gate [diagnostic only]
Binary Benefit Gate + Joint Verifier
G1 Scoped Benefit Gate [main method]
Always Joint Verify
Always Each Specialist
Always All Specialists [cost upper bound]
optional Oracle Gate / Oracle Route [analysis only]
```

所有非 Oracle policy 必须在 matched verification budget 下比较。

## P11 — Full V3.1-API Runtime

组合：

```text
Track/Workflow
→ API Perception
→ Signals
→ frozen G1 scoped Benefit Gate + frozen tau_B/tie-break
→ at most one routed Specialist Agent
→ Deterministic Coordinator
→ Provenance-capped Reliability Estimation
→ WorkflowStateStore + EventMemory
```

完整 per-video Gold-free streaming rollout。Test 禁止 refresh、D1 重建、G1 重训或 cap/threshold 调整。

## P12 — Paper Experiments + Cost + Statistics

按以下顺序运行三层实验并分别归因。不得用后一层结果替代前一层主结论。

### Layer 1 — Backbone-level Framework Gain（主结果）

固定：

```text
same exact API model_identifier
same test videos
same causal visual window
same task ontology
same output schema
same EvaluationEngine
```

比较：

```text
Single-pass Baseline:
  one causal API perception call
  no Gate decision
  no extra Verification call

Full V3.1-API Agent:
  complete Track/Workflow context
  frozen scoped G1 + tau_B + route tie-break
  at most one candidate-constrained same-backbone Specialist
  deterministic Coordinator
  WorkflowStateStore + provenance-aware EventMemory

Framework Gain = Performance(Full Agent) - Performance(Single-pass Baseline)
```

输出逐任务和 aggregate paired per-video delta/confidence interval，并同时报告两支的 calls、tokens、latency 与 cost。该 delta 只代表完整 framework effect。

### Layer 2 — Verification Policy（matched-budget 核心归因）

固定：

```text
same backbone and model_identifier
same frozen perception outputs / execution definition
same causal context and prompt versions
same Track/Workflow
same scope-specific candidate generator
same Specialist pool and deterministic Coordinator
same WorkflowState/EventMemory/reliability configuration
same ontology and evaluation
```

仅改变哪些状态进入额外 verification 及 route policy，并比较：

```text
Never Verify
Random @ G1-matched rate/budget
Confidence Proxy Gate
V1 Rule Gate
Binary Benefit Gate + Joint Verifier
G1 Scoped Benefit Gate [main]
Always Joint Verify
Always Each Specialist
Always All Specialists [cost upper bound only]
Oracle Gate / Oracle Route [analysis only]
```

G0 Bootstrap Gate 仅作 `D0 -> G0 -> D1 -> G1` 训练诊断，不进入主 policy 排名。所有非 Oracle policy 必须在 matched verification budget 下比较。

### Layer 3 — Context / Memory / Reliability Ablation

Context 固定 single-pass backbone 与 causal visual input：

```text
A0 causal frames only
A1 + predicted Track
A2 + historical Workflow
A3 + downstream surgical soft-prior signals
```

Memory 固定 G1 + Specialist pool + Coordinator，并保持 Workflow/EventMemory 开关独立：

```text
No Workflow / No EventMemory
Workflow Only
Workflow + Raw Recent EventMemory
Workflow + Finalized-only EventMemory
Workflow + Reliability-weighted EventMemory
```

`No EventMemory` 不得隐式关闭 Workflow。

Reliability 固定 G1、Workflow、Finalized-only EventMemory、Specialist pool 与 Coordinator：

```text
No reliability weighting
Base reliability
Base reliability + provenance cap
Final provenance-aware retrieval
```

最后一项包含唯一一次 retrieval-time TemporalWeight。

### Multi-backbone Scope

主 backbone 运行 Layer 1、Layer 2 与 Layer 3 完整矩阵。若加入第二 backbone，只运行 `Single-pass Baseline / Always Joint Verify / V1 Rule Gate / Full Sparse V3.1-API Agent` 验证核心趋势；不得跨 backbone 相减生成 Framework Gain。

---

# 15. 必须报告的 Metrics

## Framework Gain（Main Results first）

```text
single-pass performance B, per task and aggregate
full-agent performance A, per task and aggregate
same-backbone framework gain A - B
paired per-video delta and confidence interval
delta in calls / tokens / latency / estimated cost
```

主结果表先报告 Framework Gain，再报告 policy attribution 与 component ablation。任何 framework delta 必须由相同 exact `model_identifier` 的两支计算，禁止跨 backbone 聚合或相减。

## Recognition

- Instrument F1；
- Verb F1（mask-aware）；
- Target F1（mask-aware）；
- IVT F1（明确 support）；
- Phase accuracy / macro-F1。

实际 micro/macro 以现有 ontology/task semantics 为准，不自行发明。

## Gate

```text
verification rate
scope selection distribution
per-scope benefit regression error
per-scope benefit rank correlation
positive-benefit route capture rate
route regret against Oracle Route
same-budget task performance
```

## Verification

```text
per-Specialist KEEP / REPAIR count
per-Specialist repair precision / repair success rate / harm rate
scope-wise CandidateRecall@K
scope violation count (must be 0)
```

## Streaming

```text
per-video metrics
temporal prediction switch rate
phase transition error
optional track continuity metrics
```

## Memory

```text
retrieval hit/support
weighted vs unweighted delta
memory size
retrieval latency
```

## API Efficiency

必须区分：

```text
base perception calls
extra verification calls
total logical calls
actual provider calls
cache hits
input/output tokens
image count
latency P50/P95
retries
estimated monetary cost
```

论文中：

```text
verification rate = extra verification rate
```

不能写成“只用了 X% API calls”。

没有真实 end-to-end latency 前，不称 `real-time`，只称：

```text
online causal streaming
streaming inference
```

---

# 16. API 公平性主实验 Contract

## 16.1 Framework Gain Contract

Single-pass Baseline 与 Full V3.1-API Agent 必须满足：

```text
same exact model_identifier
same test videos
same causal visual window
same task ontology
same output schema
same EvaluationEngine and metric implementation
```

Single-pass Baseline 只执行一次 causal API perception，不使用 Gate 或额外 Verification。Full Agent 执行完整 V3.1-API pipeline。二者差值是 aggregate framework effect；不能把它解释为 Gate、Memory、Context 任一单组件贡献，也不能用不同模型间差值替代。

## 16.2 Verification Policy Attribution Contract

论文核心 policy 对比必须满足：

```text
same exact model_identifier
same frozen Perception outputs / execution definition
same causal frames and context
same Track/Workflow context
same scope-specific candidate generator
same Specialist registry/prompts/backend
same deterministic Coordinator
same WorkflowState/EventMemory/reliability configuration
same ontology
same evaluation
matched verification rate/budget
```

只有进入额外 verification 的 policy 及其 route choice 不同。Binary Joint Gate 对照使用同一 frozen perception/context/candidates 和同一 additional-call budget；它用于判断 scoped routing 是否比“只决定调不调用统一 Verifier”更有价值。

否则不能把差异归因于 verification policy。

异构更强 verifier 或每状态调用多个 Specialist 可以做 extension，但必须单独表格，不能混入核心贡献。

## 16.3 Multi-backbone Contract

主 backbone 运行完整矩阵。第二 backbone 仅要求 `Single-pass / Always Joint / V1 / Full Sparse V3.1` 核心趋势，且每个 backbone 内部独立计算其同 backbone delta。跨 backbone 比较只能描述模型敏感性或趋势，不得命名为 Framework Gain。

---

# 17. 必须具备的 Tests

至少包括：

```text
test_parser / target masks
test_split_safety
test_gold_free_inference
test_causal_windows
test_video_reset
test_api_schema
test_api_cache
test_api_request_hash
test_api_usage
test_api_retry
test_track_safety
test_workflow_causality
test_prior_train_only
test_candidate_no_gt
test_candidate_constraints
test_memory_commit_contract
test_memory_snapshot
test_memory_bounded
test_no_self_retrieval
test_joint_verifier_candidate_constraint
test_specialist_role_registry
test_specialist_candidate_constraint
test_specialist_touched_tasks_allowlist
test_specialist_router_selects_at_most_one
test_coordinator_rejects_scope_mismatch
test_coordinator_rejects_out_of_scope_update
test_gate_pair_same_snapshot
test_gate_delta_mask
test_gate_scoped_delta
test_gate_route_observed_mask
test_unobserved_route_excluded_from_loss
test_gate_delta_missing_excluded_from_numerator_and_denominator
test_gate_normalization_train_only_frozen
test_gate_artifact_reconstructs_delta
test_gate_refresh_train_only
test_g0_rollout_causal
test_d1_all_encountered_states
test_d1_pair_same_snapshot
test_no_test_gate_refresh
test_workflow_without_event_memory
test_workflow_memory_independent_switches
test_workflow_memory_independent_reset
test_workflow_memory_finalized_causal_only
test_reliability_cap_nonincreasing
test_reliability_range
test_reliability_provenance_trace
test_reliability_policy_frozen_on_test
test_temporal_weight_monotonic
test_temporal_decay_applied_once
test_retrieval_score_reconstruction
test_budget_causal
test_evaluation_isolation
test_framework_comparison_same_model_identifier
test_framework_baseline_has_no_verification
test_framework_gain_rejects_cross_backbone_pair
test_policy_comparison_matched_components
test_policy_comparison_matched_budget
```

API integration tests 应使用 cached/mock fixtures，避免 CI 每次真实付费调用。

---

# 18. Codex 每阶段报告格式

每阶段完成后生成：

```text
reports/P0_REPORT.md
reports/P1_REPORT.md
...
reports/P12_REPORT.md
```

至少包含：

```text
Stage
Status: PASS / FAIL / BLOCKED / PARTIAL
Files created
Files modified
Commands actually executed
Tests actually executed
Observed outputs
Artifacts generated
Known issues
Data/API assumptions verified
Unresolved BLOCKED items
Next stage readiness
```

没有实际运行不得写 PASS。

如果 `FAIL/BLOCKED` 影响下一阶段：

```text
STOP HERE
```

不要继续生成后续“看似完整”的代码。

---

# 19. P8 关键研究闸门

P8 是论文逻辑的双重硬门槛。

P8-A 不要求 Always Joint Verify 每个任务都提升，而是：

> 在预先定义的主指标或困难子集上，same-backbone Joint Verifier 必须出现**可重复的净 benefit**，且 harm rate 可接受、repair 行为可解释。

如果结果接近：

```text
Never Verify ≈ Always Joint Verify
```

或者：

```text
Always Joint Verify worse
```

则：

```text
P8-A: FAIL
GATE TRAINING: BLOCKED
```

应首先检查：

- candidate recall 不够；
- Joint Verifier prompt 只是 retry；
- context 没提供新增证据；
- memory evidence 质量低；
- repair constraint 错误；
- API schema parse；
- ontology candidate mapping。

禁止为了赶流程直接训练 Gate。

P8-B 继续检查 scoped Specialist 是否提供 Joint Verifier 没有的任务价值或更优效率。如果所有 Specialist 相对 Joint Verifier 都没有可重复增量价值，则：

```text
P8-B: FAIL
SPARSE MULTI-AGENT CLAIM: NOT SUPPORTED
FALLBACK: Joint Verifier + Binary Benefit Gate
```

该结果不否定基础 verification pipeline，但阻止把多 Agent routing 写成主贡献。只有 P8-A 与 P8-B 都通过，才允许 P9 构建 scoped D0 并在 P10 训练三 head Gate。

---

# 20. 最终 Definition of Done

只有以下全部满足，才能称“V3.1-API 代码架构与主 pipeline 已实现”。

```text
P0 AUDIT/DOCS/SCAFFOLD: PASS
ENGINEERING HARD GATES: PASS
LOCAL SMOKE: PASS
API CLIENT/CACHE: PASS
API SINGLE-PASS BASELINE: PASS
SAME-BACKBONE SINGLE-PASS/FULL COMPARISON: PASS
BACKBONE-LEVEL FRAMEWORK GAIN TABLE: PASS
NO CROSS-BACKBONE FRAMEWORK GAIN: PASS
TRACK/WORKFLOW CONTEXT: PASS
TRAIN-ONLY PRIORS: PASS
MEMORY/CANDIDATES: PASS
JOINT VERIFICATION PROBE (P8-A): PASS
SPECIALIST PROMOTION PROBE (P8-B): PASS
SPECIALIST SCOPE/COORDINATOR SAFETY: PASS
SCOPED D0 BOOTSTRAP DATASET: PASS
MULTI-HEAD G0 BOOTSTRAP GATE: PASS
POLICY-MATCHED TRAIN REFRESH: PASS
D1 POLICY-MATCHED DATASET: PASS
G1 FINAL SCOPED BENEFIT GATE: PASS
NORMALIZED TASK-WISE MASK-AWARE SCOPED BENEFIT: PASS
ROUTE OBSERVED MASK: PASS
WORKFLOW/EVENTMEMORY SEPARATION: PASS
PROVENANCE-AWARE RELIABILITY: PASS
NO DOUBLE TEMPORAL DECAY: PASS
TEST FROZEN G1 + TAU_B / NO REFRESH: PASS
FULL STREAMING RUNTIME: PASS
MATCHED-BUDGET POLICY/ROUTING ATTRIBUTION: PASS
SPECIALIST/CONTEXT/MEMORY/RELIABILITY ABLATION: PASS
PAPER FRAMEWORK-GAIN/ABLATION/COST PIPELINE: PASS or explicitly PARTIAL with missing expensive runs
GOLD-FREE INFERENCE: PASS
STRICT CAUSAL: PASS
PARTIAL LABEL MASK: PASS
PREDICTED TRACK MAIN CONFIG: PASS
API PROVENANCE/CACHE: PASS
EVALUATION ISOLATION: PASS
```

其中 `BACKBONE-LEVEL FRAMEWORK GAIN TABLE` 只有在 Single-pass 与 Full Agent 的 exact `model_identifier` 及其余 Layer 1 固定项一致时才能 PASS；不同 backbone 的两次运行不得配对计算该差值。`NORMALIZED TASK-WISE MASK-AWARE SCOPED BENEFIT` 只有在 canonical per-sample task error semantics 已与唯一 EvaluationEngine 对齐后才能 PASS；否则 Gate dataset 及其后续阶段必须保持 `BLOCKED`。P8-A Joint Verification 与 P8-B Specialist Promotion 是 P9-P11 的顺序先决硬门槛；一次 policy refresh 不能绕过 `Never Verify ≈ Always Joint Verify`、不可接受 harm 或 Specialist 无增量价值的失败结论。

最终生成：

```text
reports/V3_1_API_IMPLEMENTATION_REPORT.md
```

包括：

- 最终目录树；
- 模块状态；
- P0-P12 状态；
- tests；
- API backend 配置；
- prompts/schema versions；
- data leakage guards；
- artifact/provenance；
- experiments implemented；
- unresolved limitations；
- paper-ready experiment matrix；
- 下一步建议。

---

# 21. 明确禁止事项

Codex 不得：

1. 未审计 repository 就重构；
2. 硬编码数据集绝对路径；
3. 把 API key 写进 source、git 或 artifacts；
4. 将 test GT 放进 API prompt；
5. 使用 GT track 作为主结果；
6. 当前 event 检索自己；
7. 使用 future frames；
8. 将缺失 IVT/V/T 当 negative；
9. 将 API self-reported confidence 当 calibrated probability；
10. verifier 自由生成 candidate pool 外标签；
11. 使用不同 VLM 比较 Gate 后声称 Gate 更强；
12. 全视频 global top-k verification 冒充 online policy；
13. 未测 latency 就声称 real-time；
14. 未运行 test 就写 PASS；
15. API 请求失败时静默伪造预测；
16. 不确定 ontology 时自行创造医学 label name；
17. 使用 val/test labels 更新 priors；
18. 在 P8 Agent 无收益时继续训练 Gate；
19. 一次性生成所有模块并跳过 phase gates；
20. 把工程模块数量包装成论文创新点；
21. 使用不同 backbone 的性能差值宣称 Framework Gain；
22. 将 Full Agent 相对 Single-pass 的总体 Framework Gain 单独归因给 Gate、Memory 或 Context。

---

# 22. 开发原则

代码目标是：

```text
Correct
Causal
Gold-free
Auditable
Reproducible
Ablatable
Research-student maintainable
```

而不是追求：

```text
maximum abstraction
maximum framework complexity
maximum number of agents
```

## 2026-08-26 Joint Perception single-pass implementation status

- The canonical gold-free Joint Perception core is implemented and locally
  verified with exact mock/OpenRouter configurations, three ordered synthetic
  32x32 RGB frames, deterministic evidence, `NeverVerify`, KEEP-only
  coordination, and paired prediction/evidence persistence.
- The runner executes the canonical pipeline exactly once and probes the same
  rebuilt no-prior request through the same cached client. Mock evidence shows
  one origin provider call and one zero-call/zero-current-cost cache replay.
- Runtime persistence is allowlisted: usage metadata, image identifiers and
  hashes, model/request identity, paired prediction/evidence, and manifest.
  Parsed/raw provider payloads, prompt text, image bytes, credentials, headers,
  argv, and evaluation targets are not retained.
- The one authorized real synthetic OpenRouter attempt requested
  `openai/gpt-5.6-sol` and failed safely with HTTP 404 after exactly one
  provider call. No returned model, response ID, token/cost, cache-hit, paired
  artifact, or real-success claim exists for that attempt.
- No CholecTrack20 image or paper-performance experiment ran. Instance
  detection, predicted tracking, Gate learning, Specialist verification,
  deterministic repair coordination, and EventMemory remain outside this
  implementation slice.

优先：

- dataclass / typed config；
- Protocol/ABC；
- small testable functions；
- explicit schemas；
- file-based auditable artifacts；
- deterministic cached API tests；
- config-driven ablation。

避免 giant `solve()`。

`systems/v3_1_api_streaming.py` 只负责 orchestration，不实现所有业务逻辑。

---

# 23. Codex 开始时的第一条回复要求

在修改任何算法文件前，先回复：

```text
1. 我已阅读哪些文档
2. 当前 repository 简要目录
3. 当前可复用模块
4. 与 V3.1-API 目标的主要差距
5. 计划执行的 P0-A / P0-B / P0-C
6. 需要标记 BLOCKED 的信息
7. 接下来实际运行的第一组命令
```

然后开始 P0。

若没有遇到 FAIL/BLOCKED，不需要等待用户逐阶段确认，可以继续下一阶段；但必须逐阶段真实测试并写报告。

若遇到影响正确性的 FAIL/BLOCKED，立即停止并报告。
