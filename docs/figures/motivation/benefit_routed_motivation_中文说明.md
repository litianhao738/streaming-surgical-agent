# 动机图与方法架构说明

配套文件：`benefit_routed_motivation.pptx`。PPT 中的文字、节点、箭头和颜色均为原生可编辑对象。

这张图只用于解释研究动机和核心方法差异，不承担完整架构展示。左侧是对一类已有方法的抽象，右侧是本项目方法的总结性表达；完整架构在第二部分按照一次在线处理从输入到输出的顺序用文字说明。只有路由分支和跨时间状态写入这两个不易仅靠文字理解的环节保留局部简图。

## 一、左侧 Previous Methods 表示什么

`Previous Methods` 不是特指某一篇论文，而是概括常见的单帧、短 clip、多任务预测以及 sample-wise agentic pipeline。它也可以覆盖“任务专用 Experts + 一致性检查或 reflection”这一类组织方式。

其典型结构包括：

- 输入当前帧或一个较短的手术 clip；
- 使用多个 task heads 或 task-specialized Experts 预测 Instrument、Verb、Target、IVT 和 Phase；
- 可选地检查当前样本内部的任务一致性，或对冲突任务进行 reflection；
- 输出当前样本的最终预测。

这类方法适合局部视觉识别和当前 clip 内的多任务推理，结构也较直观。左侧提出的不是“已有方法完全没有时间信息”，而是以下更具体的研究边界：

1. **状态通常以当前样本为中心。** 即使输入包含数帧 temporal context，也不等于维护了整个手术视频中的持续状态。
2. **一致性修正主要发生在当前样本内部。** 过去已经确认的阶段、器械轨迹和历史事件不一定以显式、可检索的状态继续参与后续决策。
3. **验证对象和调用预算缺少联合决策。** 固定调用所有 Experts 或进行一次通用 reflection，不能直接回答“当前是否值得增加调用，以及应该复核哪个任务”。
4. **修正范围可能不够显式。** 如果二次调用可以自由重写全部输出，一项错误修正可能影响原本正确的其他任务。

因此，左侧的核心概括是 **sample-wise clip understanding**：重点是当前帧或短片段的识别与一致性，而不是显式的因果长视频状态管理。

## 二、Ours 的完整架构（文字说明）

本项目不是“每帧固定调用多个 Agent”，而是一个 **Stateful Long-Video Surgical Agent**：以 Joint Perception Agent 作为基础感知入口，以 Benefit Gate 决定是否追加验证及验证范围，再由一个受限 Specialist 完成针对性复核。

核心研究问题是：

> 在严格因果的手术视频流中，系统应在什么时候、调用哪个 Specialist，才能在额外调用预算下获得有依据的修正，同时避免不可靠历史和越权修改造成错误传播？

### 2.1 从输入到输出的完整衔接

完整系统不再额外绘制一张包含全部模块的大图，而是按每个时间点的一次在线处理顺序理解：

1. **读取严格因果输入。** `DatasetAdapter` 和 `MediaBackend` 读取 CholecTrack20 当前样本，`CausalWindowBuilder` 只组织当前帧与过去帧，不允许使用未来帧。真实标签被单独封装为 `EvaluationTarget`，不进入在线推理链路。
2. **构造当前上下文。** `StructuredContextBuilder` 合并因果视觉窗口、预测得到的器械轨迹、上一时刻的 `WorkflowState`，以及从可靠历史中检索出的 `EventMemory`。
3. **产生统一的初始预测。** `Joint Perception Agent` 一次输出 Instrument、Verb、Target、IVT 和 Phase 的结构化结果，同时保留各任务的 Top-K 候选与不确定性信号。该结果记为初始假设，而不是未经检查就写入历史的最终事实。
4. **准备可验证候选和证据。** `CandidateGenerator` 从初始预测、合法 ontology 组合及仅由训练集得到的 soft priors 构造候选池；`SignalExtractor` 汇总任务闭包、阶段兼容性、时间稳定性、轨迹连续性和历史分歧等证据。
5. **判断额外调用是否值得。** `Task-wise Benefit Gate` 比较直接保留初始预测与调用不同 Specialist 的预期收益。若额外验证不值得，则输出 `ACCEPT`；否则只选择一个 `VERIFY(scope)`。
6. **执行受限专家复核。** 被路由到的 Specialist 只能查看与其职责相关的证据，只能在给定候选池中选择 `KEEP` 或 `REPAIR`，并且只能修改被授权的任务字段。
7. **执行确定性写回检查。** `Deterministic Coordinator` 检查 Specialist 的角色、候选、修改字段、schema、ontology 和粒度是否合法。任何越权或非法结果都会回退为 `KEEP`，不能写入持续状态。
8. **最终化、提交状态并离线评估。** 合法结果形成 `Validated Final State`，经过可靠性估计后封装为 `Finalized Event`，再写入下一时刻可见的 `WorkflowState` 与 `EventMemory`。与此同时，`PredictionRecord` 才与隔离的 `EvaluationTarget` 配对，计算识别质量、路由质量、修复收益、错误伤害、流式性能和 API 成本。

因此，系统的前后关系是：前一时刻已经最终化的预测为当前时刻提供因果上下文；当前时刻先感知、再判断是否值得验证、再受限修复；只有通过确定性检查的结果才能成为后一时刻的历史。真实标签始终位于这条在线闭环之外。

其中，ontology、仅由训练集统计得到的 priors、训练完成后冻结的 Gate、prompt/schema 以及 API cache provenance 都是运行前固定的研究 artifact。测试阶段只允许读取，不能根据测试视频重新统计、更新或调参。

路由部分最容易产生误解，因此保留一张局部简图。`ACCEPT` 表示不增加调用，直接保留初始预测；`VERIFY(scope)` 才会调用一个与该范围匹配的 Specialist：

```text
InitialPrediction + candidates + causal evidence
                         ↓
                Task-wise Benefit Gate
          ┌──────────────┴─────────────────┐
          ↓                                ↓
       ACCEPT                       VERIFY(scope)
          │                                ↓
          │                    one routed Specialist
          │                                ↓
          │                    Deterministic Coordinator
          └──────────────┬─────────────────┘
                         ↓
               Validated Final State
```

状态更新与标签评估也容易混淆，因此保留第二张局部简图。历史状态只能接收已经最终化的预测事件，真实标签不能进入下一帧的上下文：

```text
Validated Final State → Reliability Profile → Finalized Event
                                               ├→ WorkflowState for t+1
                                               └→ EventMemory for t+1

PredictionRecord + isolated EvaluationTarget → EvaluationEngine
```

### 2.2 Causal Context 与 Joint Perception Agent

Joint Perception Agent 使用当前及过去帧构成的严格因果窗口，不允许读取未来帧。除了视觉输入，它还可以接收三个结构化上下文：

- **Predicted Track Context**：来自预测 tracker 的器械轨迹、连续性、运动和可见性信息；
- **Workflow State**：由过去 Finalized Events 形成的紧凑手术流程状态；
- **Reliable Event Memory**：从已最终化历史事件中检索出的、带可靠性和 provenance 的证据。

Joint Perception Agent 统一产生 Instrument、Verb、Target、IVT 和 Phase 的初始结构化预测，同时保留 task-level Top-K hypotheses 和可审计的 uncertainty proxies。它负责“当前发生了什么”，但不负责自行决定是否需要额外调用。

论文主比较中，Joint Perception 与后续 Specialists 使用同一个 exact API model identifier。这样可以尽量把差异归因于架构、路由与验证机制，而不是换用了更强模型。

### 2.3 Candidate Generator 与软证据信号

图中没有单独画出 Candidate Generator 和 Signal Extractor，但它们是实际架构中的必要模块。

Candidate Generator 根据初始预测、ontology-valid Top-K 组合和 train-only soft priors，为不同任务 scope 构建较小的候选池。候选必须携带 `scope_id`、task、ontology ID 和来源；测试时不能使用 GT 构造候选。

Signal Extractor 将以下信息转换为 Gate 特征：

- IVT 与 I/V/T 的内部一致性；
- Phase-IVT compatibility 和阶段转移异常；
- 预测变化与短期 temporal instability；
- track continuity 或 track conflict；
- 当前预测与可靠历史事件之间的 disagreement；
- 可用但不被当作真实概率的 uncertainty proxies。

这些知识是 soft evidence，不直接硬编码成“必须验证”或“必须修正”。

### 2.4 Task-wise Benefit Gate

Benefit Gate 不是普通错误检测器。它不只预测“当前结果是否错误”，而是分别估计调用三个 Specialist 后可能获得的额外修正收益：

`benefit_hat(spatial_track)`、`benefit_hat(interaction)`、`benefit_hat(workflow)`。

Gate 的输出只有两类：

- `ACCEPT`：保留 Joint Perception 的结果，不增加 API 验证调用；
- `VERIFY(scope)`：选择预期收益最高且超过冻结阈值的一个 scope。

主配置中每个 streaming state 最多调用一个 Specialist。这样，多 Agent 的作用来自动态任务分工，而不是把多个角色在每一帧全部执行。

### 2.5 Sparse Specialist Verification

三个 Specialist 使用相同 backbone，但具有不同 prompt、证据范围、候选池和允许修改字段：

|Specialist|主要检查|允许修改|明确禁止|
|---|---|---|---|
|`spatial_track`|器械语义、已有 proposal/track 的空间与语义连续性|已有实例或轨迹 scope 内的器械相关语义；关联调整必须服从 tracker 合同|新增 bbox、实例或 track ID|
|`interaction`|Verb、Target、IVT closure 及实例内组合|当前 `prediction_id` 范围内的 V/T/IVT candidate|修改 Phase、bbox、track identity 或跨实例拼接|
|`workflow`|Phase、阶段转移和 Phase-IVT soft compatibility|当前帧的 Phase candidate|修改实例级 I/V/T/IVT 或读取未来状态|

Specialist 不能自由生成答案，只能执行：

- `KEEP(H0)`；
- `REPAIR(H0 → Hk)`，且 `Hk` 必须属于当前 scope 的 CandidateSet。

这种设计把“再次调用模型”变成有任务边界的 falsification-oriented verification，而不是简单 retry。

### 2.6 Deterministic Coordinator

Coordinator 不是额外 LLM Agent，而是确定性安全层。它检查：

- Gate 选择的 scope 是否与 Specialist role 一致；
- `selected_candidate_id` 是否属于对应 CandidateSet；
- `touched_tasks` 是否在该 Specialist 的允许范围内；
- 输出 schema、ontology 和 granularity 是否合法。

任何越权修改、候选池外标签或格式错误都不能静默写入状态，而是记录 reason code 并回退为 `KEEP`。因此，Specialist 提供推理能力，Coordinator 提供不可绕过的写回约束。

### 2.7 Reliability、Workflow State 与 Event Memory

经过 Coordinator 的结果先形成 Validated Final State，再计算 task-wise Reliability Profile，随后封装为 Finalized Event。

Reliability 不是未经校准的“正确概率”，而是由 temporal agreement、track agreement、workflow compatibility、candidate stability、verification evidence 和 provenance cap 等信息形成的可靠性分数。

持久状态分成两个独立模块：

- **WorkflowStateStore**：维护紧凑的因果手术流程状态，服务下一时刻的 context 和 temporal/workflow features；
- **EventMemory**：保存有界的 episodic Finalized Events，服务候选支持、历史证据检索和 memory disagreement。

两者虽然消费同一 Finalized Event 流，但接口、状态语义和消融开关相互独立。当前事件必须在最终化之后才能写入；它不能检索到自己，所有状态也必须在 video boundary 重置。