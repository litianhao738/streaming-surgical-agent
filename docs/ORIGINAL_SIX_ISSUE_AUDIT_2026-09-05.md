# 原始六项问题复核与第五、第六项根因

日期：2026-09-05。范围：当前三帧因果窗口、V8 Verifier、证据接纳与 task-wise v2。

## 编号与结论

本文件严格使用用户本轮引用的六项：①H0/候选召回；②Verifier 视觉重排；③Repair 接纳目标；④hard-valid H0 保护；⑤一次只检查一个 scope；⑥Phase–IVT 严格检查未启用。

`FOUR_PROBLEM_RESOLUTION_STATUS_2026-09-05.md` 的编号包含 GT mask 和任务级可靠性，属于另一份问题清单，不能用它证明这里的前四项全部解决。

当前结论：①本次小样本的候选缺失已补齐；②视觉识别尚未解决；③有证据接纳规则但没有获得普遍语义增益保证；④绝对保护已改为有条件放行，有实测正例但覆盖仍有限；⑤⑥均未解决，且存在直接的路由合同耦合。

## 连续六帧测试

- Run：`artifacts/preflight/smoke_original_six_issue_audit_v8_20260905/`
- Validation：VID110，4301、4326、4351、4376、4401、4426；逐帧使用当前与前两个可用因果帧，连续更新状态。
- 配置：`joint_openrouter_final_fixed3.yaml`、`targeted_openrouter_gpt56sol_evidence_first_fixed3.yaml`、`b_tracker_smoke_force_verify.yaml`。
- 这是低阈值 RULE Gate 冒烟，不代表 Learned Gate 的正式表现。
- 10 次逻辑 API 调用，7 次缓存命中，3 次实际 provider 调用，新增账单成本 USD 0.033243。新调用为 4301 的 H0、Verb 与 Target 专家；其他 H0 与 4351 组件结果复用缓存。
- 六帧完成，执行错误 0；102 项针对性单元测试通过。
- GT 仅用于完成推理后的 Validation 离线核对。六帧均为 I={0}、V={9}、T={14}、IVT={94}、Phase=0，所有任务 mask 均有效；此段均为同类 null interaction，不能外推到复杂活动交互。

| 帧 | 实际 scope | 最终 Instrument | 最终 IVT | 最终 Phase | 本轮现象 |
|---|---|---|---|---|---|
| 4301 | interaction | {0,3,6} | {7,13} | 4 | 生成 IVT94 的提议，但无证书，保留 H0 |
| 4326 | instrument_presence | {0,3} | {7,72} | 0 | Tracker 同样认为有 scissors，KEEP |
| 4351 | interaction | {0} | {94} | 4 | null fallback 修复结构/标签，错误 Phase 未检查 |
| 4376 | instrument_presence | {0} | {10} | 0 | 证书接纳器械删除，其他交互标签仍错 |
| 4401 | instrument_presence | {0} | {21} | 0 | 同上 |
| 4426 | instrument_presence | {0} | {4} | 2 | 器械修正，交互和阶段仍错 |

| 离线任务集合 exact match | H0 | 最终 |
|---|---|---|
| Instrument | 0/6 | 4/6 |
| Verb | 0/6 | 1/6 |
| Target | 0/6 | 1/6 |
| IVT | 0/6 | 1/6 |
| Phase | 3/6 | 3/6 |
| 五头同时正确 | 0/6 | 0/6 |

这些是冒烟样本的集合 exact match 计数，不是完整数据集 mAP。

## 前四项目前的实际状态

### 1. H0/候选召回

本次六帧 H0 的原始 IVT top-8 只有 3/6 包含 GT94；当前候选 v4 在 6/6 中包含94，并通过组件闭包纳入 I0/V9/T14，Phase 全七类，因此五头 GT 均可由整个候选池表示。

原始“2/5”来自旧冒烟 4326/4351/4376/4401/4426/4451，其中4376是执行失败，五个语义结果中只有4326/4351 top-8 含94。本次样本集合不同，不能把 2/5→6/6 当作严格配对指标。

实现是保留视觉 top-8，加 Training 的 `(H0.phase, candidate.instrument)` 先验，最多20个候选，再补齐 I/V/T；仍受错误 H0.phase、器械候选缺失、先验覆盖和容量影响。因此仅能说本次候选缺失得到解决，不能宣布全数据集召回问题解决。

代码：`src/surgical_agent/research/verification/hypotheses.py:107`；`src/surgical_agent/systems/final_pipeline_factory.py:299`。

### 2. Verifier 视觉重排

旧 V7 在4351将IVT94排为0.08的记录真实存在，不能继续当作V8的直接机制描述：V8改为组件专家与本地 exact join。

但当前4351的 Verb 专家仍选1（0.78，9仅0.35），Target专家选13（0.92，14不在其返回topk）。最终94来自 exact join 无匹配后的 null fallback。该结果没有组件共识证书，不能算专家已经看对图像。4326仍继承Tracker的scissors误检。

代码：`src/surgical_agent/research/verification/verifier.py:164`、`:190`、`:203`。

### 3. Repair 接纳目标

hard-valid H0 现在要求 scope 匹配证书，确实不再仅凭候选/scope/闭包放行。但是证书是证据代理，不等价于 `loss(H1,GT)<loss(H0,GT)`；Tracker可连续误检，组件专家可共同看错，Workflow还允许Accepted历史阶段支持新证书。

此外证书保护只在 H0 原本 hard-valid 时要求。H0 hard-invalid 时，候选修成 hard-valid 仍可在无证书情况下被接纳；4351正是这一分支。它解决了已知结构问题，但没有证明一般情况下接纳语义最优。

代码：`src/surgical_agent/research/verification/repair.py:387`；`src/surgical_agent/research/verification/verifier.py:361`。

### 4. hard-valid H0 被完全保护

绝对保护已经解除：4376/4401/4426三帧均通过两帧Tracker纯删除证书，把Instrument修成GT0。无证书仍保留H0：4301提议的IVT94事后对GT更好，但无充分证书被拒绝。这是保守接纳的覆盖限制，不能以一个正例为由直接放宽全部提议。

当前需要训练集上的 rescue/harm/coverage 评估，不能声称这四项的语义问题已全部解决。

## 第五项：检查覆盖不足的精确根因

1. `GateAction`只包含一个scope；RULE和LEARNED都取单个最大风险/收益scope。配置中列出三个scope仅表示可选动作，并不表示依次执行。
2. RULE按每个scope的最大重叠风险打分。TRACKER_CONFLICT同时给instrument_presence和interaction打1分，固定顺序让前者胜出。高置信但错误的Phase可能根本没有触发风险。
3. `BoundedVerifyRepairLoop`在第一次KEEP/REPAIR通过硬检查后立即返回。`max_attempts`当前为1，即使改为2也用于失败重试或剩余硬违规，不能自动覆盖剩余语义任务。
4. Pipeline仅执行一次route/loop便finalize。任务级可靠性标记能防止未检查头被错误背书，但不会增加检查次数。

无API的构造诊断复现：器械和阶段风险同为1时选择instrument；max_attempts=2时，器械修复第一次过硬检查便结束，实际attempts=1，Phase保持不变。

代码：`research/gate/formal_policy.py:24`、`:102`；`research/gate/contracts.py:28`；`systems/final_pipeline.py:262`；`research/verification/repair.py:387`、`:422`。（以上均位于 `src/surgical_agent/`。）

若后续修复，需要显式的剩余冲突/任务覆盖调度与总预算，不能把重试次数当成多scope调度。该调度可以确定性实现，训练本身不是前置条件。

## 第六项：Phase–IVT 联合后检未接通，且直接开启会无法路由

### 未接线

`SafetyValidator.strict_phase_allowed_ivt`默认None，factory只接受Python参数透传。正式CLI没有该参数，调用factory也未传入；shared YAML没有对应表。默认运行不会产生STRICT_PHASE_CONSTRAINT。

训练阶段转移图只检查 `previous Phase→current Phase`。候选先验只按 `(H0 Phase, Instrument)` 补IVT。两者均不能替代最终同帧 `Phase–IVT` 关系检查。旧pipeline里有同名相似检查，也不能算作当前FinalStreamingPipeline已启用。

### 关系端点被错误等同于必须同时修改的任务

严格检查把冲突标记为 `(ivt, phase)`，`legal_repair_scopes`却要求一个scope覆盖全部affected任务。interaction没有phase，workflow没有ivt，因此没有可用scope。

构造诊断结果：

```text
默认validator：无违规
注入严格表：STRICT_PHASE_CONSTRAINT(ivt, phase)
候选中已有只改Phase就可完全合法的解
legal_scopes = ()
MandatoryGuard -> PENDING / NO_LEGAL_REPAIR_SCOPE
```

这意味着即使第六项接线成功，第五项的scope合同也会阻止修复。检测关系涉及两个头，不代表每次修复都必须同时修改两个头；路由应能尝试修改任一可修复端，并对完整关系后检。

代码：`research/safety.py:93`、`:125`、`:282`；`systems/final_pipeline_factory.py:185`；`scripts/run_final_dataset_pipeline.py:219`。

### 知识来源的限制

已有`build_phase_ivt_compatibility_from_training_adapter`生成的是训练中观测到的共现集合；未观测不等于不可能。把该表直接作为硬禁止表会错误拒绝合法但稀有的组合。训练统计适合作软风险，明确验证过的禁配规则才能作硬约束。

即使启用兼容性检查，也不能保证纠正4351：错误Phase与null IVT仍可能是允许共现的组合。结构/知识一致性不能替代阶段视觉判断。

## 本次变更边界

本次执行真实/缓存混合连续冒烟、102项现有测试及构造诊断，新增此报告。没有修改运行时实现或训练模型。正式Gate训练仍需等待当前策略的Training-only能力评估与策略冻结。
