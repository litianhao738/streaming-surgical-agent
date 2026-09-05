# 前四个问题的闭环状态

日期：2026-09-05
范围：三帧因果窗口、Joint H0、Evidence-first V8、证据证书 Repair admission、task-wise reliability v2

## 总结

前四个问题已经形成一条逻辑一致的工程链：完整时间轴推理使用任务 mask 离线评分；Verifier 只在候选和 scope 内提出字段级结果；Repair 只有在结构后检和匹配证书通过后才能替换 hard-valid H0；最终可靠性只授予实际被证书覆盖的任务，不再从局部 scope 扩张到整帧。

这表示运行时合同和已知失败模式已经修复，不表示 VLM、Tracker 或 Repair 已在统计意义上达到论文可用性能。正式 Learned Gate 仍需等待 Training-only capability pilot。

## 问题一：GT 不完整时如何连续推理和评分

状态：`ENGINEERING RESOLVED`。

- Pipeline 在完整 Test 时间轴上连续进行 gold-free 推理，不因某一任务缺失 GT 而删除时间点；
- GT 只在推理完成后的离线评价阶段读取；
- 每个任务独立使用 `FrameTaskMask`，缺失字段不作为负类，也不进入该任务分母；
- Test GT 评价必须显式授权，截断 engineering run 不具备 paper eligibility。

当前严格 all-instance completeness 规则下，本地审计覆盖是：Instrument/Phase 各 15,282 帧，Verb/Target 各 6,631 帧，IVT 5,953 帧。早期讨论中的 7,153/7,061 是较宽松口径，不应与当前实现混用。

## 问题二：Verifier 产生字段合法但联合非法或错误排序

状态：`KNOWN STRUCTURAL FAILURE RESOLVED; CAPABILITY NOT YET PROVEN`。

- Interaction 不再分别选择 I/V/T/IVT 后任意拼接；V8 分别获得组件判断，再由本地 ontology 做 exact IVT join；
- Instrument 修复只允许一个确定性副作用：删除已被移除器械对应的 IVT，不允许借机添加其他 IVT；
- Workflow 使用仅由 Training 构建并冻结的 phase transition graph；
- 所有 Verifier 输出仍必须通过候选池、scope、touched-task 和 IVT closure 后检。

仍未被算法合同消除的是感知源自身错误，例如 Tracker 把器械识别错、组件专家共同看错图像。此类问题需要更大 Training-only capability pilot；若错误主要来自 Tracker，才考虑重新训练或校准 Tracker，而不是继续放宽 Repair。

## 问题三：hard-valid H0 阻止正确 Repair

状态：`MECHANISM RESOLVED; THREE REAL POSITIVE INSTRUMENT SMOKES`。

hard-valid H0 不再被绝对保护。H1 在候选、scope 和闭包合法之外，还必须取得与 scope 匹配的本地证据证书：

- Instrument：最近两个因果 Tracker 帧精确支持同一纯删除结果；
- Interaction：I/V/T 组件结论均为 Verified，并产生 exact ontology IVT；
- Workflow：冻结训练图明确禁止 previous→H0，同时允许 previous→H1。

无证书或证书类型错误时仍返回 `HARD_VALID_H0_PROTECTED`。真实 Validation 冒烟中，`VID110:4376`、`4401`、`4426` 都把 Instrument 从包含错误器械的 H0 修复为 GT `{0}`。新增的 4401/4426 冒烟均为缓存命中，新增 provider cost 为 0。

Interaction 和 Workflow 证书已通过单元测试，但还缺分层 Training-only 净收益、harm rate 和覆盖率统计，不能据此训练正式 Gate。

## 问题四：局部验证被错误升级为整帧 Verified

状态：`ENGINEERING RESOLVED WITH TASK-WISE RELIABILITY V2`。

最终输出为每个任务记录以下状态：

- `Accepted`：当前输出被保留，但没有本次任务级证书；
- `Checked`：Specialist 检查过该任务，但没有本地证据证书；
- `Verified`：该任务被匹配的本地证据证书覆盖；
- `Derived`：由确定性闭包级联修改，但没有获得该任务的语义验证；
- `Pending` / `Rejected`：任务尚未裁决或整体被拒绝。

只有五个任务全部为 `Verified` 时，整帧状态才能是 `Verified`。局部 Repair 的整帧状态为 `Accepted`，但仍保留 `verified_tasks`、`checked_tasks` 和 `derived_tasks` 供审计。

Memory 将带证书的任务保存在可靠集合中，同时携带完整 task mask；时序 Gate 只使用上一条 Memory 中对应的 `Verified` 任务计算跳变；Workflow 历史携带 Phase 的任务状态；事件只有组成语义的 Phase 和 IVT 在组内全部 Verified 才能报告 `DEFINITE`。旧 v1 整帧 Verified 状态在恢复时保守迁移为全任务 `Checked`，不会继续作为五头可靠事实。

连续 `VID110:4376→4401→4426` 冒烟结果：三帧均为整帧 `Accepted`；Instrument=`Verified`，IVT=`Derived`，Verb/Target/Phase=`Accepted`；报告均为 `OBSERVED`。输出、状态和报告 schema 分别升级为 `final_pipeline_output_v2`、`streaming_finalization_v2` 和 `reliability_aware_event_report_v2`。

## 训练与正式实验边界

上述四项逻辑修复本身不需要训练。后续需要训练或重新采集的部分是：

1. 先运行分层 Training-only capability pilot，确认 Interaction/Workflow/Instrument 的 rescue、harm、coverage、provider failure 和成本；
2. 只有 Repair 在预注册 harm 上限下表现出稳定净正收益，才重新采集与当前三帧、V8、task-wise v2 完全匹配的 Gate counterfactual 数据；
3. 随后训练 Learned Gate。旧 Gate 数据或旧 Gate 模型不能直接用于当前策略；
4. Tracker OOF artifact 不因本次逻辑修改而必须重训；只有独立误差分析证明 Tracker 是主要瓶颈时才重训或校准。

## 验证记录

- 全量 Pytest：`1000 passed, 14 skipped`；
- Ruff：通过；
- `git diff --check`：通过；
- 第三问题新增冒烟：
  - `artifacts/preflight/smoke_repair_evidence_third_4401_20260905/`
  - `artifacts/preflight/smoke_repair_evidence_third_4426_20260905/`
- 第四问题修复后连续冒烟：
  - `artifacts/preflight/smoke_taskwise_reliability_contiguous3_v2_20260905/`
