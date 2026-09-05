# VideoHV-Agent 后续消融建议（暂缓实施）

日期：2026-09-05

状态：`PARKED / NOT IMPLEMENTED`
适用版本：当前三帧因果窗口、Evidence-first V8 Verifier、证据证书 Repair admission

## 1. 决策

保留 VideoHV-Agent 的“候选假设 → 区分性线索 → 局部验证”思想，作为现有证据证书无法裁决时的候选后备实验；当前不接入正式 Pipeline，也不替换现有确定性 Repair admission。

原因：VideoHV-Agent 面向长视频单选 VideoQA。当前项目输出 Instrument、Verb、Target、IVT 和 Phase 五个相互约束的结构化头，候选是多标签集合而不是互斥的自然语言选项。直接移植其 Answer Agent 或多 Agent 投票不能证明 Repair 后语义优于 H0，反而会重新引入 VLM 自我确认、位置偏置和额外 API 成本。

论文与代码：

- Paper: <https://arxiv.org/abs/2603.04977>
- Official repository: <https://github.com/Haorane/VideoHV-Agent>

## 2. 当前机制不得被改变的边界

当前 hard-valid H0 只在取得 scope 匹配的本地证据证书时允许替换：

- `instrument_presence`：最近两个因果 Tracker 观测精确支持同一纯删除结果；
- `interaction`：Instrument/Verb/Target 独立组件结论均为 Verified，并能通过本地 ontology 得到精确 IVT 闭包；
- `workflow`：历史 Phase 到 H0 的转移被冻结训练图禁止，而到 H1 的转移被允许；
- 证书缺失或类型不匹配：保留 H0，返回 `HARD_VALID_H0_PROTECTED`。

VideoHV-Agent 后备实验不得放宽上述证书，不得把 VLM 的理由文本当作证书，不得因结构合法、候选分高或多数投票就接纳 H1。

## 3. 候选后备实验

仅当现有证书无法裁决 H0/H1 时，进行字段级、匿名、对称的比较：

```text
H0/H1 在某个 scope 上存在差异，且现有证书无法裁决
        ↓
只提取发生变化的字段，随机匿名为 A/B
        ↓
Judge 生成一个能区分 A/B 的可观察命题
        ↓
Verifier 只在相同三帧因果窗口或预先规定的局部 ROI 中取证
        ↓
输出 A / B / ABSTAIN + 结构化 evidence locator
        ↓
本地检查候选池、scope、IVT 闭包和证据定位
        ↓
满足预注册的接纳阈值才允许选择 Repair，否则 KEEP H0
```

必要控制：

1. 隐藏哪个候选来自 H0、Verifier 或 Tracker；
2. A/B 顺序随机化，并测试交换顺序后的结论一致性；
3. 不向 Verifier提供上游 confidence、GT、损失或候选来源；
4. 允许 `ABSTAIN`，不得强制二选一；
5. 只验证变化字段，不能据此给未验证字段升级可靠性；
6. 最终接纳仍由本地确定性代码完成，不能由 Answer Agent 自行改写五头输出；
7. API 调用数、时延和成本作为正式指标记录。

## 4. 何时才值得启动

必须先在 Training-only 的分层样本上证明以下现象同时存在：

- H0 错误且正确答案可由候选池表示；
- 当前 scope 证书覆盖不足，导致大量 `HARD_VALID_H0_PROTECTED`；
- Verifier 提议在离线 GT 下经常优于 H0，而不是偶发个例；
- 主要剩余错误确实来自缺少区分性视觉证据，而不是 Tracker 误检、候选缺失或字段级可靠性传播错误。

第四个问题已在 task-wise reliability v2 中修复：局部证书只升级对应任务，闭包级联标记为 `Derived`，整帧不再自动升级为 `Verified`。即便如此，在现有证书完成更大规模 Training-only 净收益评估前仍不启动本实验，避免把尚未证明必要的额外 Agent 和 API 成本加入正式版本。

## 5. 评估与晋级标准

采用配对 Training-only 消融，冻结相同帧、候选池、API 模型与预算，比较：

- A：当前 V8 + 证据证书；
- B：当前 V8 + 证据证书 + 匿名区分性验证后备。

至少报告：按 scope 的 rescue rate、harm rate、net masked utility、abstention rate、candidate representability、API 成功率、调用数、成本和延迟。阈值只在开发子集校准，在隔离的 Training-only 确认子集复核。若不能在预注册的有害修复率上限内获得稳定正增益，则保持 `PARKED`。

## 6. 当前结论

VideoHV-Agent 对“证据不足时如何比较两个结构合法假设”有研究价值，但不是当前第三问题的直接补丁。字段级可靠性与 Memory 传播现已完成 v2 工程修复；当前优先级是用更大规模 Training-only 数据评估现有证据证书的覆盖率、净收益和有害修复率。
