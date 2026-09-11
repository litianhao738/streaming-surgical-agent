# 交接：先验门控 + 联合 Phase 主线的整理与冻结

日期：2026-09-11。写给接手的 Codex 会话。本文件只定义任务、约束和验收标准，不执行任何实验，也不改动任何已有文件。默认版本仍是 `parallel-phase-repair-v1.3.0-glm-low`，是否替换默认由用户决定。

## 一、目标版本

在 `prior_gated_joint_phase_v1`（报告 `docs/PRIOR_GATED_JOINT_PHASE_2026-09-11.md`）基础上**只删除盲投 Phase 面板**，其余不变。每目标 13 次调用：

| 步骤 | 调用 | 说明 |
|---|---:|---|
| H0 | 1 | 联合五头终标签 |
| 图谱补候选 | 1 | 原提示与原先验提示 |
| 精简四头五席审核 | 5 | v1.3.0 精简 prompt，请求逐字节不变 |
| 先验门控 | 0 | `prior_gated_repair`，veto 0.01 / add 0.70，H0 Phase 桶 |
| Phase 推荐 | 1 | 不带先验关系提示（沿用 `prior_gated_joint` 的防泄露设定） |
| 联合五头审核（只取 Phase 决策） | 5 | 与四头面板并行 |
| `phase_apply` | 0 | 唯一最高、均分 ≥ 4、严格高于当前 Phase 才替换 |

删除理由：盲投 Phase 的结果只被 v1.3.0 control 臂使用，不进入主候选；它在三个批次上累计改对 6、改坏 9。联合 Phase 在三个批次（Training 40、VID110 32、VID110 新 16）上累计改对 2、改坏 0。

## 二、不可改动的约束

1. **基座模型上下一致，不换模型。** H0、图谱补候选、Phase 推荐均为 `google/gemini-3.8-flash`（Google AI Studio 路由）。五席审核保持 v1.3.0 名单：`z-ai/glm-5.3-flash`、`qwen3.5-35b-a3b`、`openai/gpt-5.6-luna`、`google/gemini-3.5-flash-lite`、`deepseek/deepseek-v4-flash-vision-exp`，路由见 `scripts/run_glm_parallel_repair.py`。**不做强模型专判 Phase 的实验。**
2. 门控阈值 0.01 / 0.70、Phase 阈值 4 冻结，不在任何确认数据上调整。
3. 审核后的 Phase **不回流**给门控；`assert_ungated_request` 的防泄露检查保留。
4. 先验表排除当前视频（LOVO），读取时断言。
5. **不采集 Gate 训练数据。** 最后一步只冻结版本。

## 三、任务

### T1 整理正式入口（零调用）

- 基于 `src/surgical_agent/research/verification/prior_gated_joint.py` 提供不含盲投面板的主线形态和单一入口脚本。
- **验收：** 用已保存的原始回答离线重放 `artifacts/preflight/prior_gated_joint_vid110_confirm_20260911_v1_resume1`，主线预测与归档 `gated_control_jointphase` 在 16 个目标上逐字节一致；四头与 `gated_control` 逐字节一致。
- 更新根目录 `BEST_PIPELINE_VERSION.json`。当前内容的 `recommended_form` 写的是"H0 Phase"，需改为本版本（联合 Phase），证据段补充联合 Phase 三批的改对/改坏统计。

### T2a 账本写入改为单线程排队（零调用）

- 两次确认都因三个面板并行写 `budget.json` 触发 Windows `PermissionError`，靠恢复脚本补救。改为单一写入者（队列或锁内合并写），而不是写入重试。
- **验收：** 单元测试模拟多线程并发写入，无异常、无记录丢失；现有测试不回退。

### T2b 门控"加入"加一道保险（零调用离线验证，不通过则不采用）

- 现状：否决在 VID110 两批新帧上 12/12 正确，加入只有 3/6 正确（Training 为 25/29）。
- 候选规则：先验加入额外要求该 IVT 的精简审核均分 ≥ 3（即审核员未明确否定）；否决不变。
- **预注册标准（离线重放 Training 136 目标 + VID110 已归档 64 帧）：** 错误加入减少、正确加入不减少、平均 F1 不降、总错漏不增。任一不满足则保持原规则，并如实记录。

### T4 独立确认（需付费，方案由接手方定）

- Validation 已用尽（VID110 三批共 64 帧），VID30 经审计为 VID17 副本，不可用。
- 注意：门控阈值是在 VID103/23/31/96 全部四个视频上选的，直接在其中一个视频上"留出确认"并不独立。可选做法：嵌套留一视频（在其余三个视频上重选阈值，再确认留出视频），或等待 Testing 的正式协议。
- 要求：确认数据不得参与任何阈值、规则或模型选择；预注册标准写入 `plan.json`；接口失败如实保留、不重试、不剔除目标。

### T5 冻结版本（零调用）

- 在版本清单中固定：版本号、全部运行源码 SHA-256、阈值、模型与路由、prompt 版本、证据归档路径与哈希、已知局限。
- 不自动替换 `DEFAULT_PIPELINE_VERSION.json`。

## 四、已知局限（写入报告时须保留）

- 门控收益主要来自把 Phase 3 的 `hook/dissect/cystic_plate` 纠正为数据集惯用的 `hook/dissect/gallbladder`，是标注惯例对齐，不是新的视觉证据。
- 联合 Phase 触发率低：五席在 Calot 三角与胆囊床之间与 H0 犯同类错误（本批错误 Phase 均分 3.8～4.6，真值 1.6～3.0），方向在不同批次相反。
- 离线上界：即使给门控真值 Phase 并允许加入池外关系，新 16 帧四头错漏也只从 74 降到 73，Phase 与门控基本脱钩。
- 只在一个 Validation 视频上确认，覆盖 Phase 1/2/3；Phase 4/6 在该视频先验中无合格类别。

## 五、代码与报告路径

**代码**
- `src/surgical_agent/research/verification/prior_gated_repair.py` — 先验门控
- `src/surgical_agent/research/verification/prior_gated_joint.py` — 门控 + 联合 Phase 统一流程
- `src/surgical_agent/research/verification/phase_extension.py` — `phase_apply`
- `scripts/run_prior_gated_joint_confirmation.py` — 确认入口
- `scripts/resume_prior_gated_joint_confirmation.py` — 中断恢复入口
- `scripts/run_prior_gated_confirmation.py` — 门控单独确认入口
- `scripts/run_joint_phase_feedback_trial.py` — `joint_wire` / `phase_proposal` / `select_joint`
- `scripts/run_glm_parallel_repair.py` — 五席名单与路由
- `tools/audit/prior_gated_replay.py`、`tools/audit/prior_gated_joint_replay.py` — 零调用重放
- `tests/unit/test_prior_gated_repair.py`、`tests/unit/test_prior_gated_joint.py` — 单元测试

**报告**
- `docs/PRIOR_GATED_JOINT_PHASE_2026-09-11.md` — 本版本
- `docs/PRIOR_GATED_IVT_ADMISSION_2026-09-11.md` — 先验门控
- `docs/VALIDATION_VID110_CONFIRMATION_2026-09-11.md` — v2.0.1 的 VID110 确认
- `docs/REPAIR_CEILING_DIAGNOSIS_2026-09-10.md`、`docs/ADMISSION_RULE_AND_EXTRACTION_PROBE_2026-09-10.md` — 瓶颈诊断与已否定方向
- `BEST_PIPELINE_VERSION.json`（待 T1 更新）、`DEFAULT_PIPELINE_VERSION.json`（不改）

**证据归档（本地磁盘，被 .gitignore 排除）**
- `artifacts/preflight/prior_gated_joint_vid110_confirm_20260911_v1_resume1/`
- `artifacts/preflight/prior_gated_vid110_confirm_20260911_v1/`
- `artifacts/research/prior_gated_replay_20260911.json`
- `artifacts/research/prior_gated_joint_replay_20260911.json`
