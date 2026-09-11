# 先验门控 + 联合 Phase 审核修复：统一五头流程与 VID110 一次确认

日期：2026-09-11。默认版本仍为 `parallel-phase-repair-v1.3.0-glm-low`，本轮不替换。在先验门控候选（[报告](PRIOR_GATED_IVT_ADMISSION_2026-09-11.md)）之上，把 Phase 也纳入"提案 → 五席审核 → Python 接纳"的同一流程；一次单次付费确认（288 次调用，中断后恢复）。**预注册标准未通过：联合 Phase 审核在 16 个新 VID110 目标上一次也没有改动 Phase，候选与先验门控候选逐字节相同。**

## 一句话

四头照旧：图谱补候选 → 五席精简审核 → 冻结均分选择器 → 先验门控（veto 0.01 / add 0.70，H0 Phase 桶）。Phase 新增同构的三步：一次视觉 Phase 推荐（不带先验提示）→ 同五席在同一次请求里对四头候选池和全部七个 Phase 打分 → `phase_apply`（唯一最高、均分 ≥ 4 且高于当前 Phase 才替换）。模块 `src/surgical_agent/research/verification/prior_gated_joint.py`，版本 `prior_gated_joint_phase_v1`。

## 为什么这样接、为什么不指望四头受益

零调用重放（`tools/audit/prior_gated_joint_replay.py`，结果 `artifacts/research/prior_gated_joint_replay_20260911.json`）先回答了"更准的 Phase 能不能让门控更准"：

| 送进门控的 Phase 桶 | 40 个 Training 目标四头总错漏 | 32 个已归档 VID110 目标四头总错漏 |
|---|---:|---:|
| H0 Phase（候选现状） | 201 | 140 |
| 五席盲投多数票 | 202 | 144 |
| 联合面板 Phase | 201 | 141 |
| **GT Phase（上界，推理时不可用）** | **201** | **142** |

即使用真值 Phase，四头错漏也不下降。原因是先验表只有 Phase 3 桶存在 ≥ 0.7 的可加关系，Phase 4/6 桶无合格类别直接回退全局。因此门控继续以 H0 Phase 为桶，被审核过的 Phase **不回流**到四头，这既是数据结论也是防闭环的设计。

同一份重放里 Phase 头本身的证据：联合面板的 Phase 决策相对 H0 在 40 个 Training 目标和 32 个 VID110 目标上各改对 1、改坏 0；盲投多数票在两批分别改对 3 改坏 3、改对 1 改坏 4。把联合 Phase 叠到先验门控候选上（四头不动）：Training 40 目标平均 F1 64.40 → 64.90、错漏 219 → 217；VID110 32 目标 68.26 → 68.88、154 → 152。预期收益就是"每批 1 帧"。

## 防泄露规则（代码强制，不只是文档）

1. **审核员和 Phase 推荐只看 H0 和原始候选池。** 门控在全部回答落盘之后才运行；`assert_ungated_request` 在预检和正式执行时逐条检查每个审核/推荐请求包：`current_prediction(_hypothesis)` 必须等于 H0，且不得含 `gt`、`gated_prediction`、`vetoed`、`prior_added` 等键。否则门控把 `hook/dissect/cystic_plate` 改成 `hook/dissect/gallbladder`（Phase 3 惯例）会把审核员推向 Phase 3，再喂回门控形成自证闭环。
2. **Phase 推荐不带先验关系提示。** `run_joint_phase_feedback_trial.phase_proposal` 原本把 `candidate_relation_hints`（同一张 P(IVT|phase) 表）交给推荐模型；这里删掉，`assert_ungated_request` 也把该键列为禁止。门控与 Phase 决策不共用先验表。这是与已归档联合实验相比的唯一机制差异，预检已固定。
3. **先验表排除当前视频**，`select_prior_gated` 读取时断言 `excluded_video`。
4. **新帧**：16 个 VID110 目标按时间分位选取，三帧全部与此前四个 VID110 实验（含先验门控确认）用过的任何图片相距 > 175 原始帧，只用 mask 不用标签值。
5. **阈值冻结**：门控 0.01 / 0.70 来自 136 个 Training 目标；Phase 阈值 4 来自 2026-09-09 的 `phase_apply`。本轮不调。GT 只在 `score` 且 `completion.json` 存在后读取；`score` 先用保存的原始回答重算全部臂并要求与落盘预测一致。

## 每目标 18 次调用与五个臂

H0、图谱补候选、Phase 推荐各 1 次；精简四头审核、盲投 Phase、联合五头审核各 5 席。三个面板并行。

| 臂 | 四头 | Phase |
|---|---|---|
| h0 | H0 | H0 |
| control（v1.3.0） | 精简审核均分 | 盲投多数票 |
| gated_control（先验门控候选） | 精简审核均分 + 门控 | H0 |
| **gated_control_jointphase（主候选）** | 与 gated_control 逐字节相同（结构断言） | 联合面板 `phase_apply` |
| gated_joint_jointphase（单面板变体） | 联合面板四头均分 + 门控 | 联合面板 `phase_apply` |

预注册标准（写入 `plan.json`）：主候选平均 F1 高于且总错漏低于 h0、control、gated_control，平均 Precision 高于 h0 与 control，Verb/IVT F1 不低于 control，且不改坏任何 H0 已正确的 Phase。四头与 gated_control 相同，所以通过与否完全取决于 Phase 净纠正是否 > 0。

## VID110 新 16 目标结果（付费，单次）

入口 `scripts/run_prior_gated_joint_confirmation.py`；归档 `artifacts/preflight/prior_gated_joint_vid110_confirm_20260911_v1`（中断）与 `..._v1_resume1`（完成并评分）。

| 臂（16 目标） | I | V | T | IVT | Phase | 平均 F1 | 平均 P | 总错漏 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 85.71 | 40.91 | 51.16 | 27.27 | 68.75 | 54.76 | 55.86 | 95 |
| v1.3.0 control | 85.71 | 52.00 | 57.78 | 40.82 | 68.75 | 61.01 | 59.75 | 88 |
| 先验门控候选 gated_control | 85.71 | 52.00 | 57.78 | 46.81 | 68.75 | 62.21 | 61.14 | 84 |
| **主候选 gated_control_jointphase** | 85.71 | 52.00 | 57.78 | 46.81 | 68.75 | 62.21 | 61.14 | 84 |
| 单面板 gated_joint_jointphase | 85.71 | 50.00 | 60.87 | 35.56 | 68.75 | 60.18 | 59.60 | 87 |

**预注册标准未通过。** 主候选与 gated_control 完全相同：16 个目标里联合面板 12 次判"当前 Phase 最佳"、3 次面板不完整（保留 H0）、1 次"无 ≥ 4 的替代"。相对 H0 改对 0、改坏 0。

Phase 逐目标看：H0 错 5 个（GT 全是 3 胆囊床分离，H0 给 1 或 2）。这 5 个目标里五席给错误的 H0 Phase 均分 3.8～4.6，给真值 Phase 只有 1.6～3.0；Phase 推荐模型 4 次与 H0 相同，唯一一次推荐正确（VID110_41576 推荐 3）被面板打 3.0 而未接纳。也就是说，五个视觉模型在"Calot 三角还是肝床"上和 H0 犯同样的错，与 `PHASE_FAILURE_CAUSES_2026-09-10.md` 的结论一致；联合审核没有引入独立证据。盲投多数票这批改对 2、改坏 2（VID110_30726、41576 改对；23001、28701 改坏），仍是零净收益。

3 个不完整面板全部来自 Qwen 在 `phase_2` 行返回词表外的 `finding: "LIKELY_REFUTED"`，按冻结规则整行无效、不补分；这三例的其余四席与 H0 一致，即使有效也不会改 Phase。

单面板变体（省去精简审核，只用联合面板四头均分做门控）IVT 从 46.81 掉到 35.56，与 32 目标重放中"联合四头 + 门控在 VID110 上低于精简四头 + 门控"的方向一致，不能作为省调用的替代。

先验门控这批仍然有效：相对 control 平均 F1 +1.20、错漏 −4；门控否决 6 次全部正确，加入 IVT 9 次 5 对 4 错。这是第二批独立 VID110 帧上门控保持正向，但样本仍小、仍是单视频。

## 工程故障与恢复

1. 首次执行在第 15 个目标（VID110_42776）的联合面板启动前，写 `budget.json` 时触发 Windows 文件锁 `PermissionError`（三个面板并行落盘）。此前 14 个目标（252 次调用）完整落盘，第 15 目标已保存 13 次（缺 5 次联合审核），第 16 目标未开始；`completion.json` 记录 `fatal_error: PermissionError`，共 265 次调用。
2. 恢复入口 `scripts/resume_prior_gated_joint_confirmation.py` 与此前 VID110 确认的恢复同构：零调用预检逐条核对 252 份请求与响应哈希，复现 14 个已落盘目标的全部臂预测（逐字节一致），只允许此前从未尝试的 23 个 (目标, 阶段, 席位) 元组发请求；成功、拒绝、失败历史全部保留，不重试，累计限额不变。预检时发现重放必须使用传输层同一个宽容解析器（允许审核项内完全相同的重复键），否则 6 份回答会被判无效；已改为同一函数。
3. 恢复执行发出 23 次新请求，总计 288 次、无致命错误。整轮 285 次 JSON 正常、1 次去代码围栏后解析、2 次 GPT 盲投 Phase 403（只影响 control 臂）。费用：OpenRouter 原生 $0.4386，另 $0.0312 未核实预留；阿里云估算 ¥0.4396。
4. 正式入口在恢复完成后增加了同样的本地写入重试（`LedgerCalls`，最多 40 次、间隔 50 ms；不是 API 重试）。归档 `frozen_source/` 保存的是执行时的版本；`score` 不校验源码哈希，只校验原始回答重放。
5. 时间以首次执行的 14 个完整目标计：每目标墙钟中位数 23.6 秒（H0 5.7 + 补候选 4.0 + 三面板并行 14.2），最大 47.6 秒；恢复目录 `metrics.json` 里的 `mean_timing_seconds` 混入了缓存重放的零耗时，不作为时间依据。

## 这个结果说明什么，不说明什么

- 把 Phase 接进"提案 → 审核 → 接纳"的统一流程在工程上成立，防泄露检查全部通过，四头结果不受影响；但 Phase 头没有收益。五席在这批目标上与 H0 犯同一类定位错误，接纳规则只是没有放行错误，也没有机会放行正确。
- 联合审核每目标多 1 次推荐 + 5 次联合审核（OpenRouter 约 $0.013/目标），换来 0 帧纠正。若要保留"统一流程"的形态，最省的做法是保留联合面板但把盲投 Phase 5 次调用去掉（control 臂之外没有消费者）；这不改变任何预测。
- Phase 的瓶颈仍在视觉判断与流程语义，不在接纳机制；先验反推 Phase 已有负结果（`phase_prior_inversion_20260910.json`），本轮也刻意不用。下一步若继续做 Phase，需要改变证据来源（更长的时序上下文或独立的位置证据），而不是再换聚合规则。
- 仍是单一 Validation 视频、16 帧、只覆盖 Phase 1/2/3；先验门控在两批新帧上都正向，但联合 Phase 的"每批改对 1"这次没有复现。

## 复现

```powershell
.venv-p2\Scripts\python.exe -X utf8 tools\audit\prior_gated_joint_replay.py
.venv-p2\Scripts\python.exe -X utf8 scripts\run_prior_gated_joint_confirmation.py score --output artifacts\preflight\prior_gated_joint_vid110_confirm_20260911_v1_resume1
.venv-p2\Scripts\python.exe -m pytest tests\unit\test_prior_gated_joint.py -q
```

`prepare` / `preflight` 为零调用；`execute` 单次付费、有锁，不要对已有目录重发。恢复目录的 `recovery_plan.json` 记录来源归档哈希、缓存 265 次与新增 23 次。
