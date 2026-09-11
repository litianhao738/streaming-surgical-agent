# 修复天花板诊断：瓶颈不在审核，在候选覆盖

日期：2026-09-10。**全部离线：0 次 API 调用、0 次预测重新生成、未写入任何已封存归档、未读取 Testing。** 只读取已关闭实验目录及其已完成评分的 Training GT。

结论先行：

1. **Verifier/Repair 机制不是瓶颈。** 在候选池装得下的范围内，五席审核确实有效：IVT recall 从 H0 的 21.82% 提到 30.91%，Verb 从 46.15% 提到 59.62%。
2. **瓶颈是候选覆盖。** 32 目标上，55 个 GT IVT 里 **29 个从未进入过任何候选池**，任何完美审核都够不着。IVT 天花板 47.27%，最好的臂已经跑到 30.91%。
3. **覆盖缺口集中在 Target 和 Verb 分量**，不是器械，也不是三元组组装。
4. **提案上限根本没被用满** —— 上限不是约束，proposer 自己不提。
5. **Phase 的问题不是缺时序证据。** 32/32 目标的三帧窗口完整落在单一阶段内部，错误目标里 5/7 距最近边界 194–869 秒。错的是相邻阶段的视觉区分。

## 一、天花板分解

对每个 GT 标签，判定它是"从未进池"（`NOT_PROPOSED`，审核无从补救）还是"进了池但没被选中"（`REJECTED`）。天花板 recall = 池覆盖率 = 完美审核的上界。

工具：[candidate_ceiling_diagnosis.py](tools/audit/candidate_ceiling_diagnosis.py)，证据：`artifacts/research/candidate_ceiling_20260910.json`。

### `expanded_split_review_32_20260910_v1`（32 目标）

| 头 | GT 标签 | 进池 | **天花板** | H0 recall | control recall | 从未进池 | 被审核否掉 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Instrument | 49 | 44 | **89.80%** | 89.80% | 89.80% | 5 | 0 |
| Verb | 52 | 35 | **67.31%** | 46.15% | 59.62% | 17 | 4 |
| Target | 47 | 28 | **59.57%** | 51.06% | 57.45% | 19 | 1 |
| IVT | 55 | 26 | **47.27%** | 21.82% | 30.91% | 29 | 9 |

IVT 的 38 个漏检：**29 个（76%）从未进池**，9 个（24%）进池后被否。

即使把审核做到完美，IVT 最多再涨 16.4 个百分点；而修好覆盖的空间是 52.7 个百分点。**机制迭代的收益上限只有覆盖问题的三分之一。**

### 16 目标确认队列

| 归档 | IVT 天花板 | 各臂实际 | 从未进池 | 被否 |
|---|---:|---:|---:|---:|
| `joint_phase_feedback_fresh16_20260910_v2` | 42.31% | 11.54%（全部臂） | 15 | 7–8 |
| `joint_phase_alternative_repair_dev16_20260910_v1` | 57.69% | 11.54–15.38% | 11 | 11–12 |

这解释了 [JOINT_PHASE_FEEDBACK_TRIAL](JOINT_PHASE_FEEDBACK_TRIAL_2026-09-10.md) 里 B/C/D 四轮迭代为何全部未过确认：它们在争夺一个只有 42% 上限的空间里的几个标签。

## 二、漏掉的 IVT 是什么性质

对每个"从未进池"的 GT IVT，看它自己的 I/V/T 分量当时是否各自在池里。工具：[missing_ivt_anatomy.py](tools/audit/missing_ivt_anatomy.py)，证据：`artifacts/research/missing_ivt_anatomy_20260910.json`。

| 归档 | 从未进池 IVT | 组装缺口<br>（分量齐全，三元组没提） | 部分缺口 | 感知缺口<br>（一个分量都没有） | 缺失分量计数 |
|---|---:|---:|---:|---:|---|
| 32 目标 | 29 | **3（10.34%）** | 24（82.76%） | 2（6.90%） | target 21、verb 18、instrument 5 |
| fresh16 | 15 | 2（13.33%） | 10（66.67%） | 3（20.00%） | target 12、verb 9、instrument 4 |
| alt16 | 11 | 2（18.18%） | 8（72.73%） | 1（9.09%） | target 7、verb 6、instrument 1 |

两个推论：

- **"多枚举三元组"不是答案。** 纯组装缺口只占 10%，就算全补上，32 目标的 IVT recall 也只从 30.91% 升到约 36.4%。
- **真正缺的是 Target 和 Verb 分量本身。** 29 个漏检里 target 缺 21 次、verb 缺 18 次、instrument 只缺 5 次。器械认得出，**"对什么做了什么"认不出**。Target 自己的天花板就只有 59.57%。

## 三、提案上限没有被用满

既然覆盖不够，先查上限是不是约束。证据：`artifacts/research/proposer_quota_saturation_20260910.json`。

| 头 | 每轮上限 | 用满上限的目标数 | 平均提案数 | 空列表目标数 |
|---|---:|---:|---:|---:|
| Instrument | 2 | 0（0%） | 0.03 | 31/32 |
| Verb | 2 | 2（6%） | 0.72 | 11/32 |
| Target | 2 | 1（3%） | **0.28** | **24/32** |
| IVT | 4 | 1（3%） | 1.16 | 5/32 |

**上限完全不是约束。** Target 上 32 个目标里 24 个 proposer 返回空列表，平均只提 0.28 个 —— 与此同时正确 target 有 19 次不在池中。放宽 `MAX_NEW` 不会有任何效果。

对照 proposer 的提示原文（[run_candidate_panel_trial.py:56](scripts/run_candidate_panel_trial.py:56)）：

> "Do not add alternatives merely to fill a quota. Return empty lists if no new candidate has visual support."

proposer 被明确要求保守，它照做了。这是一个**可单因素检验的假设**：保守指令正在压制覆盖率。

## 四、Phase：不是时序证据问题

工具：[phase_evidence_window.py](tools/audit/phase_evidence_window.py)，证据：`artifacts/research/phase_evidence_h0_20260910.json`。

对 32 个目标，从 Training GT 还原每个视频的阶段时间线，量出目标帧在自己阶段段落中的位置。

- **32/32 目标的三帧窗口 `[-2,-1,0]` 秒完整落在单一阶段内部**，一次都没有跨越边界。
- 到最近阶段边界的距离：最小 5 秒、中位数 **148 秒**、最大 1304 秒；所在段落长度中位数 625 秒。
- H0 阶段正确 25/32（78.13%）。7 个错误目标：

| 目标 | GT | 预测 | 距最近边界 | 段落长度 |
|---|---:|---:|---:|---:|
| VID31_62501 | 1 | 3 | **869 秒** | 2956 秒 |
| VID23_4351 | 1 | 3 | **451 秒** | 625 秒 |
| VID23_8651 | 1 | 3 | **279 秒** | 625 秒 |
| VID103_13426 | 1 | 3 | **229 秒** | 551 秒 |
| VID23_34176 | 5 | 0 | **194 秒** | 421 秒 |
| VID31_84851 | 2 | 1 | 24 秒 | 439 秒 |
| VID103_24001 | 2 | 1 | 5 秒 | 198 秒 |

7 个错误里 **5 个距任何阶段边界 194–869 秒**，深在同质段落内部。给更长历史或更多帧不可能修好这些。

混淆分布：`1→3` ×4、`2→1` ×2、`5→0` ×1。主错误 `1→3` 是 CalotTriangleDissection 误判为 GallbladderDissection —— 也就是"在胆囊颈/Calot 三角操作"与"在肝床分离胆囊"的区分。**两版 compact 提示都已经逐字要求过这个区分**（`compact_phase_v2.txt`："Distinguish work at the gallbladder neck, separation at the liver bed, and field cleanup"），模型仍然分不开。

Phase 修复的两种形态在 32 目标上：

- 多数票（control）：改了 10 个，**改对 3、改坏 7**，净 −4。
- 联合评分（joint_r1/r2）：改了 0–1 个，基本**惰性**，既不帮也不害。

所以 [PHASE_MECHANISM_COMPARISON](PHASE_MECHANISM_COMPARISON_2026-09-10.md) 里"联合方案 Phase 与 H0 持平"不是因为它判得准，是因为它几乎不触发。

## 五、这份诊断否定了什么

| 方向 | 已有尝试 | 本诊断的判定 |
|---|---|---|
| 改审核提示/阈值/quorum | compact、weighted、flexible_quorum、split_review | 天花板之下的空间只有 16pp，且审核已用掉大半 |
| Phase 机制（多数票/均分/联合/反馈） | 6+ 变体，见两份 Phase 文档 | 错误是同质段落内的视觉混淆，机制改不动 |
| 更长/更多历史帧 | 未系统做 | 窗口从不跨边界，5/7 错误远离边界，无依据 |
| 放宽候选上限 | 未做 | **上限未被用满，无效** |
| 多枚举三元组组合 | targeted_joint_repair 部分涉及 | 纯组装缺口仅 10%，上限约 +5.5pp |
| ROI 裁剪 | roi_current / roi_temporal / verifier_current | 已试，[未通过确认](DEFAULT_IMPROVEMENT_TRIAL_2026-09-09.md) |

## 六、离线量化：扩大先验候选能把天花板抬到多少

上一节指出上限不是约束、proposer 不肯提，那就绕开 proposer：**直接用已冻结的 LOVO 先验批量补候选**。这一步不需要视觉、不需要新调用，纯 Python。

工具：[prior_coverage_simulation.py](tools/audit/prior_coverage_simulation.py)，证据：`artifacts/research/prior_coverage_simulation_20260910.json`。

按先验自身频率排序（H0 阶段桶可用时优先，否则全局），每个头补前 N 个，量出天花板变化。查询视频对自己的先验被排除，脚本里重新校验过。

| 每头新增 | Verb 天花板 | Target 天花板 | **IVT 天花板** | 32 目标总新增候选 |
|---:|---:|---:|---:|---:|
| 0（当前） | 67.31% | 59.57% | **52.73%** | 0 |
| 1 | 75.00% | 72.34% | **60.00%** | 19 |
| 2 | 78.85% | 74.47% | 60.00% | 50 |
| 3 | 88.46% | 76.60% | **67.27%** | 89 |
| 5 | 96.15% | 87.23% | **78.18%** | 198 |
| 8 | 100.00% | 97.87% | **89.09%** | 388 |

> 本表 depth 0 的 IVT 天花板 52.73% 含组装可达项（三分量齐全即算可达），故高于第一节按"IVT 是否进池"算的 47.27%，两者口径不同、互不矛盾。

**每头补 5 个先验候选，IVT 天花板从 52.73% 抬到 78.18%**，代价是每目标约多 6 个候选（32 目标共 198 个）。当前池规模约 10 个命题，`MAX_POOL` 是 64，容量充足。**提案阶段 0 次新增 API 调用**，只是审核请求变长。

必须同时说明的风险：**这抬的是天花板，不是成绩。** 候选变多同时给了面板更多接纳错误标签的机会，Precision 可能下降。已测的审核行为偏保守（32 目标上池内只否掉 4 verb / 1 target / 9 IVT），但这不能保证在 5 倍大的池上仍然成立。因此这条路的预注册指标必须**同时**包含覆盖率、Precision 和 FP+FN 总数，而不是只看 recall。

## 七、唯一由数据指向的下一步

**提高 Target 与 Verb 的候选覆盖**，这是唯一能抬高天花板的方向。按可检验性和成本排序：

1. **单因素检验 proposer 的保守指令。** 只改 proposer 提示中"无视觉支持就返回空列表"这一段，其余（模型、图片、上限、审核、阈值、接纳规则）全部不动。预注册指标改为**覆盖率**（GT 进池率）而非最终 F1 —— 因为覆盖率才是被干预的量，F1 受审核否决率混杂。若覆盖率不升，这条路直接排除。
2. **用 LOVO 先验批量补 Verb/Target 候选（第六节已离线量化，建议优先）。** [prior_panel.build_universe](src/surgical_agent/research/verification/prior_panel.py:127) 当前只加 **1 个** `INDEPENDENT_PRIOR` target，先验表里的 verb/target 频率基本没被用。扩到每头 5 个可把 IVT 天花板从 52.73% 抬到 78.18%，提案阶段 0 新增调用。预注册指标必须同时含覆盖率、Precision 与 FP+FN 总数。
3. **确认 Target 的 null 语义与 GT 口径是否一致。** 32 目标里 4 个漏检 IVT 是 null 关系（id ≥ 94），而 `retrieve_candidate_hints` 显式排除 `ivt >= 94`。

第 2 项的天花板收益已经离线算出，且不增加调用次数，是成本最低、上限最高的一条；第 1 项作为它的对照可同批进行。

## 复现

```powershell
.venv-p2\Scripts\python.exe -X utf8 tools/audit/candidate_ceiling_diagnosis.py artifacts/preflight/expanded_split_review_32_20260910_v1 artifacts/preflight/joint_phase_alternative_repair_dev16_20260910_v1 artifacts/preflight/joint_phase_feedback_fresh16_20260910_v2 --output artifacts/research/candidate_ceiling_20260910.json
.venv-p2\Scripts\python.exe -X utf8 tools/audit/missing_ivt_anatomy.py artifacts/preflight/expanded_split_review_32_20260910_v1 artifacts/preflight/joint_phase_alternative_repair_dev16_20260910_v1 artifacts/preflight/joint_phase_feedback_fresh16_20260910_v2 --output artifacts/research/missing_ivt_anatomy_20260910.json
.venv-p2\Scripts\python.exe -X utf8 tools/audit/phase_evidence_window.py artifacts/preflight/expanded_split_review_32_20260910_v1 --arm h0 --output artifacts/research/phase_evidence_h0_20260910.json
.venv-p2\Scripts\python.exe -X utf8 tools/audit/prior_coverage_simulation.py artifacts/preflight/expanded_split_review_32_20260910_v1 --output artifacts/research/prior_coverage_simulation_20260910.json
```

## 局限

- 全部样本来自同四个 Training 视频（VID103/VID23/VID31/VID96），已被反复分析，**不是独立验证集**；覆盖率数字不外推到 Testing 或新视频。
- 天花板按"该次运行的候选池并集"计算。不同归档的池由不同提示/模型产生，跨归档的天花板不可直接相比。
- `NOT_PROPOSED` / `REJECTED` 的划分依据接纳结果，不解释模型为何这样判断；"审核有效"只说明它在池内提高了 recall，不证明其视觉判断正确。
- Phase 时间线来自 mask 有效的标注帧，边界距离精度受标注采样率限制。
- 第六节是**天花板模拟，不是实验结果**。它只回答"正确标签会不会出现在池里"，不回答面板会不会选中它，也不包含扩大池后新增的假阳性。任何采用都必须先做真实配对实验。
- 未改动任何默认版本、历史结果或已封存归档。
