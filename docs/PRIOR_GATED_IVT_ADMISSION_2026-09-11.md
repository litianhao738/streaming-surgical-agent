# 先验门控 IVT 接纳：设计、离线选阈值与 VID110 一次确认

日期：2026-09-11。默认版本仍为 `parallel-phase-repair-v1.3.0-glm-low`，本轮不替换。新增模块、审计工具、确认入口和 15 项单元测试；一次单次付费确认（192 次调用）。历史归档未改写。

## 一句话

在 v1.3.0 的五席四头审核之后加一步纯 Python：**按"排除当前视频、以 H0 阶段为条件"的 Training 先验频率，否决罕见 IVT、接纳高频 IVT，Phase 冻结为 H0。** 不增加任何模型调用，输出 JSON 形状不变，每目标反而少 5 次 Phase 调用。

## 为什么是这个方向

三项零调用诊断（`artifacts/research/*_20260911.json`，全部来自已付费归档）：

1. **审核均分对 IVT 对错几乎无分辨力。** 40 个 Training 目标：control 面板 IVT 均分 AUC 0.425，联合面板 0.463；VID110 32 目标分别 0.455 / 0.457。组件均分也不行（0.45～0.56）。
2. **先验频率有分辨力。** 同一批候选，以 H0 Phase 为条件的排除当前视频 Training 频率区分正确/错误 IVT：Training AUC **0.753**，VID110 **0.761**；Target 组件 0.83 / 0.83。
3. **GT 的 I/V/T 在全部 72 个已归档帧上都恰好等于 GT IVT 的组件投影。** 因此四个交互头由 IVT 集合决定，修 IVT 就是修全部四头。

此外，两套已付费面板（v1.3.0 control 与 v2.0.x 联合）的 14 种纯 Python 重组规则（`tools/audit/panel_recombination_replay.py`）在 40 个 Training 目标上没有一个通过预注册标准，说明再折腾聚合方式没有出路；这与 `ADMISSION_RULE_AND_EXTRACTION_PROBE` 的结论一致，但把结论推到了两个面板的组合。

已排除的重复方向：放宽席数、中位数、split review、抽取式审核、prompt 改写、第二轮 Repair、先验扩池（`pool_expansion` 实测 64.73→63.87）、Phase 时序/均分/多数票变体，均已有负结果记录。

## 规则

模块：`src/surgical_agent/research/verification/prior_gated_repair.py`，版本 `prior_gated_ivt_v1`。

```
四头 = recent_mean_panel.select(H0, 候选池, 五席均分, 阈值4)     # v1.3.0 原选择器，逐字节不变
对四头中每个非 null IVT：  先验率(IVT, H0 phase) < 0.01  → 删除（veto）
对候选池中每个非 null IVT：先验率(IVT, H0 phase) ≥ 0.70  → 加入，并补齐其 I/V/T 组件（add）
Phase = H0 Phase
```

- 先验率取该视频排除后的 Training 表中 H0 阶段桶的频率；该桶无任何合格类别时回退到全局频率。null 关系（id ≥ 94）永不被先验增删。
- 表来自各归档已冻结的 `priors/*.json`，`excluded_video` 与 `fit_videos` 在读取时重新校验。
- 可选的"剪掉无 IVT 锚定的 Verb/Target"在开发重放中降低 Verb/Target F1，默认关闭。

## 离线选阈值（Training，零调用）

`tools/audit/prior_gated_replay.py`：7 个归档、136 个去重 Training 目标（VID103/23/31/96 各 34），网格 veto ∈ {无, .005, .01, .02, .05} × add ∈ {无, .5, .6, .7} × prune ∈ {关, 开}。预注册标准：候选平均 F1 高于且总错漏低于"同基础无门控"和"v1.3.0 control"，各头 F1 不低于二者较低值；通过者取错漏最少、再取平均 F1 最高。结果 `artifacts/research/prior_gated_replay_20260911.json`。

| 臂（136 目标） | I | V | T | IVT | Phase | 平均 F1 | 总错漏 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 85.37 | 55.35 | 43.78 | 21.27 | 77.21 | 56.60 | 888 |
| v1.3.0 control | 85.16 | 57.02 | 46.97 | 23.98 | 74.26 | 57.48 | 904 |
| control 四头 + H0 Phase | 85.16 | 57.02 | 46.97 | 23.98 | 77.21 | 58.07 | 896 |
| 门控 H0（零审核） | 85.16 | 55.22 | 50.35 | 34.38 | 77.21 | 60.46 | 798 |
| **门控 control（选定 veto .01 / add .70）** | 85.16 | 57.02 | 52.09 | **37.04** | 77.21 | **61.70** | **800** |

选定设置在 7 个归档、4 个视频上每一组都优于 control（VID31 最弱：55.61→56.91）。136 目标逐帧配对：改善 56、不变 78、变差 2。门控做了什么：否决 64 次，63 次正确（其中 `hook/dissect/cystic_plate` 37 次）；加入 29 次 `hook/dissect/gallbladder`，25 次正确。control 与联合两个面板的四头都能复现归档预测（136/136、40/40）。

同一设置在此前已评分的 VID110 32 目标上（事后核对，不是选阈值依据）：平均 F1 61.98（默认）→ **68.26**，IVT 34.86→49.02，总错漏 184→154，逐帧 13 改善、19 不变、0 变差。

## VID110 新目标一次确认（付费，单次）

入口 `scripts/run_prior_gated_confirmation.py`，归档 `artifacts/preflight/prior_gated_vid110_confirm_20260911_v1`。16 个新 VID110 目标按时间分位选取，三帧全部与此前 VID110 实验用过的任何图片相距 >175 原始帧，只用 mask 不用标签值。阈值来自上节，未在此调整。每目标 12 次调用（H0、补候选、5 席四头、5 席 Phase），全部 192 次完成、无致命错误、无重试。预注册标准写入 `plan.json`：候选在平均 F1、平均 Precision、总错漏上均严格优于 H0 与 control，且 Verb、IVT F1 不低于 control。

| 臂（16 目标） | I | V | T | IVT | Phase | 平均 F1 | 平均 P | 总错漏 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 95.24 | 47.62 | 34.15 | 19.05 | 81.25 | 55.46 | 55.30 | 91 |
| v1.3.0 control | 95.24 | 47.83 | 38.10 | 21.74 | 68.75 | 54.33 | 52.87 | 98 |
| control 四头 + H0 Phase | 95.24 | 47.83 | 38.10 | 21.74 | 81.25 | 56.83 | 55.37 | 94 |
| 门控 H0（零审核） | 95.24 | 47.62 | 37.21 | 31.58 | 81.25 | 58.58 | 58.84 | 83 |
| **门控 control（候选）** | 95.24 | 47.83 | 37.21 | **33.33** | 81.25 | **58.97** | 57.72 | **87** |

**预注册标准通过。** 相对 control：平均 F1 +4.64，平均 Precision +4.85，总错漏 −11；相对 H0：+3.51 / +2.42 / −4。逐帧配对相对 control：7 改善、7 不变、2 变差。门控动作：否决 8 次全部正确（`hook/dissect/cystic_plate` 5、`bipolar/dissect/cystic_plate`、`bipolar/coagulate/cystic_plate`、`hook/dissect/peritoneum` 各 1）；加入 `hook/dissect/gallbladder` 4 次，2 对 2 错。Target 比 control 低 0.89 个百分点，来自 1 次错误加入。所有 80 个臂预测均可由已保存原始回答离线重放复现（0 不一致）。

必须一并说明：

- **接口失败。** 5 次 GPT Phase 403、1 次 DeepSeek Phase 520、3 次补候选 Gemini 429、1 次 Gemini 审核去代码围栏后解析。Phase 失败只影响 control 臂的多数票；3 个补候选失败的目标候选池只有 H0 标签。剔除任一调用失败的目标后只剩 7 个，候选仍优于 control（52.59 vs 47.24，错漏 40 vs 45）但样本太小，且此时零审核的门控 H0（54.33 / 36）反而更高。
- **v1.3.0 的 Phase 多数票再次净伤害**（改对 1、改坏 3），这是第四个批次出现同一方向。
- **门控 H0 与门控 control 差距很小**（58.58 vs 58.97；Training 60.46 vs 61.70）。五席审核在门控之后的边际贡献只剩约 1 个百分点。
- **费用与时长。** OpenRouter 原生 $0.2210，另有 $0.6136 未核实预留（多为 429/403 失败的保守预留），阿里云估算 ¥0.1852。每目标墙钟中位数 19.2 秒（H0 5.7 + 补候选 2.7 + 两面板并行 9.7），最大 42.3 秒；若去掉 Phase 分支则只剩四头面板，估算约 8～9 秒。

## 这个结果说明什么，不说明什么

收益几乎全部来自一件事：五个视觉模型在"胆囊床分离"阶段一致把 `hook/dissect/cystic_plate` 打 4～5 分，而 Training 标注在该阶段几乎从不用这个关系（VID110 先验率 0.002），改标 `hook/dissect/gallbladder`（0.759）。这是**数据集标注惯例**与模型解剖判断之间的系统偏差，门控用 Training 统计把预测拉回标注惯例。它不是新的视觉证据，也不能证明模型看错了组织。

因此：

- 收益集中在 Phase 3（胆囊床分离）帧，且依赖 H0 Phase 正确。H0 Phase 错时收益很小（Training 31 帧：37.34 vs 36.07）。
- 仍是同一个 Validation 视频，两批共 48 帧，不是跨手术泛化；Testing 未参与。
- Phase 4、6 在 VID110 先验中无合格类别（全局回退），Phase 0、5 样本极少；这些阶段的行为未经验证。
- 阈值 0.01 / 0.70 在网格内很稳（veto .005～.05 结果几乎相同），但仍是在被反复分析过的 4 个 Training 视频上选的。

## 建议

1. 作为候选 `prior-gated-ivt-v1` 保留，**不自动替换默认**。若采用，建议的默认形态是"v1.3.0 四头 + 先验门控 + H0 Phase"，每目标 6 次调用（比现默认少 5 次）。
2. 下一次独立确认应换视频：Testing 之外只剩 VID30，而 VID30 标注已被审计判定为 VID17 副本，不能用。可行做法是在 Training 内按视频留出（用 VID02/04/11/13/17/37 中 Verb/Target 有标注的帧，如有），或等待 Testing 阶段的正式协议。
3. 若继续开发，唯一有数据支撑的下一步是让先验参与**补候选**而非只做门控：当前 19 个池外正确 IVT 中 15 个先验合格，但补候选模型没提。这与 `REPAIR_CEILING_DIAGNOSIS` 一致，且 `pool_expansion` 的失败恰是因为扩池后仍用审核均分接纳；改为先验门控接纳后值得重测。

## 复现

```powershell
.venv-p2\Scripts\python.exe -X utf8 tools\audit\panel_recombination_replay.py
.venv-p2\Scripts\python.exe -X utf8 tools\audit\prior_gated_replay.py
.venv-p2\Scripts\python.exe -X utf8 scripts\run_prior_gated_confirmation.py score --output artifacts\preflight\prior_gated_vid110_confirm_20260911_v1
.venv-p2\Scripts\python.exe -m pytest tests\unit\test_prior_gated_repair.py -q
```

确认归档内 `frozen_source/` 保存执行时的源码；执行后对 `scripts/run_prior_gated_confirmation.py` 与 `tools/audit/prior_gated_replay.py` 只做了 Ruff 导入排序和一处闭包改写，`score` 输出与执行时逐字节一致（已重跑核对）。
