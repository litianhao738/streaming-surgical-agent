# 方案 4 输出模块 v2.2：两处修正与结果（2026-09-14）

## 结论

在方案 4 的出口后处理中加了两处修正。Gate、路由、逻辑调用和 API 请求都没有改：

1. **本体过滤：** 只输出至少出现在一个三元组里的器械。Tracker 检测到的"标本袋"不再当作器械输出。它是 CholecTrack20 的框类别，但在三元组体系里是目标。
2. **闲置标签清理：**
   - Qwen 初审明确给 1 分（且回答有效）的闲置三元组（器械-无动作-无目标），直接删除；
   - 如果输出里已经没有闲置三元组，"无动作""无目标"这两个标签也一并删除。

**结果：**

- **6,059 帧全量回放：** 五头平均 F1 从 66.40 升到 **67.02**（+0.62），FP+FN 从 32,398 降到 **30,774**（少 1,624 个）。4 个视频的 F1 和错漏都改善了。
- **一致性核对：**
  - 逻辑调用次数和 Gate 路由与原方案 4 相比，0 处差异；
  - 用 `--output-modules v2.1` 回放，逐帧复现原方案 4 的预测，0 处差异。
- **两帧实际运行：** 用已保存的实际回答重新计算，F1 从 51.43 升到 61.33。三元组仍然是 0，原因是肝脏被认成胆囊，本次没有处理。

## 依据

**本体过滤：**
- 6,059 帧里，GT 的器械集合和 GT 三元组里出现的器械完全一致；没有任何三元组以标本袋作为器械。
- `src/surgical_agent/data/label_policy.py` 也写明：Track20 的标本袋框，对应的是三元组"抓钳-抓-标本袋"里的目标。
- Tracker 在 102 帧里报告了标本袋，其中 57 帧在包装阶段，识别本身是合理的，但作为器械一律算错。

**闲置标签清理：**
- GT 里的"无动作""无目标"只和闲置三元组一起出现，没有例外。
- Qwen 对已有的闲置三元组给 1 分时，约 83% 确实不存在。
- Qwen 对有动作的三元组给 1 分时，却有 77% 其实是存在的。例如"电钩-分离-胆囊"，这个比例是 85%。所以这条规则只用在闲置类上。

**测试过但没有采用的做法：**

| 做法 | 结果 |
|---|---|
| 删除所有被 Qwen 打 1 分的已有标签 | F1 65.66，三元组 32.79，明显变差 |
| 新增器械要求更强证据（置信度 ≥ 0.5、≥ 0.7，或前一秒也检到） | 整体只多 0.05–0.07，且有视频变差 |
| 闲置清理只用在 Gate 跳过的帧 | 有视频变差 |

## 结果（全量模型在训练集回放，非嵌套外层）

| 头 | v2.1 | v2.2 |
|---|---:|---:|
| 器械 | 93.99 | 94.47 |
| 动作 | 65.02 | 66.38 |
| 目标 | 50.71 | 51.59 |
| 三元组 | 36.94 | 37.32 |
| 阶段 | 85.34 | 85.34 |
| **五头平均 / FP+FN** | **66.40 / 32,398** | **67.02 / 30,774** |

| 视频 | v2.1 | v2.2 |
|---|---:|---:|
| VID103 | 67.22 / 6,150 | 67.42 / 5,969 |
| VID23 | 65.83 / 4,069 | 65.94 / 3,936 |
| VID31 | 67.26 / 17,713 | 68.28 / 16,507 |
| VID96 | 61.32 / 4,466 | 61.39 / 4,362 |

**各模块实际触发的帧数：**
- Tracker 标本袋被排除：102 帧；
- 删除闲置三元组：484 帧；
- 删除"无动作"：821 帧；删除"无目标"：817 帧。

v2.2 一共改动了 941 帧，其中 814 帧变好、127 帧变差。

## 局限

- "只用在闲置类"这个限制，是看到数据之后才定的。全部评估都在 4 个反复用于开发的训练视频上，Gate 用的是全量拟合模型，属于训练集内结果，必须做独立验证。
- 这条规则依赖 Qwen 初审的打分习惯。以后如果更换初审模型或改提示词，需要重新核验。
- 三元组仍然偏低（37.3），主要瓶颈是肝脏/胆囊混淆：GT 的"抓钳-牵拉-肝脏"出现在 392 帧，只预测对了 20 帧。
- 嵌套外层数字（论文用）还没有用 v2.2 重算。需要在 `scripts/assess_tracker_scheme4.py` 里接入 v2.2 的输出模块。

## 代码改动

- **`src/surgical_agent/research/gate/tracker_pipeline_v2.py`：**
  - 新增 `TRIPLET_INSTRUMENTS`、`NULL_IVTS`、`null_label_cleanup`，以及 `OUTPUT_MODULES`（v2.1 / v2.2）；
  - `output_modules(..., ontology_filter=...)`；
  - `run_interaction` 额外返回 `probe_ratings`，这些是 Gate 判断之前就已经付费拿到的 Qwen 评分；
  - `run_target(..., output=...)`。
  - `VERSION` 保持不变，因为 Gate 模型文件会核对它。
- **`scripts/run_tracker_scheme4_pipeline.py`：**
  - 新增 `--output-modules`，默认 v2.2；
  - prepare 把选择写入计划，execute 按计划执行，旧计划按 v2.1 执行；
  - 回放收据和评分都记录输出模块版本。
- **`scripts/run_pipeline.py`：** 透传 `--output-modules`。
- **测试：** `tests/unit/test_tracker_pipeline_v2.py` 新增 2 项；方案 4 的两个测试文件共 18 项全部通过。
- **`DEFAULT_PIPELINE_VERSION.json`：** 新增输出模块版本字段，并注明如何回退。
  - `tools/package_scheme4_pipeline.py` 已同步这些版本字段，重新生成清单会保留 v2.2 选择。

## 复现（全部零 API）

```powershell
.venv-p2\Scripts\python.exe scripts/run_tracker_scheme4_pipeline.py replay --output-modules v2.1 --output <新目录A>
.venv-p2\Scripts\python.exe scripts/run_tracker_scheme4_pipeline.py replay --output-modules v2.2 --output <新目录B>
.venv-p2\Scripts\python.exe scripts/run_tracker_scheme4_pipeline.py score --output <新目录B> --annotations artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json
.venv-p2\Scripts\python.exe tools/audit/scheme4_output_modules_v22_compare.py
```

本次输出目录：`artifacts/preflight/scheme4_output_v21_parity_20260914` 和 `artifacts/preflight/scheme4_output_v22_all_20260914`。
