# 五头视觉修复与上一版的同样本对照（2026-09-09）

本实验单独保存，未替换默认入口或原图谱版本。结果不是完整 Tracker/Gate Pipeline 的正式 Testing 指标。

## 执行前冻结的方案

复用八个既有 Training 目标、原来的三帧因果图片和缓存 H0。四个视频各两个目标：VID103/18326、33576；VID23/13176、28876；VID31/40701、73701；VID96/15051、26051。样本已经被用于开发诊断，因此不把这次提升当作独立泛化验证。

本次配对基线的缓存 H0 模型为 OpenRouter `google/gemini-3.8-flash`，原版联合预测 prompt、final-only 五头合同、temperature 0、reasoning low、4096 输出 token 上限；同一份 H0 在所有对照中复用。

共享起点为原最佳图谱一轮输出：固定 H0 → 图谱提供备选关系 → 单模型补候选 → 五模型看图评分 → Python 局部接纳。随后增加一次联合五头视觉修订，同时形成两个对照结果：

1. `paper_style`：视觉 LLM 返回完整五头答案；Python 检查 ID、头容量、Phase 单选，以及 IVT 的组件闭包。合法答案替换旧答案；四个交互头同时为空时保留旧交互头，Phase 仍独立处理。
2. `panel_five`：同一份模型修订用于扩充候选；原五个审核席看原三帧图片，独立评分四头候选和全部七个 Phase。Python 再接纳局部修改。

辅助输出 `llm_raw` 是闭包处理前的原始模型答案；`panel_four_shadow` 复用同一批新审核，只把 Phase 固定为原图谱版，用于观察 Phase 更新本身的影响。它不是一次独立 API 对照。

## 借鉴母体的具体范围

参考作者公开代码固定提交 `3028fce4a8eb7ba181314b593c0336ad69b9b67b`：

- [reflection.py](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/reflection.py)：检查 IVT 与组件、阶段先验等问题；可让同一个视觉模型根据图片、原预测、问题和候选选项返回修订后的五头答案。
- [orchestrator.py](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/orchestrator.py)：对返回结果做规则处理与更新，包括 IVT 组件并集和独立阶段更新。公开实现没有在这次模型修订后再追加本项目的五模型评分。

本次借鉴“发现约束问题 → 看图联合修订 → 再检查”的思路。母体使用每道题的 MCQ 选项，本项目使用完整本体和帧级多标签集合，任务不完全一致，不声称复现其论文数值。母体论文描述的多轮流程也不能直接等同于该提交中一次条件分支触发的 `model_repair`。

## 五头从输入到输出

修订模型接收三帧图、图谱一轮答案、旧审核的具体观察与拒收原因、全本体、最多两条原图谱关系提示，以及阶段兼容性软提示。无 GT 答案、无测试集统计。反馈中的模型评分已去掉，错误票不会变为有效票。

模型输出：

```json
{
  "prediction": {
    "instrument": [0],
    "verb": [0],
    "target": [0],
    "ivt": [7],
    "phase": [3]
  },
  "observations": {
    "instrument": "Short current-image observation.",
    "verb": "Short current-image observation.",
    "target": "Short current-image observation.",
    "ivt": "Short current-image observation.",
    "phase": "Short current-scene observation."
  }
}
```

示例仅说明格式，不是实际预测或 GT。单头解释上限 1000 字符。四头容量沿用已冻结的 final-only 合同：I ≤ 3、V ≤ 4、T ≤ 5、IVT ≤ 8；Phase 必须一个合法 ID。英文提示和 JSON 开头保留学术医疗视频用途说明。

审核保持原五个系列：Grok `grok-4.6`、Qwen `qwen3.8-flash`、GPT `openai/gpt-5.6-luna`、Gemini `google/gemini-3.5-flash-lite`、DeepSeek `deepseek/deepseek-v4-flash-vision-exp`。修订模型为 OpenRouter `google/gemini-3.8-flash`，固定 Google AI Studio 路由，medium reasoning。以上具体 ID 由本地接口预检确认；不把不可关闭的模型推理称为“完全不推理”。

每个候选返回唯一 rating（1—5）、finding、scope、image_indices、observation。新增标签要求五张有效票平均 ≥ 4；删除要求平均 ≤ 2，且不能删除保留 IVT 仍需的组件。局部反例不能证明整帧不存在某标签。

Phase 使用单独的原子替换规则：必须有当前阶段的有效均分，另一个阶段的五票均分 ≥ 4、唯一最高且高于当前阶段，才能替换。并列、证据不足或当前阶段评分不完整时保留原阶段。Phase 不是七个可同时新增的多标签。

阶段软提示来自原图谱实验已经冻结的 Training 统计表，每个查询视频整段排除；按当前非空 IVT 的 P(IVT|Phase) 提供最多两个有足够统计支持的阶段。它不是 P(Phase|图像)，当前 IVT 也可能错，因此不允许由频率直接覆盖阶段。五模型审核不接收此提示或修订模型的解释，避免将其当作视觉证据。

## 调用、评分与复现

执行前上限：8 次联合修订 + 40 次审核，共 48 次新增同步 API 调用；每目标五个审核并发，不重跑 H0、不重试、不追加第三轮。预算上限分别为 OpenRouter $2、xAI $2、阿里云 ¥2，不合并不同币种。

输出目录：`artifacts/preflight/five_head_repair_eight_20260909_v1/`。`plan.json` 冻结代码、输入图片、历史回答、先验和请求；`calls/` 保存脱敏请求、原始响应、使用量；`completion.json` 封存推理，之后独立算分。评分器逐调用回放、重算审核均值，并独立核对各头 TP/FP/FN。缺失 GT 按头 mask 排除；失败目标仍保留并回退。

```powershell
# 新实验只能用新的输出目录；不要覆盖已经付费执行过的目录。
.venv-p2/Scripts/python.exe scripts/run_five_head_repair_trial.py prepare --output artifacts/preflight/NEW_FIVE_HEAD_RUN
.venv-p2/Scripts/python.exe scripts/run_five_head_repair_trial.py execute --output artifacts/preflight/NEW_FIVE_HEAD_RUN

# 已完成实验的离线重新评分，不调用 API。
.venv-p2/Scripts/python.exe scripts/score_five_head_repair_trial.py --output artifacts/preflight/five_head_repair_eight_20260909_v1
```

新增实现位于 `src/surgical_agent/research/verification/five_head_repair.py`，新运行入口和评分器如上。执行前 145 项相关测试及四个改动 Python 文件的 Ruff 检查通过。旧运行与评分代码未改。

## 实测结果

已完成 48 次新增请求与封存后的逐调用回放。所有五头的有效 GT 数均为 8。8 次联合修订均合法；37 次审核返回可解析结果；另 3 次 DeepSeek 返回 HTTP 429（Fireworks 上游共享池限流）。这三目标保留原图谱输出，仍进入主表的八目标分母。记录中的 `REVIEWED` 表示已经走过审核流程，不能据此声称每个目标都取得了五份有效回答。

**结论：这批八个开发目标上，原图谱一轮版仍优于此次五头修复；新增机制未替换默认版。**

### F1 对照（micro-F1，%）

| 版本 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 固定 H0 | 66.67 | 61.54 | 51.85 | 29.63 | 62.50 |
| 原图谱一轮版 `graph_r1` | **69.23** | **66.67** | **57.14** | **40.00** | 62.50 |
| 上一轮四头视觉修复 `previous_four` | 69.23 | 66.67 | 57.14 | 38.71 | 62.50 |
| 本次：五头 LLM 修订＋Python 检查 `paper_style` | 66.67 | 66.67 | 53.85 | 32.26 | 62.50 |
| 本次：再经过五模型审核 `panel_five` | 69.23 | 66.67 | 55.17 | 38.71 | 62.50 |

相对原图谱一轮，最终 Target −1.97 个百分点、IVT −1.29 个百分点，其余三个头不变。相对上一轮四头视觉修复，最终 Target −1.97 个百分点，其余头相同。`llm_raw` 与 `paper_style` 此次完全相同：模型已满足组件闭包，Python 没有额外增补组件，也未触发四头全空保护。

### Precision（micro，%）

| 版本 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| H0 | 69.23 | 66.67 | 58.33 | 33.33 | 62.50 |
| 原图谱一轮 | 75.00 | 62.50 | 61.54 | 40.00 | 62.50 |
| 上一轮四头修复 | 75.00 | 62.50 | 61.54 | 37.50 | 62.50 |
| 五头 LLM＋Python | 69.23 | 62.50 | 63.64 | 31.25 | 62.50 |
| 五头 LLM＋五模型审核 | 75.00 | 62.50 | 57.14 | 37.50 | 62.50 |

### Recall（micro，%）

| 版本 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| H0 | 64.29 | 57.14 | 46.67 | 26.67 | 62.50 |
| 原图谱一轮 | 64.29 | 71.43 | 53.33 | 40.00 | 62.50 |
| 上一轮四头修复 | 64.29 | 71.43 | 53.33 | 40.00 | 62.50 |
| 五头 LLM＋Python | 64.29 | 71.43 | 46.67 | 33.33 | 62.50 |
| 五头 LLM＋五模型审核 | 64.29 | 71.43 | 53.33 | 40.00 | 62.50 |

### 集合 Accuracy（整头预测集合与 GT 完全相同，%）

| 版本 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| H0 | 37.50 | 25.00 | 25.00 | 0.00 | 62.50 |
| 原图谱一轮 | 37.50 | 25.00 | 25.00 | 0.00 | 62.50 |
| 上一轮四头修复 | 37.50 | 25.00 | 25.00 | 0.00 | 62.50 |
| 五头 LLM＋Python | 37.50 | 25.00 | 12.50 | 0.00 | 62.50 |
| 五头 LLM＋五模型审核 | 37.50 | 25.00 | 25.00 | 0.00 | 62.50 |

Phase 为单选且均有 GT，因此这里 micro Precision/Recall/F1 与 Accuracy 相等。IVT 集合 Accuracy 为零表示没有一帧的整组 IVT 全对，不表示所有单个 IVT 都错。最终 IVT 的 TP/FP/FN 为 6/10/9，原图谱一轮为 6/9/9。

### 改对、改坏和无收益替换

按完整目标、相对原图谱一轮统计：

| 结果 | 完整改对 | 部分改善 | 改坏 | 混合变化 | 仅无收益替换 | 不变 |
|---|---:|---:|---:|---:|---:|---:|
| 上一轮四头修复 | 0 | 0 | 1 | 0 | 0 | 7 |
| 五头 LLM＋Python | 0 | 1 | 3 | 0 | 0 | 4 |
| 五头 LLM＋五模型审核 | 0 | 0 | 2 | 0 | 0 | 6 |

另按 Phase 独立统计：0 次改对、0 次正确改坏、1 次错→错的无收益替换、7 次保持。该无收益替换和 Target 的改坏发生在同一目标，所以完整目标表被归为“改坏”，不能再加算一个目标。

修订模型共提出 6 项交互标签变化：1 项有益删除、5 项有害变化。审核拦下 4/5 有害变化，放行 1/5；唯一有益删除所在目标发生 DeepSeek 429，未获得五票，故未应用。这不能证明五模型已经否定了有益删除。审核还从旧候选池另加了一个错误 Target。

对 Phase 的错→错替换，集合操作会记为“删掉一个错误标签＋加入另一个错误标签”，但不能把前半步单独宣称为一次阶段修复成功。

## 具体问题定位

1. **候选没有补出正确答案。** 原池覆盖 15 个 GT IVT 出现项中的 7 个，五头重写之后仍是 7/15；没有任何 IVT 超出旧候选池。其余 8 个正确关系仍未进入可选范围。Phase 的七种选项全部在审核范围内，这是枚举覆盖，不是识别成功。
2. **LLM 全量修订会改坏原有正确关系。** 一个修订把 `hook–dissect–cystic_duct`（58）换成 `hook–dissect–gallbladder`（60），并删除了原本正确的 cystic_duct Target。原始回答明确这样判断，Python 没有错误映射 ID；本次全部 `derived_components` 为空，不能归因于程序强行重建三头。
3. **平均高分仍可能是共同看错。** 错误 `bipolar–coagulate–cystic_plate`（27）得到 `[3,4,5,5,4]`，均分 4.2；错误 gallbladder Target 得到 `[4,5,3,5,5]`，均分 4.4。五个系列并不等于五个独立正确证据源。
4. **扩大重审范围会重新激活旧错误候选。** 本次审核所有旧池与新修订的并集。上述错误 Target 并不是修订模型这次提出的 ADD，而是旧候选重新得高分后被 Python 加入。这里是审核范围与接纳策略带来的行为，不是 JSON 解析错。
5. **Phase 已接通，仍缺正确判断。** LLM 本次八帧全部保留旧 Phase；五模型把一帧从 4（packaging）改为 3（dissection），GT 是 5（cleaning/coagulation）。错误阶段 3 获 4.0，正确阶段 5 仅 1.4，因此不是正确答案被阈值卡住。另一错误阶段样本中正确阶段均分为 2.8，当前错误阶段为 3.8；降低通过阈值也不能据此可靠找回正确阶段。
6. **接口可用性影响严格五票策略。** VID31/40701、VID31/73701、VID96/26051 的 DeepSeek 被限流，相关均分为缺失并回退。成功响应里还存在 27 条“局部否定不足以证明整帧不存在”的无效审核，原因被保留；不是 1000 字符长度限制。

API 全部成功的五目标仅作接口成功子集诊断，同样没有语义净收益：原图谱 IVT F1 47.06% → 新五头审核 44.44%；Target 50.00% → 47.06%；Phase 60.00% → 60.00%。主结果仍保留全部八目标。失败子集的未知五票结果不能凭空补出，因此本实验不能估计接口完全可用时整批的精确分数。

## 新增费用、耗时与保留决定

- 新增请求 48 次：8 次视觉修订、40 次审核；45 次可解析，3 次 HTTP 429，无重试。没有新 H0 请求。
- OpenRouter 已返回的 native usage 费用 **$0.13150678**，其中修订模型 **$0.06925350**；xAI native 费用 **$0.199638**。已报告美元费用合计 **$0.33114478**。
- 阿里云按用量保守估计 **¥0.115377**，不与美元合并。
- 三次 429 未提供用量，账本另外保留 **$0.12173304** 的未知费用预留。这不是已证实扣费；含此预留的美元占用上限为 **$0.45287782**。
- 本次新增推理墙钟 **238.16 秒，约 3.97 分钟**；上一轮四头视觉修复为 **220.69 秒**，本次约增加 **17.47 秒（7.92%）**。两次审核范围、调用成功率不同，不能外推为全量 Pipeline 的固定增幅。该时长不含已缓存的 H0、图谱第一轮和离线算分。

保留原图谱一轮作为这批同样本对照中效果更好的版本；新增五头路径保留为独立实验入口。接通 Phase 是功能完成，不能等同于准确率改善。下一项值得独立验证的是阶段所需的更长因果历史或阶段区分证据，先检查当前短窗口能否辨别阶段；不要仅降低评分阈值或继续全量重写五头。

原始数值见实验目录中的 `metrics.json`、`frame_deltas.json`、`stage_diagnosis.json`。`scored_truth.json` 与推理调用分开保存；未上传新代码或密钥。
