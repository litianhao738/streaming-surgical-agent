# Gemini 纯 API baseline 三目标冒烟与测试集费用估算

日期：2026-09-06。真实调用已完成；只运行三次推理，没有运行完整 Testing。

## 实验设置

- OpenRouter 请求模型：`google/gemini-3.8-flash`；账单记录实际版本：`google/gemini-3.8-flash-20260902`。
- 供应商固定 Google AI Studio，禁止回退，单次尝试，无重试。
- 每个目标输入 `[t-50,t-25,t]`，一次调用联合输出 Instrument、Verb、Target、IVT、Phase。
- 三个目标来自 Training / VID103：25101、25126、25151。相邻目标间隔 25 个原视频帧，即 1 秒；不是原视频相邻三帧。
- 使用当前主版本请求构造器及完整 prompt / final-only JSON schema。只在独立实验请求中替换模型、路由和实验版本标识；项目默认 Qwen 配置保持不变。
- 参数：temperature=0，reasoning.effort=low，max_tokens=4096；图像 detail 为 low、low、high。
- 无 Tracker、Gate、Verifier、Repair、历史预测或跨窗口记忆。样本事先固定，推理未读取 GT 标签。
- 系统 prompt SHA-256：`c1009de8429158eab6b9c0f4ee39b12f51e5e970335f0c554866198191b6e3e5`。
- 计划预算 $0.10，最大三次调用；预算是本地调度预留，不是供应商强制封顶。

## 实际消耗

| 目标帧 | 输入原视频帧编号 | 输入 tokens | 输出 tokens | 实际费用 USD |
| --- | --- | ---: | ---: | ---: |
| 25101 | 25051、25076、25101 | 4075 | 62 | 0.00328875 |
| 25126 | 25076、25101、25126 | 4075 | 118 | 0.00349875 |
| 25151 | 25101、25126、25151 | 4075 | 118 | 0.00349875 |
| 合计 | 三个三图窗口，三次调用 | 12225 | 298 | **0.01028625** |

平均每个目标 $0.00342875；平均调用延迟 4.70 秒。三次均成功返回并通过五头 JSON 校验，finish_reason 均为 stop。没有缓存命中，也没有未计价调用。

虽然请求设置 low reasoning，本次供应商回报的 reasoning_tokens 均为 0；不能据此保证其他输入的推理消耗也为 0。本实验核对通路和费用，没有评估识别准确率。

费用采用每次响应中的 usage.cost，并逐笔通过 generation 查询核对，两种账单一致。OpenRouter 说明 usage.cost 是账户实际扣费，见[费用核算文档](https://openrouter.ai/docs/cookbook/administration/usage-accounting)。

## 完整 Testing 费用

按本地正式 Testing 的全部推理目标重新计数，不使用 GT mask 删减调用数：

| 视频 | 预测目标数 |
| --- | ---: |
| VID01 | 1516 |
| VID06 | 1962 |
| VID07 | 4213 |
| VID111 | 1657 |
| VID12 | 827 |
| VID25 | 1915 |
| VID39 | 1462 |
| VID92 | 1730 |
| 合计 | **15282** |

每个目标调用一次，预计 15,282 次 API，不能因为一次输入三张图就再除以三。实际窗口有 375 个单图、317 个双图、14,590 个三图窗口，来自视频边界和采样缺口。以下统一按完整三图输入外推，没有提前扣除短窗口的潜在节省。

价格快照对应标准 Google AI Studio 路由：输入 $0.75 / 百万 tokens，输出 $3.75 / 百万 tokens。来源：[OpenRouter 模型端点元数据](https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints)。未使用 Flex、Priority 或 Batch。实际三次账单与这组单价一致。

| 估算方式 | 全部 Testing 费用 USD |
| --- | ---: |
| 按本次平均每目标费用 | **52.3981575（约 $52.40）** |
| 每次均按本次最贵一次 | **53.4678975（约 $53.47）** |
| 每次输入按 4075 tokens，输出全部用满 4096 tokens，无缓存折扣 | **281.4371325（约 $281.44）** |

输出打满情景公式：

`15282 × (4075 × 0.75 / 1000000 + 4096 × 3.75 / 1000000)`

输出预算包含计费推理 token，不再额外重复加一笔推理费用，见[OpenRouter 推理 token 说明](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。

这三个数字都依赖固定模型、固定价格、每目标只调用一次；$281.44 还要求输入不超过本次 4075 tokens。它是有条件的保守预算情景，不是由三个样本证明的全测试集绝对最大费用。其他图像、输出和收费变化可能改变结果。

可先按约 $52–54 理解这三个样本的费用水平；若预留输出长度波动，可准备约 $300 的调用预算，但该预留不是承诺封顶。含本次冒烟，平均外推总计约 $52.41，输出打满情景总计约 $281.45。以上为 API 消耗，不含支付渠道或汇兑费用。

## 核验与原始记录

- 3 个独立 dispatch.lock；3 次推理；3 次只读账单查询；无补调或响应修复。
- 逐笔核对模型、供应商、max_tokens、temperature、reasoning、三张图与 detail，以及系统 prompt 字节一致性。
- 预检通过，实验脚本 Ruff 检查通过。
- 原始结果目录：`artifacts/preflight/gemini_three_frame_cost_20260906/`。
- `plan.json` / `requests/`：冻结样本、请求、图像指纹、参数和价格。
- `predictions.json` / `summary.json`：预测、native usage、未四舍五入费用和外推公式。
- `calls/`：脱敏请求、原始 SSE、账单查询和用量流水。
- `billing_audit.json`：三笔账单及实际请求核验结果。
- `test_inventory.json`：测试集目标计数与窗口大小分布。
- `run_experiment.py`：可审阅实验入口，已有执行锁，禁止重复触发本轮付费调用。

## 后续核查：Google 官方 API 与 Batch（仅估算，未实测）

2026-09-06 核对 [Google 官方价格](https://ai.google.dev/gemini-api/docs/pricing)：`gemini-3.8-flash` 在 2026-12-31 前的同步单价为输入 $0.75 / 百万 tokens、输出 $3.75 / 百万 tokens；Batch 单价为输入 $0.375、输出 $1.875。输出价格包含 thinking tokens。

以本次 OpenRouter 的实际 token 数保持不变计算：

| 项目 | OpenRouter 已测用量对应费用 USD | Google 官方同步估算 USD | Google 官方 Batch 估算 USD |
| --- | ---: | ---: | ---: |
| 三个目标 | 0.01028625 | 0.01028625 | 0.005143125 |
| 全 Testing，按样本平均 | 52.3981575 | 52.3981575 | 26.19907875 |
| 全 Testing，每次按样本最贵一次 | 53.4678975 | 53.4678975 | 26.73394875 |
| 全 Testing，每次 4075 输入 + 4096 输出 | 281.4371325 | 281.4371325 | 140.71856625 |

表格第一行只有 OpenRouter 一列是实际扣费；其余项目均为外推。官方接口的 token 计数、图像分辨率映射、返回长度和缓存情况可能不同，实际 Batch 账单不能直接断言等于上述数值。约 $150 可作为保守预算预留参考，仍不是无条件的费用上限。

[官方 Batch 文档](https://ai.google.dev/gemini-api/docs/batch-api)说明：Batch 异步处理独立请求，目标周转时间为 24 小时，很多任务会更快；支持标准请求配置、多模态和结构化输出。小批次可内联提交（总请求小于 20 MB），大批次适合使用 JSONL（单文件最大 2 GB），多模态输入可引用已上传文件。

对本项目应保持每条请求三帧因果输入、只预测一个目标、一次五头输出，并用 video_id/frame_id 标识结果。每条请求均不读取前一条预测，所以可批量提交，不依赖服务端完成顺序。批处理只是调度与计费改变，本身不增加用于识别的视觉或时序证据；这是基于接口行为的判断，不是本项目准确率实验结论。

本次 OpenRouter 账单的实际供应商已经是 Google AI Studio，实际模型为 `gemini-3.8-flash-20260902`。直接使用 Google 官方接口不能被当作换了更强模型。比较识别效果仍需匹配模型版本、相同图片、prompt/schema、thinking、图像分辨率及输出上限，再对照 GT 计算五头指标。三个目标只适合检查接口、用量及输出差异，不能证明两种调用方式在测试集上等效。

当前进程未配置 GEMINI_API_KEY、GOOGLE_API_KEY 或 GOOGLE_APPLICATION_CREDENTIALS；项目现有 API 凭据文件中也未找到 Google 官方 key。因此本轮没有提交官方同步或 Batch 请求，没有新增推理费用，也没有官方 Batch 的 F1/Accuracy 结果。实测需要可用且开通付费 Batch 配额的官方凭据。

## 后补：三目标 GT 评分（OpenRouter 已存预测，零新增调用）

对冻结的 VID103 / 25101、25126、25151 预测离线评分。五头 GT 均有效，每头分母都是 3；缺失任务 GT 才剔除，空标签集合不自动视为缺失。

| 任务 | micro-F1 | 集合完全匹配 Accuracy | 正确目标数 |
| --- | ---: | ---: | ---: |
| instrument | 100.00% | 100.00% | 3/3 |
| verb | 100.00% | 100.00% | 3/3 |
| target | 0.00% | 0.00% | 0/3 |
| ivt | 0.00% | 0.00% | 0/3 |
| phase | 100.00% | 100.00% | 3/3 |

三帧预测和 GT 均各自相同：预测 I=2、V=2、T=1、IVT=59、Phase=3；GT I=2、V=2、T=0、IVT=60、Phase=3。靶点把 gallbladder（胆囊）判成 cystic_plate（胆囊板），相应 IVT 错误。每帧五头全部正确为 0/3。

结果只代表 Training 中一个短片段的三个目标，不是完整测试集成绩，也不是官方 Batch 结果。详细 GT、mask、TP/FP/FN、预测文件指纹见 artifacts/preflight/gemini_three_frame_cost_20260906/scores.json。前文“未评估准确率”为最初费用实验的状态，本节为随后应用户要求补算。
