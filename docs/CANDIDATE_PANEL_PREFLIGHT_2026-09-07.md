# 候选审核机制与五品牌接口预检（2026-09-07）

> 2026-09-08 文档发布说明：本页保留实验结束时的结果及 Git 状态；“未提交”等表述不是本次发布状态。完整机制索引与当前发布范围见 [汇总](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)。近期本地脚本、冻结源码与原始实验目录未随这次文档提交发布，文中相关复现命令需要本地产物。

后续更新：已自行找到 OpenRouter 的 DeepSeek 视觉模型，并完成四目标五品牌端到端开发测试。下文保留本次预检时的历史状态；最新结果见 [端到端测试报告](CANDIDATE_PANEL_END_TO_END_2026-09-07.md)。

本次没有完成五品牌端到端修复实验，也没有证明 IVT 提升。已完成新的隔离审核核心、离线测试与单目标接口预检。默认 H0 未改变。

## 凭据问题已解决

用户本次提供的最新阿里云工作空间 Key 与原本地文件不同。更新 `docs/aliyun_API KEY` 后，指定工作空间的 qwen3-vl-flash 文本探针 HTTP 200，31 输入 / 6 输出 token；随后三帧图像审核也成功。原先的 401 不能继续归因于最新 Key。没有把密钥写进源代码、报告或请求档案。

## 对流程的判断及实际实现范围

候选池加多模型审核可以作为实验，但不能把平均分称为正确率，也不能只挑一个最高分 IVT：同一帧可以有多个 IVT。五个模型看同样的图片仍可能共同看错；如果候选里没有正确答案，协调器无法凭评分补出正确答案。

冻结的 final-only H0 只含最终五头标签，并没有 top-k 候选池。新实验核心允许基座模型额外提出候选，Python 协调器统一编号，五个指定品牌分别对每个候选评分。候选提案只扩展候选池，不能直接修改输出。

`src/surgical_agent/research/verification/candidate_coordinator.py` 已实现：

- 候选按任务和标签 ID 返回评分映射，避免 100 项数组错位或少一项。评分必须是 1–5 的整数。
- 新输出不要求解释文字。若模型多写辅助字段，只提取完整合法的评分、记录忽略字段；这叫数值契约通过，不叫完整 JSON Schema 通过。原始响应保留。
- 必须齐备 Grok、Qwen、GPT、Gemini、DeepSeek 五个席位。缺少一个、漏候选 ID、出现非法分数，均不接纳该轮结果。
- 均分达到 4 支持存在，低于等于 2 支持删除，中间区域保留 H0。阈值是待验证的规则，不是校准后的概率。
- 多标签接纳，新增 IVT 的缺失组件也须被支持。删除 IVT 不自动重建整份 I/V/T；仍被保留 IVT 使用的共享组件不删除，Phase 保留 H0。
- 最多三次完整审核；修复器仅补候选。没有新候选就停止；接口失败保留上一轮有效结果。没有 GT 驱动的“直到正确”循环。

目前 `run()` 是可离线测试的回调式核心；尚未接上完整的真实 API 候选生成及三轮评估入口。`scripts/check_candidate_panel_providers.py` 只是图像接口预检，不能当作完整实验入口。

## 实际请求与结果

固定复用 Training 的 VID103 / 12501 历史冒烟输入，图像为 12451、12476、12501，原 H0 缓存复用。本次只审核 H0 形成的四个候选命题，未请求新候选，未让模型读取 GT。这不是独立新样本确认。

| 席位 | 实际模型 / 接口 | 结果 |
| --- | --- | --- |
| Qwen | 阿里云工作空间 qwen3-vl-flash，enable_thinking=false | HTTP 200，四项评分完整；2028 输入 / 39 输出 token |
| Grok | xAI grok-4.20-0309-non-reasoning | HTTP 200，四项评分完整，reasoning_tokens=0 |
| GPT | OpenRouter openai/gpt-4.1-mini，固定 OpenAI 路由 | HTTP 200，四项评分完整，reasoning_tokens=0 |
| Gemini | OpenRouter google/gemini-2.5-flash-lite，固定 Google AI Studio，reasoning disabled | 第一次嵌套 JSON Schema 返回空 scores；第二次仅改为 JSON 对象模式，四项评分完整，reasoning_tokens=0 |
| DeepSeek | 同一阿里云工作空间 deepseek-v4-flash-vision-exp | HTTP 404，model_not_found / Model not exist |

Grok 选用的是该账户可用的非思考型号，不能据此声称它是小参数模型。GPT 本次未触发此前的图像拦截，但单次成功不代表其他目标都不会被拦截。Gemini 的一次输出格式对照支持继续检查适配问题，尚不能证明空评分问题已普遍消失。

DeepSeek 官方已有图像模型，不能笼统说 DeepSeek 不支持视觉。但该阿里云空间的模型列表没有此视觉版，且实际调用 404。普通 DeepSeek 文本模型或 OCR 模型不能冒充第五位视觉语义审核者。参见 [DeepSeek 官方视觉文档](https://api-docs.deepseek.com/guides/vision/)。

共 6 次图像 POST（五席位各一次，Gemini 格式对照再一次），另有上述一次成功文本探针。没有重跑 H0，没有发布任何修复。

- OpenRouter 实际返回费用合计 $0.0015262，包含 Gemini 空评分失败的已计费请求。
- Grok 实际返回 21,184,000 USD ticks，即 $0.0021184；换算依据 [xAI Cost Tracking](https://docs.x.ai/developers/cost-tracking)。
- 已知美元实扣合计 $0.0036446。阿里云仅返回 token 用量，没有返回实扣金额；不把缺失账单写成零费用，不与美元合计混算。DeepSeek 404 未返回费用。

## 测试、版本和证据

新核心 12 项测试通过；相关旧核心与接口回归合并运行 40 项通过。增加请求适配测试后，新核心与新适配共 14 项通过（不与前述数量相加为独立测试总数）。四个新增 Python 文件 Ruff 通过。

核实 HEAD 为 `7a44312477b9415c34c6f307ef943e9f087d9dff`。五个冻结 H0 入口/配置/提示/契约文件与 `25fee9485a8f740aadf7921d5312e9c488955cbb` 对比无差异。本次没有提交或推送，没有修改 Tracker 训练产物及其既有删除状态。

证据目录：

- `artifacts/preflight/candidate_panel_provider_check_20260907/aliyun_latest_key_probe.json`
- 同目录 `aliyun_models_latest_key.json`、`visual_preflight_audit.json`
- `artifacts/preflight/candidate_panel_visual_interfaces_20260907_v1/`：五个原始响应、脱敏图像请求、固定样本和源文件哈希。
- `artifacts/preflight/candidate_panel_visual_interfaces_20260907_v2/`：仅 Gemini JSON 对象模式对照。

v1 驱动在评分验证失败时未将 usage 提升至响应摘要顶层，但完整原始 body 已保存费用。后续驱动已修正计费记录先于评分验证；独立 audit 从所有原始 body 汇总，包含该失败费用。旧响应未覆盖。

## 下一步的最小实验条件

首先补齐可调用的 DeepSeek 视觉接口，通过相同图片与数值契约预检。未经补齐，不把四席平均包装为五席实验，不用其他品牌静默替代。

接口齐备后，先对既有开发样本固定复用 H0，冻结候选扩展提示、评分阈值和每轮输入，再接通完整 driver。候选生成、三轮评分及补池均需单独记账。小批量结果应逐头统计有效 GT 数、F1、集合 Accuracy、改对、改坏、部分改善、无收益替换，以及候选覆盖率、错误候选放行率和新增费用。缺失 GT 按任务 mask 排除。Testing 不用于选择方案。

当前这次预检不产生上述语义指标；四个接口能返回数字不代表修复有效。
