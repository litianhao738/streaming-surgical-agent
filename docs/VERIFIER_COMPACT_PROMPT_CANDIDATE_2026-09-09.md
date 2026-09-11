# Verifier 精简提示候选：固定输出契约

当前状态：后续用户已明确选择精简 v2 为默认，接入版本为 `parallel-phase-repair-v1.1.0-compact-prompt`；见[默认运行说明](DEFAULT_REPAIR_PIPELINE_2026-09-09.md)。本文件下述“候选/未启用”描述保留原离线实验阶段的历史状态。

后续已完成真实对照，发现并修正 Qwen Phase JSON 模式缺少 `JSON` 单词的问题，候选升级到 v2；原 v1 文本与下述离线计数保留为历史记录。实测结果见 [配对测试报告](VERIFIER_COMPACT_PROMPT_TRIAL_2026-09-09.md)：输入减少，四头 F1 持平，未升级默认。

2026-09-09，版本 `verifier_compact_wording_v1_candidate`。已完成离线实现与长度审计，尚未证明准确率提高，也未替换默认或历史冻结流程。没有调用推理 API。

## 修改范围

四头审核把原 `instructions`、`proposition_semantics` 和 `output_contract_clarification` 的重复说明合并为一段按“命题含义、时间证据、评分、整帧否定、输出”组织的指令。保留独立 `label_boundaries` 原文，保留 null 定义，不增加案例、图像、候选或输出字段。

Phase 只改 `instructions` 的措辞，继续根据活动、目的和位置判断，保留阶段持续性、单个器械不足以证明阶段、当前帧引用、null 弃权等要求。七阶段定义不变。

两分支的完整输入内 Schema 和 API `response_format` 都原样保留；Gemini 四头仍用 `rows`，其他席仍使用其原协议。评分枚举、观察文字长度、解析器、五份有效审核要求、接纳阈值和 Phase 投票规则不变。图片字节/归档哈希、detail、顺序、帧号、候选名称与组件、模型、路由、reasoning、max_tokens 全部保留。

## 离线证据

使用 `artifacts/preflight/parallel_phase_eight_20260909_v1` 已保存的 40 份四头和 40 份 Phase 请求。逐份还原修改前的首段文本后，整个请求必须与原请求完全相同；另外逐字段核对文本内证据与 Schema 没有变化。

| 文本分词估算 | 原文本 Token 总量 | 新文本 Token 总量 | 减少 | 每请求最少减少 |
|---|---:|---:|---:|---:|
| 四头 / cl100k_base | 105395 | 97515 | 7.48% | 197 |
| 四头 / o200k_base | 105984 | 97904 | 7.62% | 202 |
| Phase / cl100k_base | 29520 | 28440 | 3.66% | 27 |
| Phase / o200k_base | 29520 | 28440 | 3.66% | 27 |

这些是整个提示文本包（含内嵌 Schema）的代理分词计数，不含图片、提供方消息封装或 API 额外 Schema 的计费。未验证它们是五个部署模型的准确分词器，因此不构成五家实际输入 Token、总费用或延迟保证。图片等固定开销使完整请求的节省比例小于文本部分；固定 JSON 结构也不固定 observation 长度或隐藏推理量。

审计文件：`artifacts/preflight/verifier_compact_prompt_20260909_v1/audit.json`，保存 80 份请求的源路径、前后文本哈希、分词计数和审计限制。

## 使用方式

模板位于 `src/surgical_agent/research/verification/prompts/compact_interaction_v1.txt` 和 `compact_phase_v1.txt`。纯函数 `compact_review_wire(body, branch, count_tokens=...)` 接收原构造器生成的请求，返回副本；branch 为 `graph` 或 `phase`。调用者提供目标模型的文本 Token 计数函数，若改后超过原预算则抛出错误，不发送请求。使用代理计数函数时，这个限制也仅对该代理成立。

这是独立候选接口，没有修改现有 runner 的默认请求。正式接入前，应使用准确的提供方计数确认硬预算，再在固定 Training 图像和候选池上做配对对照，保持模型与思考强度不变。比较 IVT F1、Verb/Target 误报、有效审核率、总 Token、费用和延迟；Testing 不参与选提示。不能直接把旧评分当作新提示结果。

既有 `docs/VERB_PROMPT_TRIAL_2026-09-09.md` 中追加 Verb 观察提示未通过综合标准，因此本候选仅提出“重组重复说明”的待验证假设，不宣称新增了已验证的准确率收益。官方也建议通过评测监控提示修改的表现：[OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering)。

离线复现：在包含 tiktoken 的 Python 环境运行 `tools/audit/audit_compact_prompt.py --output <新的审计路径>`。本次 tiktoken 仅安装到 `tmp/verifier_prompt_token_tools`，未修改项目依赖。
