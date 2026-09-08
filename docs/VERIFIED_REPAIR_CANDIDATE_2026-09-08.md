# 当前保留的 Verifier／Repair 研究候选：发布与离线复现

日期：2026-09-08。本次发布的是**单模型补候选＋五席逐项视觉审核＋格式容错＋具体观察反馈**的研究代码、完整对照结论和可下载的离线重放资料。它是当前保留供继续验证的候选，**不是已经证明稳定最优的方案，也没有替换默认 H0 或完成 Tracker／Gate 集成。**

入口导航：[完整实测报告](REPAIR_REVISION_RESULTS_2026-09-08.md) · [聚合评分 JSON](experiments/repair_revision_aggregate_results_20260908.json) · [四目标离线重放包](experiments/verified_repair_candidate_20260908.json) · [GraphRAG 设计，尚未实现](GRAPH_RAG_VERIFIER_DESIGN_2026-09-08.md)。

## 现有实现与调用链

1. H0 使用真实因果图片 `[t-50,t-25,t]`，联合预测 Instrument、Verb、Target、IVT、Phase；开头或缺帧后仅用真实可用历史。本研究实验的 H0 为 OpenRouter `google/gemini-3.8-flash`。公开默认纯 API 入口仍按自己的冻结配置运行。
2. 同一 Gemini 看图片、当前预测与已有问题，提出少量额外候选。Python Coordinator 把它们与当前预测合并、去重，绑定完整 IVT 及其组件。提案不直接修改最终答案。
3. Grok、Qwen、GPT、Gemini、DeepSeek 五席分别看同一组当前图片，逐项返回评分、观察、图片引用和存在／反驳／不清楚判断。它们没有其他席的答案。
4. Python 检查 ID 与图片绑定、观察的语义范围和 JSON。无冲突的辅助字段及有限格式差异可以规范化；矛盾绑定、不同值重复字段、错误图片引用、局部未看到却声称整帧不存在等仍拒绝。
5. 五席均有效时，均分 ≥4 可支持新增、≤2 可支持删除；新 IVT 必要组件也须通过。独立 I／V／T 也按这些门槛局部增删，删除组件不得破坏保留 IVT；证据不足时保留当前标签，Phase 始终不修复。**修复的最终接纳是 Python 规则，不是另一位 LLM 自由重写整份答案。**
6. 未解决时，把五席原来的具体观察和问题清单交回提案模型，再提案、审核。观察只是模型意见，不是 GT。普通修订实验最多三轮；最新观察反馈对照最多两轮。`MODEL_PASS` 只表示池内命题满足规则，不保证池外没有漏检或答案正确。

最近实测的模型配置如下，仅记录当时精确版本与路由，不能保证未来可用性：

| 用途／席位 | 模型 | 通道 |
|---|---|---|
| H0／单提案 | `google/gemini-3.8-flash` | OpenRouter，Google AI Studio |
| Grok | `grok-4.6` | xAI；旗舰模型例外 |
| Qwen | `qwen3.8-flash` | 阿里云工作空间 |
| GPT | `openai/gpt-5.6-luna` | OpenRouter，OpenAI |
| Gemini | `google/gemini-3.5-flash-lite` | OpenRouter，Google AI Studio |
| DeepSeek | `deepseek/deepseek-v4-flash-vision-exp` | OpenRouter，Fireworks |

源码入口：[观察反馈对照](../scripts/run_evidence_feedback_trial.py)、[单提案／分布式提案对照](../scripts/run_repair_revision_trial.py)。核心规则：[recent_mean_panel.py](../src/surgical_agent/research/verification/recent_mean_panel.py)、[review_normalization.py](../src/surgical_agent/research/verification/review_normalization.py)、[review_feedback.py](../src/surgical_agent/research/verification/review_feedback.py)。

## 实测支持到哪里

下面三个批次来自四个 Training 视频，不能跨批按绝对分数排名，也不能当成 Testing 或完整 Pipeline 消融。

| 批次 | 同批 H0 IVT F1 | 单提案／反馈最终 IVT F1 | 修改收益与局限 |
|---|---:|---:|---|
| 开发 8 目标 | 16.67% | 24.00% | 2 个目标部分改善、0 改坏；收益没有继续随轮数增加 |
| 另 8 个确认目标 | 41.67% | 33.33% | 0 改善、3 改坏；开发收益未复现 |
| 再 4 个观察反馈目标 | 37.50% | 47.06% | 1 个目标部分改善、0 改坏；全部改善已在共享首轮发生 |

最新四目标反馈最终答案与共享首轮逐帧完全相同；原问题清单对照在第二轮多加错一个器械，具体观察反馈没有发生该错误。只有两个目标触发第二轮，因此**只能说它避免了一次对照组的错误新增，不能说第二轮反馈提高了 IVT**。

最新四目标的完整同批评分如下，数值为百分比；每个任务有效 GT 数均为 4。P／R／F1 为 micro，集合 Accuracy 要求该头集合完全相同。

| 任务 | H0 P／R／F1／集合 Accuracy | 共享首轮＝反馈最终 P／R／F1／集合 Accuracy |
|---|---|---|
| Instrument | 100／100／100／100 | 100／100／100／100 |
| Verb | 37.50／37.50／37.50／0 | 44.44／50／47.06／0 |
| Target | 57.14／57.14／57.14／25 | 57.14／57.14／57.14／25 |
| IVT | 37.50／37.50／37.50／0 | 44.44／50／47.06／0 |
| Phase | 50／50／50／50 | 50／50／50／50 |

本批修复后仍无 IVT 集合全对目标。各头标签错误数 FP＋FN 合计：H0 30、共享首轮 28、原反馈最终 29、具体观察反馈最终 28。两次提前 `MODEL_PASS` 的目标 IVT 也仍错误，不能用通过率替代真实评分。

三批合计实际 530 次 POST（包括失败试探），USD 原生账单合计 $2.376736994，另有阿里云 Token 估价 ¥0.912747；未知费用预留另记为 $0.0238012 与 ¥0.286426，不混入已知账单。最新四目标占 47 次 POST，USD $0.183892074、阿里云估价 ¥0.069048。**本次整理发布和离线重放不新增模型调用。**

## 下载后无需密钥的离线重放

按 [安装说明](API_EXPERIMENT_QUICKSTART_2026-09-06.md) 安装项目与依赖；本研究脚本额外依赖在 `requirements-prior-panel.txt`。可执行：

```powershell
python -m pip install -r requirements-prior-panel.txt
python scripts/replay_verified_repair_candidate.py
```

重放读取仓库内四目标包，使用真实 Coordinator、格式规范化、均分、接纳和反馈函数，核对保存的候选与输出。它不会连接模型服务，不要求手术图片或密钥。包内包含模型原始判断文本、模型预测与必要状态；**不包含图片、逐帧 GT 标签、凭证或原始请求 headers**。

发布前在独立、仅含已跟踪文件与明确发布清单的工作树验证：287 项原有相关测试、9 项公开重放测试均通过；发布 Python 文件 Ruff 通过；两个历史入口 `--help` 通过；真实归档的 4 目标、6 轮轨迹离线重放通过。此处共 296 项通过是本次发布检查，不把测试数量当成模型效果证据。

这能复现“保存的模型回答如何变成最终预测”，不能代替再次看图调用模型。评分部分由已审计的聚合 TP／FP／FN、有效目标数及全对数重算；公开包不含 GT，因而不能独立验证每一帧的标注。原本机独立 GT 审计已通过，聚合摘要保留其源文件哈希。完整视觉评测仍需自行取得数据集并使用独立评分器。

## 历史付费入口的运行边界

两个实验入口与其实际传递依赖一起发布，但它们是冻结的研究运行脚本。`prepare` 依赖本机历史 selection、此前计划和封账记录；数据集路径与原先的阿里云工作空间地址仍记录历史配置。**不能在干净 clone 中直接启动原先那次付费实验，也不能用新路径改写旧计划后声称仍是原实验。**

`--help` 和随附合成测试可离线运行。重新开展研究需要独立的新 selection、计划、路径与接口配置，明确样本和调用上限后再执行；不要误运行旧 `run_*` 入口以尝试重放。四目标公开重放入口是上面的 `replay_verified_repair_candidate.py`。

`artifacts/...` 在历史报告中表示本机证据位置，其原始调用、训练输出和图片没有随本次提交公开。公开的聚合摘要与四目标包在 `docs/experiments/`。本次只发布本候选的依赖、测试和说明，其他本地 Tracker／Gate 准备及历史输出保持原状。

## 后续研究的边界

保留已测试的工程修正，并把具体观察反馈作为待确认选项。若继续测 GraphRAG，先按 [位置与接口设计](GRAPH_RAG_VERIFIER_DESIGN_2026-09-08.md) 检查资料是否真有新增价值；固定候选后辅助审核，比较无检索、普通检索和图检索。该设计尚未接入或获得提升结果。
