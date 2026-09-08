# Verifier / Repair：代码修订与真实 API 对照（2026-09-08）

> 2026-09-08 文档发布说明：本页保留实验结束时的结果及 Git 状态；“未提交”等表述不是本次发布状态。完整机制索引与当前发布范围见 [汇总](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)。近期本地脚本、冻结源码与原始实验目录未随这次文档提交发布，文中相关复现命令需要本地产物。

已落实证据约束、失败回退和接口兼容修订，但**没有解决 IVT 语义准确性问题，不替换默认 H0**。
最终四个完整标注 Training 目标中，新审核仅改善一个目标的 Verb；Target、IVT 没有提升。
另测“不看 H0 的独立候选生成”，最终标签集合相同，候选覆盖更低。不能宣称五模型投票已验证答案正确。

## 代码与实际链路

默认 `scripts/run_dataset_api_pipeline.py`、默认配置和原始 H0 prompt 保持冻结。
本轮研究入口 `scripts/run_complete_gt_semantic_trial.py` 实际执行：

```text
原版 Qwen H0（三张真实因果图，final-only 五头）
  → Gemini 3.8 Flash 提供候选（同三张图，low reasoning）
  → Python 合并 H0 / 新候选 / IVT 必要组件，绑定合法 ID
  → Grok、Qwen、GPT、Gemini、DeepSeek 独立视觉审核
  → Python 验证逐项证据，计算平均分、处理冲突与依赖
  → 通过的局部修改；其余保留 H0；Phase 保持 H0
```

审核者不知道哪些候选来自 H0，也不接收 GT、任务 mask 或其他审核者的答案。
当前五席与上一轮相同：Grok `grok-4.20-0309-non-reasoning`、阿里云 `qwen3-vl-flash`、
OpenRouter `openai/gpt-4.1-mini`、`google/gemini-2.5-flash-lite`、
`deepseek/deepseek-v4-flash-vision-exp`（Fireworks）。Grok 的“小模型”身份未获证实。
仅候选模型更换为 OpenRouter `google/gemini-3.8-flash`，固定 Google AI Studio。

| 实际问题 | 修订 | 不能由此推出的结论 |
|---|---|---|
| Target 的“存在”易被当成组织可见 | 显式区分操作对象、背景可见、邻近组织受力 | 不能保证模型找对组织 |
| 数字分与观察依据脱节 | 每个候选仅一个 rating，绑定 finding、scope、图片索引和 observation | 结构一致不证明文字描述真实 |
| 一把器械未做某动作被当成整帧不存在 | 局部否定不能支持删除整帧标签；不明确则保留 | 模型声称检查全帧也不是客观证明 |
| 高平均分掩盖明确反证 | 同一候选同时出现 ≥4 支持与 ≤2 反驳时阻止该项修改 | 有益修改也可能被挡住 |
| 不合格证据仍参与接纳 | 任一席该项证据不合格，不能用其余席重算平均后接纳 | 保守回退不是修复成功 |
| JSON 模式重复到截断；嵌套严格 Schema 返回空对象 | Gemini 审核采用统一 rows 数组 + JSON-object；本地检查精确 ID、重复、缺项及字段 | 仍可能发生内容或语义契约错误 |
| 只按时间抽样，未提前核实交互 GT | 完整标注补充入口强制先检查五头 mask，缺少 evaluation 行也排除 | mask 可用不等于模型看得见全部标注关系 |
| 列表次序变化被日志当成修改 | 修改标志与独立审计按标签集合比较 | 不计为“无收益替换” |

新核心为 `src/surgical_agent/research/verification/semantic_coordinator.py`。
候选级错误隔离不会直接抹掉其他候选的合格结果。旧 scores-only 接口仍可复现；
`candidate_coordinator.run` 增加可注入审核器和 1–3 轮上限，旧默认参数不变。
新增 IVT 仍检查必要组件；删除 IVT 不自动删除所有组件，保留 IVT 需要的共享组件不会被删除。
不强制用 IVT 投影重建独立 I/V/T 头。观察文字上限为 1000 字符。

多轮仅在尚有低支持/分歧且候选能扩展时继续，最多三轮，无新候选即停止。
这是停止条件，不是“判断答案已经正确”的 oracle。完整 GT 对照特意固定一轮，隔离审核变化。

## 实验与保留的失败

所有目录位于 `artifacts/preflight/`，原始 request / response / usage / 模型身份 / 图片哈希均保留。
请求日志移除图片字节但保留 SHA256；密钥不进入请求日志或代码。

| 目录 | 请求数 | 用途与结果 |
|---|---:|---|
| `semantic_candidate_20260908_v1` | 0 | 准备记录，尚未调用时修订了代码，未覆盖 |
| `semantic_candidate_20260908_v2` | 70 | 四个时间采样目标，实际 2/1/2/2 轮；缺 Verb/Target/IVT GT，只能做接口与 I/P 诊断 |
| `semantic_review_contract_20260908_v1` | 20 | 相同首轮候选重测，澄清枚举、图片索引、局部/全帧范围；仍有 Gemini 重复输出截断 |
| `semantic_gemini_strict_probe_20260908` | 1 | 嵌套严格 Schema 返回空对象，失败保留 |
| `semantic_gemini_rows_probe_20260908` | 1 | 数组严格 Schema 仍返回空对象，失败保留 |
| `semantic_gemini_object_rows_probe_20260908` | 1 | 数组 JSON-object 返回完整字段，仍有一项局部否定被阻止 |
| `semantic_complete_gt_20260908_v1` | 48 | 四个完整五头 GT 目标：H0 + 共享候选 + 旧/新审核，一轮 |
| `semantic_blind_candidate_20260908_v1` | 24 | 同四目标、缓存 H0，独立看图生成候选 + 相同新审核，一轮 |

第一次新目标选择漏做 mask 预检是本轮实验设计失误，不能将其缺失的交互指标解释为零分。
后续按规则选 VID103 / VID23 时间线 1/4、3/4 附近最近的完整标注目标，等距选较早者。
最终固定 `VID103/14576、37226；VID23/9801、31001`。最后一个原时间锚点 31026 缺交互 GT，
按可用性规则选到 31001。选择只检查 mask，未根据 GT 标签或分数挑帧。Testing 未参与。

独立候选对照复用了已经算过分的四个 Training 目标，因此明确属于开发集消融，不能包装成新验证集。
没有训练或使用 OOF Tracker，也没有新增记忆、RAG 或新审核席位。

## 四个完整 GT 目标：结果

每格为 **micro-F1 / 集合 Accuracy（%）**；每个头有效目标数均为 4。

| 任务 | H0 | 原数字审核 | 新证据审核 | 独立候选 + 新审核 |
|---|---:|---:|---:|---:|
| Instrument | 76.92 / 25 | 76.92 / 25 | 76.92 / 25 | 76.92 / 25 |
| Verb | 76.92 / 25 | 71.43 / 0 | **85.71 / 50** | **85.71 / 50** |
| Target | 16.67 / 0 | **30.77 / 0** | 16.67 / 0 | 16.67 / 0 |
| IVT | 14.29 / 0 | 14.29 / 0 | 14.29 / 0 | 14.29 / 0 |
| Phase | 75 / 75 | 75 / 75 | 75 / 75 | 75 / 75 |

这里原数字审核的 4 目标结果包含一次 GPT HTTP 403 后的 H0 回退，不是四个成功审核样本。
新审核四个目标的 20 个请求均解析成功；逐项证据校验仍可能失败，相关修改被阻止。
另存 `successful_transport_paired_comparison.json`，只比较两边都取得完整审核响应的 3 个目标：
H0 / 原审核 / 新审核的 Verb F1 为 80 / 72.73 / 90.91，Target 为 20 / 36.36 / 20，
IVT 均为 18.18；Target 和 IVT 集合 Accuracy 仍全为零。结论仍是取舍，不是新方案全面获胜。

| 相对 H0 的目标级变化 | 原数字审核 | 新证据审核 | 独立候选 + 新审核 |
|---|---:|---:|---:|
| 部分改善 | 1 | 1 | 1 |
| 整帧所有有效头由错到全对 | 0 | 0 | 0 |
| 改坏 | 1 | 0 | 0 |
| 等损失的无收益替换 | 0 | 0 | 0 |
| 语义不变 | 2 | 3 | 3 |

共享候选的一轮对照中，有益标签修改机会 16 个：原审核与新审核均仅放行 1 个（6.25%），
但不是同一项。错误修改机会 22 个：原审核放行 1 个（4.55%），新审核放行 0 个。
这些全样本运行指标包含上面的旧审核失败回退，不能当作无失败的独立模型能力估计。

新审核在 38 个候选中，仅 14 个取得五席全部合格的逐项证据；24 个被不合格证据阻止，
另 5 个合格但有明确支持/反驳冲突，被保留处理。拦截过多本身就是当前局限。
其中 Gemini 行格式解决了整份空对象问题，但完整 GT 轮仍出现 3 个字段契约错误；
独立候选轮无此类字段错误，仍有局部否定、评分/结论冲突。不能称接口永不失败。

## 剩余问题在哪里

1. **候选漏掉答案。** 带 H0 的候选池仅覆盖 8 个 GT IVT 中的 2 个；不看 H0 的候选池仅覆盖 1 个。
   独立生成没有改善这批样本，不能把主要问题简单归因于被 H0 引导。
2. **动作对了，关系仍错。** VID103/37226 新增 `retract` 对了，但审核文字仍常将其对象说成 gallbladder。
   GT 中相关关系是 `grasper/retract/liver`，另一个关系是 `hook/dissect/gallbladder`，二者均未进入该候选池。
   因此 Verb 提升不等于 IVT 修复。这里 GT 确实有 retract；此前 grasp 案例不能推广为“都改成 grasp”。
3. **新审核会挡住有益修改。** 原审核在该目标新增 gallbladder Target 是有益的；新审核未达到接纳条件。
   防误改与召回存在实际取舍，不能只展示错误修改放行率下降。
4. **视觉误判与语义契约失败共存。** 多个模型仍会把局部否定用于整帧，或在输出里自相矛盾。
   Schema 和程序能拒绝不可靠修改，不能替模型找到漏看的器械或判断正确组织。

本轮没有发现足以解释上述 IVT 低分的 ID 映射/评分器错位证据。
原始 H0 的低分也是四个小目标上的结果，不推翻此前更大范围内保留该版本的基线决策。
不能据此断言全部是模型能力问题，图像可辨性、标签定义和候选/审核策略仍有共同影响。

## 费用、验证与复现

本轮合计 **165 次 POST**：164 次 HTTP 200，1 次 HTTP 403；HTTP 200 不等于语义/Schema 合格。
有原生计费依据的新增金额为 **OpenRouter $0.27879919 + xAI $0.16688455 = $0.44568374**。
阿里云按 token 保守估算 **¥0.243952**，并非账单金额；另保留 OpenRouter **$0.0241336 未知费用预留**，
不是已确认收费。以上均不包含历史实验费用，历史占用逐轮承接。
总额未超本轮新增 OpenRouter $1.20 / xAI $0.50 / 阿里云 ¥1 的限额。

`artifacts/research/semantic_verifier_repair_20260908/combined_audit.json` 核对全部账本连续承接。
各完整实验的 `independent_audit.json` 独立复算五头 TP/FP/FN、集合 Accuracy、图片一致性、
候选绑定、Gemini 行到候选 ID 映射以及原生计费。预测文件在算分前冻结，算分后哈希不变。
`round_comparisons.json` 中为兼容历史报告保留三个状态快照；真实调用轮数以 `history` 为准，不把填充快照当真实轮次。

最终 **53 项相关测试通过，Ruff 通过**。单元/集成检查覆盖局部否定、显式冲突、缺字段、候选 ID 重复/缺失、短历史索引、
共享组件、轮数上限、完整 mask 选样、失败费用保留和冻结 H0；Ruff 检查本轮修改文件。
付费实验结束后仅修正日志/审计对标签顺序的处理，以及 Gemini 行格式说明中的字段数量笔误
（实际为 candidate_id 加五个证据字段）；没有重写已冻结的请求或预测。
没有修改默认 H0，没有提交/推送，也没有清理用户其他本地文件、Tracker 训练输出或历史实验。

离线复核（不调用 API）：

```powershell
.venv-p2/Scripts/python.exe tools/audit/audit_semantic_candidate.py artifacts/preflight/semantic_complete_gt_20260908_v1
.venv-p2/Scripts/python.exe tools/audit/audit_semantic_candidate.py artifacts/preflight/semantic_blind_candidate_20260908_v1
.venv-p2/Scripts/python.exe tools/audit/summarize_semantic_trials.py
```

若以后明确要复现付费对照，使用新目录，先 prepare，再显式 execute：

```powershell
.venv-p2/Scripts/python.exe scripts/run_complete_gt_semantic_trial.py prepare --output artifacts/preflight/semantic_review_next --previous artifacts/preflight/semantic_blind_candidate_20260908_v1
.venv-p2/Scripts/python.exe scripts/run_complete_gt_semantic_trial.py execute --output artifacts/preflight/semantic_review_next
```

这是同一固定四目标的复现，不是独立新样本。每次 prepare 保存代码与请求元数据；
已启动过的目录拒绝付费重放。旧版本源码另保存在对应 `frozen_source/`，不会用新代码冒充旧实验。

本周建议保留 H0 完成主实验与复现材料。新增代码防护和失败分析可以保留，
但目前不值得继续靠调阈值、增加投票轮数或复杂模块追求这几个开发帧的分数。
