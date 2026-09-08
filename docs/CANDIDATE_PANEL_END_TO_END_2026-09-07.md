# 五品牌候选审核端到端开发测试

> 2026-09-08 文档发布说明：本页保留实验结束时的结果及 Git 状态；“未提交”等表述不是本次发布状态。完整机制索引与当前发布范围见 [汇总](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)。近期本地脚本、冻结源码与原始实验目录未随这次文档提交发布，文中相关复现命令需要本地产物。

后续针对 Target 下降完成了同图、同候选的 30 次诊断请求，见 [Target 交互诊断](TARGET_INTERACTION_DIAGNOSIS_2026-09-08.md)。本文保留原端到端实验结果。

本次已完成 4 个固定 Training 开发目标的真实 API 测试：复用冻结 H0 → 基座补候选 → 五品牌评分 → Python 接纳 → 有问题时再补候选，最多三轮。结果不支持替换默认 H0：一个器械标签改对，两个 Target 标签改坏，IVT 无改善。

## DeepSeek 型号与渠道已找到

用户指定的是 DeepSeek 品牌，型号和可用渠道应由助手检索。先前仅因阿里云空间未上架就要求用户补接口，检索不充分。

官方有 [DeepSeek V4 Flash Vision Exp](https://api-docs.deepseek.com/guides/vision/)，[OpenRouter 也已上架](https://openrouter.ai/deepseek/deepseek-v4-flash-vision-exp)。该模型支持图像，公开说明为 284B 总参数、13B 激活参数的 MoE；不能把它写成总参数只有 13B 的模型。

使用现有 `docs/API.txt` 的 OpenRouter Key 即可。DeepSeek 官方托管路由被账户现有 paid-model-training 数据策略排除，返回路由 404；保留账户设置后，选择 Fireworks 托管的同一 DeepSeek 模型，三帧审核 HTTP 200，完整评分，reasoning_tokens=0。没有换成其他品牌，没有修改账户隐私设置。

最终五席：

| 席位 | 模型 | 渠道 |
| --- | --- | --- |
| Grok | grok-4.20-0309-non-reasoning | xAI 直连 |
| Qwen | qwen3-vl-flash，enable_thinking=false | 用户指定阿里云工作空间 |
| GPT | openai/gpt-4.1-mini | OpenRouter，OpenAI 固定路由 |
| Gemini | google/gemini-2.5-flash-lite，reasoning disabled | OpenRouter，Google AI Studio 固定路由 |
| DeepSeek | deepseek/deepseek-v4-flash-vision-exp，reasoning disabled | OpenRouter，Fireworks 固定路由 |

Grok 是账户可用的非思考型号，尚无依据称其为小参数模型。全部席位实际收到同一目标的相同三帧图像。

## 协议与代码

入口：`scripts/run_candidate_panel_trial.py`。协调核心：`src/surgical_agent/research/verification/candidate_coordinator.py`。

H0 原始结果来自 `artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/predictions.json`，SHA256：`69c5c6b5b4d88fe34643994fb72d11732caf4f6004d536d6860e1a54f3ac0a11`。本次没有重新请求 H0，也没有把候选生成提示替换进默认 H0。

额外候选由同一基座 `qwen/qwen3.8-max-0902`、OpenRouter Alibaba 路由生成，temperature=0、reasoning=low、max_tokens=4096。五个审核者最多输出 1536 token，独立读取候选 ID 和图像，不知道哪些候选属于 H0 或上一轮接受结果，也看不到其他审核者的评分。协调器按五席算术平均处理多标签；均分≥4支持新增，≤2支持删除，区间内保留 H0。共享组件保护与最终集合上限仍生效，Phase 固定。

修复调用只能补充候选，不能直接发布修改。只有新候选才触发下一轮；没有新候选就停止，不强行重复至三轮。每次 JSON 首字段包含真实的学术医疗视频分析说明。缺失 GT 不进入提示，最终统一按任务 mask 排除。

样本固定为 VID103 / 18426、33676；VID23 / 13076、27726。它们是此前已分析的开发样本，不是独立确认集。最后一个样本仅 Instrument、Phase GT 有效；其新增 Target 未计入正误。

## 请求适配问题与处理

首次端到端尝试的 4 个基座请求全部被 Alibaba 拒绝，原因是数组 Schema 含不支持的 `uniqueItems`，没有生成候选、没有进入审核。保留在 `candidate_panel_development_20260907_v1`。这是本次新增接口适配代码的问题，不能归因于模型视觉能力。

修正版只从 API 结构化输出 Schema 去掉此不支持字段；本地仍严格拒绝重复 ID，文本中的完整 Schema 也保留。加入 400/401/402/404 参数或配置错误后停发的处理。候选提示、图片、模型和接纳阈值没有根据 GT 调整。v1 的 $0.6620765 未知费用预留完整带入 v2 预算，没有清零。

修正版路径：`artifacts/preflight/candidate_panel_development_20260907_v2`。40 次真实请求全部 HTTP 200：基座 10 次（4 次初始补池、6 次后续补池），五个审核者各 6 次，共 30 次审核。

四个目标分别完成 2、2、1、1 轮，共 6 轮有效五席审核。全部因没有新候选停止，无第三轮调用。两个第二轮均未改变第一轮已发布的输出；报告中的第三轮快照只是最终结果延用，不能说实际跑过第三轮。

## 结果：H0 → 最终输出

F1 是各头 pooled micro-F1；Accuracy 是该头预测集合与 GT 集合完全一致的比例。

| 头 | 有效目标 | F1 | 集合 Accuracy |
| --- | ---: | ---: | ---: |
| Instrument | 4 | 93.33% → 100.00% | 75.00% → 100.00% |
| Verb | 3 | 50.00% → 50.00% | 33.33% → 33.33% |
| Target | 3 | 50.00% → 42.86% | 0.00% → 0.00% |
| IVT | 3 | 16.67% → 16.67% | 0.00% → 0.00% |
| Phase | 4 | 100.00% → 100.00% | 100.00% → 100.00% |

按每帧有效任务的集合对称差统计：改善 1、改坏 2、无收益替换 0、不变 1、改善与恶化混合 0。改善帧的器械集合从不全对变为全对，额外的“改善但未全对”帧为 0。该帧没有有效 IVT GT，不能把它写成一帧五头全部修复成功。

可算分标签修改共 3 项：改对 1 个器械标签，新增 2 个错误肝脏 Target；另有 1 个 Target 新增因该头缺失 GT 而不评价。没有 IVT 修改。

按每个唯一候选相对 H0 的一次 ADD/REMOVE 机会统计（不是按轮累计重复票数）：

| 头 | 有益修改放行 / 机会 | 错误修改放行 / 机会 |
| --- | ---: | ---: |
| Instrument | 1/1 | 0/7 |
| Verb | 0/3 | 0/4 |
| Target | 0/3 | 2/5 |
| IVT | 0/5 | 0/7 |

其中“机会”仅涵盖进入审核池的候选；完全没进入候选池的漏检不在该分母内，另见覆盖率。

## 机制诊断：事实与推断

**事实 1：正确候选仍然缺失。** 三个有 IVT GT 的目标共 6 个正确标签，审核池合计仅覆盖 1 个，与 H0 覆盖数相同，5 个遗漏没有被补池找回。增加投票无法从池中选出不存在的答案。即使能完美删去所有错误 IVT，本池也没有能力得到任一目标完整的正确 IVT 集合。

**事实 2：多模型高分不能当作语义正确。** 两个新增的 `target_8=liver` 均为 GT 中的假阳性。对应第一轮均分为 4.8、4.6；第二个目标第二轮五席全部给 5 分。提高平均分阈值不能普遍解决此类共同错误。

**代码层面的待验证原因：审核命题仍有歧义。** `body_for()` 的总任务写的是候选在整帧中是否存在，而 Target 候选名称只是 `liver`；这容易被理解成“画面里有没有肝脏”。尽管已附加明确的全局 tool-contact 类别边界，它未必足以让每个单独评分都针对“器械正在交互的目标”。高分的具体成因不能仅靠无解释的数字响应确认，不能声称已经证明模型一定采用了背景存在判断。

**事实 3：审核没有纠正既有错误 IVT。** 候选池内相对 H0 的 5 个有益 IVT 修改机会都是删除既有假阳性，最终一个都没执行。部分既有错误得到高分，部分降到 2.4 等不确定区间，被固定保留策略留下。这既涉及评分准确性，也涉及保守接纳规则；不能将其全部归为解析失败。

**事实 4：后续循环未产生最终增益。** 两个第二轮没有改变第一轮输出，额外补池也没有提高正确 IVT 覆盖。本次不能用“运行到多轮”证明 reflection 有效。

因此继续保留默认 H0。若后续再做一个最小实验，优先把 Target/Verb 的每条命题绑定到实际器械交互，单独验证“可见组织”与“交互目标”混淆；需要固定新 Training 开发目标和共享候选对照。此建议尚未实现或启动新付费实验。不继续堆模型、加轮数或按本次帧号写特判。

## 费用与复现

修正版 40 次调用：OpenRouter 原生费用 $0.18128656，Grok 原生费用 $0.01833040，已知美元合计 $0.19961696。Qwen 6 次共输入 15,168 / 输出 644 token，接口没有返回实扣金额；预算按保守费率占用 ¥0.021608，这不是阿里云实扣账单。

本轮找 DeepSeek 的成功冒烟另花 $0.00038214。因此本轮已知美元费用合计 **$0.19999910**，约 $0.20。另有 1 次 OpenRouter 路由 404 和 4 次 Alibaba Schema 400 未返回费用，不能当成已确认零费用。全轮 POST 数为 46（2 次 DeepSeek 预检、4 次失败端到端请求、40 次修正版请求）。未计入上一轮已报告的费用。

相关测试合并 45 项通过；Schema 适配修正后，核心及新接口测试 17 项通过。相关 Python 文件 Ruff 通过。独立审计确认：6 轮五席均分正确、所有调用图像一致、五头 mask 后 TP/FP/FN 与集合 Accuracy 一致、原生账单汇总一致。

主要证据：`plan.json`、`frozen_source/`、`calls/`、`targets/`、`predictions.json`、`comparison.json`、`round_comparisons.json`、`candidate_coverage_and_edits.json`、`mechanism_diagnosis.json`、`budget.json`、`independent_audit.json`。最终预测 SHA256：`8027394d7c35fdc109a83001ee53e326ab5157512c909b9dde1bbcd6f717c5d4`。

离线审计命令：

```powershell
.venv-p2/Scripts/python.exe tools/audit/audit_candidate_panel.py artifacts/preflight/candidate_panel_development_20260907_v2
```

当前 HEAD 仍为 `7a44312477b9415c34c6f307ef943e9f087d9dff`。默认五个 H0 文件与最初冻结提交对比无差异。本次没有提交、推送、修改 Tracker 产物或删除历史实验；密钥未加入 Git。
