# 必要输出、LLM Repair 与第二轮续跑

> 2026-09-08 文档发布说明：本页保留实验结束时的结果及 Git 状态；“未提交”等表述不是本次发布状态。完整机制索引与当前发布范围见 [汇总](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)。近期本地脚本、冻结源码与原始实验目录未随这次文档提交发布，文中相关复现命令需要本地产物。

2026-09-08，本轮按用户要求检查母体 Repair，并实现差分修复、执行额外一轮的小对照。
没有重新调用 H0；使用上一实验的 Gemini/OpenRouter H0、原三帧和第一轮审核记录。

## 母体论文和公开代码分别做什么

用户 PDF 的第 5 页 Fig.2 和第 6 页 Reflection 段描述：程序先检查 IVT 组件闭包、
Phase/IVT 一致性，再针对有问题的任务重新调用专家，之后重新检查，约束通过或达到轮数上限时停止。
这是“规则检查＋模型修订”，不是由 LLM 或 GT 保证答案正确。

已核实原 PDF SHA256：`30abb5d2caba3f9068e88187692bcfec5c5d6b4c77fce6e60ac166d4479a95f0`。
本轮检查并渲染了完整相关方法页，未修改原 PDF。

作者公开代码固定提交 `3028fce4a8eb7ba181314b593c0336ad69b9b67b` 中：

- `IVTVerifyExpert` 调用基础多模态模型，看三帧并筛选候选。
- `reflection.py:model_repair()` 再调用基础多模态模型，输入三帧、当前答案、问题列表、类别选项和可选规则/先验，
  输出整份五头答案及报告。
- `orchestrator.py` 在 reflection/model_repair 开启、`max_rounds>0` 且 issues 非空时执行这一次模型修复；
  然后映射选项、重新施加规则，再接纳非空修订。没有额外独立视觉裁判证明改好了。
- 虽然变量叫 `max_rounds`，这份公开实现的模型修复是一次 `if` 分支，
  **没有完整实现论文文字里的多轮定向专家循环**。issues 中也包含流程事件，不能等同于已检测到视觉错误。

[作者模型修复代码](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/reflection.py#L248)，
[调用与接纳代码](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/orchestrator.py#L445)。
本轮对本地四个关键作者源码快照的 SHA256 与下载清单核对一致，未执行作者代码。

## 本项目的最小优化

H0 已经只输出最终五头标签，因此保持完整本体、Schema、三帧和原提示词。
新增差分模块 `src/surgical_agent/research/verification/delta_repair.py`：

1. LLM Repair 只返回 `changes`，不重复未修改标签、完整预测和报告。
2. 每个修改仍绑定候选 ID、ADD/REMOVE、评分、判词、图片引用、范围和一句观察。
   没看清、没有返回某标签，均不能被程序解释为删除。
3. 可从完整本体提出旧候选池外的新标签；程序验证 ID、实际差异和组件依赖，重建临时答案。
4. 若有合法修改，五个原审核模型只复核实际修改及新增 IVT 的必要组件，仍看到完整三帧。
   审核请求不携带 Repair 的评分或“谁提出该答案”的信息。
5. 仍采用原来的严格证据聚合：非法项或明确支持/反驳冲突，使该候选有效分为 3。
   新增需 ≥4，删除需 ≤2；新增 IVT 还要求必要组件均获支持。
   未请求的修改不能生效，部分通过的独立修改可以生效，Phase 保持原值。

这不是把一份短 JSON 当成“已看对”的证明。只降低重复输出与不必要的审核范围。
图片、当前帧/全帧证据与类别定义仍有必要；不能只让审核模型阅读 H0 文本而不看图。

## 冻结对照

入口：`scripts/run_second_round_delta_trial.py`。
源实验：`artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1/`。
新目录：`artifacts/preflight/gemini_second_round_delta_20260908_v1/`。

同四个 Training 开发目标：VID103/14576、37226；VID23/9801、31001。
五头 mask 全部有效；预测冻结并关闭推理后才读 GT 评分。Testing 未使用。
两臂共享同一份 H0/R1，B 不使用 A 第二轮的结果。

- A，原流程第二轮：原 Gemini 候选补充（low reasoning）→若池有新增候选，再执行五模型审核→原 Python 接纳。
  严格复现原来的“无新候选则停止”规则。
- B，LLM 差分修复：Gemini 3.8 Flash（medium reasoning）读取 R1 状态、仅未决问题及各席意见、原三帧、完整本体，
  输出最小修改→Python 校验→若非空，五模型只复核必要项→受限局部应用。
- B 的临时修订与最终接纳结果分别评分，不用 GT 选择哪个版本。

上限事先固定为 48 次新增调用：A 最多 4＋20，B 最多 4＋20；新增 H0 调用为 0。
新增预算上限 OpenRouter $1.20、xAI $0.50、阿里云估算 ¥1.00，承接历史账本而非归零。
两臂在修复角色、候选范围、审核范围和 reasoning 上不同，**不能把差异归因于单独压缩 JSON**。

## 实际结果

共 **18 次真实 POST，全部 HTTP 200、JSON 可解析**：
4 次原候选续补＋10 次原第二轮审核＋4 次差分 LLM 修复。

| 目标 | A 实际总审核轮数 | A 第二轮结果 | B 的 LLM 返回 |
|---|---:|---|---|
| VID103/14576 | 2 | 新增正确 Target 0，其他标签不变 | 空 changes |
| VID103/37226 | 2 | 不变 | 空 changes |
| VID23/9801 | 1 | 补充无新候选，按原规则停止 | 空 changes |
| VID23/31001 | 1 | 补充无新候选，按原规则停止 | 空 changes |

因此不能声称四个目标都实际完成了两轮审核。
B 四次都是合法空修改，**没有触发 B 的后续五模型 API**；
差分解析、局部审核及受限接纳路径有单元/接口测试，但非空修改的这一新路径尚未获得真实付费样本覆盖。

下表均为百分数，单元格为 **micro-F1 / 集合 Accuracy**，每头有效目标数为 4：

| 方案 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 缓存 Gemini H0 / 第一轮结果 | 80 / 50 | 57.14 / 25 | 33.33 / 0 | 13.33 / 0 | 50 / 50 |
| A，原流程续跑 | 80 / 50 | 57.14 / 25 | 46.15 / 0 | 13.33 / 0 | 50 / 50 |
| B，LLM 临时修订 | 80 / 50 | 57.14 / 25 | 33.33 / 0 | 13.33 / 0 | 50 / 50 |
| B，最终结果 | 80 / 50 | 57.14 / 25 | 33.33 / 0 | 13.33 / 0 | 50 / 50 |

相对共享 H0/R1，A 部分改善 1、改坏 0、混合 0、无收益替换 0、不变 3；
新增正确标签 1、错误标签 0，没有删除标签，整帧由错到五头全对仍为 0。
B 改对/部分改善/改坏/无收益替换均为 0，不变 4。

A 的 Target 提升来自 VID103/14576 新增 gallbladder（Target 0）。
它并未删除原错误 cystic_plate（Target 1），所以集合 Accuracy 仍为 0。
该目标第二轮 target_0 的有效分 4.0；ivt_17/ivt_60 都是 3，未获接纳。
原错误 target_1 仍获 4.2，说明一致意见仍可能支持错误组织。
总体候选覆盖由 2/8 增至 3/8 个 GT IVT，但最终 IVT 完全未变。

B 的四个空 JSON 只能证明“本次提示与输入下没有提出修改”。
不能从空输出判定模型到底仍看错，还是证据要求/保守倾向导致它不愿修改；
更不能把四个 KEEP 当作确认 H0 正确。

## 输出精简与费用

本次 B 的实际可见输出均为 `{"changes":[]}`，避免了重复整份预测，也跳过了无修改时不必要的五模型审核。
但是四次 B 合计输入 28,788 tokens、输出计费 6,538 tokens，其中 reasoning 为 6,510 tokens。
因此**可见 JSON 很短不等于费用很低**；没有配对的“完整输出”付费对照，不能报告压缩带来的真实费用下降百分比。
非空差分时的审核项减少比例仍待实际样本验证，不能用本轮零修改声称审核质量已优化。

- 原候选续补原生费用 $0.01507125。
- 原第二轮审核美元原生费用 $0.02525999，另有阿里云 token 保守估算 ¥0.023022。
- 四次差分 LLM 修复原生费用 $0.04610850。
- 全部新增：OpenRouter **$0.07089169**、xAI **$0.01554805**，美元合计 **$0.08643974**；
  阿里云估算 **¥0.023022**，不是原生账单。没有新增未知费用预留。

59 项相关测试通过，新增 Python 文件 Ruff 通过。
`independent_audit.json` 独立复算五头计数、验证原图与缓存 H0/GT/mask 一致、实际修改绑定、聚合算术及连续账本。
`mechanism_diagnostic.json` 保存第二轮候选、有效分、预测差异和阶段费用。
源预测、旧实验、冻结基线、Tracker 输出及其他用户本地修改保留；没有提交 Git 或密钥。

离线复核：

```powershell
.venv-p2/Scripts/python.exe tools/audit/audit_second_round_delta.py artifacts/preflight/gemini_second_round_delta_20260908_v1
```

本轮证实多一次候选与审核可以出现局部 Target 收益，但核心 IVT 问题仍未解决。
差分 LLM 方案保持研究状态，不能因接通流程就替换成“已验证更好”的默认修复方案。
