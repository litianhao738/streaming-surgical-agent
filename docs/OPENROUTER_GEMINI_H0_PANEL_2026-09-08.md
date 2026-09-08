# OpenRouter Gemini H0 接入与四目标小测试

> 2026-09-08 文档发布说明：本页保留实验结束时的结果及 Git 状态；“未提交”等表述不是本次发布状态。完整机制索引与当前发布范围见 [汇总](VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)。近期本地脚本、冻结源码与原始实验目录未随这次文档提交发布，文中相关复现命令需要本地产物。

2026-09-08，按照用户“不使用阿里云基座，H0 也改用 OpenRouter”的要求，
沿用此前指定的 Gemini 3.8 Flash，新增研究入口
`scripts/run_openrouter_gemini_h0_trial.py`。
该入口已经真实跑通 H0 → 候选补充 → 五模型逐项审核 → Python 局部修改。
它不代表修复已经有效，也没有覆盖已发布的 Qwen 基线或历史结果。

## 实际接口和协议

- H0：`https://openrouter.ai/api/v1/chat/completions`，`google/gemini-3.8-flash`，
  `provider.only=[google-ai-studio]`，禁止回退，读取 OpenRouter 凭据。
- 候选补充：同一个 OpenRouter Gemini 3.8 Flash，独立调用。
- 审核：原来的 Grok、轻量 Qwen、GPT、Gemini、DeepSeek 五席。
  只有其中的轻量 Qwen 审核席使用阿里云直接接口；H0 和候选模型均不使用该接口或 Alibaba 路由。
- 维持原 H0 的三帧因果图片、low/low/high detail、完整提示词/本体/最终五头 Schema、
  temperature=0、reasoning=low、4096 输出上限。实际请求与旧 H0 逐字段比较，
  **只有 model 和 provider.only 不同**。
- 新请求的规范化元数据包含 Gemini 模型与 Google 路由，不能命中旧 Qwen 请求身份。
  本轮四份 H0 均重新调用，没有使用旧预测冒充 Gemini。
- 沿用此前严格的逐项证据审核与局部接纳规则，最多一轮。
  扩池、放宽有效票、反思、参考图等未获证实的消融没有混入本次更换 H0 的实验。

原 `joint_openrouter_h0.yaml` 本身已经使用 OpenRouter，只是其上游是 Alibaba 的 Qwen。
官方 Batch 又是另一条历史入口。两者均与直接使用阿里云轻量审核接口区分。
旧 `run_dataset_api_pipeline.py` 的默认基线保留；新研究链路从本文入口启动。
旧 Gemini pure-H0 配置使用过不同的 compact 协议，没有拿它冒充当前 final-only 协议。

模型与参数支持在调用前查验了 [OpenRouter 官方模型页](https://openrouter.ai/google/gemini-3.8-flash)
及官方 endpoints API；实际固定 Google AI Studio 的模型元数据保存在实验目录。

## 样本和结果

固定 Training 目标：VID103/14576、37226；VID23/9801、31001。
这是已分析过的开发样本，不是独立确认集。四个目标五头 GT mask 均有效。
抽样及调用只使用 GT 可用性，不把目标 GT 标签放入请求；所有推理结束、预测冻结后统一评分。
Testing 未参与。

28 次实际 POST 全部 HTTP 200、JSON 可解析：4 H0＋4 候选＋20 审核。
可解析不等于每项证据合法，更不等于视觉判断正确。
下表为百分数，单元格是 **micro-F1 / 集合 Accuracy**，每头有效目标数均为 4。

| 预测 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 历史 Qwen H0，OpenRouter/Alibaba | 76.92 / 25 | 76.92 / 25 | 16.67 / 0 | 14.29 / 0 | 75 / 75 |
| 本轮 Gemini H0，OpenRouter/Google | 80 / 50 | 57.14 / 25 | 33.33 / 0 | 13.33 / 0 | 50 / 50 |
| 本轮 Gemini H0 经 Verifier/Repair | 80 / 50 | 57.14 / 25 | 33.33 / 0 | 13.33 / 0 | 50 / 50 |

相对本轮 Gemini H0：改对 0、部分改善 0、改坏 0、混合收益/损伤 0、无收益替换 0、不变 4。
五头集合全部正确的目标仍为 0。四个目标均实际完成一轮审核，没有把占位快照计为三轮。
旧数字评分审核没有重跑，`old_arm_run=false`；通用评分器的缺失 H1 回退字段不是一次额外模型实验。

## 为什么换了 H0 仍未修好

候选池只覆盖 **2/8 个 GT IVT**，审核无法接受未进入池中的其余六个正确关系。
此外，14 个候选因至少一席证据字段不合格而冻结，6 个候选因明确支持/反驳冲突而冻结。
这是跨四目标、各任务候选的计数，不能误读成 20 个 IVT 或 20 个 API 失败。
其余候选也没有使新标签达到接纳条件或原标签达到删除条件。

例如：

- VID103/14576：Gemini 与旧 Qwen 同样输出 IVT 59（hook/dissect/cystic_plate），
  GT 是 17、60。仍有边缘器械漏检和操作组织错配。
- VID103/37226：Gemini 找到了两类器械及两类动作，但输出 17、59，GT 为 19、60。
  组件数量增加不等于完整关系正确。
- VID23/9801：Gemini 输出 instrument 4/verb 4/IVT 79，GT 对应关系中需要 hook/dissect。
  本轮不能因 Target 聚合分数上升就忽略其他头新增错误。
- VID23/31001：器械类预测正确，但动作和对象转成 null，IVT 为 94、99；GT 为 19、88。

上述事实支持“换 API 路由/模型本身不能保证解决视觉辨认与候选缺失”。
它们不足以证明 Gemini 全面劣于 Qwen，也不足以证明这套审核在新样本上必然无效。
当前不能宣称新 H0 已超过旧基线，或 Verifier/Repair 已修复成功。

## 费用、验证与复现

新增预算在调用前冻结：OpenRouter $0.60、xAI $0.25、阿里云保守估算 ¥0.30。
承接上一实验完整占用，不归零历史未定价预留。

- OpenRouter 原生费用 **$0.04291719**，其中 H0 $0.01367325、候选 $0.01240200、审核 $0.01684194。
- xAI 原生费用 **$0.02639610**。
- 美元原生费用合计 **$0.06931329**；阿里云审核按 tokens 保守估算 **¥0.037719**，非原生账单。
- 没有新增未定价失败预留。`unpriced_calls` 中四项是阿里云估算记录，不是未成功调用。

38 项相关测试通过，改动 Python 的 Ruff 通过。
独立审计验证请求只变模型/路由、图片绑定、GT/mask 一致、五头 TP/FP/FN/F1/集合 Accuracy、
逐项接纳算术、原生账单和连续预算均一致。
原默认 H0 五个文件与提交 `25fee9485a8f740aadf7921d5312e9c488955cbb` 比较无差异。
本轮未提交 Git，未改动训练输出或其他用户本地修改，未输出或提交密钥。

实验目录：`artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1/`。
保留 `plan.json`、`frozen_source/`、`h0_preflight/`、全部 `calls/`、预测、费用、
`independent_audit.json` 和 `h0_switch_audit.json`。

离线复核：

```powershell
.venv-p2/Scripts/python.exe tools/audit/audit_openrouter_gemini_h0.py artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1
```

需要再次运行时使用新的目录，不能覆盖本轮付费结果：

```powershell
.venv-p2/Scripts/python.exe scripts/run_openrouter_gemini_h0_trial.py prepare --output artifacts/preflight/openrouter_gemini_h0_panel_new_run --previous artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1
.venv-p2/Scripts/python.exe scripts/run_openrouter_gemini_h0_trial.py execute --output artifacts/preflight/openrouter_gemini_h0_panel_new_run
```

prepare 会重新固定源代码、图像、请求元数据及价格；execute 才会付费，仍为最多 28 次调用。
