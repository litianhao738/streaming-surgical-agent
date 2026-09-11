# 默认五模型组合：v1.2.0

按用户“那就用这五个轻量级模型”的选择，默认修复版本设为`parallel-phase-repair-v1.2.0-lightweight-reviewers`。本地已接入，未提交或推送GitHub。

| 实际审核模型 | 模型ID | 路由 |
|---|---|---|
| Ministral 3 8B | `mistralai/ministral-8b-2512` | OpenRouter，仅Mistral，无回退；使用原OpenRouter凭证 |
| Qwen3.5 35B-A3B | `qwen3.5-35b-a3b` | 阿里云原工作空间，关闭thinking |
| GPT轻量 | `openai/gpt-5.6-luna` | OpenRouter，仅OpenAI，reasoning none |
| Gemini轻量 | `google/gemini-3.5-flash-lite` | OpenRouter，仅Google AI Studio，reasoning minimal |
| DeepSeek视觉轻量 | `deepseek/deepseek-v4-flash-vision-exp` | OpenRouter，仅Fireworks，reasoning disabled |

Qwen是35B总参数、约3B激活，并非总参数3B。其他Flash型号不以名称推断精确参数量；Gemini保持原minimal设置，不能把本组合描述成五家一律零推理。历史内部席位键`grok`保留，但新版本对应的是Mistral，日志的`model`、计划`reviewer_families`和供应商信息决定实际身份。

## 默认流程

固定H0 → 两路并行：

1. 图谱提供备选关系 → 原单模型补候选 → 上述五模型看图逐项评分 → Python局部修复I／V／T／IVT。
2. 上述五模型独立看原三帧、判断Phase → 五份有效且至少三票同阶段 → Python合并Phase。

两路内部的五模型均并行。四头仍为新增均分≥4、删除均分≤2；Phase仍为原默认单选多数票。本次只选定模型组合，没有把独立实验的Phase七阶段均分模板一起推广，也没有新增轮次。Phase修复保持启用。

纯API H0、补候选模型、图片、精简提示、JSON协议、原接纳规则不变。原默认范围仍是归档8目标与缓存H0；本次没有新增完整数据集、Tracker或Gate集成。

## 入口、旧版与验证

`scripts/run_pipeline.py`读取`DEFAULT_PIPELINE_VERSION.json`，新计划转入`scripts/run_lightweight_parallel_repair.py`。适配器在本次CLI进程内绑定两路新模型、Ministral的OpenRouter地址／凭证／原生费用归属；运行结束后恢复绑定，不修改历史执行器文件。它拒绝把Ministral的推理输出作为正常无推理审核。

已准备的旧compact或原始计划仍委托原执行器，保持旧模型和提示；旧默认清单完整保留在`configs/defaults/parallel-phase-repair-v1.1.0-compact-prompt.json`。也可直接使用`run_compact_parallel_repair.py`准备旧版新实验。不要把新版本配置套在旧请求上重放。

```powershell
# 查看当前默认，不调用API。
.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py info

# 准备已有归档范围内的新计划；不发起模型推理。
.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py prepare --output <new-directory> --dataset-root D:/cholec_dataset
```

本轮模型切换不追加付费推理。测试覆盖两路五席模型、提示／Schema不变、正确凭证与账户、拒绝意外推理、异常后的绑定恢复、版本冻结和历史计划委托。

实测验证：相关单元测试44项通过，改动Python文件Ruff通过。通过默认入口完成8个历史目标的真实数据准备，并离线核对80份审核请求（8目标×5模型×2分支）：图片、提示及Schema保持一致，Phase请求指纹匹配，新模型及OpenRouter费用归属正确。未发送新的模型推理请求，因此不产生新准确率成绩。

准备记录：`artifacts/preflight/default_lightweight_prepare_20260910_v1/offline_roster_preflight.json`；冻结计划SHA256：`ebe5d3f60aead13d505b199cbbc6dc20c1fed5bef60a998a310f82ef3ce1e5f2`。

## 准确率与速度证据的范围

[上一轮8目标替换实验](LIGHT_PHASE_SEATS_TRIAL_2026-09-10.md)仅对Phase均分机制比较两席替换，另外三席缓存。32次请求有效，两席并行等待减49.72%；Phase仍75%，低于该批H0 87.5%。该结果不等于当前完整默认的四头／Phase新成绩，也不能等同于完整Pipeline提速50%。

本次默认选择来自用户对模型组合的明确要求，不声称Phase问题已解决或本组合准确率最优。历史成绩、训练输出与旧模型归档保留。
