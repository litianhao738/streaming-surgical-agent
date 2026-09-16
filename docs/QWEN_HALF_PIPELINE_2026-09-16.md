# Qwen 同范围半量 Testing 完整流程

> 本文记录最初的 Qwen H0 / Gemini 提案版本。该轮已按用户要求暂停；当前续跑与 Qwen 提案配置见 [Qwen 候选提案续跑](QWEN_PROPOSER_CONTINUATION_2026-09-16.md)。不要再启动下面的旧目录。

已准备的本地运行目录：`artifacts/experiments/qwen38_half_probe_complete_20260916_r2`。
使用新 54 维 Phase Gate，保持 Gemini 实验的相同时间段、8,578 帧 pipeline、7,823 帧评分和 480 帧 H0 预热。
7,964 条 Qwen H0 直接复用，1,094 条缺失或短上下文 H0 由 `qwen3.8-max` 补齐。
只替换 H0 基座，Gemini 候选提案、评审模型、Tracker、Report/Judge 配置保持不变。
Gate 仍为 Gemini Training 拟合的固定 Gate：这是跨基座迁移实验，没有用 Qwen Testing 标签重训。

## 当前电脑直接启动（PowerShell）

```powershell
Set-Location D:\PythonProject7
.\.venv-tracker-gpu\Scripts\python.exe -X utf8 scripts/run_pipeline.py testing-execute --output artifacts/experiments/qwen38_half_probe_complete_20260916_r2 --workers 8 --allow-paid
```

此命令实际收费。执行到 Report、Rule、Judge、GSR，最终结果位于输出目录的 `reports/results/summary.json`。
本轮准备没有发送付费推理请求；执行时缺失 H0 只调用 Qwen，不回退到 Gemini。
Qwen 使用既有北京阿里云端点，已通过免费的 `/models` 查询确认型号可用；未做付费在线烟测。

## 时间与费用估算

8 并发、GPU Tracker：预计 **12–18 小时**，不是实测保证。依据新初审缓存各席延迟，推算纯评审理想耗时约 7.6 小时，再加候选提案、补齐 H0、图片/Tracker、Report/Judge 和运行波动。
跨基座 Gate 打开率可能不同，不能把 Gemini 上的路由比例当作 Qwen 实测值。

- OpenRouter（候选提案、相关评审、Report/Judge）：约 **60–90 USD**。
- 阿里云（补齐 Qwen Max H0、Qwen Flash 初审）：约 **80–120 CNY**。
- DeepSeek：预计数美元；GLM 按实际账单另计，未取得可核实的国内官方价目文本，未冒充为零费用。
- 实际停止上限：核心 OpenRouter 120 USD、Report/Judge 30 USD、阿里云 200 CNY；GLM 和 DeepSeek 各最多 8,578 次请求（这两个是次数限额，不是金额限额）。

阿里云估算采用公开北京 Qwen Max 原价输入 12、输出 36 CNY/百万 tokens；不预扣缓存折扣，专有端点实际账单可能不同。
来源：[Qwen Max 官方价格](https://help.aliyun.com/zh/model-studio/qwen3-8-max)、[DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)。
已有 H0 的历史费用不重复计入；估算不含重试。

## 重新准备另一轮

scope 已在本机生成。其他环境需要先恢复本地数据和权重，再生成 scope：

```powershell
.venv-p2/Scripts/python.exe -X utf8 scripts/prepare_qwen_half_scope.py
.venv-tracker-gpu/Scripts/python.exe -X utf8 scripts/run_pipeline.py testing-prepare --scope artifacts/evaluation/qwen38_testing_half_probe_20260916 --output artifacts/experiments/qwen_half_new --report-budget-usd 30
```

准备后再对新目录执行 `testing-execute --allow-paid`。查看进度用 `testing-status`；不要重复启动同一目录。
原 r1 是本地接入检查目录，使用上面的 **r2**。
