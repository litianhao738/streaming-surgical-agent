# 新 Gate 完整入口（2026-09-16）

默认 Gate：`five-head-probe-gate-20260916`，54 维特征，阈值 `0.011637720680921813`。
Qwen 初审同时看器械、动作、目标、IVT 和 Phase，Phase 的 12 个新特征进入 Gate。
Gate 打开后五头联合复审复用这次 Qwen 回答，随后执行 Tracker 输出模块和 60 秒因果 Phase 平滑。
完整 Testing 入口继续生成 Report、Rule、Judge 和 GSR。

## 数据归属

默认基座是 Gemini 3.8 Flash；路径见 `configs/defaults/gemini_probe_inputs.json`。
6,059 帧新五头初审 Training 缓存可用于当前 Gate 离线回放。
已有 Gemini H0 是可复用输入，H0 本身不经过 Gate。
`gemini38_report_testing_half_20260916` 的历史 FULL/Report 分数仍属于当时的 Gate，不能改名作为新 Gate 的成绩。
新 Gate 的 Testing 分数须在新的输出目录执行后取得。
其他基座待提供本地路径后，先检查模型身份、逐帧覆盖和图像协议，再创建对应 scope；不能静默用 Gemini 补齐缺帧。
当前半段 Testing runner 仍只接受指定 Gemini 时间线，`--scope` 本身不是任意基座数据导入器。

## 本机命令（项目根目录，PowerShell）

查看当前默认版本：

```powershell
.venv-p2/Scripts/python.exe scripts/run_pipeline.py info
```

离线预检（输出目录必须是新目录；不会调用 API）：

```powershell
.venv-p2/Scripts/python.exe scripts/run_pipeline.py preflight --output artifacts/experiments/probe_preflight_new
```

本机已建立新 Gate scope。其他环境先恢复 manifest 指向的本地模型、Training 缓存和 Testing 输入，再执行一次：

```powershell
.venv-p2/Scripts/python.exe scripts/freeze_probe_testing_scope.py
```

从 Gemini H0 跑新 Gate 完整 Testing（产生 API 费用，使用 CUDA Tracker）：

```powershell
.venv-tracker-gpu/Scripts/python.exe scripts/run_pipeline.py testing-run --output artifacts/experiments/gemini_probe_complete_new --workers 8 --allow-paid
```

也可先用 `testing-prepare` 代替 `testing-run`，不加 `--allow-paid`，只做本地准备和公开端点检查；之后 `testing-execute --allow-paid` 执行。
预算默认读取 scope 下的 `suggested_budget_limits.json`，Report/Judge 上限默认 30 USD；预算不足会明确停止，不能保证默认预算覆盖整个实验。
自定义上限用 `--budget-limits 路径 --report-budget-usd 金额`。
查看进度用 `testing-status --output 输出目录`。
中断后检查已保存失败原因，再用 `testing-resume --output 输出目录 --allow-paid`；未知结果的已发送请求不会自动重复计费。
修改源码或默认模型后，不要继续执行原封存计划，须检查差异或新建实验。

## 质量界限与发布

本次接入验证：66 项相关单元测试通过；6,059 帧离线回放通过，54 维特征与采集缓存逐帧一致。
CUDA Testing 准备通过（8,578 帧 pipeline、7,823 帧评分）；Report/Judge 完整准备通过。
这些检查没有发送付费推理请求，也没有产生新 Gate 的 Testing 成绩。

新 Gate 是 Training 研究版本：`harm` 标签、`overall` 阈值规则；只约束汇总表现，不保证每个视频都不退步。
`selection_feasible=true` 不等于独立验证通过，`deployable=false` 保留。
GitHub 发布源码、测试、默认版本和数据来源索引；原始数据、训练权重、API 密钥及实验导出留在本地。
