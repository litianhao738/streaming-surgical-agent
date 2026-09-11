# 默认修复版本：图谱一轮＋独立 Phase＋精简审核提示

历史说明：2026-09-10默认模型组合已按用户要求切换为[v1.2.0轻量审核席](DEFAULT_LIGHTWEIGHT_REPAIR_2026-09-10.md)。本页保留v1.1.0机制与当时结果；统一入口读取当前默认，直接调用`run_compact_parallel_repair.py`可使用本页旧模型版本。

2026-09-09，先将 `parallel-phase-repair-v1.0.0-experimental` 设为默认修复，随后按用户“那更新到默认修复”的要求，升级到 **`parallel-phase-repair-v1.1.0-compact-prompt`**。入口读取仓库根目录的 `DEFAULT_PIPELINE_VERSION.json`，prepare/execute/score 均转入 `run_compact_parallel_repair.py`，由它复用原并行执行器与离线评分器。

审核提示使用实际测试过的 `verifier_compact_wording_v2_json_mode_candidate`（该字符串保留实验身份，现已被默认流程采用）：四头 `compact_interaction_v1.txt`，Phase `compact_phase_v2.txt`。Phase 含明确 JSON 指令，已修复阿里云 JSON object 模式的兼容问题。

最新完善：默认入口在execute/score成功后自动生成`review_diagnostics.json`，恢复具体审核拒绝原因，不改变票数和预测、不增加API调用。[多方案对照结果](DEFAULT_IMPROVEMENT_TRIAL_2026-09-09.md)：裁剪未通过新24目标确认，加权审核在原提示上小幅改善但精简提示旧8目标明显退化，均未启用。该轮原提示成绩不能代替当前精简版成绩。

## 采用的机制

固定 H0 后，原图谱关系提示→单视觉模型补候选→五模型按精简提示看图评分→Python 四头局部接纳，与精简 Phase 短历史审核并行。四头仍要求五份有效审核，新增均分至少4、删除均分至多2；Phase 用原三帧，五份响应有效且至少三票同阶段才采用。最终只合并 Phase，不反向改变四头或图谱。输出 Schema、图片、候选组件、模型、思考强度与上限均保持原设置。

未启用新修改项审核、transactional 接纳、额外轮次或联合 LLM 重写。H0 生成入口继续是 `scripts/run_dataset_api_pipeline.py`，纯 H0 的协议和配置未修改。

## 默认入口

显示当前选择，不读取密钥或发送 API 请求：

```powershell
.\.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py
```

现有执行器只支持归档的八个 Training 目标及其缓存 H0。下面展示已有范围内的新一次运行；并非任意 H0 数据集的全量入口。

```powershell
# 先准备独立目录；按现有规则核验、冻结输入和源码。
.\.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py prepare --output artifacts/preflight/default_graph_phase_NEW --dataset-root D:/cholec_dataset

# execute 会产生真实 API 费用；本次设置默认时没有执行此命令。
.\.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py execute --output artifacts/preflight/default_graph_phase_NEW --dataset-root D:/cholec_dataset

# 推理关闭后离线算分，沿用任务 mask。
.\.venv-p2\Scripts\python.exe -X utf8 scripts/run_pipeline.py score --output artifacts/preflight/default_graph_phase_NEW --dataset-root D:/cholec_dataset
```

`NEW`目录须换成尚不存在的实验目录。也可设置 `CHOLECTRACK20_ROOT` 代替传入数据路径。入口沿用原账户配置、分支预算与禁止付费覆盖规则；当前每次八目标最多88次新增请求，不含缓存H0的历史费用。完整数据集、Tracker、Gate接入仍未完成。

## 保留记录与含义

以下 v1.0.0 成绩和原版核验是历史记录，并非精简提示的结果。新版没有新建 Git 发布标签，也未把旧 release commit 标成新版本提交。

- 历史最佳观察结果仍在 `BEST_PHASE_RESULT.json`：原图谱四头缓存＋`phase_short`，IVT F1 40%、Phase F1 75%，五头平均F1 61.61%。
- 最新并行真实重跑的 IVT F1 是38.71%，其余头F1相同。默认选择的是机制，不保证每次API响应复现历史最高分，也不会给新图片套用旧答案。
- 原图谱标签 `best-repair-2026-09-09`、两个历史最佳结果清单、原始实验和未提交修改均保留。本次只新增默认选择及入口，不移动Git标签或修改历史评分。
- 默认选择不等于全量验证成功；八目标反复用于开发，IVT集合Accuracy仍是0/8。

实现核验时，原并行执行器、图谱提案、五模型审核、Phase模板及两个接纳模块共八个关键文件与发布标签逐一比较，忽略换行差异后完全相同。

本次验证：15项并行修复测试通过（包括归档请求与结果回放），新入口Ruff通过；默认信息读取及prepare／execute／score三条路由离线检查通过。路由检查用模拟子进程，不执行付费请求。本次新增模型API调用为0，未提交或推送GitHub。

## 精简提示默认升级核验

新版 `run_compact_parallel_repair.py` 在单个 CLI 进程内绑定已测试的两套提示构造器，复用原执行器与选择器。prepare 先验证原历史 H0/图谱来源，再冻结新提示文件及实际 Phase 请求指纹；四头候选池在运行时生成，记录的是精简后的真实审核请求指纹。execute 和 score 使用同一套绑定，并在结束或异常时恢复原模块，历史源码不被改写。

新建目录采用新默认；已经准备的旧 `fixed_h0_parallel_graph_and_phase_v1` 目录仍按旧提示执行和评分。遇到其他实验 profile 会拒绝混用，需使用该实验自己的入口。

本地长度检查使用 UTF-8 字节数代理，不称为所有提供方的精确 Token 计数。实测 76 对输入减少的证据见[完整测试报告](VERIFIER_COMPACT_PROMPT_TRIAL_2026-09-09.md)，仍不承诺每次费用或准确率改善。

升级核验：59 项相关离线测试通过，Ruff 通过；默认 info 已显示 `parallel-phase-repair-v1.1.0-compact-prompt`。本次设置默认不发起推理 API 调用。

真实数据 prepare 已通过，冻结计划位于 `artifacts/preflight/default_compact_repair_ready_20260909_v1/plan.json`，上限沿用 8 目标、88 次请求。本次只准备，未执行。随后离线回放核对 8 份原补候选请求和 80 份精简审核请求，新默认生成的请求与实测归档逐份完全一致，8 个最终预测全部复现实测精简组；证据为同目录 `offline_acceptance.json`。没有为接入验证重复付费。
