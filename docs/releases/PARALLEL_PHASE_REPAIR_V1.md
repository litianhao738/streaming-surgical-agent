# parallel-phase-repair-v1.0.0-experimental

日期：2026-09-09。分支 `research/parallel-phase-repair-v1`；同名标签固定本次代码和证据。流程的首要阅读入口是根目录 **[LATEST_PIPELINE.md](../../LATEST_PIPELINE.md)**，含并行图、每模块输入输出、JSON 示例、规则、模型、成绩与时间。

本版发布固定 H0 后的并行修复实现及零费用重放。原图谱标签不移动，默认纯 API H0 不替换。最新真实并行重跑与此前最高观测结果分别保存，不能因为版本较新就声称更准确。

## 下载后先做离线重放

Python 3.11–3.12。新目录下载；跳过本实验不使用的历史 Git LFS 训练权重：

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --branch parallel-phase-repair-v1.0.0-experimental --single-branch https://github.com/litianhao738/streaming-surgical-agent.git
cd streaming-surgical-agent
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-repair-replay.txt
.venv/Scripts/python.exe scripts/replay_parallel_phase_repair.py
```

Linux 使用 `GIT_LFS_SKIP_SMUDGE=1 git clone ...`，Python 路径改为 `.venv/bin/python`。重放无需 GPU、API Key、原始图片或 GT；安装会下载项目依赖，既有模块包含 Torch 导入。

[公开重放包](../experiments/parallel_phase_replay_20260909.json)保存八目标 H0、模型回答、候选、审核记录、Phase 决策、预期输出及汇总计数。重放重新解析回答、构建候选池、校验五席、计算均分、局部修复和 Phase 合并，核对最终输出及文件哈希。还保存并检查“原图谱缓存＋独立短历史 Phase”的最高观测结果。

它不会重新请求视觉模型，也不读取逐帧 GT。公开评分从归档 TP/FP/FN/整集合命中计数重算，**不是在新机器独立重评原始 GT**。图谱提示沿用原实验归档；图谱统计重建的旧版重放仍可单独运行 `scripts/replay_prior_graph_candidate.py`。

## 真实实验入口与依赖边界

| 文件 | 作用 |
|---|---|
| `scripts/run_parallel_phase_trial.py` | 实测的两分支并行实验；`prepare` / `execute` |
| `src/surgical_agent/research/verification/parallel_phase.py` | 并行调度、异常传播、计时与 Phase 合并 |
| `scripts/run_phase_extension_trial.py` | 独立 Phase 的请求格式；原短／长历史对照 |
| `src/surgical_agent/research/verification/phase_extension.py` | Phase 单选、弃权、五份有效与三票规则 |
| `scripts/score_parallel_phase_trial.py` | 本地完整归档的请求审核、离线 GT 评分、时费统计 |
| `scripts/replay_parallel_phase_repair.py` | 新 clone 可直接运行的无图像离线重放 |

**原始 `prepare/execute/score` 是固定八目标的冻结实验脚本，不是通用数据集命令。** 新 clone 的 `--help` 可运行；真实执行还需要：

- 外部 CholecTrack20 图片与标注，以及授权使用的接口凭证。
- 原八目标 H0／图谱／Phase 等多代历史归档；审核链递归校验旧计划、源码、请求和响应。
- 归档中所记录的本地绝对路径与原文件字节哈希。不能只改一个 `--source` 就声称已移植全部历史路径。

这些大体量本地归档没有上传，图片、逐帧 GT 和凭证也没有上传。所以本发布**不提供“新 clone 一条命令重跑原 88 次 API”的承诺**。本机完整原环境可继续使用原入口；新机器付费跑之前，需要单独准备可移植的新计划、输入和账户配置，重新核价。当前模型 ID 是实验实用型号，不保证未来接口仍可用。

最多 88 次新调用：8 次候选＋40 次四头审核＋40 次 Phase。原脚本分账户总上限为 OpenRouter $2、xAI $2、阿里云 ¥2，无自动重试；这不是实际账单。已有目录禁止重复执行或覆盖。发布过程中没有执行付费命令。

入口复用部分旧试验函数，因此包含历史五头／局部 LLM 实验依赖及 Gate 抽样工具。**被导入不等于该模块在最新推理中运行**；判断实际链路请以流程图和 `run_parallel_phase_trial.py` 的调用路径为准。

## 版本与证据

- [PIPELINE_VERSION.json](../../PIPELINE_VERSION.json)：最新并行版本，IVT F1 38.71%、Phase F1 75%。
- [BEST_PHASE_RESULT.json](../../BEST_PHASE_RESULT.json)：同批最高观测结果，IVT F1 40%、Phase F1 75%；原图谱四头缓存与独立 Phase 组合。
- [并行实测报告](../PARALLEL_PHASE_PIPELINE_TRIAL_2026-09-09.md)：真实耗时、费用、失败、回退与退化原因。
- [独立 Phase 报告](../PHASE_EXTENSION_TRIAL_2026-09-09.md)：阶段修复及短／长历史对照。
- [发布指标快照](../experiments/parallel_phase_summary_20260909.json)：各头完整指标、损益、候选覆盖、时费及来源 SHA-256。
- [原版保留标记](https://github.com/litianhao738/streaming-surgical-agent/tree/best-repair-2026-09-09)：仍指向 `a0406baa7e5aeb8313403e792a8e609e2fceccef`。

当前只在同八个 Training 目标开发验证；IVT 整集合准确率仍为 0%。完整 H0 导入、Tracker/Gate 串联、全量调度和四组消融尚未完成。本次发布范围是修复实验代码、直接依赖、测试和脱敏复现材料，保留本地其余改动及训练输出。

## 发布检查

隔离发布工作树中，233 项相关测试通过，2 项因未复制本机原始归档而跳过；38 个发布 Python 文件 Ruff 通过。新增公开回放的 25 项检查包含篡改身份、候选、均分、席位、源码和四头输出的负向验证。四个运行／评分入口 `--help` 通过，130 个项目模块均从发布目录导入，没有借用原工作区文件。

公开重放确认 8 个图谱审核面板、16 个 Phase 面板、8 次生产并行合并与封存预测一致。新文档相对链接、提交文件白名单、凭证和内嵌图片模式已检查。它们是工程与归档检查，不是新增视觉效果实验。机器记录见[发布检查摘要](../experiments/parallel_phase_release_checks_20260909.json)。
