# prior-graph-repair-v1.0.0-experimental

日期：2026-09-08。分支：`research/prior-graph-repair-v1`。基于 `main` 的 `517a89a2982ec6b6819397f35b2d41bbdf84da98` 创建独立实验快照；精确发布提交由同名 Git tag 固定。

本版保存最新已测的 **Gemini H0 → Training 先验提示 → 单模型补候选 → 五席审核 → 局部修复**。它是小样本出现有限收益的研究版本，尚未证明稳定最优，也不是已经集成 Tracker／Gate 的完整 Pipeline。默认 Qwen 纯 API H0 与以前的实验均保留。

## 输入到输出

1. H0 看真实因果三帧 `[t-50,t-25,t]`，OpenRouter Gemini 联合输出 I／V／T／IVT／Phase 最终标签。
2. 本地统计图谱依据 H0 器械、完整 IVT 组件关系和预测阶段，给最多两个完整 IVT 提示。每个查询视频使用排除该整个视频的 Training 统计；无效 GT 按任务 mask 排除。先验不是视觉证据。
3. 原 Gemini 补候选调用同时看图片、H0 和提示；最多新增四个 IVT。Python 合并候选池并补齐待审核组件，不直接接纳图谱标签。
4. Grok、Qwen、GPT、Gemini、DeepSeek 五席并行看图，分别审核同一候选池。五席不接收统计频率或图谱来源；本实验不是五个 OpenRouter 席位，Grok 使用 xAI 直连，Qwen 使用阿里云。
5. Python 校验逐项证据与 JSON，五席均有效才计算均分。新增阈值为 4；新 IVT 的组件也须通过。删除需要原有反证规则；局部增删，不按 IVT 重建整帧独立头。证据不足保留 H0，Phase 不修改。

本版本固定一轮，每目标单分支最多 7 次模型调用（1 H0＋1 补候选＋5 审核），图谱查询不增加 API 调用。它没有将以前的两／三轮研究自动合入。

## 已测效果与时间

同批八个 Training 目标、四个视频，每头有效 GT 数均为 8。下表为 micro-F1；Precision、Recall、集合 Accuracy、费用及逐项损伤见[完整报告](../PRIOR_GRAPH_CANDIDATE_TRIAL_2026-09-08.md)。

| 方案 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 同批 H0 | 66.67% | 61.54% | 51.85% | 29.63% | 62.50% |
| 同批无图谱修复 | 69.23% | 66.67% | 55.17% | 34.48% | 62.50% |
| 本版图谱辅助修复 | 69.23% | 66.67% | 57.14% | 40.00% | 62.50% |

Precision（查准率）是预测标签中正确标签的比例 `TP / (TP + FP)`，与整组标签全对的集合 Accuracy 不同。同批 micro-Precision 如下：

| 方案 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 同批 H0 | 69.23% | 66.67% | 58.33% | 33.33% | 62.50% |
| 同批无图谱修复 | 75.00% | 62.50% | 57.14% | 35.71% | 62.50% |
| 本版图谱辅助修复 | 75.00% | 62.50% | 61.54% | 40.00% | 62.50% |

Verb 正确预测从 8 个到 10 个、错误预测也从 4 个到 6 个，因此 Recall 与 F1 提高，但 Precision 下降。不能用 F1 的改善声称每项指标都提高。

三组 IVT 集合 Accuracy 均为 0/8。相对 H0，本版 4 个目标部分改善、2 个改坏、2 个不变，没有全部改对目标；相对无图谱修复有 2 个部分改善、6 个不变。不能据此声称 Verifier 已修好。关键 IVT 得分提升存在审核波动或候选上下文变化的解释，尚未做独立确认。

四个独立配对目标的补候选＋并行审核＋检索合计，从 93.76 秒到 103.41 秒，增加 **9.65 秒／10.29%**；含相同 H0 请求则从 119.46 到 129.11 秒，增加 **8.08%**。五席审核逐目标增幅中位数为 14.32%，未达到原定 ≤10% 目标。计时未包含完整 Tracker／Gate，也没有全量吞吐测试；另外四组复用面板不算独立计时。

原实验共 84 次 POST，已知费用 $0.35605404，另有阿里云估价 ¥0.112322。本次整理、离线验证和 GitHub 发布没有新增模型调用。

## 下载与离线重放

本重放入口使用 Python 3.11–3.12（依赖的既有模块使用 `datetime.UTC`）。在新目录下载分支；`GIT_LFS_SKIP_SMUDGE` 可避免本任务不使用的历史训练权重下载。

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --branch research/prior-graph-repair-v1 --single-branch https://github.com/litianhao738/streaming-surgical-agent.git
cd streaming-surgical-agent
git checkout prior-graph-repair-v1.0.0-experimental
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-repair-replay.txt
.venv/Scripts/python.exe scripts/replay_prior_graph_candidate.py
```

Linux 可在 clone 命令前设置 `GIT_LFS_SKIP_SMUDGE=1`，将 Python 路径换为 `.venv/bin/python`。依赖清单安装完整项目及研究依赖；既有模块导入会用到 Torch，所以不能只安装 `jsonschema`。重放不使用 GPU、模型权重、数据集或凭证。已有完整项目环境可直接运行重放命令。

公开的[八目标重放包](../experiments/prior_graph_candidate_replay_20260908.json)包含模型原始回答文字、H0、候选、审核状态、Training 按视频聚合计数、四张留出先验表、图像哈希和评分聚合计数。重放重新拟合先验、检索提示、合并候选、规范化五席结果、计算均分并应用修复；核对最终预测与归档一致，并检查源码及包的校验和。

包不包含图片、逐帧 GT、API Key 或请求 headers。公开指标是依据归档 TP／FP／FN 等计数重新计算，**不是拿 GT 再独立算分**；真实图像、GT 与请求核验已经在原本机做过，源证据哈希保存在[机器摘要](../experiments/prior_candidate_summary_20260908.json)。校验和只能验证文件一致性，不能证明模型判断或 GT 本身正确。

## 真实 API 入口与当前边界

`scripts/run_prior_candidate_trial.py` 是保存的八目标 Training 两组配对实验入口，提供 `prepare`／`execute`。真实数据与接口依赖按[安装说明](../API_EXPERIMENT_QUICKSTART_2026-09-06.md)及 `requirements-prior-panel.txt` 安装。`--help` 可离线查看。

```powershell
python scripts/run_prior_candidate_trial.py prepare --output artifacts/preflight/prior_candidate_release_new --dataset-root D:/cholec_dataset
python scripts/run_prior_candidate_trial.py execute --output artifacts/preflight/prior_candidate_release_new --dataset-root D:/cholec_dataset
python scripts/score_prior_candidate_trial.py --output artifacts/preflight/prior_candidate_release_new --dataset-root D:/cholec_dataset
```

`prepare` 会访问模型路由和价格元数据，并新建样本／源码／统计／图片冻结记录；`execute` 会付费，最多 104 次 POST，原脚本预算为 OpenRouter $2、xAI $2、阿里云 ¥2，无自动重试。须自行配置可用凭证；脚本保留本次实际使用的模型及阿里云工作空间接口，其他用户应在新实验中提供自己的可用配置。这些是历史已测型号，不保证未来仍可调用。

新 clone 没有本机历史抽样排除名单，因此重新 `prepare` 的具体帧号可能不同；不要把新实验声称为原八帧重跑。既有输出目录禁止覆盖，`execute` 也不支持断点续跑。本发布没有启动上述付费命令。

运行时依赖中包含 `prepare_final_only_gate_training.py` 与 `final_only_training.py`，只是因为试验入口借用了历史身份和时间抽样函数，不表示 Gate 被调用或正式训练完成。原实验 `plan.json` 广泛冻结过一些未调用的本地模块；本次只发布实际依赖，不捎带无关修复／训练改动，因而不能将原本机计划直接当成新 clone 的全源码校验清单。

已测的候选、审核、检索与接纳代码保持原样。发行时新增离线重放及安装依赖清单，另修正一项破坏性测试：先复制候选再故意写坏字段，避免污染其他测试共享的本体。没有重跑或改写历史模型输出，旧四目标重放包也保持原样；发布包记录实际重放源码校验和。

以下工作仍未接入本入口：已有全量 H0 导入、Tracker／Gate 串联、Testing 调度、多目标并发和恢复执行。五席在单个目标内已并行。完整 Pipeline 的四组 Tracker×Gate 消融也尚未完成。

## 历史版本与发布范围

发布检查在独立工作树完成：215 项相关测试通过（含旧四目标与新八目标离线重放），发布 Python 文件 Ruff 通过，五个运行／评分／审计入口 `--help` 通过。十一项新增运行／审计源文件与实测工作区内容一致；仅测试隔离和发布材料另有变更。提交清单检查不包含密钥、原图、模型权重或训练输出。这些是工程检查结果，不代表新增模型效果实验。

[候选后图检索的六目标实验](../GRAPH_RAG_REVIEW_TRIAL_2026-09-08.md)与本版“候选前先验提示”分别留档，不把早期设计当成当前实现。[此前单提案／观察反馈版本](../VERIFIED_REPAIR_CANDIDATE_2026-09-08.md)和主分支保留。根目录 [PIPELINE_VERSION.json](../../PIPELINE_VERSION.json)标明入口、参数与未集成功能。

本次提交只含上述研究链实际依赖、测试、审计工具、结果摘要与离线重放。Tracker 权重、训练输出、其他本地未提交修改和密钥不在发布清单中。
