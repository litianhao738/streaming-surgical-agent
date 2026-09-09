# Latest update: parallel graph repair and independent Phase

2026-09-09. See [the clear end-to-end flow](LATEST_PIPELINE.md), [release and replay guide](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/docs/releases/PARALLEL_PHASE_REPAIR_V1.md), and [complete new metric snapshot](https://github.com/litianhao738/streaming-surgical-agent/blob/parallel-phase-repair-v1.0.0-experimental/docs/experiments/parallel_phase_summary_20260909.json).

All rows below share the same eight Training targets and cached H0; each task has eight valid GT targets. Values are micro-F1 (%).

| Version | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| H0 | 66.67 | 61.54 | 51.85 | 29.63 | 62.50 |
| Original graph R1 | 69.23 | 66.67 | 57.14 | 40.00 | 62.50 |
| Retained graph cache + independent short Phase | 69.23 | 66.67 | 57.14 | 40.00 | 75.00 |
| Latest concurrent rerun | 69.23 | 66.67 | 57.14 | 38.71 | 75.00 |

The latest rerun used 88 new calls after cached H0, 203.965 seconds, native USD $0.26604873 plus estimated Aliyun CNY 0.096176. Phase changed two targets: one corrected, one wrong-to-wrong. IVT exact-set accuracy remains 0/8. This publication itself makes zero model calls.

The original graph tags remain fixed. New best-observed Phase results and the latest execution version are separate records. The historical comparison below predates isolated Phase repair; its statements about five-head LLM variants concern those earlier variants.

---

# 实验记录：当前保留版本与对照

更新：2026-09-09。本次仅整理、核对和发布记录，**新增模型 API 调用 0 次**。详细本地日志、失败记录、冻结源码及训练输出保留。

## 当前保留版本

**图谱一轮版：固定 H0 → Training 图谱关系提示 → 单模型补候选 → 五模型看图评分 → Python 局部修复。**

- 原冻结版本：[prior-graph-repair-v1.0.0-experimental](https://github.com/litianhao738/streaming-surgical-agent/tree/prior-graph-repair-v1.0.0-experimental)。
- 本次保留标记：[best-repair-2026-09-09](https://github.com/litianhao738/streaming-surgical-agent/tree/best-repair-2026-09-09)，同一提交 `a0406baa7e5aeb8313403e792a8e609e2fceccef`；原标签不移动。
- [版本与参数清单](BEST_REPAIR_VERSION.json) · [下载、运行与离线重放](https://github.com/litianhao738/streaming-surgical-agent/blob/a0406baa7e5aeb8313403e792a8e609e2fceccef/docs/releases/PRIOR_GRAPH_REPAIR_V1.md)。本次重新重放 8 个目标、4 张留出先验表、16 组审核，结果一致，无 API 调用。

“当前保留”指下面**同批八目标对照中**的选择，不是全局最优或正式 Testing 结论。保留版只运行一轮、阈值 4，Phase 保持 H0；尚未集成最新 Tracker/Gate。五头修复已单独测试，但没有替换保留版。

## 同样本评分

四个 Training 视频、共 8 个目标；所有版本共享缓存的 OpenRouter `google/gemini-3.8-flash` H0 和 `[t-50,t-25,t]` 三帧。各头有效 GT 均为 8；缺失 GT 按头 mask 排除，接口失败目标保留在分母中。以下为 **micro-F1，%**。

| 方案 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| H0 | 66.67 | 61.54 | 51.85 | 29.63 | 62.50 |
| 无图谱一轮修复 | 69.23 | 66.67 | 55.17 | 34.48 | 62.50 |
| **图谱一轮：保留** | **69.23** | **66.67** | **57.14** | **40.00** | **62.50** |
| 图谱＋第二轮观察反馈 | 69.23 | 66.67 | 57.14 | 40.00 | 62.50 |
| 图谱＋第二轮问题清单 | 69.23 | 66.67 | 55.17 | 40.00 | 62.50 |
| 一轮改四席有效（离线）／五席直接重审 | 69.23 | 66.67 | 57.14 | 38.71 | 62.50 |
| 四席有效＋直接重审 | 69.23 | 64.52 | 57.14 | 38.71 | 62.50 |
| 四头 LLM 局部修改＋五模型审核 | 69.23 | 66.67 | 57.14 | 38.71 | 62.50 |
| 五头 LLM 联合修订＋Python 检查 | 66.67 | 66.67 | 53.85 | 32.26 | 62.50 |
| 五头 LLM 联合修订＋五模型审核 | 69.23 | 66.67 | 55.17 | 38.71 | 62.50 |

同分不代表同一机制；合并行仅节省表格空间。全部变体、辅助输出、TP/FP/FN、Precision/Recall/F1、集合 Accuracy、损益与源文件 SHA-256 见[完整指标快照](docs/experiments/repair_comparison_20260909.json)。

保留版的 **H0 → 修复** Precision、Recall 与集合 Accuracy（%）：

| 头 | Precision | Recall | 集合 Accuracy |
|---|---:|---:|---:|
| Instrument | 69.23 → 75.00 | 64.29 → 64.29 | 37.50 → 37.50 |
| Verb | 66.67 → 62.50 | 57.14 → 71.43 | 25.00 → 25.00 |
| Target | 58.33 → 61.54 | 46.67 → 53.33 | 25.00 → 25.00 |
| IVT | 33.33 → 40.00 | 26.67 → 40.00 | 0.00 → 0.00 |
| Phase | 62.50 → 62.50 | 62.50 → 62.50 | 62.50 → 62.50 |

Precision 是预测标签中正确的比例；Recall 是找回真值的比例；F1 综合误报与漏检；集合 Accuracy 要求一帧该头整组标签全对。IVT 集合 Accuracy 为零，不代表所有单个 IVT 都错。

## 为什么没有采用后续变体

| 实验 | 相对图谱一轮的实际结果 | 决定 |
|---|---|---|
| 第二轮观察反馈 | 8/8 最终答案不变，无新增收益；问题清单臂另加一个错误 Target | 不增加默认轮次 |
| 四席有效、旧候选直接重审 | 多加错误 IVT；四席直接重审还多加错误 Verb | 不放宽席数 |
| 四头 LLM 局部修改 | 最终 1 个目标改坏、7 个不变；无改善 | 保留为诊断实验 |
| 母体思路的五头联合修订 | LLM＋Python：1 个目标部分改善、3 个改坏、4 个不变；加五模型审核：2 个改坏、6 个不变 | 不替换保留版 |

五头实验确实审核全部 7 个 Phase，并实际替换一次阶段；但属于“错→错”，没有阶段收益。该替换与 Target 改坏在同一目标，不能重复计数。三次 DeepSeek 429 导致缺票回退；仅看接口全部成功的五目标，仍没有改善，不能把退化全归因于限流。

主要局限：正确 IVT 候选仍只覆盖 **7/15**；模型会共同看错作用组织或动作；高均分不是正确概率；重审会重新激活旧池中的错误候选。图谱一轮相对 H0 也有 **4 个部分改善、2 个改坏、2 个不变**，没有五头全部改对的目标。收益有限，尚未完成新样本独立确认。

## 实际新增调用与费用

以下为各次原实验的增量，不重复计算共享 H0/原轮次。美元为 API 返回的费用，人民币为 token 估算；未知预留不是已确认扣费。

| 原实验 | 新增调用 | 已报告美元费用 | 人民币估算 | 另有未知美元预留 |
|---|---:|---:|---:|---:|
| 无图谱／图谱一轮配对 | 84 | $0.356054 | ¥0.112322 | 0 |
| 第二轮反馈／问题清单 | 44 | $0.239144 | ¥0.070571 | $0.078041 |
| 四／五席直接重审 | 35 | $0.163365 | ¥0.067684 | 0 |
| 四头 LLM 局部修改 | 43 | $0.251421 | ¥0.063160 | 0 |
| 五头 LLM 联合修订 | 48 | $0.331145 | ¥0.115377 | $0.121733 |

## 历史与尚未完成

- [此前 H0、Tracker、Gate 和修复实验完整记录](https://github.com/litianhao738/streaming-surgical-agent/blob/517a89a2982ec6b6819397f35b2d41bbdf84da98/EXPERIMENT_RESULTS.md)与[早期修复机制汇总](docs/VERIFIER_REPAIR_EXPERIMENT_SUMMARY_2026-09-08.md)保留；不同样本不能直接横向排名或相加为独立样本量。
- 默认纯 API H0 的安装与运行仍见[快速开始](docs/API_EXPERIMENT_QUICKSTART_2026-09-06.md)。本页 Gemini 配对研究与默认 Qwen H0 分开记录。
- 最新完整 Pipeline 集成、当前策略的正式 Gate 训练及四组 Tracker×Gate 消融尚无完成结果；本页模块小测试不能替代它们。
- 本次主分支发布的是精简记录、聚合指标和最佳版本指向。最佳版代码／重放包在固定标签；未采用变体的源码、完整请求和详细报告继续保留在本地原目录。未发布图片、逐帧 GT、凭证或训练输出。
