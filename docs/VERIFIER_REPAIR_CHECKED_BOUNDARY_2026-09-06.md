# Verifier / Repair：final-only 边界修复与离线回放

2026-09-06。状态：独立修复链路的工程检查接口和回放已完成；没有新增付费调用，没有新的语义准确率提升证据。

## 范围与当前初始预测

Tracker OOF 训练按用户反馈正在进行。本工作不需要其训练结果，也没有操作训练环境、权重或训练输出。

已发布 OpenRouter H0 和本地已冻结的官方 Batch H0 都返回同一个 final-only 五头契约。新增 `src/surgical_agent/research/verification/final_only_grounded.py` 可读取原始 final-only JSON，或 Batch 导入后的五头 `selected_ids` 字典；不生成 top-k、分数或新的 H0。

新接口复用历史 grounded 定位、实例 IVT 提案和复核契约。旧 `grounded_repair.py`、旧六目标实验 runner、原始实验目录、H0 prompt 和 Batch 锁文件均保留原样。纯 API 默认入口仍只运行 H0。旧完整 Gate 依赖排名的限制保持不变。

**这里完成的是标签契约兼容、候选/接纳边界修复和离线回放，不是官方 Batch 的付费 Repair runner，也没有把修复接入默认 Pipeline。**

## 已复现并修复的工程问题

1. **标签顺序变化会触发无收益复核。** 例如 H0 的 IVT 为 `[60,17]`，候选为 `[17,60]`，各头集合相同，旧字典比较仍认为存在变化；满足复核布尔条件时甚至会 ACCEPT。新接口严格验证后按集合比较，返回 `NO_LABEL_CHANGE`，不要求 review；保留原 H0 标签顺序。
2. **实例提案合法不保证聚合输出合法。** 三个实例各提出三个合法 IVT，旧实现可接受总计九个 IVT、七个 Target 的结果，超过 final-only 的 8 / 5 项上限。新接口在投影后检查完整五头契约；超限拒绝整个候选，不截断标签。
3. **H0 / 可选阶段的错误边界不完整。** 新接口拒绝缺头、重复、错误类型、越界、多 Phase 等无效 H0；无效定位、提案或复核则保留已验证的 H0，并记录原因。包括旧校验器被超大整数触发 `OverflowError` 的情况。

H0 合法性检查不要求五头等于 IVT 的严格投影，不改写已冻结的独立头预测。候选继续沿用旧实验的 IVT 投影机制，Phase 保留 H0；这不表示投影一定符合所有标注作用域。

## 调用顺序

```python
from surgical_agent.research.verification.final_only_grounded import (
    prepare_grounded_review,
    finalize_grounded_repair,
)

prepared = prepare_grounded_review(
    h0, locator, proposal, proposal_slot="FIRST"
)
# 只有 prepared["review_required"] 为 True 时，才需要一次视觉复核。
# 复核必须使用 prepared["hypotheses"] 和同一目标的原图/裁剪。
result = finalize_grounded_repair(
    h0, locator, proposal, review, proposal_slot="FIRST"
)
final = result["final"]
```

这些函数不创建 API 客户端，不接收 GT。实际调用层必须保存并绑定同一套 H0、H1、hypothesis slot、图片与请求哈希；不能把历史六目标的 review 移用于新的 Batch H0。接纳依据仍是同一模型的自评证据，不能视为独立语义裁判。

## 离线收益报告

新 `repair_comparison.py` 分别计算 H0、候选策略、最终接纳三臂，以及有候选的共同样本子集。无候选时，候选策略回退 H0，不将它虚构为空预测。每头按有效 GT mask 统计 micro-Precision/Recall/F1 和集合 Accuracy；失败预测保留有效 GT 分母，不获得 exact，即使该头 GT 为空。

配对变化按每头集合对称差区分部分改善、变坏、改变但损失相同、未变化，并另计 wrong→exact / exact→wrong。一个目标不同头有得有失单列 mixed。报告中的 `all_valid_heads_exact` 指该目标所有有效头同时正确，不应在有缺失标注时称为“五头同时正确”。

回放命令，不调用 API：

```powershell
.venv-p2/Scripts/python.exe scripts/replay_grounded_contact_repair.py --run artifacts/preflight/grounded_contact_qwen0902_20260905 --output artifacts/preflight/grounded_contact_checked_replay_new
```

必须用新的输出目录，不能写入源实验目录。回放核对预测、保存 GT/mask 与计划目标的完整对应，记录源文件和当前实现哈希，明确区分历史费用与本次零费用。此 CLI 专用于已有 OpenRouter grounded smoke 目录；GT 来自此前已审计的保存记录，本轮没有重新加载源数据集。

## 六目标回放结果

来源仍为旧 Validation / VID110 六目标，使用历史 `grounded_contact_h0_v1`，不是最新官方 Batch 的新增实验。新检查没有改变这六目标的最终标签或接纳结论：

- 候选可用 6/6；接纳 2 次；4 次无标签变化。
- 1 个目标仅部分 Verb 改善；0 个目标由所有有效头未全对变成全对。
- 0 个目标按对称差出现改坏；1 次无收益替换；没有 mixed 目标。
- H1 和最终结果的 IVT 都没有 TP。仅调整接纳无法从这批候选中获得正确 IVT。

| 任务 | H0 micro-F1 | 最终 micro-F1 | H0 / 最终集合 Accuracy |
|---|---:|---:|---:|
| Instrument | 95.65% | 95.65% | 83.33% / 83.33% |
| Verb | 34.78% | 43.48% | 0% / 0% |
| Target | 52.17% | 52.17% | 0% / 0% |
| IVT | 0% | 0% | 0% / 0% |
| Phase | 100% | 100% | 100% / 100% |

| 历史阶段 | 调用数 | 原始 ledger 费用 USD |
|---|---:|---:|
| H0 | 6 | 0.079164 |
| 定位 | 6 | 0.042792 |
| 提案 | 6 | 0.090102 |
| 复核 | 2 | 0.031004 |
| 修复新增合计 | 14 | 0.163898 |
| 全链路合计 | 20 | 0.243062 |

本轮实际新增 API 调用 **0**，费用 **0**。不能把回放的历史 usage 计为新消费，也没有依据将此回放宣传为省费或提高准确率的新实验。

最终回放文件：`artifacts/preflight/grounded_contact_checked_replay_verified_20260906/report.json`。

## 验证与下一步边界

- 56 项相关测试通过：新边界、新收益统计、回放 CLI、历史 grounded 回归和既有初始预测锁测试。
- 本轮新增的三个 Python 实现和三个测试文件全部通过 Ruff。
- 官方 Batch 已保存的三个真实 H0 逐项通过新接口，无修复证据时最终标签与原记录完全相同。它是接口兼容检查，不是新的 Repair 语义实验。
- 旧实验、H0 和 Tracker 文件没有因本轮修改；没有 Git 提交。

本轮修复可避免无效输出与纯顺序变化带来的多余复核，但六目标的错误仍主要存在于候选视觉判断和复核可信度。当前没有依据启动重复六目标、增加轮数或训练 Gate 的付费尝试。若设计新的语义对照，应另行固定 Training-only 样本、同一冻结 H0、唯一改动、各阶段调用数和预算；Testing 不参与选择。
