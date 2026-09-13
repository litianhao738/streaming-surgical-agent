# 最新数据与历史数据索引

本索引用于区分“当前结果”“未采用的有效实验”“已替代的尝试”和“临时副本”。标注不代表删除，历史原始证据保留用于复现；本次没有批量清空数据目录。

## 应该使用的数据

| 用途 | 位置 | 说明 |
|---|---|---|
| 当前 Gate 模型、输入清单 | `artifacts/training/gate/tracker_scheme4_20260914/` | 已上传；以两个 DEFAULT 清单为准 |
| 当前 OOF Tracker 预测 | `artifacts/training/tracker_clip_v2_oof5_20260906/oof/` | 已上传；旧 tracker/tracker_oof5 不是当前版本 |
| 方案 4 研究汇总 | [research_report.json](scheme4_pipeline_20260914/research_report.json) | 嵌套外层 F1 66.3849、24047 次调用 |
| 方案 4 逐帧研究输出 | [scheme4_research_predictions.jsonl](scheme4_pipeline_20260914/scheme4_research_predictions.jsonl) | 6059 帧，分别包含旧 Gate、v2 外层和全量拟合预测 |
| Gate 研究数组 | [scheme4_gate_arrays.npz](scheme4_pipeline_20260914/scheme4_gate_arrays.npz) | 特征、路由、分数、阶段等；labels 是预测差异训练目标，不是 GT 文件 |
| 当前完整 pipeline 输出 | [scheme4_pipeline_predictions.jsonl](scheme4_pipeline_20260914/scheme4_pipeline_predictions.jsonl) | 6059 帧；逐帧验证匹配全量模型，23600 次调用；不能冒充外层结果 |
| 当前完整 pipeline 评分 | [pipeline_fullfit_scores.json](scheme4_pipeline_20260914/pipeline_fullfit_scores.json) | F1 66.4002，全量拟合在 Training 的回放 |
| 方案 5、6 最终配对输出 | [schemes56_predictions.json](scheme4_pipeline_20260914/schemes56_predictions.json) | 16 帧三组输出；两方案未通过检查，不是当前默认 |
| 方案 5、6 费用与完成记录 | [schemes56_collection_receipt.json](scheme4_pipeline_20260914/schemes56_collection_receipt.json) | 208 次请求全部结算；原始账本及回复本地保留 |
| 方案 5、6 评分与审计 | [quality_report.json](scheme4_pipeline_20260914/quality_report.json)、[completion_audit.json](scheme4_pipeline_20260914/completion_audit.json) | 包含失败情况、逐视频结果与提示改变子集 |
| 与无 Tracker 比较 | [baseline_comparison.json](scheme4_pipeline_20260914/baseline_comparison.json) | 历史参考、配对比较与费用口径分开 |
| 原探针复现 | [probe_reproduction_receipt.json](scheme4_pipeline_20260914/probe_reproduction_receipt.json) | 仅证明历史报告复现，不代表当前 pipeline 的评分 |

文件来源、SHA-256、大小和数据状态见 [latest_data_manifest.json](scheme4_pipeline_20260914/latest_data_manifest.json)。上次已上传汇总与模型；本次补齐逐帧数据、研究数组和采集完成凭据，且核对原有六份汇总与本地最新结果一致。

## 已标注为历史、失败或已替代的数据

| 位置 | 状态／处理 |
|---|---|
| `artifacts/research/tracker_scheme4_20260914_r1/` | SUPERSEDED：使用 r2，不能作为最终真实 Tracker 接口验证结果 |
| `artifacts/research/tracker_schemes56_pilot_20260914_r1/`、`r2/` | FAILED_PREPARATION：离线准备失败，无付费采集 |
| 同系列 `r3/`、`r4/` | SUPERSEDED_PREPARATION：旧预览，最终采集是 r5 |
| 同系列 `r5/` | COMPLETED_NOT_ADOPTED：保留，不删；这是方案 5/6 的正式小试验原始证据 |
| `artifacts/preflight/tracker_scheme4_default_20260914_r1/` | LEGACY_PREFLIGHT_NOT_SCHEME4：虽然名称含 scheme4，实际是切换前的旧 PGP 检查 |
| `artifacts/preflight/scheme4_prepared_zero_20260914/` | PREFLIGHT_ONLY：零预算准备检查，不能当质量结果 |
| `artifacts/training/tracker/`、`artifacts/training/tracker_oof5/` | LEGACY_NOT_CURRENT：保留历史引用；工作区原有缺失文件不在本次同步中删除 |
| `docs/experiments/tracker_pipeline_status_20260914/` | HISTORICAL_RESEARCH：早期探针，参看其 DATA_STATUS.md |

这些目录已增加 DATA_STATUS.md；ignored 的本地目录标记通过公开 manifest 和本索引同步其状态，原结果字节未改。

## 可清理的临时数据

`tmp/scheme4_release_index_20260914.zip`、对应导出目录、`tmp/scheme4_code_snapshot_20260914/`、零预算临时配置和临时估时脚本已标为 DISPOSABLE。它们不是运行依赖或原始实验记录，本次选择标注而非删除。本地清单为 `tmp/SCHEME4_DISPOSABLE_DATA.md`。

原图、视频、GT 原文件、API 请求/回复日志和预算数据库没有上传；它们仍在本地。不能据此声称 GitHub 包含从原始数据开始复现的一切输入。

## 尚不存在的新结果

[H0／R／G／T 新消融设计](../PIPELINE_MODULE_ABLATION_DESIGN_2026-09-14.md)仅完成设计同步，新的四格消融尚未跑。不能把当前结果换名称后填入新表。
