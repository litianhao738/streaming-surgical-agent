# Verifier 逐项审核小测试 — 2026-09-07

本次按用户新指令执行一次独立的小测试。**第一笔真实审核再次超时，有效审核为 0；尚无新 Verifier 的语义效果结果。**没有修改 H0、审核提示词、候选、接纳规则或超时设置，没有自动重试。

## 冻结范围与执行

- 输入复用 `artifacts/preflight/grounded_gemini_semantic_supplement_20260906` 的已有 Gemini H0、定位、候选和原始图片；没有重新调用这些阶段。
- 模型 `google/gemini-3.8-flash`，OpenRouter 同步请求，严格 Google AI Studio 路由；temperature=0、reasoning=low、max output tokens=4096。
- 方案 A：每项增删均通过核验，才整份接纳候选。方案 B：由同一审核结果只应用允许的局部修改及依赖约束。Phase 保留 H0。
- 最大 3 次新审核，单目标最多 1 次；本次预算 $0.20，每笔预留 $0.05。预算是本地调度限制，非供应商强制封顶。此前未定价请求单独保留，不记为免费，也不并入本次已知费用。
- 价格快照在调用前从 OpenRouter 公开接口刷新；预检与 53 项相关测试通过。
- `plan_sha256`: `a898b15235d67d83d1191a2b65c7282f2e7a9084b2c067bfbae3a66bb0d609c1`。

| Training 目标 | 本次行为 | A/B 输出 |
| --- | --- | --- |
| VID103 / 18501 | 第一次审核超时，实测约 196.09 秒（传输配置 180 秒） | 失败回退 H0 |
| VID103 / 33751 | 费用保护停止后，未派发 | H0 |
| VID96 / 14951 | 费用保护停止后，未派发 | H0 |
| VID96 / 25951 | 原本无有效候选，计划零调用 | H0 |

实际派发 1 次，成功审核 0 次，未定价调用 1 次，停止原因 `UNPRICED_CALL_STOP`。没有收到 HTTP 响应记录、SSE 或 generation ID，无法凭现有记录查询该笔生成费用。

## 评分与解释

所有四个原目标都计入评分，各头有效 GT 数量均为 4。GT/mask 在新预测持久化后关联；通用评分器按任务 mask 排除缺失 GT。Testing 未参与。下表 H0、候选和旧审核来自历史记录；**新 A/B 列只是失败或未调用时的 H0 回退值，不能作为成功审核后的模型成绩。**

每格为 micro-F1 / 集合 Accuracy。

| 任务 | H0 | 原候选策略 | 原审核结果 | 新 A/B 回退值 |
| --- | --- | --- | --- | --- |
| Instrument | 90.91% / 75% | 90.91% / 75% | 90.91% / 75% | 90.91% / 75% |
| Verb | 57.14% / 0% | 57.14% / 0% | 57.14% / 0% | 57.14% / 0% |
| Target | 72.73% / 50% | 60.00% / 0% | 72.73% / 25% | 72.73% / 50% |
| IVT | 42.86% / 0% | 28.57% / 0% | 42.86% / 0% | 42.86% / 0% |
| Phase | 75.00% / 75% | 75.00% / 75% | 75.00% / 75% | 75.00% / 75% |

新 A/B 相对 H0：部分减错 0、改坏 0、无收益替换 0、mixed 0、未改变 4；完整改对 0，各头 wrong→exact / exact→wrong 均为 0。原因是没有得到任何有效审核，不能解释为 Verifier 成功避免了误改。五头同时全对仍为 0/4。

独立只读复算核对了所有对照臂的 TP/FP/FN、集合正确数和任务 mask，与保存报告一致。上述四个目标已被检查过，仅用于机制诊断，不能证明泛化收益。

## 费用与连接检查

本次 `known_added_cost_usd=0.0` 仅表示没有取得任何已知价格；`added_cost_usd=null`、`accounting_complete=false`。**实际新增费用未知，不能写成 $0。**加上前一次独立实验，累计两笔请求费用待确认；历史目录未覆盖。

失败后仅进行了不生成内容的 `GET /api/v1/key` 检查，HTTP 200，鉴权成功；没有输出或记录密钥及账户详情，也没有新增模型请求。价格与鉴权接口可访问，只能确认部分连接及凭据正常，不能证明推理路径、路由或模型服务正常。当前证据不足以确定超时根因。

下一步先排查推理调用链路并核对未定价记录。暂不据此改变审核方案、增加轮数或宣布语义成功/失败；默认 H0 继续冻结。

## 记录和复现

本次完整记录：`artifacts/preflight/diff_review_gemini_smalltest_20260907/`。

- `plan.json`、`frozen_source/`、`requests/`：冻结样本、代码、图片和请求哈希。
- `calls/`、`calls_summary.json`：真实派发与超时记录。
- `predictions.json`、`scored_predictions.json`、`summary.json`：四目标输出、独立 GT/mask、评分及未知费用标记。
- `connectivity_check.json`：失败后的只读鉴权检查。

本次使用命令如下。该目录已有执行锁，不应删除锁或重复执行。后续付费运行必须另建目录并明确范围和费用；省略 `--execute` 与凭据参数只做预检。

```powershell
.venv-p2/Scripts/python.exe scripts/run_diff_review_trial.py --source artifacts/preflight/grounded_gemini_semantic_supplement_20260906 --output artifacts/preflight/diff_review_gemini_smalltest_20260907 --pricing-snapshot artifacts/preflight/diff_review_readiness_rerun_20260907/endpoints.json --max-targets 3 --budget-usd 0.20 --reserve-usd 0.05 --execute --api-key-file <本地凭据文件>
```

本轮未修改实验 Python 代码，未暂存、提交或推送 GitHub；历史实验和原有本地修改均保留。
