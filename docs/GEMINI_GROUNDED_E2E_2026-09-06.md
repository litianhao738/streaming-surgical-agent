# Gemini 联合 H0 → grounded Verifier / Repair 小批量实验

日期：2026-09-06。本报告对应用户授权的 OpenRouter Gemini 同步 API 测试，不使用 Batch。

**已真实完成两批共十二个目标、四十次调用，累计 $0.13957350。完整五头 GT 的四目标上，最终五头 micro-F1 均未高于 H0，Target 集合 Accuracy 从 50% 降至 25%。当前证据不支持将该修复链路替换默认基线。**

## 实际实现与冻结边界

独立端到端入口为 `scripts/run_grounded_api_pipeline.py`，运行模块为
`src/surgical_agent/research/verification/grounded_pipeline.py`。实际顺序：

`当前联合 H0 → 盲接触区域定位 → 原图与裁剪生成候选 H1 → 双假设 Verifier → 接纳 H1 或保留 H0`。

- H0 复用 `JointPerceptionRequestBuilder`、主版本完整 prompt、本体和 final-only 五头 Schema。
- 本实验仅替换模型为 `google/gemini-3.8-flash`，严格 Google AI Studio 路由、禁止供应商回退；各阶段都使用同一 Gemini。不是 Qwen 与 Gemini 准确率对比。
- 三帧因果输入 `[t-50,t-25,t]`；开头或缺帧后使用真实短历史。历史图片 low detail、目标 high detail。temperature=0、reasoning=low、max_tokens=4096。
- H0 系统 prompt SHA-256 为 `c1009de8429158eab6b9c0f4ee39b12f51e5e970335f0c554866198191b6e3e5`。
- Locator 和 Proposal 不接收 H0 答案。Proposal 和 Review 接收因果原图与目标图实际像素裁剪；Review 比较本次原始 H0 与 H1，保存假设顺序和请求哈希绑定。
- 每目标最多四次调用、无自动重试、没有循环。候选与 H0 各头集合相同则跳过 Review；可选阶段失败保留 H0。Phase 始终保留 H0。
- 接纳仍使用历史 grounded 视觉条件，经 final-only 边界校验；没有按本次 GT 调整 prompt、接纳阈值或特判。
- 同模型的盲调用不代表错误独立，接纳条件通过也不证明语义正确。

已发布的默认 `run_dataset_api_pipeline.py` 仍是纯 Qwen H0。旧 Tracker/Gate 完整研究入口没有被强行接入 final-only 硬标签。本工作没有操作 Tracker 训练、权重或输出，没有 Git 提交或推送。

## 第一批：工程链路测试与标注覆盖问题

原计划从 Training 的 VID02、VID04、VID11、VID17 各取可用时间线的 `floor(n/3)`、`floor(2n/3)`，共八目标，最多 32 次调用、预算 $1。本批固定目标没有读取任务标签；所有模型输出保存后才评分。

| 视频 | 目标帧 |
|---|---|
| VID02 | 31076、53401 |
| VID04 | 18676、29551 |
| VID11 | 26376、53526 |
| VID17 | 13401、23051 |

八目标全部得到合法 H0，产生六个候选、三次 Review、两次 ACCEPT。但评分发现八目标的 Verb、Target、IVT mask 全为 false，只能评分 Instrument 和 Phase。两次 ACCEPT 修改的都是缺失 GT 的任务，应记为 **两次不可评分替换**，不能称为改对、改坏或无收益。

Instrument 在 H0、H1 策略及 final 均为 TP=11、FP=0、FN=0，集合 Accuracy 为 8/8；Phase 均为 TP=8、FP=0、FN=0，集合 Accuracy 为 8/8。两头 micro-F1 均为 100%。其余三头有效目标数为零，指标是缺失值，不能写成 0%。

两次 Locator 输出框坐标反向（VID11/26376 与 VID17/23051，left > right），校验拒绝并保留 H0，没有付费重试。预测、原始 SSE usage、调用记录与费用逐笔一致。

| 阶段 | 实际调用 | USD |
|---|---:|---:|
| H0 | 8 | 0.02736900 |
| Locator | 8 | 0.01472475 |
| Proposal | 6 | 0.02601600 |
| Review | 3 | 0.01461150 |
| 修复新增 | 17 | 0.05535225 |
| 合计 | 25 | 0.08272125 |

本批只证明调用、回退、记录和评分管道跑通，不能回答 IVT 修复有无收益。原目录和预测保持不变；在增加标注预检前，已按原 plan 的哈希冻结八个执行源码文件至 `frozen_source/`，逐项验证一致，记录于 `frozen_source_manifest.json`。

## 补充批：固定完整标注样本

为完成语义效果检查，补充四个 Training 目标：VID103/18501、33751；VID96/14951、25951。视频来自已有文档确认有动作标注的 Training 范围，时间点仍按可用时间线 `floor(n/3)`、`floor(2n/3)` 固定。预检仅核查标注可用性，五头 mask 均为 true；没有按标签值、模型错误或期望收益挑选时间点。

本批单独目录、最多 16 次调用、预算 $0.50，与首批已发生费用合计仍在原 $1 计划内。全部提示词、模型、图像机制和接纳规则与首批相同。新增 `--require-all-task-gt` 在付费调用前检查固定样本的任务 mask，不满足则直接停止；GT 值不进入模型请求。

指标在保存本批全部预测后计算：逐头排除缺失 GT，以相同目标比较 H0、候选策略 H1、最终策略。没有候选时 H1 策略保留 H0，同时另报有候选的共同子集。改对/改坏按各头集合对称差减少/增加统计，另列由错误变为完全正确、由完全正确变错；多个头一好一坏单列 mixed。换标签但损失相同记为无收益替换，不与不可评分替换混淆。

### 四目标语义结果

每格为 **micro-F1 / 集合 Accuracy**。五头各有四个有效 GT，H0 均成功；只有三个目标有有效 H1，候选策略在另一个目标回退 H0。三臂使用同一份 H0，不重新抽样初始预测。

| 任务 | H0 | 候选 H1 策略 | Verifier 后 final |
|---|---:|---:|---:|
| Instrument | 90.91% / 75% | 90.91% / 75% | 90.91% / 75% |
| Verb | 57.14% / 0% | 57.14% / 0% | 57.14% / 0% |
| Target | 72.73% / 50% | 60.00% / 0% | 72.73% / 25% |
| IVT | 42.86% / 0% | 28.57% / 0% | 42.86% / 0% |
| Phase | 75.00% / 75% | 75.00% / 75% | 75.00% / 75% |

三臂五头同时完全正确均为 0/4。F1 相同不代表每个目标结果相同：Target 的一处减错与另一处新增错误在汇总 TP/FP/FN 上抵消，但原本完整正确的目标减少了一个。

| 目标 | 候选与复核行为 | 最终相对 H0 的变化 |
|---|---|---|
| VID103/18501 | H1 将 IVT 58 换成 60，丢掉正确 Target 2；Verifier 选择 H0 | 拦住一次有害候选，保留 H0 的已有错误 |
| VID103/33751 | H1 新增 null IVT 94，Verifier 接纳 | 改坏：Verb/Target/IVT 各新增一个错误标签；Target 从集合全对变错 |
| VID96/14951 | H1 去掉部分错误标签并更换 IVT，Verifier 接纳 | 部分减错：Verb/Target/IVT 对称差各减少 1；没有补中任何新 GT，IVT TP 仍为 0 |
| VID96/25951 | Proposal 对可见区域返回 INSUFFICIENT，无有效 H1，不调用 Review | 保留 H0；器械、动作、目标、IVT 漏检和 Phase 错误仍在 |

最终接纳两次：**部分减错 1、改坏 1、等损失的无收益替换 0**；保留 H0 两次。完全修复为五头全对 0，mixed 0。逐头统计中 Verb、Target、IVT 各改善 1、恶化 1，Instrument 和 Phase 均未改。Target 有一次 exact→wrong，没有任何头出现 wrong→exact。

三个候选本身是两次恶化、一次部分减错。Verifier 拒绝其中一个恶化候选，接纳另一个恶化候选和部分减错候选。本次观察到一次过滤成功，也观察到一次错误接纳；不足以证明 Verifier 有稳定的净收益。

有效 H1 没有增加任何正确 IVT。VID96/14951 的“改善”来自减少错误标签，不应宣传为正确识别了其 GT IVT。所有可评分目标的 IVT 集合 Accuracy 仍为零。

### 补充批与本轮总费用

| 阶段 | 补充批实际调用 | 补充批 USD |
|---|---:|---:|
| H0 | 4 | 0.01387350 |
| Locator | 4 | 0.00755700 |
| Proposal | 4 | 0.01934100 |
| Review | 3 | 0.01608075 |
| 修复新增 | 11 | 0.04297875 |
| 合计 | 15 | 0.05685225 |

两批合计 **40 次调用 / $0.13957350**：H0 为 12 次 / $0.04124250，Verifier/Repair 新增 28 次 / $0.09833100。全部请求费用可核对；无重试、无未定价调用。本报告不计入此前其他会话的 Gemini 三目标纯 H0 费用或旧 Qwen 六目标实验费用。

### 判断与下一步

事实：链路真实跑通；最终结果没有获得微平均 F1 净增益，并损失了一处 Target 集合正确率。工程校验能拦住反向框、无效候选和不充分证据，但不能保证视觉标签正确。

从本批行为推断，当前语义限制同时存在于候选和复核：候选未补中新的正确 IVT，复核也会认可有害的 null 替换。VID96/25951 的证据不足是模型返回的判断，不能直接当成图像客观上无法识别；仅靠这四个目标也不能区分视觉误判与标注类别定义理解偏差的全部来源。

因此保留已冻结的纯 H0 基线，保留本入口作为可复现实验，不提升为默认方案。下一步优先整理固定 H0 实验与复现材料；若继续研究 Repair，应先离线审查保存的错误案例及本体定义，确认候选是否可能增加正确 IVT，再另行预注册样本与预算。没有依据继续修改 prompt、增加轮数或根据这四帧 GT 写特判。

## 原始记录

- 第一批：`artifacts/preflight/grounded_gemini_e2e_20260906/`。
- 补充批：`artifacts/preflight/grounded_gemini_semantic_supplement_20260906/`。
- 各目录的 `plan.json`、`requests/` 冻结样本、请求、源代码与图像哈希、模型、路由、参数和价格快照。
- `predictions.json` 保存 H0/H1/final、定位、候选、复核及回退原因；`scored_predictions.json` 在推理后关联 GT/mask；`summary.json` 保存三臂指标和配对统计。
- `calls/` 保存脱敏请求、原始 SSE、响应和费用；执行锁禁止原目录重复付费派发。
- `artifacts/preflight/grounded_gemini_e2e_audit_20260906/combined_audit.json` 独立复算两批三臂 TP/FP/FN/集合正确数、实际替换与不可评分替换，并核对累计账目。两批均已保存与各自 plan 哈希一致的 `frozen_source/`。
- 预算属于本地调度限制，每次调用前预留 $0.04，不是供应商强制封顶。费用以真实响应 usage.cost 逐笔计账，含未通过 Schema 的已付费响应。

Testing 未用于本轮选样、调参或选方案；这些小样本不能作为完整 split 的模型成绩。

## 复现入口

工程验证：72 项相关单元与集成测试通过，覆盖因果短历史、真实像素裁剪、H0 盲候选、Review 请求绑定、可选失败回退、费用记录、final-only 边界、任务 mask、配对评分及付费前标注覆盖检查。相关 Python 文件通过 Ruff。真实调用与离线测试分别记录，测试通过不作为语义效果证据。

下列命令只做预检，不发送 API。复现实验必须换一个不存在的输出目录；已执行目录有锁，不应删除锁后重跑。

```powershell
.venv-p2/Scripts/python.exe scripts/run_grounded_api_pipeline.py --dataset-root D:/cholec_dataset --output artifacts/preflight/grounded_gemini_reproduction_new --pricing-snapshot artifacts/preflight/grounded_gemini_e2e_readiness_20260906/endpoints.json --budget-usd 0.50 --reserve-usd 0.04 --videos VID103 VID96 --require-all-task-gt
```

确认计划里的样本、五头覆盖、调用数和预算后，使用同一命令追加 `--api-key-file <本地凭据文件> --execute` 才会真实付费。凭据内容不进入请求日志；本轮产物完成密钥扫描。环境安装与数据路径配置沿用 `API_EXPERIMENT_QUICKSTART_2026-09-06.md`，真实 API runner 使用已安装项目依赖的 Python 环境。

离线重算可直接把 `scored_predictions.json` 的列表传给 `compute_repair_comparison`；它不建立 API 连接。如需重新核对源 GT，可从保存的预测调用 `score_saved` 与同版本数据适配器，不能再调用模型。首次八目标执行源码以其 `frozen_source/` 为准；补充批以自己的 plan 记录为准。
