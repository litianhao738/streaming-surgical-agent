# Tracker 作为审核证据：r3 配对试验协议（在任何付费请求和结果之前冻结）

## 背景

r2（`artifacts/training/gate/tracker_review_evidence_20260914_r2`）只完成了 2 对，没有精度结论。中断原因是 `VID96_5901` **对照组**（不带 Tracker）的 GLM 返回 HTTP 400、错误码 `1301`（内容安全拒答），触发了全局停止。

这类拒答并不新：历史 6,059 行里有 267 行出现过 `CONTENT_REJECTED`，正式流水线一直按首轮 fallback 把它当作该席位无效，而不是停掉整批。r2 的停止条件比正式流程更严。

## 设计：全部沿用 r2

- 协议：`docs/PGP_TRACKER_REVIEW_EVIDENCE_PROTOCOL_2026-09-14.md`
- 脚本：`scripts/assess_tracker_review_evidence.py` 的 `prepare / collect / score`

沿用的内容：
- 冻结 Gate 放行、且 Qwen 初审后仍需交互审核的帧；
- 每个 Training 视频按时间分位取 16 帧，共 64 帧。`prepare` 是确定性的，与 r2 同一批帧；
- 每帧两组：不带证据的对照组、带 Tracker 证据的处理组。组的先后按样本轮换；
- 同一帧两组的 H0、候选池、图像、Qwen 初审、模型、参数、输出格式和阶段回答都相同。唯一区别是给剩下 4 位交互审核员的两项 Tracker 提示字段；
- 上限：OpenRouter $5，GLM 128 次请求，DeepSeek 128 次请求，其余账户为 0；最多 512 次新请求；
- 不自动重试，不换模型，不替换或删除失败样本；
- 主要判定沿用 r2 的五项检查（`pilot_screen_pass`）：
  - 处理组五头平均 F1 不下降；
  - FP+FN 不增加；
  - 至少一项严格改善；
  - 每个视频都不变差；
  - 已知美元成本不超过对照组的 110%。

## 唯一修订

官方路由（GLM / DeepSeek / Qwen）上的**内容拒答**（阿里云 `data_inspection_failed`、GLM `1301`）记为 `API_FAILED`，按正式流水线首轮 fallback 视为该席位无效，**不再触发全局停止**。

其他全局停止条件都不变：
- 官方路由上其他 400–404；
- OpenRouter 路由上的 400–404；
- 身份或路由不一致；
- 预算耗尽；
- 结果不确定的请求。

实现方式：`scripts/assess_tracker_review_evidence_r3.py` 继承 `Calls`，只替换拒答判断这一行，其余调用、记账、写盘逻辑与原实现逐行相同。r3 的协议文件和脚本哈希都写入计划，由 `verify` 在执行前核对。

## 预先声明的次要分析（不改变主要判定）

1. 两组各自的拒答率、无效率，以及各服务方的请求数。
2. 两组结果都叠加 v2 出口模块（M1 Tracker 融合 + M2 单功能器械定动作，见 `HANDOFF_TRACKER_PIPELINE_V2_2026-09-14.md`）后，比较交互四头和五头结果。
3. 按"Tracker 看到了器械、但答案里没有对应三元组"与否分层，只做描述。

## 局限

- 64 帧的条件性试验只能决定是否值得扩大，不能用来替换默认，也不能称为论文级结论。
- 不访问 Testing 和 VID110，不部署，不修改任何冻结文件或默认指针。
- r2 目录及其停止标记保持原样。

## 执行

应用户 2026-09-14 的要求执行，输出目录为 `artifacts/training/gate/tracker_review_evidence_20260914_r3`。
