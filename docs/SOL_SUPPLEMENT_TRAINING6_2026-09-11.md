# GPT 5.6 Sol 补测：同一组六个 Training 目标

主线仍为 `prior-gated-joint-mainline-v1.0.0`，基座选择仍为 Gemini 3.8 Flash。未启动四视频全量采集或 Gate 数据收集。

本次对原来 H0 被 403 拒绝的 VID103_18501、VID23_28551 各补一次，两次均成功，依赖的主线审核也完成。原来四个成功目标直接保留。模型 `openai/gpt-5.6-sol`、OpenAI 严格路由、图片和提示不变；预检逐项比对原 H0 请求。先前的拒绝不能据此认定为永久限制，也无法仅凭这次成功确定其内部审核原因。

| 基座 | 完成目标 | H0 五头平均 F1 | 主线五头平均 F1 | 主线 IVT F1 | 主线 Phase F1 |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 6/6 | 74.70 | 75.08 | 66.67 | 50.00 |
| Qwen 3.8 Max | 6/6 | 61.80 | 61.28 | 30.00 | 83.33 |
| GPT 5.6 Sol | 6/6 | 49.88 | 58.99 | 36.36 | 50.00 |

指标单位为百分数。各头采用汇总 TP/FP/FN 的 micro-F1，再对五头取平均；此处三者均有六个 Phase 输出，Phase F1 等于准确率。最终总错漏分别为 24、43、41。

本次新增 26 次实际调用，全成功，全部通过零调用原始请求/响应重放。新增 API 执行耗时合计 107.41 秒（约 1 分 47 秒，不包含准备及处理中断）；OpenRouter 原生费用 $0.08745022，阿里云费用保守估算 ¥0.050956。加上 Sol 原先实际调用费用，累计约 $0.26596122 + ¥0.164879。原先两次 403 由服务商明确声明不扣费，原账本中的未知预占仍保留在历史档案，不计入已确认支出。

第一次补测在 8 次响应后触发本地预算预占上限。恢复时直接重放这 8 次，沿用原六样本实验每模型 $4 + ¥0.7 的预算上限，只发送剩余 18 次；没有重复发送已完成的请求。v1 中间评分不代表完成的补测，最终结果以 resume1 为准。

结论：这组六目标上 Gemini 仍领先，继续推荐其作为后续四个 Training 视频的采集基座。Sol 获得第二次尝试，不能将此合并表改写为原始单次无重试实验；原比较保持不变。六样本仅支持开发阶段初筛，不证明普遍最优。

证据：

- 最终报告：`artifacts/preflight/mainline_sol_supplement_20260911_v1_resume1/comparison.json`
- 六目标合并预测：同目录 `predictions_composite.json`
- 新增原始响应及预算：上述目录与 `artifacts/preflight/mainline_sol_supplement_20260911_v1/sol/`
- 两阶段 `completion.json` 证据哈希均已核验；DEFAULT、BEST、TRAINING_BASE_MODEL_SELECTION 三个根指针哈希保持不变。
- 实验入口：`scripts/supplement_sol_mainline.py`；每阶段保留 `frozen_supplement.py`，档案存在时禁止重复执行。
