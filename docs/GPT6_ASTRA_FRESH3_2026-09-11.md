# GPT-6 Astra 同主线三目标补测

GPT-5.6 Sol、Gemini 3.8 Flash 与本次 GPT-6 Astra 均使用 `prior-gated-joint-mainline-v1.0.0`。基座承担 H0、候选提议、Phase 建议三个位置，均经 OpenRouter；五头审核、先验门、Phase 修改规则、三帧图片及语义提示不变，均为 low 推理强度。OpenAI 基座去掉不支持的 temperature，Gemini 保留 temperature=0。审核席 Qwen 仍使用原来的阿里云直连；并非整条主线每个请求都经过 OpenRouter。

本次使用标准 `openai/gpt-6-astra`，严格限定 OpenAI 标准路由，禁止回退。官方图像输入及 low 支持已核实：https://developers.openai.com/api/docs/models/gpt-6-astra 。OpenRouter 路由元数据已保存在实验目录。未使用 Pro、Batch 或其他模型替代。

固定沿用上一轮的 VID103_14326、VID23_9551、VID31_32601，直接复用该轮 Sol/Gemini 结果，本次只付费运行 GPT-6。先验排除查询视频，未调提示或阈值。已有三个样本属于开发数据，不能视为独立大规模确认。

| 基座 | 完成数 | H0 平均 F1 | 最终平均 F1 | 最终 IVT F1 | Phase 正确数 |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 3/3 | 75.45 | 77.69 | 46.15 | 3/3 |
| GPT-6 Astra | 2/3 | 55.44 | 61.00 | 50.00 | 2/3 |
| GPT-5.6 Sol | 2/3 | 39.44 | 47.32 | 44.44 | 0/3 |

失败目标按空预测评分。GPT-6 在 VID31_32601 的 H0 收到 OpenAI 图片审核 403，violence/graphic，明确声明不扣费。Sol 失败的是 VID103_14326；两者完成子集不同，不能直接把整个分数差解释为识别能力差。没有重试、替换样本或更换路由。

## GPT-6 与 Gemini 共同成功的两个目标

共同目标为 VID103_14326 和 VID23_9551；该诊断子集与上一轮 Sol/Gemini 的共同成功子集不同。

| 基座 | H0 平均 F1 | 最终平均 F1 | 最终 IVT F1 | Phase 正确数 | 最终总错漏 |
|---|---:|---:|---:|---:|---:|
| GPT-6 Astra | 70.86 | 78.48 | 66.67 | 2/2 | 6 |
| Gemini 3.8 Flash | 70.86 | 78.00 | 50.00 | 2/2 | 7 |

这两个目标上 GPT-6 的 H0 汇总平均分与 Gemini 相同，经过主线后略高，IVT 更好。不能再概括为 GPT 系列所有任务项都较差；也不能凭两个成功目标的 0.48 个百分点差距认定 GPT-6 普遍更好。端到端固定三目标上 Gemini 完成率更高，本次不修改主线基座推荐。

新增 27 次实际请求（两个完整目标各 13 次，第三个失败 H0 一次），串行处理三个目标、内部审核并行，API 墙钟时间 47.14 秒。OpenRouter 原生费用 $0.062725524，阿里云保守估算 ¥0.045931。403 的账本未知预占保留，独立费用汇总依据服务商声明计零。Gemini、Sol 未重复付费执行。成本仅对应本次实际完成的工作量，不能当作三个完整目标预算。

档案：`artifacts/preflight/gpt6_astra_fresh3_20260911_v1/`。包含 plan、路由元数据、冻结入口与目标执行辅助脚本、模拟配对预检、全部原始请求/响应、账本、预测、completion 哈希、三模型 comparison、真值及 independent_audit。全部 27 次请求通过零调用重放，三模型评分经独立 TP/FP/FN 复算，封存证据哈希核验通过。DEFAULT、BEST、TRAINING_BASE_MODEL_SELECTION 未变；未采集 Gate 或启动四视频全量运行。

入口：`scripts/compare_gpt6_fresh3.py`。归档已存在时拒绝重复执行。
