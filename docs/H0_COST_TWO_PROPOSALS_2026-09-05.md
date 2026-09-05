# 三帧 H0 成本优化：两个独立方案

状态：模板、请求规范与对照计划已准备；尚未执行付费实验。默认 Pipeline、冻结版本及历史预测没有改动。

两个方案共用同一个对照，不能把精简输入和 minimal 同时打开后声称知道是哪项产生了效果。

| 组 | 输入模板 | reasoning.effort | 对照目的 |
|---|---|---|---|
| baseline_low | 原模板 | low | 两个方案共用的基线 |
| lean_low | 精简模板 | low | 只检验精简输入 |
| original_minimal | 原模板 | minimal | 只检验思考档位 |

三组均使用 OpenRouter `qwen/qwen3.8-max-0902`，输入 `[t-50,t-25,t]`，对应 `[-2,-1,0]` 秒。历史两图 low detail、目标图 high detail。图片字节、顺序、temperature=0、max_tokens=4096、完整五头输出 Schema 均一致。不使用 Tracker、Repair 或跨窗口记忆。

## 方案一：精简重复输入，保留 low

删除纯 H0 没有使用的空 track/workflow/历史预测字段及相关提示；重复的帧编号只保留一份。删除系统提示词末尾复制的完整 JSON Schema，保留 API 的严格 `response_format.json_schema`。

动态输入示例：

```json
{"frame_ids":[22901,22926,22951],"relative_seconds":[-2,-1,0],"target_frame_id":22951}
```

图片仍单独上传。video_id、源图片哈希、请求哈希和本体版本继续由本地请求来源记录管理。完整 100 类 IVT 映射、所有任务本体、null 标签边界、只预测目标帧和不得读取未来帧的要求均保留。

示例文本长度：系统提示词 6338 → 4893 字符；动态输入 578 → 86 字符；合计 6916 → 4979 字符，减少约 28%。这是字符变化，不是 token 或账单降幅，且没有计算仍保留的 response_format 与图片。

风险：移除重复 Schema 或空状态上下文仍可能影响模型行为。既检查输出合法率，也检查 IVT、Verb、Target 的真实语义评分。不能将语法通过当成成功。

## 方案二：保留原输入，只比较 low/minimal

基线参数：

```json
{"reasoning":{"effort":"low"},"max_tokens":4096,"temperature":0}
```

候选参数：

```json
{"reasoning":{"effort":"minimal"},"max_tokens":4096,"temperature":0}
```

原始输入文字、图像、输出 Schema 及所有其他实际请求字段保持完全一致。这个方案不使用精简输入，也不同时降低 max_tokens。

本次已读取的 OpenRouter 模型元数据支持 minimal，且标记思考为 mandatory；这里降低思考档位，不尝试关闭思考。此前三帧成功回复平均约 735 个思考 token、73 个最终 JSON token，思考费用约占总费用 37%。minimal 的实际节省与准确率变化尚未测量，不能预先承诺节省比例。

## 输出 JSON：三组完全一致

以下仅表示格式，不是某张图的预测：

```json
{
  "schema_version":"joint_perception_final_only_v1",
  "instrument":{"selected_ids":[0,2]},
  "verb":{"selected_ids":[1,2]},
  "target":{"selected_ids":[0]},
  "ivt":{"selected_ids":[17,60]},
  "phase":{"selected_id":3}
}
```

## 最小对照计划与预算

从原先冻结的 Training 目标序列按位置选取 8 个目标：四个视频各两个。选择依据为固定位置，没有用 GT 标签、已有预测对错或置信度挑样本。

| 视频 | 目标帧 1 | 目标帧 2 |
|---|---:|---:|
| VID103 | 22951 | 36076 |
| VID23 | 17326 | 28876 |
| VID31 | 48751 | 77776 |
| VID96 | 16676 | 25726 |

这八个目标没有参与此前完成的 32 目标付费预测，但原研究中断评分曾读取冻结目标的离线 GT；不能把它们描述为从未接触过的新测试集。本次准备过程仅读取冻结计划和请求，不读取 GT。

8 个目标 × 3 组 = 24 次调用。按原三帧实测均价粗估约 $0.29，建议规划预算 $0.50、并发 2、无自动推理重试。该金额是待执行计划，不是已产生费用或已实现的金额硬上限。执行前需检查当时余额，加载并核验原图，再计算每个新请求的规范哈希。

轮换组间调用顺序；三组应在同一轮调用，不能拿很久以前缓存的基线充当新的对照。分别比较 lean_low 对 baseline_low、original_minimal 对 baseline_low。

记录输入 token、服务端缓存读写 token、思考 token、最终输出 token、账单费用、延迟、失败原因及五头语义指标。主结果保留全部有效 GT 目标，失败按约定计入分母；另报共同成功目标的配对比较。8 个目标只用于发现明显退化和校准费用，不用于宣布全局最优或验证非劣效。

## 已准备的文件与检查

- 请求构建器：`scripts/prepare_h0_cost_optimization.py`，没有 API 发送入口。
- 待执行规范：`artifacts/preflight/h0_cost_two_proposals_20260905/plan.json`，以及同目录 24 个 `requests/*.json`。
- 同目录提供原版/精简版完整 system 文本、动态输入示例和 `output_schema.json`。
- 请求规范保留原请求来源元数据；其中旧 request_hash 只标识源请求，不是修改后的新请求哈希。
- 3 个针对性测试通过：实际网络请求内容中 minimal 只改变 reasoning；精简输入保持图像、Schema、本体和标签边界；拒绝带状态或图像顺序错误的来源。Ruff 检查通过。
