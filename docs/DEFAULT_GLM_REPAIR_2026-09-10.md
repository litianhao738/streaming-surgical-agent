# 默认审核席：GLM-5.3-Flash，v1.3.0

按用户要求，默认版本为`parallel-phase-repair-v1.3.0-glm-low`。GLM替换v1.2.0的Ministral 8B席，四头和Phase两条分支均生效。尚未提交或推送GitHub。

| 审核模型 | 实际型号 | 请求设置 |
|---|---|---|
| GLM | `z-ai/glm-5.3-flash` | OpenRouter，仅Together，禁止路由回退；`reasoning={"effort":"low","exclude":true}` |
| Qwen | `qwen3.5-35b-a3b` | 原阿里云工作空间；thinking关闭 |
| GPT | `openai/gpt-5.6-luna` | 原OpenRouter设置 |
| Gemini | `google/gemini-3.5-flash-lite` | 原OpenRouter设置，minimal推理 |
| DeepSeek | `deepseek/deepseek-v4-flash-vision-exp` | 原OpenRouter设置 |

GLM官方明确不能关闭thinking；OpenRouter模型目录也返回`mandatory=true`，支持`max/high/low`，默认`max`。因此显式使用最低`low`。`exclude=true`仅隐藏返回的推理文字，不停止内部推理、不免除推理token费用。不会发送不支持的`none`或`enabled=false`。

依据：[Z.ai官方文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)、[OpenRouter推理参数](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。本地元数据保存在`artifacts/preflight/glm_default_smoke_20260910_v1`。

## 流程与兼容

固定H0后，两路并行：图谱备选关系→原单模型补候选→五模型四头评分→Python局部修改；另一条由五模型独立看三帧判断Phase→Python合并。两路内部也并行。四头保持五份有效意见、新增均分≥4／删除≤2；Phase保持五份有效意见、至少三票一致。未增加轮次，未更改图片、提示词、JSON协议、H0或接纳规则。

入口`scripts/run_pipeline.py`由默认清单转入`scripts/run_glm_parallel_repair.py`。GLM复用OpenRouter凭据，费用记入OpenRouter；内部旧席键`grok`在该版本表示GLM，实际身份以`model`和`reviewer_families`为准。适配器只允许该席的强制推理，不放松模型身份、提供方、拒答、截断或语义约束检查。已准备的旧版本目录继续委托原执行器。

旧默认清单完整备份：`configs/defaults/parallel-phase-repair-v1.2.0-lightweight-reviewers.json`，旧执行器、冻结样本和历史结果保留。当前执行器范围仍是固定八目标缓存H0实验；本次不扩展到完整数据集、Tracker或Gate。

## 真实接口检查

固定旧八目标中按原顺序前两个Training目标：VID103/18326、33576。复用原图片和已生成候选池，GLM各做一次四头审核、一次Phase审核。无新H0／候选调用，无GT进入请求；不是修复效果对照。

先冻结Fireworks路由检查4次，均返回共享池上游429，未产生模型判断。保留失败记录，不把失败时长计为有效推理速度。首轮脚本还误用了要求五席输入的归一化函数，触发`KeyError`；已改为单席归一化，旧源码保存在冻结目录，未重发首轮请求。

随后固定Together路由，再发4次，4/4成功；四头共17项意见通过现有约束，两个Phase结果均通过单选协议检查。返回型号与提供方正确，无推理文字返回。

| GLM单席计时 | VID103/18326 | VID103/33576 | 平均 |
|---|---:|---:|---:|
| 四头审核 | 7.02秒 | 5.74秒 | 6.38秒 |
| Phase审核 | 2.71秒 | 2.05秒 | 2.38秒 |

Together四次请求以并发2运行，总墙钟9.08秒（不含此前输入构建）。这不是完整五席或Pipeline耗时，也不是与Ministral在同批任务上的配对速度对照。四头两次返回推理token为99、110；Phase两次为0，不能据此宣称关闭了推理。

Together原生费用合计**$0.00268350**。Fireworks四次429无usage，保守保留未知费用上界**$0.04807710**，不冒称实际扣费，也不宣称这部分免费。每轮上限$0.20，无自动重试。

记录：`artifacts/preflight/glm_default_smoke_20260910_v1`和`glm_default_smoke_20260910_v2`。60项相关测试通过，改动Python文件Ruff通过。该检查证明新接口可用，不证明F1／Precision提高或该型号是最优模型。

默认入口已完成八目标计划准备，未执行该八目标付费实验。目录`artifacts/preflight/default_glm_prepare_20260910_v1`；计划SHA256：`3a8285230f5ebe3eef229e9172b02a86e9c03fbe3b3770b700775370d4ece8de`。40份Phase请求的冻结指纹均匹配，v1.2.0原计划也仍通过原版本校验。
