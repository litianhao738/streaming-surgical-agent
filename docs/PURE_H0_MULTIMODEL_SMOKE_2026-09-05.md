# OpenRouter 多模型纯 H0 冒烟：2026-09-05

> 后续核查纠正：本文null交互不能被理解为必然“没有物理动作”。图像/GT审计与删除top-k的配对结果见 [后续报告](PURE_H0_NULL_AUDIT_AND_TOPK_ABLATION_2026-09-05.md)。历史预测和分数未修改。

## 范围与可用性

用户截图指定 Qwen3.8 Max (0803)、Grok4.6、Gemini3.8 Flash。
实时查询 [OpenRouter 模型目录](https://openrouter.ai/api/v1/models) 与逐模型 endpoints：

- `x-ai/grok-4.6`：存在，支持图像及 structured_outputs；固定 xai 路由，无 provider fallback。
- `google/gemini-3.8-flash`：存在，支持图像及 structured_outputs；固定 google-ai-studio 路由，无 fallback。
- `qwen/qwen3.8-max-0803`：不在当前目录，endpoints 查询返回404。
- 当前可见 `qwen/qwen3.8-max-0902`，支持图像；已询问是否替代，未收到确认前不调用，也不把0902冒充0803。

首轮实际完成 Grok 与 Gemini，未完成 Qwen。模型返回字段与请求 ID 一致。随后用户批准使用 Qwen0902 补测，见文末补测记录。

## 固定输入与评分

- Validation VID110，4076/4101/4126/4151/4176/4201，连续六个目标。
- 每次输入最近三张因果图像；第一窗4026/4051/4076。相同帧、相同顺序、history低细节、target auto。
- `scripts/run_pure_h0_smoke.py` 直接调用联合预测；不构造 Tracker/Gate/Verifier/Repair，不输入预测历史。
- 原始 compact H0 prompt、相同五头ontology、max_output_tokens4096、reasoning low。不同模型的low不能视作相同实际推理开销。
- 每帧仅一次请求，不重试，不根据GT挑选输出。GT在整批推理完成后离线读取，各任务mask独立。
- 本六帧五头GT恰好完整，均为I0/V9/T14/IVT94/P0。它们是单一null交互与单一阶段，不代表完整Test成绩。
- 以下指标为标签集合完全匹配，不是mAP。

## 原始相同 prompt 的两组运行

| 模型 | 有效输出 | I | V | T | IVT | P | 费用USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grok4.6 | 6/6 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0.081930 |
| Gemini3.8 Flash | 0/6 | 不可评估 | 不可评估 | 不可评估 | 不可评估 | 不可评估 | 0.029870 |

Gemini并非没有HTTP响应，也不是六帧语义全错：六次均返回，但被本地JSON schema校验拒绝。原批次不计入有效语义分母；保留全部失败记录。

## Gemini 格式诊断与独立重跑

增加一次同参数4076诊断，费用USD0.002924，不并入六帧成绩。保存校验前的原始JSON后发现：

```text
返回 schema_version = cholectrack20_v1
期望 schema_version = joint_perception_gate_owned_compact_v1
```

这是把输入 ontology_version 与输出 schema_version 混淆。对诊断响应作仅在内存中的版本字段替换后，其余字段通过现有校验，证明该诊断帧只有这个合同问题；没有覆盖原响应，也没有把它登记成正式成功预测。

诊断帧原始selected标签I0/V9/T14/IVT94/P0恰好全对。但不能据此推测没有留存原JSON的先前六次输出也全部正确。

在纯H0入口增加可选 `--explicit-schema-version`：仅在system prompt末尾说明输出版本字面值、不要填ontology版本。这个选项改变canonical request hash；不修改返回标签、不更换候选、不提供GT，也不放宽解析器。

使用该选项单独重新跑 Gemini 六帧，全批预先固定、不挑最好结果；原失败批次仍保留。这是**修正输出合同提示后的独立实验**，并非与原Grok/Astra完全同prompt的严格单变量比较。

| Gemini格式澄清后 | Instrument | Verb | Target | IVT | Phase |
|---|---:|---:|---:|---:|---:|
| 正确数 | 6/6 | 3/6 | 3/6 | 3/6 | 6/6 |

六次全部返回并通过校验；4126、4151、4176三帧五头全对。费用USD0.029460。

Grok4176的Instrument正确，其他任务在六帧均错。Gemini仍把4076/4101/4201的null交互判断成活跃动作，但器械与阶段均正确。不能用这六帧直接认定哪个模型全面优于其他模型。

## 与历史 Astra 的配对参考

Astra此前同六个目标有4076接口拒绝，因此只在共同成功的4101/4126/4151/4176/4201五帧比较：

| 模型/配置 | I | V | T | IVT | P |
|---|---:|---:|---:|---:|---:|
| Astra原prompt | 1/5 | 2/5 | 2/5 | 0/5 | 3/5 |
| Grok原prompt | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Gemini格式澄清prompt | 5/5 | 3/5 | 3/5 | 3/5 | 5/5 |

这个表只是小样本参考，Gemini有上述格式提示差异；没有据此修改默认模型或任何Verifier/Repair政策。

## 产物与费用

位于 `artifacts/preflight/`：

- `pure_h0_grok46_vid110_4076_4201_20260905/`
- `pure_h0_gemini38flash_vid110_4076_4201_20260905/`：原六帧合同失败批。
- `pure_h0_gemini38flash_schema_probe_20260905/`：单次诊断，含原始JSON。
- `pure_h0_gemini38flash_schema_explicit_vid110_4076_4201_20260905/`：独立格式澄清批，含原始JSON。

上述首轮总计19次真实请求、无缓存命中，已报告费用 **USD0.144184**，不含之前的Astra实验及随后批准的Qwen补测。

每个目录保留predictions.json、summary.json、api_usage.jsonl、run_status.json。新加的raw_responses在本地校验前保存JSON，避免以后格式失败只剩错误码。未记录密钥。

配置：

- `configs/perception/joint_openrouter_grok46_pure_h0_fixed3.yaml`
- `configs/perception/joint_openrouter_gemini38flash_pure_h0_fixed3.yaml`

运行入口不变，Gemini独立澄清批额外指定 `--explicit-schema-version`。新运行必须使用新output目录，避免混入历史费用和结果。

针对性校验：66项测试通过，覆盖路由、模型准入、原JSON留存、schema提醒改变缓存hash且不改变图像/input字段，以及GT mask评分。

最终全仓库回归：1021 passed，14 skipped。既有V8/V9默认配置未替换。

## 用户批准后的 Qwen0902 补测

用户确认使用 `qwen/qwen3.8-max-0902` 替代不可用的0803。固定Alibaba路由，无fallback；其余视觉输入、六个目标时间点、生成参数与Gemini格式澄清批相同，使用 `--explicit-schema-version`。不加入Tracker/Gate/Verifier/Repair，不重试、不挑选结果。

六次全新API调用均返回有效结果，缓存命中0，未计价调用0。实际报告费用 **USD0.100560**；连同上述19次调用累计25次、**USD0.244744**，仍不含历史Astra实验。

| 模型/提示版本 | 有效输出 | Instrument | Verb | Target | IVT | Phase | 五头全对帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8 Max0902 / 格式澄清 | 6/6 | 4/6 | 0/6 | 0/6 | 0/6 | 4/6 | 0/6 |
| Gemini3.8 Flash / 格式澄清 | 6/6 | 6/6 | 3/6 | 3/6 | 3/6 | 6/6 | 3/6 |
| Grok4.6 / 原prompt | 6/6 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |

以上均为每帧标签集合完全匹配，不是mAP；Grok未加版本号澄清，与前两组有提示差异。六帧属于同一短片段且GT相同，不能据此做总体模型排名。

Qwen逐帧首次输出：

| 帧 | I | V | T | IVT | P |
|---|---|---|---|---|---|
| 4076 | 0,2 | 0,1,2 | 0,1,8 | 1,16,17,60 | 1 |
| 4101 | 0,2 | 1,2,9 | 0,8,11 | 60,63,96 | 3 |
| 4126 | 0 | 0,1 | 0,7,8 | 7,17,19 | 0 |
| 4151 | 0 | 0,1 | 0,8,10 | 17,19,20 | 0 |
| 4176 | 0 | 0,1 | 0,8,10 | 7,17,19 | 0 |
| 4201 | 0 | 0,1 | 0,8,10 | 17,19,20 | 0 |

所有帧GT均为I=[0], V=[9], T=[14], IVT=[94], P=[0]。后四帧器械、阶段正确，但V/T/IVT未匹配null交互标注；前两帧还多选器械2并预测错误阶段。4101虽然包含正确Verb9，但多选了1、2，因此集合完全匹配仍失败。此处没有运行Verifier/Repair，错误属于首次H0与GT的偏差，不能归因于修复机制。

产物目录：`artifacts/preflight/pure_h0_qwen38max0902_schema_explicit_vid110_4076_4201_20260905/`，包含summary、predictions、用量及六份校验前原始JSON。

配置：`configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml`。补测仅新增Qwen模型准入、独立后端标识及严格Alibaba路由；未替换默认pipeline。

复现命令（output必须换成新目录；原缓存可复用，此次实际无缓存命中）：

```powershell
.venv-p2\Scripts\python.exe scripts/run_pure_h0_smoke.py --dataset D:\cholec_dataset --config configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml --api-key-file docs/API.txt --cache artifacts/final_pipeline_cache/pure_h0_multimodel_20260905 --output artifacts/preflight/pure_h0_qwen38max0902_schema_explicit_vid110_4076_4201_20260905 --video VID110 --start 4076 --max-frames 6 --explicit-schema-version
```

本次验证：针对性24项通过；全仓库 **1022 passed, 14 skipped**。
