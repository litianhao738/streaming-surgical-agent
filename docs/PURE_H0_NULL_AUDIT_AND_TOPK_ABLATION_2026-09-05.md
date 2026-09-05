# 五头 H0：图像/GT 核查与 top-k 消融

## 结论

六帧没有发现代码层面的图像错配、五头漏评或把缺失GT补成null。删除top-k没有解决交互识别偏差。此前将null解释为“无动作/无交互”过于简单，需要纠正：null还包含不在数据集规定类别/组合内的交互。

这不是已证明模型纯粹“看不见动作”，也不是已证明GT错误。当前能定位的是最终视觉语义选择与封闭本体标签之间的偏差，且现有prompt对null边界说明不足。语义说明缺口的因果影响尚未实验验证。

## 图像与GT审计

原始源：`D:/cholec_dataset/Validation/VID110/vid110.json`。

- 逐一复核4076/4101/4126/4151/4176/4201的PNG文件名、causal frame IDs和实际构建的上传图像SHA256，全部与原Qwen调用ledger一致。
- 每次三张因果图像；无未来帧。首窗4026/4051/4076，末窗4151/4176/4201。
- 原始实例标签直接包含I0/V9/T14/IVT94/P0，离线集合聚合与原始JSON一致；不是把-1、缺失或空列表填成null。
- 前五帧各一个实例；4201有两个grasper实例，类别集合合并后仍为I=[0]，不是遗漏第二件器械。
- 已目视检查上述六张目标图及4026、4051两张历史图。可见器械运动、细长物体与线状物；标注框位置随器械移动，无明显串帧迹象。但这不是对临床类别及全部GT的专家复标。
- PNG数值文件名精确对齐，未使用测试MP4的-1解码偏移；本次未发现这种off-by-one问题。

可复现审计：`scripts/audit_pure_h0_alignment.py`。产物：`artifacts/preflight/pure_h0_qwen38max0902_alignment_audit_20260905.json`。

## null语义：纠正之前的说明

官方CholecTriplet FAQ说明：有效器械存在，但动作、靶点或联合组合不在规定的100类内时，可以保留器械标签而将Verb/Target标为null。因此不能将null等同于“物理上静止或绝无接触”。这是相关triplet本体的官方定义；不能凭该定义单独断言这六帧的具体临床交互类型。

来源：[官方FAQ](https://cholectriplet2021.grand-challenge.org/FAQ/)。[CholecTrack20官方仓库](https://github.com/CAMMA-public/cholectrack20)说明其与CholecT50等数据的关联；本地本体使用相应100类映射。

当前`ontology_prompt.py`只说IVT94-99表示可见器械加null_verb/null_target，没有解释哪些非空交互应归入null。这是确认存在的说明缺口。为不混淆消融，本次没有更改此公共prompt，也没有根据这六帧答案定制提示。

官方CholecTriplet2021指标还排除了null triplet类别。因此本项目这组六帧含null的集合完全匹配率，不能直接与该挑战的非null AP比较。并未据此删除本项目null评分或修改现有GT mask。

## 单变量输出合同消融

模型：`qwen/qwen3.8-max-0902`，Alibaba单一路由，无fallback。

对照：复用上一轮已固定的六次真实Qwen top-k结果，不另付费重跑对照。
实验：新增六次首次请求，仅输出五项最终标签。

相同：六个时间点、三图及顺序、图像detail、输入JSON、医学背景、本体、视觉推理说明、max_output_tokens4096、reasoning low、无历史预测、无Tracker/Gate/Verifier/Repair。

变化：删除top-k候选数量/排序/包含关系要求；JSON schema移除topk字段，使用独立schema_version。保留其余最终标签数量上限；没有把最终标签伪造成带置信度候选，没有放松旧解析器。

`--final-only`为独立消融入口，不是一个新的五头训练模型。请求文本和schema标识另行落盘，canonical hash改变；配置与默认pipeline不变。

## 结果

| 组别 | 有效响应 | I | V | T | IVT | P |
|---|---:|---:|---:|---:|---:|---:|
| 原最终标签+top-k | 6/6 | 4/6 | 0/6 | 0/6 | 0/6 | 4/6 |
| 仅最终标签 | 5/6 | 5/5 | 0/5 | 0/5 | 0/5 | 4/5 |

为了不混淆分母，共同成功的4101/4126/4151/4176/4201五帧：

| 组别 | I | V | T | IVT | P |
|---|---:|---:|---:|---:|---:|
| 原最终标签+top-k | 4/5 | 0/5 | 0/5 | 0/5 | 4/5 |
| 仅最终标签 | 5/5 | 0/5 | 0/5 | 0/5 | 4/5 |

Phase总正确数不变：4101由错变对、4201由对变错。不能声称阶段整体提高。两个组的五头全部正确帧均为0。

仅最终标签的原始输出：

- 4076：`response_content_invalid`，在生成有效ProviderResponse前失败；没有可评分JSON。该错误类别不足以区分空内容、非JSON或响应字段异常，未重试猜测。
- 4101/4126/4151/4176：I=[0], V=[1], T=[0], IVT=[17], P=[0]。
- 4201：I=[0], V=[1], T=[0,8], IVT=[17,19], P=[1]。

实验没有修复交互类别选择，模型更简洁地输出了同类错误。单次小样本且非同时随机化运行，不足以证明top-k对总体质量无影响。

六次新增调用、0缓存命中；已记录费用 **USD0.058802**，另有一次失败调用费用未知，不能把该数字称为完整最终账单。成功响应的原始JSON有5份，失败仍保留ledger记录。

## 产物与验证

- 实验产物：`artifacts/preflight/pure_h0_qwen38max0902_final_only_vid110_4076_4201_20260905/`。
- 入口：`scripts/run_pure_h0_smoke.py --final-only --explicit-schema-version`，其余参数沿用Qwen固定三帧配置及相同六个目标。
- 新schema：`src/surgical_agent/perception/final_only.py`，仅在API schema注册表注册；不使正式JointApiVlm接受无候选数据。
- 单元测试验证五头字段、非法ID/重复/额外字段拒绝、图像及输入不变、hash隔离，不依赖GT修正输出。
- 全仓库回归：1024 passed，14 skipped。对齐审计脚本在实际六帧上全部断言通过。

## 后续优先级（未执行）

1. 基于公开本体定义补充通用null及有效交互边界说明，不提供测试帧答案或帧特定例子。
2. 用含null与非null的预先固定、多片段样本验证说明是否有效；不要继续只在这六个相同GT时间点调prompt。
3. 分开报告字段集合完全匹配、候选召回及按明确类别集合定义的正式AP；不临时剔除难例改善成绩。

当前证据不支持直接增加Agent数量或repair轮数，也不支持直接启动付费Gate训练。
