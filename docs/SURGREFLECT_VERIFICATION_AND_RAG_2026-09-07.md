# 审核说明 1000 字符更新、SurgReflect 代码核验与 RAG 判断

当前 A/B 实验入口的审核说明已放宽至 **1000 字符**，保持历史 300 字符契约。关于 RAG，当前更值得先验证的是明确展开标签名称和 IVT 元组；没有证据支持直接引入通用医学文献 RAG 就能提高本项目 Verifier 准确率。

后续补记（2026-09-07）：用户现已提供母体完整 PDF。新的 [全文、五篇示例论文与反馈反思核查](VERIFIER_PAPER_METHODS_AND_REFLECTION_2026-09-07.md) 区分论文描述和公开代码，并提出尚未测试的最小方案。下文保留当时的代码核验与实验事实；其中“全文尚不可得”的描述仅指本历史核验当时的证据范围。后续标签名称、分拆审核和历史补图开发结果见 [完成记录](VERIFIER_DEVELOPMENT_AND_CONFIRMATION_2026-09-07.md)，不能继续把本页当时的优先建议写成待完成任务。

## 1000 字符修改已经完成

`scripts/run_presence_review_trial.py` 新运行默认 `--review-observation-max-chars 1000`；显式传入 `300` 可选择历史契约。纯 API H0 入口不启用审核，未受影响。旧独立研究入口保留其历史默认，避免静默重解释旧记录。

| 审核类型 | 历史 300 字符契约 | 新 1000 字符契约 |
| --- | --- | --- |
| 逐修改 Diff | `frame_label_diff_review_v1` | `frame_label_diff_review_v2` |
| 中性存在性 Presence | `frame_label_presence_review_v2` | `frame_label_presence_review_v3` |
| 双假设 Contrast | `contact_contrast_review_v1` | `contact_contrast_review_v2` |

修改同时覆盖供应商输出 Schema、本地验证、Presence 转 Diff 后的二次校验、Grounded 内部接纳。只改审核的 `observation` 和 `distinguishing_observation` 上限；候选 `contact_observation` 仍为 300。没有截断原文，没有改变判词、图片引用、标签依赖、H0 或视觉判断提示的语义；提示中的契约版本标识随新版本更新。

独立离线链路验证确认：1000 字符通过，1001 拒绝；旧 300 仍生效；坏枚举、引用、缺项、空白说明仍拒绝，旧审核至少 12 个去空白字符的原检查保留。H0、定位、候选的请求哈希在 300/1000 两种配置下完全相同，只有两种审核请求的版本和对应 Schema 改变。

相关回归测试的一批为 **187 通过、1 跳过**，另有重叠的 159 项检查通过，不相加计数；涉及 Python 文件 Ruff 通过。旧实验文件和 `frozen_source/` 保留，未暂存、提交或推送 Git。

187 项检查覆盖 `test_presence_review_trial.py`、`test_review_observation_limits.py`、`test_p3_api_config_schema.py`、`test_p3_api_client.py`、`test_grounded_contact_repair.py`。唯一跳过的是缓存符号链接防护测试，因本机创建符号链接抛出 `OSError`，不是审核功能失败。24 目标的新入口离线预检也已通过，计划明确记录 1000 字符、Contrast v2、Presence v3、候选说明仍 300；目录为 `artifacts/preflight/presence_review_limit1000_preflight_20260907/`，调用数为零。

使用上一轮实际 HTTP 200 原文，明确迁移契约版本并按 1000 上限离线重新验证：**16/16 旧审核、16/16 新审核通过，最长新审核说明 857 字符，未裁剪任何文字**。所有派生预测落盘后才关联 GT，原始请求、回复与预测未改写。这不是重新调用模型得到的新成绩。

结果与此前长度回放的最终标签逐项一致：H0 IVT F1 为 26.83%，A 为 26.83%，B 为 25.00%。新增 API 调用、费用均为零；说明放宽修复了工程误拒，却没有独自带来语义提升。

材料：`artifacts/preflight/presence_review_qwen24_limit1000_20260907/` 中的 `replay.py`、`revalidation.json`、`predictions.json`、`presence_audit.json` 和新的 `frozen_source/`。原 300 字符严格结果仍在 `presence_review_qwen24_20260907/`。旧严格回放的源码哈希保护不会被关闭；精确复现旧实验须在独立目录使用其冻结源码，不能直接用新源码覆盖旧协议。

## 母体证据范围

核查的是用户提供的作者仓库，当前提交为 `3028fce4a8eb7ba181314b593c0336ad69b9b67b`。已读取原始代码、README 并保存 19 份文件快照；没有执行第三方代码。

精确标题、SurgReflect、arXiv 和 OpenReview 检索未找到可核验的对应论文正文，作者 README 也没有正式论文链接。所以下面是**公开实现机制与代码等价伪代码**，不能冒充论文原文伪代码或已核实论文实验。固定链接与逐项定位保存在 `artifacts/research/surgreflect_20260907/audit_findings.json`。

## 母体 Verifier／Repair 靠什么运行

**视觉 Verifier 是同一个基础多模态模型的再次 API 调用。**输入是三张原图、当前题目的 IVT 字符串候选，以及可选的器械动作兼容提示。它返回 `selected` 和 `maybe`；实现把两者合并为选中结果，目标偏向减少漏检。它不是独立训练的二分类器，也没有直接接收 RAGTool 检索结果。[IVTVerifyExpert](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/experts.py#L628)

送审候选由 Python 从每题已有选项中排序产生：结合当前 I/V/T、阶段、标签共现次数，保留原 IVT 并扩充候选，通常至 24 项。这里没有检索相似手术图片。[候选工具](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/tools.py#L201)

**Repair 混合程序规则与一次模型重写。**程序先删除选项外三元组；遇到器械动作不兼容，就从选项中找同器械、同组织的兼容动作替代，找不到则删除。再把 IVT 中的组件并入独立 I/V/T，并可能按统计更新阶段。这些规则检查一致性，不能证明画面中确实发生该动作。[规则修复](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/reflection.py#L200)

可选模型修复把原图、当前答案、问题清单、选项、兼容规则和阶段先验交给同一模型重写。结果再经过规则检查；I/V/T/IVT 任一头非空就可替换这四个头，合法 Phase 另行更新。代码没有再调用独立视觉裁判，也没有逐项证明比原答案更好。[模型修复](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/reflection.py#L248)、[接纳](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/orchestrator.py#L474)

白话：先看图答题，再拿一张扩充候选清单让同一模型勾选，程序纠正不合规则之处，必要时让同一模型重写一次。它没有一个已经知道真实答案的“老师”。

## 对应伪代码

以下省略选项字符串映射和报告细节，各步骤仍受配置开关控制。第一行忠实体现公开代码的数据接口，没有替作者补上它未强制执行的训练/评估隔离。

```text
统计库 = 从调用者传入样本的 GT 构建共现统计
H = 多模态模型(三帧图像, 每题选项)       # 默认一次联合五头预测
issues = []

if 启用 reflection:
    阶段提示 = 从 H.IVT 与共现统计估计
    候选 = 保留 H.IVT，并用当前 I/V/T 和统计扩充
    selected, maybe = 同一模型看图审核候选
    H.IVT = selected ∪ maybe
    issues += “完成了候选审核”           # 常规事件，不一定表示发现错误
    H.IVT = 程序按选项和器械动作规则删改
    H.I/V/T = 原集合 ∪ H.IVT 的组件
    H.Phase = 按统计条件保留或更新

报告 = 模型生成并由规则清理
if reflection and model_repair and max_rounds > 0 and issues 非空:
    R = 同一模型(原图, H, issues, 选项, 规则和阶段提示)
    R = 映射到合法选项，并重做兼容性/组件检查
    if R 的 I/V/T/IVT 任一头非空:
        H.I/V/T/IVT = R.I/V/T/IVT
    if R.Phase 合法:
        H.Phase = R.Phase
    报告 = 按更新结果重新生成
return H, 报告
```

虽然参数叫 `max_rounds`，当前公开实现只有一次 `if` 分支，没有“反复审核直到正确”的循环；`issues` 也包含常规审核事件，不能将其视为可靠的语义错误检测器。[控制流程](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/orchestrator.py#L445)

此外，母体用每个样本自己的 MCQ 选项，本项目用固定完整本体识别，任务难度和分母不同；不能直接用两边的分数证明谁的审核机制更强。公开 README 的运行参数和目录组织也与源码不完全一致，应以源码核验为准，不能称其已完整复现。

## 母体的 RAG 是什么

它主要把标注统计写成短文本，例如某种器械常对应哪些动作、组织，某阶段常出现哪些 IVT，再按词语重合检索。没有医学论文数据库、向量检索器或相似帧检索。统计也直接用于候选排序和 Phase 更新；RAG 通过这些渠道、初始预测提示及模型修复先验间接影响结果。[StatsRAGStore](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/rag_store.py#L25)

逐样本提示没有直接插入该帧标准答案，但全局建库读取调用者传入样本的 `answer/gold_context`，代码没有强制排除评估集。如果把待评估样本也传入，就会通过统计泄露标签信息；这并不能证明尚未核实的论文实验一定这样运行。本项目若用统计或标注例子建库，只能使用允许的 Training 参考数据，评估 Training 时还须排除对应视频；Testing 不参与建库或调参。[统计来源](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/3028fce4a8eb7ba181314b593c0336ad69b9b67b/rag_store.py#L49)

## 本项目是否需要加入 RAG

**目前不把通用 RAG 作为首选修复。**RAG 可以补充类别知识，却不能仅凭“这种动作常见”证明它发生在当前图像里。原始 RAG 研究主要针对知识密集型语言任务；其结果不能直接当成手术 IVT 视觉核验的提升保证。[RAG 原论文](https://arxiv.org/abs/2005.11401)

本项目的直接证据是：已把完整、正确的本体送入审核，但仍出现审核 IVT17 时检查错误器械、以及反证与 IVT59 对象不对应的问题；Target 判断也会把“组织可见”当成“组织正在被操作”。因此要先检验它是否正确使用已知定义，不能直接归因于知识库不够大。

| 已观察到的问题 | RAG 可能帮助的范围 | 当前更小的处理 |
| --- | --- | --- |
| 数字 ID 与审核对象错配 | 检索到相关定义可能减少查表负担 | Python 从冻结字典直接展开完整名称和 IVT 元组，精确对应，无需向量检索 |
| 看见组织就认为它是 Target | 经审核的类别边界卡可能澄清“交互对象”定义 | 将对应任务定义紧邻命题提供；区别可见性与交互关系 |
| 看错接触、动作或组织 | 文字知识无法补出当前画面缺失的证据 | 保留当前图像为判断依据，无法确定时返回 UNCLEAR |
| H0/H1 共同漏掉正确 IVT | 候选端检索可能提出新候选 | 属于候选生成实验，不能记成只改善 Verifier 的效果 |

相关手术研究也应区分用途。SurgRAW 的正式 v2 方法将医学资源 RAG 用于动作预测、结局、患者信息等认知推断任务；视觉器械/动作识别主要使用结构化观察、知识图谱兼容检查和讨论。其消融没有直接证明医学文本 RAG 能提高我们这个 IVT Verifier。[SurgRAW 方法与消融](https://arxiv.org/html/2503.10265v2#S3.SS3)

本周若继续一个最小验证，建议只比较“1000 字符、数字命题”与“相同审核加程序解码的完整标签名称”，其余 H0、候选、图像、模型、接纳规则保持一致。先看审核是否更准确地识别有益/有害修改，再看最终各头 F1、集合 Accuracy、改对/改坏和费用。用新的固定 Training 目标，不按已看过的 24 个 GT 设计特判或重报调参后成绩。

如果这一步仍表明缺少类别定义知识，再考虑固定的小型本体定义卡库：按 task/label ID 精确取卡、来源可查、无在线网页搜索、无目标帧 GT。此时可以称为检索式知识补充，但无须先建大型向量数据库。只有这类补充出现可重复的语义净收益，才值得扩大；当前未实现 RAG，也未启动新的付费实验。
