# 初始预测推荐：母体实现、同任务论文与本地实测的综合判断

日期：2026-09-06。只读研究与方案整理；没有训练、付费 API 调用、默认 Pipeline 修改或冻结版本变更。

## 推荐与证据强度

在已选定 OpenRouter API 的项目路线下，推荐继续以**一次 API、三帧因果图像、联合输出五头**为主干，第一优先级只检验**独立 Training 视频的少量标注示例能否改善视觉类别判断**。当前可直接使用的仍是原版三帧 H0；加入示例是待验证候选，尚不能称为准确率已获胜版本。

具体协议：

- 当前窗口 `[t-50,t-25,t]`，真实时间 `[-2,-1,0]` 秒；只预测 t。
- 输出目标每次前进 25 个原视频帧，即一个 1 Hz 标注点；滑动窗口重叠两张图。这是输出采样计划，不意味着 API 已满足每秒返回一次的延迟要求。
- 同一 Qwen 模型、原版提示词、完整本体与当前五头输出保留。新候选仅加入预先固定的领域示例；不同时改变模型、图片编码、候选范围或输出结构。
- 示例来自不同于被评估视频的 Training 数据，并经标注及视觉对应核实。建议第一轮固定两组对照、共四个示例，覆盖组织靶点与 null 边界；数量是小实验预算选择，不是论文最优值。原图加简短标签说明是本项目的视觉示例提案。
- 示例与待预测三帧明确分开标记；示例标签不能作为当前帧候选限制。当前目标和近邻 GT 不进入请求，也不用于挑选示例。
- 一次调用仍同时输出 I/V/T/IVT/Phase。示例增加上下文，不更新模型权重，不是微调。

这项改动直接对应已见的 Target 误报和 null 漏识别。它不是以通过投影闭包为目标，也没有证明能同时解决 Phase、动作运动证据不足和遮挡。

## 母体 SurgReflect：已核实公开实现，而非推测名称

已找到[公开仓库](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding)。本次确认的是仓库当前实现；未找到可独立核查的论文全文及完整实验消融，不把 README 的设计说明当成已证实的性能优势。

- [config.py](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/main/config.py#L13)：`joint_predict=True`，`k_joint=1`。
- [orchestrator.py](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/main/orchestrator.py#L142)：默认一次 JointPredictExpert 生成五项结果；另有 I→V→T→IVT→Phase 的顺序分支，并传递上游预测。
- 因而“母体多专家架构”不等于“默认初始预测需要五次互不通信的 API”。本项目旧 C 组五个独立调用没有传上游答案，并非母体顺序分支的严格复现。
- [experts.py 的作者注释](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/main/experts.py#L524)说明联合默认的动机：旧单任务专家过于保守、漏选 V/T/IVT。这是实现动机说明，不是公开的配对消融结果。
- 公开 [benchmark JSON](https://github.com/AlexZhihao/SurgReflect-Hierarchical-Agentic-Verification-for-Multi-Triplet-Surgical-Scene-Understanding/blob/main/cholect50_bench_tiers_1500x6.json) 经只读解析共 1500 个样本、50 个视频，每样本三张图、IVT 题均有 12 个候选。当前项目对完整 100 类预测，两者难度不能直接等同。JSON 帧序号相邻，但缺少 FPS 元数据，不将它解释为相邻原视频帧。
- README 的 RAG 是统计共现/文本检索，不是已经验证的视觉 few-shot。其描述使用整个 benchmark 的 gold 构建统计；本项目只能使用 Train 来源的先验，不能照搬这一数据访问协议。

## 各论文真正提供的依据

| 来源 | 核心机制 | 对当前 H0 的启示及边界 |
|---|---|---|
| [RDV / Rendezvous](https://arxiv.org/html/2109.03223v2) | 训练共享视觉表示；器械注意引导动作与靶点，联合学习交互关系 | I/V/T/IVT 彼此关联，有共享证据的必要。神经网络多头联合训练不能等同于五次 API，也不能等同于在提示词里写“关注接触点” |
| [RiT / Rendezvous in Time](https://arxiv.org/html/2211.16963) | 1 fps 因果序列，训练时序注意，仅监督末帧 | 支持稀疏因果末帧任务。6 帧是其模型的消融选择；五折 IVT AP 29.4→29.7，grasp/retract/coagulate 未改善，不证明六图 API 更好 |
| [SurgSTU](https://arxiv.org/html/2604.00784v2) | 手术视频 QA；提示添加候选词表及 Train 示例，另有领域微调 | 是 API 领域示例值得尝试的直接相关证据。但其词表与示例一起变化，且部分收益来自格式遵从；本项目已提供词表且 80/80 JSON 有效。论文未明确示例是否携带图像，所以本项目视觉对照仍是推断 |
| [StreamingVLM](https://arxiv.org/html/2510.09608v1) | 重叠片段 SFT；短视觉、长文本 KV 保留与复用 | 支持按时间范围分配上下文；普通无状态图片 API 不等于其缓存机制，也没有证明手术三/六图最优 |
| [Flash-VStream](https://arxiv.org/html/2506.23825v1) | 压缩历史特征并保留关键细节；使用 LoRA 指令训练 | 提醒不要为了历史容量丢掉细粒度视觉信息；其特征记忆容量不是 API 图片窗口 |
| [TDC-Video](https://arxiv.org/html/2504.10443v1) | 静态关键帧与训练过的动态上下文压缩 | 静态组织细节与动态动作信息有不同需求；无法通过改 JSON 直接获得其编码能力 |
| [M3-Agent](https://arxiv.org/html/2508.09736v1) / [Vgent](https://arxiv.org/html/2510.14032v1) | 情景/语义记忆或图检索用于长视频问答；M3 包含训练，Vgent 离线建图 | 适合研究更长流程的检索与记忆；不能据此修正当前组织分类，更不能在严格因果任务里检索未来 |
| [VideoARM](https://arxiv.org/html/2512.12360v2) | 按问题定位区间，自适应采样并查询分层记忆 | 提醒图像预算要服务于所缺证据；其长视频问答的定位工具与多步预算不是每秒五头初始预测的固定配置 |
| [TeCNO](https://arxiv.org/html/2003.10751) | 训练因果时序卷积，利用较长历史预测 Phase | Phase 的流程上下文值得独立研究，但不证明把之前预测写进 API 提示就有同样收益；RDV/RiT 也没有本项目的 Phase 头 |

论文共同启示是“学习或提供与任务相关的证据和类别知识”，而不是“多加角色、多加帧必然更好”。其中单次联合 API 是工程和现有实验共同支持的保守选择，不是所有上述论文都直接研究过的结构。

## 本地实测比跨任务参数迁移更直接

1. 窗口实验：四组都成功的 28 个共同目标中，三帧 IVT F1=31.58%，六帧=30.09%，单图=31.86%。三帧并未显著胜过单图，但六帧没有提供稳定 IVT 收益。因此此前基于 RiT 推荐六帧的假设，已经被本地后续实测修正。
2. 五专家：部分 null IVT 被找回，Target 仍无整组正确，五头全部正确仍为 0/6；存在响应失败和跨头冲突。它不足以支持把五次独立调用设为默认。
3. prompt：40 目标原版 IVT F1=33.33%，只删 Schema=28.80%。不能再称只删 Schema 是保证效果的最佳版。
4. 接触裁剪/复核：此前六目标实验已经试过独立定位、局部提案和双假设复核，IVT 仍为零。不能将同类“再看一遍接触点”作为已证实的解决方案。
5. 低分审计：原版胆囊板 21 次预测只有 3 次正确，7 个 GT null IVT 全漏；原始回答和计分重建无差异。这使领域类别示例比继续纯格式修改更值得检验，但不保证示例有效。

本地来源：`H0_FRAME_STRATEGY_TRAINING32_QWEN_2026-09-05.md`、`FIVE_EXPERT_ABC_QWEN_PILOT_2026-09-05.md`、`H0_SCHEMA_CONFIRMATION_SMOKE_2026-09-06.md`、`GROUNDED_CONTACT_REPAIR_SMOKE_2026-09-05.md`、`H0_LOW_ACCURACY_DIAGNOSIS_2026-09-06.md`。

## 如何验证，何时考虑训练

下一次只比较原版三帧联合 H0 与“相同 H0 + 固定领域示例”。在未参与示例选取的视频上预先固定目标；冻结样本、请求、顺序和评估后运行。主要看完整 IVT micro-F1 与各头集合 Accuracy，另查 Target 误报、null 召回、其他类别退化、完整响应率和费用。失败不得剔除来制造高分，不以结构闭包提升代替语义提升。当前未创建付费实验；示例会增加图片/token，需按实际请求重新估计预算，不能沿用原三图单价或承诺缓存命中。

如果独立验证仍无收益，应承认提示示例未解决问题。更有同任务文献支持的下一层是训练/适配共享手术视觉表示与 I/V/T/IVT 交互分类，学习组织、动作与 null 的视觉边界；Phase 用因果历史目标单独检验。这不是重训 Tracker 身份关联就能替代的任务。若使用 CholecT50/Cholec80 或手术预训练权重，先按视频身份排除 Track20 Validation/Test 的重叠。

不能由此断言必须训练才能获得任何提升，也不能承诺 API 提示改动会达到领域训练模型的精度。现阶段最推荐的是固定主干、验证一个直接针对语义错误的小改动。
