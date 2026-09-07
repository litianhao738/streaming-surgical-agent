# 当前 Verifier / Repair 机制与实测边界

**本轮保留冻结的 H0，不进入 Validation 确认。** 逐项审核、局部修复、名称展开、单命题独立核验和一次条件触发的历史补图均已完成开发测试，所有修复变体都未通过预先写下的语义收益筛选。补图相对“原图再审核一次”没有带来 IVT 指标增量，不继续增加新模块。

本说明按 2026-09-07 已核实代码及已完成的 Training 实验更新。历史实验、独立补充分析与补图主试验分别记录，不把工程接通或格式恢复当成准确率提高。[完整开发报告](VERIFIER_DEVELOPMENT_AND_CONFIRMATION_2026-09-07.md)、[最终决定](../artifacts/preflight/verifier_goal_readiness_20260907/final_decision.json)

## 哪个入口实际运行什么

| 入口 / 模块 | 实际行为 |
| --- | --- |
| [默认纯 API 入口](../scripts/run_dataset_api_pipeline.py) | 只做冻结的三帧联合 H0；没有接入 Verifier、Repair、Tracker、旧 Gate 或跨窗口记忆。 |
| [旧 grounded 链路](../src/surgical_agent/research/verification/grounded_pipeline.py) | H0 → 不看 H0 的定位 → 原图与裁剪提出 H1 → 将 H0/H1 隐去身份、打乱顺序后整份比较 → 接纳 H1 或保留 H0。它仍是双假设审核，不能把它称为新 A/B。 |
| [新审核对照入口](../scripts/run_verifier_variant_trial.py) | 复用并保存同一 H0、定位、H1 和图片，再把有分歧的标签送入中立存在审核；同一份审核由 Python 分别计算 A、B。 |
| [历史补图对照入口](../scripts/run_temporal_memory_trial.py) | 复用已完成的首次名称审核，符合固定条件时额外审核一次；比较原图重复审核与增加真实过去图片。已完成对照、未通过筛选，没有替换默认入口。 |

H0 是三帧因果联合预测，正常窗口为原视频 `[t-50,t-25,t]`，即 `[-2,-1,0]` 秒。定位、候选和视觉审核仍由模型调用完成；Repair 的接纳和合并是 Python 规则。它们没有使用 GT 在线判错，也没有把 final-only 硬标签当作旧 Gate 所需的 top-k 置信度。

## 新 Verifier 怎样审核，Repair 怎样修改

定位调用先找器械尖端或接触区域，候选调用再看原图和裁剪提出 IVT。当前 [H1 构造函数](../src/surgical_agent/research/verification/grounded_repair.py) 会由这些 IVT 重建 Instrument、Verb、Target，Phase 取 H0。定位声明、覆盖声明和候选支持声明都来自模型，不能当作接触真值；H1 也可能漏掉 H0 中正确的独立标签。

[中立存在审核](../src/surgical_agent/research/verification/presence_review.py) 只问 H0/H1 不一致的标签是否在目标帧成立。Python 按任务和标签 ID 排序，生成 `p001` 等中立编号；请求不展示 H0/H1 身份或 ADD/REMOVE 方向。名称变体由程序从冻结本体直接补上标签名称和 IVT 的器械、动作、组织名称，系统说明、本体、图片、输出 Schema 和接纳规则不变。当前开发试验使用 v3，`observation` 上限为 1000 字符；历史 v2 的 300 字符限制保留在原实验中。

模型逐项返回 `PRESENT / ABSENT / UNCLEAR`、观察文字、图片引用、判断范围及是否检查目标全图。Python 检查格式、完整命题对应和引用是否合法。**这些检查不能证明模型真的看对了器械、动作或组织，也不能证明“已检查全帧”的声明真实。** null 类别仍按原本体及边界定义处理；看不清应答 UNCLEAR，不能用 null 代替不确定。

[A/B 执行规则](../src/surgical_agent/research/verification/diff_review.py) 如下：

- 增加标签：必须回答 PRESENT，并引用目标全图。删除标签：必须回答 ABSENT、范围为 FRAME、声明检查了全帧，并引用目标全图。UNCLEAR 不授权修改；PRESENT 绝不授权删除。
- A：全部差异都获准才整份采用 H1，否则保留 H0。
- B：从 H0 开始，只应用获准的变化。删除 IVT 不自动删除组件；增加 IVT 所缺的组件必须独立获准，H0 已有组件可复用；删除组件还要确认没有保留的 IVT 依赖它。其他独立头标签保留，不把全帧 I/V/T 重新投影为 IVT 的并集。
- 整份审核格式、命题对应或引用不合法，A/B 都整体回 H0，不提取其中看似合法的几项。B 合并后超出最终输出约束时，也整体回 H0。H0 本身无效属于初始预测失败，不能伪称 KEEP 成功。Phase 始终保留 H0。

```text
h0 = frozen_joint_prediction(real_causal_images)
require_valid(h0)                       # H0 失败不能伪造回退答案
h1 = grounded_candidate(images, crops)  # 模型定位与候选；Python 构造标签
if h1 无效或没有实际标签差异:
    return A=h0, B=h0

questions = neutral_presence_propositions(h0, h1)
review = model_review(images, same_crops, questions)
if 整份 review 不通过格式、命题对应和图片引用校验:
    return A=h0, B=h0

for change in Python 保存的差异:
    ADD 仅在 PRESENT 且引用目标全图时获准
    REMOVE 仅在 ABSENT、FRAME、full_frame_reviewed=true
             且引用目标全图时获准
    其他情况不获准

A = h1 if 全部变化获准 else h0
B = 从 h0 应用获准变化，再检查组件依赖与最终输出约束
return A, B
```

删除判断必须针对整帧：一把抓钳没有牵拉，不代表另一把也没有。H0 是帧级标签集合，没有可靠的实例对应；代码不能自行补造这种对应。A/B 只能在原 H0/H1 差异内修改，不能找回两者共同漏掉的答案。

## 已完成的实验说明了什么

历史 24 目标中，严格 A 全部 KEEP；B 的 IVT F1 从 H0 的 26.83% 降至 25.32%。另做的零调用、说明长度补充回放恢复了审核格式，B 仍只有 25.00%。原始回答还出现过审核错误器械、用另一种器械的缺失反驳当前命题。请求中的完整 100-IVT 本体及映射核对无误，所以不能把问题归结为漏传本体，也不能证明所有错误都由数字 ID 引起。[历史 24 目标报告](PRESENCE_REVIEW_RESULTS_2026-09-07.md)

随后名称、独立核验和历史补图共用另一批固定的 16 个 Training 目标及候选；Instrument/Phase 有效 GT 为 16，Verb/Target/IVT 为 15。下表每格为 **micro-F1 / 集合 Accuracy（%）**，不同批次的分数不可直接当作版本提升。

| 输出 | Instrument | Verb | Target | IVT | Phase |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 90.20 / 68.75 | 75.00 / 53.33 | 59.57 / 33.33 | 44.44 / 20.00 | 81.25 / 81.25 |
| 数字命题 B | 90.57 / 68.75 | 72.00 / 46.67 | 61.54 / 13.33 | 49.12 / 13.33 | 81.25 / 81.25 |
| 名称命题 B | 90.57 / 68.75 | 73.47 / 46.67 | 62.75 / 20.00 | 45.61 / 20.00 | 81.25 / 81.25 |
| 严格独立核验 A | 92.31 / 75.00 | 73.47 / 53.33 | 59.57 / 33.33 | 43.64 / 20.00 | 81.25 / 81.25 |
| 严格独立核验 B | 90.57 / 68.75 | 72.00 / 46.67 | 58.82 / 13.33 | 40.00 / 13.33 | 81.25 / 81.25 |
| 独立核验 B 长度补充回放 | 90.57 / 68.75 | 72.00 / 46.67 | 58.82 / 13.33 | 40.00 / 13.33 | 81.25 / 81.25 |
| 原图重复审核 B | 90.57 / 68.75 | 72.00 / 46.67 | 61.54 / 13.33 | 45.61 / 13.33 | 81.25 / 81.25 |
| 历史补图 B | 88.46 / 62.50 | 73.47 / 46.67 | 61.54 / 13.33 | 45.61 / 13.33 | 81.25 / 81.25 |

名称 A 全部保持 H0，不能算修复成功。数字 B 虽提高 IVT F1，但 IVT 集合 Accuracy 和 Verb F1 下降；名称 B 的 IVT F1 小幅提高，同时 Verb F1 下降。两者均未通过预写筛选条件。名称展开减少了已观察到的明确编号错配，但有效 GT 范围内 IVT 的 UNCLEAR 从数字版 2 条增至 14 条；更清楚的输入没有自动变成更可靠的视觉判断。[名称配对结果](../artifacts/preflight/verifier_goal_training_named_pair_20260907/summary.json)

单命题独立核验（factored）已完成：每次只送一个原名称命题，保留相同图片、完整本体和生成参数；任何一个回答缺失或不合法，整帧审核失败并回 H0。共 45 次调用，费用 $0.662778，42 份格式有效、3 份说明超长，13 个有变化目标中 10 个成功合并。它没有带来语义收益。候选中的 10 项有益、12 项有害 IVT 修改，名称 B 实际应用 2/10 与 3/12，严格独立核验 B 为 0/10 与 6/12。这里统计最终输出增删，不是模型自报同意。[严格独立核验结果](../artifacts/preflight/verifier_goal_training_factored_v2_20260907/summary.json)

随后完成的**零调用长度敏感性分析是补充结果**：在已经看到配对 Training GT 和长度故障后写下协议，对所有回答统一仅截短超出 1000 字符的 `observation`，其他字段、判决和接纳规则不改，再完整校验。3 份超长回答恢复，其余 42 份保持原值；原始结果没有覆盖。A/B 的五头汇总 F1 与集合 Accuracy 均未提高。补充 B 多应用一个有益、一个有害 IVT 修改，变为 1/10 与 7/12，不能将格式恢复称为语义收益。[补充协议](FACTORED_LENGTH_SENSITIVITY_PROTOCOL_2026-09-07.md)、[独立补充结果](../artifacts/preflight/factored_length_sensitivity_20260907/summary.json)

这些试验仍暴露两个限制。第一，独立 Target 的个别回答用“肝表面可见”支持肝作为作用目标，虽然输入已经要求按器械接触关系判断；可见组织与作用靶组织并不等价，当前程序不能自动核实自由文字是否建立了这种关系。第二，这批 H0 漏掉的 14 个真实 IVT 出现项中，H1 只新增了 2 个，仍有 12 个共同漏检；即使审核完美也补不出它们。GT 离线逐项择优的乐观 IVT F1 上限为 58.33%，只用于诊断候选空间，不能当作可部署成绩。

## 已完成的一次历史补图

最后一项开发对照检验的是**新增真实图像是否改善审核**。18/18 次额外审核已完成且全部有效，两臂各 9 个目标完整替换首次审核；没有额外请求失败回退。其他 7 个目标继续保留，因此主结果仍覆盖全体 16 个目标。[有限历史补图协议](VERIFIER_TEMPORAL_EVIDENCE_PROTOCOL_2026-09-07.md)、[完成结果](../artifacts/preflight/verifier_temporal_memory_v2_20260907/summary.json)

固定原 16 个 Training 目标、H0、H1、裁剪和首次名称审核。只有首次完整有效的审核对 Verb 或 IVT 返回 UNCLEAR、候选确有变化且有额外真实历史图时触发；不查看 GT 或评分 mask 决定触发。实际 9 个目标触发，两臂各最多追加一次整份审核：

- `repeat1000`：与首次名称请求发送相同内容，原图多审核一次。
- `memory1000`：仅增加原三帧之前的 `t-100,t-75`，即 -4、-3 秒的真实图像及必要帧号、引用和图片位置；其余命题、说明、本体、Schema、参数、目标图及裁剪不变。

向过去先检查 t-75，再检查 t-100，遇缺帧停止；不跨视频、不补造或重复帧、不用未来图。不保存过去的预测作为真值，不输入首次回答，也没有示例检索、Tracker 轨迹或自由 reflection 循环。历史帧中的动作不能直接复制为目标帧动作。这里只扩展两秒历史，不能代表已经检验了更密集的运动线索，也不能识别“回答确定但实际错误”的所有情况。

这一辅助层的失败回退与首次审核、严格 factored 不同，必须单独说明：**第二次完整有效则整份替换首次审核；第二次失败、非法或未派发，则保留首次完整审核及其 A/B 结果。** 不把两次回答逐项拼成有利结果，也不把首次已经合法修复的结果重新丢回 H0。

```text
first = 已绑定原请求、完整有效的首次 names 审核
for arm in [repeat1000, memory1000]:
    chosen_review = first
    if 固定条件触发:
        extra_review = 额外一次整份审核(该臂图片, 相同命题)
        if extra_review 整体通过格式、命题和引用校验:
            chosen_review = extra_review
    result[arm] = 原 A/B 规则(h0, h1, chosen_review)
```

比较时先看 memory 相对 repeat 是否有增量，再看各自是否优于 H0。两臂保留全部 16 个目标和失败回退，先保存全部预测再关联 GT，缺失 GT 按任务 mask 排除。额外失败导致保留首次结果不能记作审核质量提高。本轮仍严格使用 1000 字符上限，没有混入上面的长度补充归一化。

结果中，两种 B 的 IVT F1 均为 45.61%，高于 H0 的 44.44%，但集合 Accuracy 均从 20.00% 降至 13.33%。重复 B 实际应用 5 项有益、6 项有害 IVT 编辑；补图 B 为 4 项有益、5 项有害。两种 B 都没有把任何 IVT 错误集合完整改对，并各将一个原本全对的 IVT 集合改错。补图 A 全部保持 H0，不能算成功；重复 A 的 IVT F1 为 43.64%，也未改善。

直接比较补图 B 与重复 B，IVT 有一个目标改善、一个变坏，汇总 F1 和集合 Accuracy 相同；Verb 有一个目标改善，但 Instrument 有一个变坏。补图 B 的 Instrument F1 为 88.46%，低于 H0 的 90.20%；Verb F1 为 73.47%，仍低于 H0 的 75.00%。新增画面有局部帮助，但没有达到整体净收益条件。

补图对照原生费用共 $0.347276：重复组 9 次 $0.135524，补图组 9 次 $0.211752。本轮开发累计 136 次、$2.063434，未知费用与未结算预留均为 0。补图输入、原生费用及逐集合指标已通过 [完整独立审计](../artifacts/preflight/verifier_temporal_memory_v2_20260907/independent_temporal_audit.json)；审计本身没有新增 API 调用。76 项专项测试与相关 Ruff 检查通过，说明工程检查完成，不能替代上述语义结果。

## 当前推荐与没有接入的辅助机制

本轮检验了“固定 H0/H1 → 中立标签存在审核 → Python 限定范围的局部修复”，实际默认输出继续使用 H0。代码已经限制修改范围和增删方向，但尚未证明能稳定筛出有益变化；下一步应完成冻结基线实验与复现材料，保留这些负结果。

既定条件要求 IVT F1 高于 H0，IVT 集合 Accuracy 不低于 H0，有益 IVT 修改多于有害修改，I/V/T 各头 F1 不低于 H0；memory 还需胜过 repeat 的 IVT F1，且不降低其 IVT 集合 Accuracy 和组件 F1。此次重复审核和补图都未通过，所以不进入 Validation 方案确认，也不按这一批 GT 继续调规则。Validation / Testing 标签没有用于方案选择。结论仅覆盖这批 Training 开发样本和具体机制，不证明所有历史图像或视觉审核都无效。

标注示例记忆、自由反思循环和 Tracker 辅助均未加入当前对照。Tracker OOF 能提供未训练该视频的器械定位线索，不能提供动作、组织或接触真值。若以后研究示例记忆，样本只能来自允许的 Training 数据，开发期还需排除目标视频，Validation/Testing 标签不能入库；过去预测也不能作为自证答案。Reflection 若没有可核实的新证据或程序能识别的具体矛盾，只是同一模型再次猜测，目前没有可靠的自由文字语义判错器来自动触发这类修正。

## 论文依据与适用边界

CoVe 将具体核验问题与原回答分离，并研究每个问题独立回答；其消融也提示开放问题可能优于 yes/no。事实问答结果不能直接等同于图片中的 IVT 核验。本项目已经测试的 factored 仍是存在性问题，而且增加了调用次数和总生成预算，不能把任何变化全归因于上下文隔离。[原文方法与消融](https://arxiv.org/html/2309.11495v2#S3.SS3)

Woodpecker 使用视觉工具组织核查和修订；CRITIC 用工具反馈辅助修正，Reflexion 用任务反馈形成反思记忆。这些工作支持检验额外证据与反馈的价值，但不能证明器械框足以判断手术动作，也不能把未经验证的模型结论存起来后视为真值。[Woodpecker](https://arxiv.org/abs/2310.16045)、[CRITIC](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html)、[Reflexion](https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)

另外，缺少外部反馈的自我修正可能失败或退化；相关研究针对推理任务，并非直接的手术视觉结论。[ICLR 2024 研究](https://arxiv.org/abs/2310.01798)
