# H0 低分原因：40 个开发目标的证据分析

本次只离线分析，没有新增 API 费用，没有修改预测、GT、默认提示词、Pipeline 或冻结版本。来源为严格重算的 80 份回答：原版和只删重复 Schema 版各 40 个目标，四个 Training 视频各 10 个目标。以下错误计数以原版为主，单位是“某标签在某个目标帧出现一次”，不是器械实例数。

## 结论

主要问题是细粒度语义识别：靶点偏向胆囊板/胆囊，动作偏向 retract/dissect，null 交互漏识别，部分器械类别漏识别，以及阶段严重集中在两种解剖阶段。严格的集合 Accuracy 会进一步降低数字，但实际误报、漏报也很多。不能据此认定“只要加帧”“只要改 JSON”或“加一次修复”就能解决。

## 已排查的评分因素

- 从实际 HTTP/SSE 可见回答重建 JSON，80/80 有效；与旧预测零差异。
- 重新读取源 GT/已有明确来源的 VID31 修复标注；40 个目标五头 GT 都有效，各头分母均为 40。
- IVT：22 TP、46 FP、42 FN，micro-Precision 32.35%，micro-F1 33.33%。4 帧整组全对，14 帧部分命中，22 帧没有任何正确 IVT。
- Phase 是单分类，17/40=42.50%；其低分无法用多标签集合苛刻来解释。
- 这些是开发目标的 micro-F1/集合 Accuracy，不是官方完整测试集 mAP。

## 错误集中在哪里

| 类型 | 可复核的计数 | 能得出的结论 |
|---|---|---|
| Target | cystic_plate 预测 21 次，仅 3 TP、18 FP；占全部 38 个 Target FP 的 47.4% | 胆囊板过度预测是明确弱点 |
| Verb | grasp 有 8 个 GT，仅 2 TP；coagulate 有 4 个 GT，0 TP；retract 预测 24 次，其中 10 FP | 动作分界和凝血识别不足，不能仅归因于工具定位 |
| null | 7 个 GT null IVT 标签均未命中；唯一预测的 null IVT 出现在另一帧 | 模型经常把 null 交互替换为具体动作/靶点；null 不是缺失 GT |
| Instrument | bipolar 有 7 个 GT，仅 1 TP | 器械识别也有明显类别短板；不能因为总体 I F1=82.76% 就认为这部分已解决 |
| IVT | grasper-retract-gallbladder 有 14 FP；hook-dissect-cystic_plate 有 9 FP；合计占 IVT FP 的 50% | 错误集中在熟悉的交互类别，并非完全随机 |
| Phase | 37/40 输出是 calot_triangle_dissection 或 gallbladder_dissection；7 个 cleaning_and_coagulation GT 全部漏掉 | 阶段区分不足；短历史缺少流程信息是待验证解释 |

这些是帧级标签统计，不能用它们直接推断一对一器械实例的混淆关系，也不能推断模型训练数据中的类别频率。

## 原图与逐帧证据

本轮实际查看以下 6 个目标原图，其中 VID96/21351 额外查看 t-50、t-25 两张历史图；其余为目标图抽查，并未完成 40 个窗口的临床重标注。

1. **VID103/25101：靶点单独错误。** GT 是 hook-dissect-gallbladder，预测 hook-dissect-cystic_plate；Instrument、Verb、Phase 都正确。原图可见工具尖端位于相邻组织的边界，原始记录的工具类别/动作/靶点分别是 2/2/0，未标记遮挡、烟雾或模糊。至少这帧的 IVT 错误不能只怪工具漏检或严重画质问题；也不能凭本次目视检查宣布 GT 解剖类别错误。同类单标签 60→59 共 5 帧。
2. **VID96/21351：动作单独混淆并伴随阶段错误。** GT grasper-grasp-gallbladder，预测 grasper-retract-gallbladder。三张图都只显示局部工具和组织，画面变化有限，不能可靠重建连续牵拉过程。提示词给出了 grasp/retract 名称，没有这组类别的标注判定实例。可能同时存在动作证据不足与模型采用日常词义的因素，目前无法分开量化。
3. **VID23/36501：器械类别和 null 语义同时错误。** GT 器械是 bipolar/scissors、IVT 为 95/97，预测 grasper/clipper、IVT 为 7/79。原图可见两器械末端靠近/接触，原始记录两实例均标记遮挡。不能仅凭“附近有组织、器械在活动”推断它正在夹胆囊或夹闭胆囊管。这帧并非缺少 IVT GT。
4. **VID31/68101：正确识别工具仍然给 null 工具强加动作。** GT [17,96]，预测 [17,60]，工具集合 [0,2] 正确；把 null hook 交互替成 hook-dissect-gallbladder。此视频使用已有来源映射的帧级标注，没有本轮可用的逐实例 GT，因此不做实例级定位正确性结论。
5. **VID103/42651：多种错误叠加。** GT bipolar/grasper 在 liver 上的 coagulate/retract，预测只剩 grasper-retract-gallbladder；Phase 5→3。原图有烟雾、部分工具被遮挡，原始记录也标记 smoke/occluded。这是实际视觉难点的例子，但一个例子不能证明烟雾解释全部低分。
6. **VID23/20776：正例。** GT 和预测五头完全一致，IVT 为 hook-dissect-gallbladder。原图有清楚的工具接触区域；这说明同类交互有时能正确识别，不能断言模型完全不认识该类别。

原图位于 `D:/cholec_dataset/Training/<VID>/Frames/<六位帧号>.png`。

## 请求机制为何可能暴露这些弱点

实际请求使用 `qwen/qwen3.8-max-0902`，三张图对应 -2、-1、0 秒，历史图 detail=low、目标 high；只预测最后一帧，Tracker/workflow 历史为空。它有时间顺序，但不是连续视频运动证据，也没有较长的手术流程历史。低细节历史图和两秒跨度是否分别影响 Verb、Phase，需要独立对照，不能凭当前结果定因。

提示词已包含完整 100 类 IVT 映射、目标帧边界、每个器械的接触关系要求、null 语义。因而不能说“完全没告诉它类别或 null 定义”。但文字规则不等于模型已学会从组织外观和动作中执行这些分类边界。现有输出只保留五头集合，无法从输出本身判断是定位错误还是看对位置后分类错误；内部解释也不能作为视觉正确性的证明。

只删 Schema 后，40 目标 IVT F1 从 33.33% 到 28.80%，并未解决低分；此前增加通用证据规则的 tuned 版在 16 目标上也未改善。样本小且每条件单次回答，不能把差异都归因于删除某一句，但已没有足够证据采用精简版替换默认。

## 下一步建议及验证边界

优先检验“模型是否理解视觉类别边界”，而不是再扩大 Agent/Repair 链路。最小候选实验是保持原版三帧联合预测，加入少量来自独立开发视频、人工核实的视觉对照示例，覆盖靶点分界、grasp/retract 和 null；示例中不得使用这 40 个评测目标或其近邻。用其他未参与示例选择的视频做配对验证，观察完整 IVT F1、Target FP、null 命中和其他类别退化，同时记录示例带来的成本。该方案只是可检验假设，不保证提高准确率，也尚未运行或估算付费预算。

若仍无法改善，应评估训练一个有监督的手术视觉识别模块，学习器械—接触组织—动作关系；Tracker 的身份关联训练本身不能替代这些分类目标。是否必须训练、训练多少数据，当前实验不足以确定。Phase 的长历史问题应单独做对照，避免同时改变多个变量。

CholecTriplet2021 将任务定义为细粒度工具—动作—组织关联识别，报告包含专门训练的空间/时间建模方法；它使用 AP/mAP，不能与本次集合 Accuracy 横向比较。论文为研究方向提供依据，并不能证明某种方法一定解决本项目当前错误：<https://arxiv.org/html/2204.04746v2>。CholecTrack20 官方说明原视频 25 FPS、标注帧 1 FPS：<https://github.com/CAMMA-public/cholectrack20>。因此“标注每秒一帧”本身不是数据处理错误。

## 复查产物

- `artifacts/preflight/h0_error_diagnosis_20260906/label_errors.json`：两版本每类 GT、预测、TP/FP/FN 和单标签混淆计数，含严格重算来源 SHA256；计数恒等式已检查。
- `artifacts/preflight/h0_strict_rescore_20260906/strict_rescore.json`：逐帧预测与 GT。
- `artifacts/preflight/h0_prompt_refinement_qwen0902_20260906/requests/VID103_25101_baseline.json`：抽查实际请求。
- `docs/H0_STRICT_RESCORE_2026-09-06.md`、`docs/H0_SCHEMA_CONFIRMATION_SMOKE_2026-09-06.md`：严格指标与两轮配对结果。
