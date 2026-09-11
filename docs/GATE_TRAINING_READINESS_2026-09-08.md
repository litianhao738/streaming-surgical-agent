# Gate 训练准备核查与 final-only 训练入口

2026-09-08。本轮完成代码与数据核查、离线准备、CPU 拟合测试以及下一批采集预检；新增付费 API 调用为 0。**工程入口已补齐，当前真实数据仍不足以开始可靠的 Gate 拟合。没有生成或部署新的真实 Gate 权重。**

## 核实到的原有状态

| 项目 | 实际状态 |
|---|---|
| corrected Tracker OOF | 已安装。重新读取索引、6 份预测产物、训练 manifest 与 checkpoint 哈希，核对 `clip_to_frame_v2` 和数据集修复哈希。10 个 Training 视频均有对应输出，目标视频不在其 detector 训练视频列表中 |
| 已有 learned Gate 权重 | `artifacts/training/gate/demo/`，200 行、仅 VID31，`demo_only=true`、`paper_final=false`。权重中的历史阶段字段不等于正式可用性证明 |
| 旧 formal Gate 代码 | 有 D0 收集、三修复 scope 的收益监督、视频分组训练与诊断脚本；没有当前 final-only H0 的正式训练结果 |
| 旧 formal 真实 pilot | 20 个计划目标，17 个 H0 完成。Instrument 与 Interaction 都是 0 正例、14 负例、3 未知；Workflow 是 2 正例、11 负例、4 未知。不能拿这个表启动完整三路 Gate 训练 |
| 当前 H0 兼容性 | 旧 `FORMAL_GATE_FEATURE_ORDER` 包含 selected confidence 和 candidate margin；当前 final-only H0 没有这些量。将硬标签填成 0/1 分数不能恢复 top-k 信息 |
| 最新五席三轮数据 | 12 个 Training 目标。收益标签可用 11 个：1 正、10 非正；另 1 个空池未完成有效审核，保留为未知。唯一正例在 VID96 |

本轮核查前 HEAD 为 `f34fbeafecf6827dd3d269a9c0254ad16914fe76`。现有未提交修改、Tracker 产物、历史实验目录和旧 Gate 保留。本轮新代码尚未提交或推送。

## 为什么不能直接开始正式训练

原目标是 **Benefit Gate：在修复前预测“运行这次修复会不会使答案更好”**。
它与 Risk Gate“这个 H0 是否可能有错”不同。H0 错了，修复可能仍然改不对，甚至更坏；两种标签不能混用。

最新缓存的离线装配结果：

| 目标 | 有效样本 | 正例 | 负例 | 未知 | 当前可做视频留出拟合 |
|---|---:|---:|---:|---:|---|
| 修复有收益 `benefit` | 11 | 1 | 10 | 1 | 否 |
| 有收益且没有任何头改坏 `safe_benefit` | 11 | 1 | 10 | 1 | 否 |
| H0 的 I/V/T/IVT 至少一头有错 `h0_error` | 12 | 11 | 1 | 0 | 否 |

把 VID96 留出后，Benefit Gate 的训练部分有 8 个负例、0 个正例；其余留出折的训练部分也只有 1 个正例。
Risk Gate 同样受限于唯一正确样本，不能通过改名绕过数据不足。
训练脚本要求至少 3 个有标签的视频，且每个视频留出后的拟合部分至少有 2 个正例和 2 个负例。这只是防止退化拟合的工程最低条件，不是统计充分性保证。

Verifier/Repair 不必先变成完美系统才能训练 Gate。可以固定一个版本，学习其有益与有害的条件；但必须有足够的实际收益标签，并在独立视频上评估。
当前修复正例很少，不能承诺再训练一个 Gate 就能解决 Target/IVT 的视觉误判。建议暂停反复修改审核机制，先冻结采集策略。

## 本轮补齐的训练工程

新增 [final-only 特征和监督契约](../src/surgical_agent/research/gate/final_only_training.py)、
[离线准备入口](../scripts/prepare_final_only_gate_training.py) 和
[CPU 训练入口](../scripts/train_final_only_gate.py)。

这是一条独立的 **final-only 整次修复收益 Gate** 训练路径：选择保留 H0 或调用既定的整次 I/V/T/IVT 修复。
它没有伪造旧 Gate 的 top-k 特征，也没有宣称完成旧 Instrument / Interaction / Workflow 三 scope 的全部迁移。
最新五席修复不改 Phase，所以此版不能训练“修复 Phase 的收益”。Phase 的 H0 错误仍单独统计。

特征版本 `final_only_gate_features_v1`，共 31 维：

- 20 个基础特征：H0 各交互头标签数、Phase one-hot、空交互输出、真实历史图片数、null IVT 数，以及 IVT 投影与独立头的差异计数。投影差异只是特征，不删除独立头标签。
- 11 个 Tracker 特征：是否可用、器械/类别数、检测器分数、与 H0 器械集合的差异、短历史轨迹新增/消失/保留。检测器分数明确属于 Tracker，不冒充 H0 置信度。
- 没有 GT 标签、任务 mask、修复候选、五席分数、最终结果、视频 ID、绝对帧号作为模型特征。
- 一份 H0 对应“无 Tracker / 有 Tracker”两组配对特征。它们属于同一观察，不把 12 帧写成 24 个独立样本。

准备过程先保存 `features_before_gt.json` 及哈希，再从原始数据集关联 GT/mask。
训练入口逐项核对有标签记录中的特征仍等于这个先前保存的版本；额外特征、跨 split、重复目标、混合 H0/修复版本会被拒绝。
同一视频始终整体留出；无 Tracker 与有 Tracker 的比较共享完全相同的观察和划分。

收益监督 `final_only_episode_utility_v1` 定义为：I/V/T/IVT 四头都具有有效 GT 时，修复后逐帧集合 F1 的四头均值是否高于 H0。
两边集合都为空时该头效用为 1。这个效用是训练标签，**不是报告中的 pooled micro-F1**。
同时保存效用差、逐头集合损失变化和是否有头改坏；缺失 GT、空池未审核、失败未完成的修复不被补成负例。

CPU 模型为标准化后的逻辑回归：`C=1`、`class_weight=balanced`、`liblinear`、`max_iter=2000`、seed 3407。
预定诊断阈值为 0.5，不在这批数据上搜索最佳阈值；输出包含视频留出 AUROC/AP、收益放行/误放、路由比例、效用差，以及 always-repair / never-repair 对照。
模型分数尚未校准，不直接当作可靠的正确率。生成物标记 `deployable=false`，不自动安装进默认 Pipeline。

**OOF 边界：**目标视频没有参加其自身 Tracker 的训练，已经核实。
但只把 Gate 按视频留出，并不等于整个 Tracker + Gate 系统做了嵌套 OOF：其他 Gate 训练行使用的 detector 可能见过该 Gate 留出视频。
因此这些分数只叫“固定上游输出条件下的 Gate 诊断”，不能冒称整个系统无泄漏的最终验证。正式 operating point 仍需冻结后独立 Validation 校准，Testing 保持不用。

## 实际准备好的本地产物

当前准备目录：`artifacts/training/gate/final_only_preparation_20260908_v2/`。
较早的同日准备目录保留，当前以 `_v2` 为准。

| 文件 | 内容 |
|---|---|
| `features_before_gt.json` | 12 个目标的冻结特征与 OOF 来源，不含 GT |
| `training_examples.json` | 同 12 个目标的特征、源预测、GT/mask、收益和风险标签，仅供开发核查 |
| `manifest.json` | 来源/源码/权重哈希、31 维特征顺序、逐视频类别计数、不能开训的具体原因 |
| `pilot_selection.json` | 下一批 40 个目标、图片哈希、标注可用性、历史目标排除与调用上限 |

当前完整五头 GT 覆盖分别为 VID103：1545 个、VID23：1168 个、VID31：3732 个、VID96：1119 个。
其他六个 Training 视频在当前可用标注中没有完整的 V/T/IVT，不能为这次整段交互收益监督补负例。

下一批从上述四视频各固定 10 个时间分位目标，只按时间与标注可用性选样。
避开已识别历史计划目标前后 75 个原视频帧；新选目标之间也保持超过 75 帧。
具体帧号与排除来源哈希均写入计划。这里的“新”指相对计划索引排除后的采集目标，不是未见过的新手术或独立测试集。

## 同步采集预检与费用范围

为准备 Gate 数据，原 [三轮同步入口](../scripts/run_recent_mean_panel_trial.py) 增加了 `--selection-manifest` 和显式分账户额度参数。
它逐项重查 Training split、任务 mask、因果窗口和图片哈希；调用上限按实际样本数计算，不再固定为 12 帧的上限。
审核提示、模型、评分阈值与三轮机制保持现有版本，仍带真实的学术手术视频语境说明。
[独立审计入口](../tools/audit/audit_recent_mean_panel.py) 同样改为核对冻结计划目标数，保留旧 12 帧兼容性。

已完成的采集预检目录：`artifacts/preflight/gate_final_only_collection_20260908/`。
内含 `plan.json`、40 份 H0 请求预检、OpenRouter 端点/价格元数据和冻结源码。
H0 仍为 OpenRouter Gemini 3.8 Flash；五席品牌与最近实验相同。Tracker 仅作为 Gate 的离线特征，不改变本次 H0 或修复输入。

- 40 个 Training 目标；最多每目标 1 次 H0 + 3 ×（1 次补池 + 5 次审核），即 **760 次新增 POST 上限**。
- 已配置本批新增本地额度上限：**OpenRouter $3、xAI $3、阿里云 ¥3**，分别接续旧账本。预留不是实扣金额，也不是供应商硬封顶。
- 根据此前 12 目标的已知用量，粗略线性外推约 $2.76 原生美元费用 + ¥1.09 token 估价；候选长度、失败和提前停止会改变实际费用，这不是报价。
- 本轮只访问公开 OpenRouter 端点元数据并构造请求；**没有发送这 760 次中的任何付费请求**。因此也没有本批新的接口成功率、收益标签或 Gate 训练效果。
- 公开元数据预检不能替代 Grok / Qwen 账号可用性或下一次真实响应的检查。执行时沿用显式失败/费用记录，不用替换型号或静默重试补齐数据。

这些新采集结果属于带空池保护的冻结版本，不能与旧付费版本的 12 帧直接混合作为同一策略的训练集。导入器以策略与源码哈希区分来源。

## 本地命令

现有 `.venv-p2` 已能运行，不需要为了这个逻辑回归训练重新启动 GPU 任务。

只检查当前准备数据，不训练、不调用 API：

```powershell
.venv-p2/Scripts/python.exe scripts/train_final_only_gate.py --prepared artifacts/training/gate/final_only_preparation_20260908_v2 --check-only
```

当前返回 `can_fit_pilot=false`，这是本次真实核查结果。

下面第一条是**付费采集**，本轮未执行；后续三条为离线审计、装配和训练：

```powershell
.venv-p2/Scripts/python.exe scripts/run_recent_mean_panel_trial.py execute --output artifacts/preflight/gate_final_only_collection_20260908
.venv-p2/Scripts/python.exe tools/audit/audit_recent_mean_panel.py --output artifacts/preflight/gate_final_only_collection_20260908
.venv-p2/Scripts/python.exe scripts/prepare_final_only_gate_training.py --source artifacts/preflight/gate_final_only_collection_20260908 --output artifacts/training/gate/final_only_collected_20260908
.venv-p2/Scripts/python.exe scripts/train_final_only_gate.py --prepared artifacts/training/gate/final_only_collected_20260908 --target benefit --output artifacts/training/gate/final_only_pilot_fit_20260908
```

代码、请求或前一费用账本变化时，旧预检会拒绝执行，需在**新目录**重新 prepare；不删除执行锁或覆盖历史目录。
若采集提前失败，保留部分记录与费用，不能把未发送样本当负例；若 40 帧仍不足以使每个拟合折有正负例，训练仍会拒绝。
不能从不同轮次按 GT 选最好答案来造正例，也不能为凑类别修改审核阈值。正式拟合要等实际类别分布和冻结策略满足条件。

## 验证记录

42 项相关单元/集成测试通过，包含 final-only 特征、缺失 GT、未知修复、视频划分、混合策略拒绝、GT 关联后特征一致性、CPU 参数序列化以及原三轮入口回归。
本轮相关 Python 文件 Ruff 通过。CPU 拟合只使用明确的合成测试夹具验证实现，不当作真实 Gate 效果。
真实 12 帧已完成 OOF 装配与训练前检查；Training 外标签未用于本轮准备。

本周下一步是采集这批已冻结策略的收益数据，再决定能否拟合；可以暂停继续改 Verifier/Repair，但不能跳过收益监督的数据准备。
