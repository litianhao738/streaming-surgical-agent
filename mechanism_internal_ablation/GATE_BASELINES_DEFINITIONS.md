# 三种 Gate 基线：定义、处理流程与伪代码

更新日期：2026-09-21。本文依据项目当前固定阈值实现整理，描述 Random、Uncertainty-only 和 Rule-based Gate。本文不修改实验代码、阈值或历史结果，也不包含此前人为随机下调的模拟分数。

## 1. 共同输入与输出

三种方法共用同一份原始答案 H0、候选池和 Qwen Probe 评分，只改变哪些帧继续进入后续多模型复审。

- **H0**：基座模型第一次输出的 Instrument / Verb / Target / IVT / Phase 答案。
- **候选池**：包含现有答案候选及新增候选；四头候选位于 `pool.propositions`，阶段另有 7 个候选。
- **Qwen Probe**：为候选项提供 1–5 分的等级评分；无有效评分记为 `None`。
- **cheap answer**：Gate 前完成候选与先验处理后的答案，不保证与原始 H0 相同。
- **送审决定**：选中帧执行共同的后续复审、聚合与修复；未选中帧保留 cheap answer。

Gate 选择过程不使用测试集 GT，也不查看后续复审结果。GT 仅用于预测封存后的评分。当前规则及阈值属于探索性设置，不代表已在独立验证集上调优。

| 方法 | 依据 | 当前参数 | 是否训练 | Verify 是否固定 |
|---|---|---|---|---|
| Random | 每帧独立的确定性伪随机抽样 | 概率 p=0.5；20 个种子 0–19 | 否 | 否，仅期望约 50% |
| Uncertainty-only | 候选评分居中或阶段竞争接近 | 不确定性分数 ≥0.5 | 否 | 否，由评分分布决定 |
| Rule-based Gate | 反对现有答案、支持新增答案、支持切换阶段 | 规则分数 ≥0.5 | 否 | 否，由规则触发情况决定 |

## 2. Random：独立随机送审

每帧以 50% 概率进入复审，选择不依赖该帧答案或 Probe 评分。代码使用 SHA-256 将实验标识、随机种子和帧 key 映射为整数，与固定阈值比较，使选择与遍历顺序无关、可以复现。

```python
def random_gate(frame_key, seed, probability=0.5):
    text = "bernoulli_fixed_v1|" + str(seed) + "|" + str(frame_key)
    value = int(SHA256(text.encode("utf-8")).hexdigest(), 16)
    cutoff = int(probability * 2**256)
    return value < cutoff

for seed in range(20):
    selected = [key for key in frame_keys if random_gate(key, seed)]
    predictions = route_and_repair(selected, shared_inputs)
    seal(predictions)
    results[seed] = evaluate_on_valid_labels(predictions, ground_truth)

# 分别评分后再对各项指标取均值；同时保留每个种子的结果。
random_result = mean_metrics(results)
```

**解释**：480 帧不要求恰好选中 240 帧；不同种子选中数和效果都可能不同。若错误集合固定、选择与错误独立，则随机送审概率为 50% 时，Error Recall 的期望也是 50%，不意味着这些错误已经被修好。

## 3. Uncertainty-only：选评分不确定的帧

### 3.1 四头候选不确定性

对非歧义的 Instrument / Verb / Target / IVT 候选，评分为 r 时定义：

`u(r) = 1 − |r − 3| / 2`

| Qwen 评分 r | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 不确定性 u(r) | 0 | 0.5 | 1 | 0.5 | 0 |

对这些候选的不确定性取平均，得到 U_candidate。缺失评分的候选按 1 计入；若没有非歧义候选，U_candidate=1。歧义判定调用现有 `original.ambiguous`，不在 Gate 中另设规则。

### 3.2 阶段不确定性

若 7 个阶段评分均有效，取最高分 r_top1 和第二高分 r_top2：

`U_phase = 1 − (r_top1 − r_top2) / 4`

分差越小，阶段越难区分；任一阶段评分缺失时，U_phase=1。

### 3.3 送审决定与伪代码

`U = max(U_candidate, U_phase)`；当 U≥0.5 时送审。该分数范围为 [0,1]，是评分构造的启发式不确定性，不是概率熵或经过校准的错误概率。

```python
def uncertainty_gate(snapshot, threshold=0.5):
    ratings = snapshot.probe_ratings_all
    values = []
    for candidate in snapshot.pool.propositions:
        if ambiguous(candidate):
            continue
        rating = ratings[candidate.id]
        values.append(1.0 if rating is None
                      else 1.0 - abs(rating - 3.0) / 2.0)

    candidate_u = mean(values) if values else 1.0
    phase = [ratings["phase_" + str(i)] for i in range(7)]
    if any(rating is None for rating in phase):
        phase_u = 1.0
    else:
        ordered = sorted(phase, reverse=True)
        phase_u = 1.0 - (ordered[0] - ordered[1]) / 4.0

    score = max(candidate_u, phase_u)
    return score >= threshold, score
```

**例子**：阶段最高分为 5、第二高分为 4，则 U_phase=0.75，达到送审阈值；最高分为 5、第二高分为 1，则该项为 0，但仍需检查候选平均不确定性。

## 4. Rule-based Gate：选有明确纠错信号的帧

该规则以 **H0** 判断候选原本是否存在，不以 cheap answer 判断。四头候选中，歧义候选和缺失评分候选均不参与两个比例的计算。

### 4.1 三项规则分数

1. **反对现有答案** R_refute：属于 H0 的有效候选中，评分≤2 的比例。
2. **支持新增答案** R_support：不属于 H0 的有效候选中，评分≥4 的比例。
3. **支持切换阶段** R_phase：仅当 7 个阶段评分全部有效时计算。取 H0 阶段之外的最高评分 best；如果 best≥4，则 `R_phase=max(0, best−H0阶段评分)/4`，否则为 0。

前两项若对应候选集合为空，该项为 0。阶段评分任一缺失，阶段项为 0。

`R = R_refute + R_support + R_phase`

三项直接相加，不再除以 3；在合法评分下 R 的范围为 [0,3]。当 R≥0.5 时送审。

### 4.2 伪代码

```python
def rule_based_gate(snapshot, threshold=0.5):
    ratings = snapshot.probe_ratings_all
    present, absent = [], []

    for candidate in snapshot.pool.propositions:
        rating = ratings[candidate.id]
        if ambiguous(candidate) or rating is None:
            continue
        if candidate.label_id in snapshot.h0[candidate.task]:
            present.append(rating)
        else:
            absent.append(rating)

    refute = count(r <= 2 for r in present) / max(1, len(present))
    support = count(r >= 4 for r in absent) / max(1, len(absent))

    phase_score = 0.0
    phase = [ratings["phase_" + str(i)] for i in range(7)]
    if all(r is not None for r in phase):
        old_phase = snapshot.h0["phase"][0]
        best_alternative = max(phase[i] for i in range(7) if i != old_phase)
        if best_alternative >= 4:
            phase_score = max(0.0, best_alternative - phase[old_phase]) / 4.0

    score = refute + support + phase_score
    return score >= threshold, score
```

伪代码中 `count(condition for ...)` 表示统计条件为真的数量。输入沿用项目已验证的候选结构、合法评分，以及包含一个合法阶段编号的 H0。

**例子**：4 个现有有效候选中有 2 个被打低分，R_refute=0.5；即使没有新增支持或阶段切换信号，仍会触发送审。

## 5. 两种评分 Gate 的关键区别

| 情况 | Uncertainty-only | Rule-based Gate |
|---|---|---|
| H0 中某个候选被评分 1 | 该候选不确定性为 0，因为评分很明确 | 计入反对现有答案的低分候选 |
| 某个候选被评分 3 | 该候选不确定性为 1 | 不触发≤2 或≥4 的四头规则 |
| 新候选被评分 5 | 该候选不确定性为 0 | 计入支持新增答案的高分候选 |
| 某个四头评分缺失 | 按不确定性 1 计入平均 | 跳过，不进入比例分母 |
| 任一阶段评分缺失 | 阶段不确定性设为 1，会触发当前阈值 | 阶段规则项设为 0；四头规则仍可触发 |

表中描述的是单项贡献，最终动作仍取决于完整分数。Uncertainty-only 关注“评分是否拿不准”，Rule-based 关注“是否有明确证据需要改答案”；两者可能选中不同的帧。

## 6. 统一实验流程伪代码

```python
# 阶段 A：不使用 GT，准备三种策略共享的输入。
for frame in frozen_testing_frames:
    snapshot[frame.key] = build_h0_candidates_cheap_and_qwen_probe(frame)

# 阶段 B：按各自固定参数选帧，不做 top-K，不强制相同复审预算。
strategies = ["random", "uncertainty", "rule"]
selected_by_strategy = {}
for strategy in strategies:
    selected = apply_gate(strategy, snapshot, frozen_parameters)
    selected_by_strategy[strategy] = selected
    for key in snapshot:
        if key in selected:
            prediction[strategy][key] = common_mar_repair(snapshot[key])
        else:
            prediction[strategy][key] = snapshot[key].cheap
    seal(selected, prediction[strategy])

# 阶段 C：仅在选择及预测封存后读取 GT，按任务有效 mask 评分。
truth = load_testing_truth()
for strategy in strategies:
    report_verify_error_recall_bvr_and_task_metrics(
        selected_by_strategy[strategy], prediction[strategy], truth)
```

此处为逻辑流程，实际可按批次执行，并复用请求与配置完全相同的缓存。Random 需按前述 20 个种子展开。

公平对比必须统一样本、模型、候选处理、复审机制及最终输出边界。480 点实验使用共同五席复审缓存；2,855 点扩大实验使用历史早停复审。两种协议应明确区分。若评估完整流程，应给所有策略应用相同的 Tracker、清理和阶段平滑。

当前公平对照中 Random 也保留公共 Probe 前缀，尽管其送审动作本身不需要 Probe 评分；不能把删除该前缀后的低成本实现与当前 Token 口径混写。

## 7. 实现位置与版本区别

- [fixed_threshold_gate.py](src/fixed_threshold_gate.py)：`uncertainty_score` 和 `choose`，定义本文件所述固定阈值选择及独立随机抽样。
- [mechanisms.py](src/mechanisms.py)：`rule_score` 定义规则评分，`digest_key` 定义可复现哈希。
- 固定阈值运行目录中的 protocol.json 记录阈值、随机种子及实验口径；该文件由本地运行生成，不随源码发布。

注意：`mechanisms.py` 还保留早期匹配预算版 `selections`，其中 Rule 和 Random 按 top-K 选出与 Learned 相同的帧数。**它不是本文描述的当前固定阈值入口。** 本文 Rule-based 也不等同于项目历史所有名称中含 Rule 的其他 Gate。

此前的随机下调表属于人为模拟；本文描述的随机性仅指 Random 的选帧过程，不允许在真实评分后随机修改指标。
