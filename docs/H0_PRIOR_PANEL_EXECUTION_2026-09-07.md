# H0先验与多模型审核：执行记录

状态：**实现、真实冒烟、8目标两臂试验、独立评分及审计完成。未达到预先约定的IVT收益条件，保留默认H0。** 无先验臂修正1个器械标签；先验臂全部保持H0；两臂IVT micro-F1均为24.00%，与同批H0相同。本结果只适用于这批Training小样本，不能与历史不同样本的分数直接比较。

用户已授权使用OpenRouter同步API执行方案，总新增预算为15美元。默认纯API入口与H0配置保持不变；新增入口为 [run_prior_panel_trial.py](../scripts/run_prior_panel_trial.py)，核心逻辑为 [prior_panel.py](../src/surgical_agent/research/verification/prior_panel.py)。未提交或推送Git，保留其他本地修改。

## 已实现边界

- Training统计逐帧同类去重、独立任务mask、逐视频等权；每个目标使用排除整段视频的先验表。Phase条件向全局收缩，保留原先约定的支持门槛和4个新增IVT／1个Target上限。
- H0全部四头标签及7类器械进入固定命题集合，先验仅扩充候选。Judge看不到先验频率、来源或其他成员的回答；Repair收到具体问题和确定性的操作角色约束。
- 每轮3位Judge返回存在／不存在／无法判断；任何成员整体失败，整轮不发布修改。单项引用或见证字段矛盾只将该项归为无法判断。
- 最多3轮审核、2次局部补丁。新增至少2票，删除需3位全帧否定；每个有效后续轮次重新审核相对H0的全部累计修改。共享组件父关系按当前修订稿重建；不全量投影独立头，不修改Phase。
- 每次发送前落盘预留费用与唯一调用身份，不自动重试。保守费用覆盖完整模型上下文、最高输入／缓存费率，并额外覆盖图片和原生推理计价。补丁和完整后续审核一起预留，避免产生无法复核的修订稿。

## 与准备稿的明确差异

1. 用户将拟定20美元预算改为15美元。所有版本的实际请求累计计入188次上限；先前未结算失败按其完整预留金额计入新版本预算，不当作免费。
2. v1 Google Vertex EU返回400：复杂Schema约束状态过多。v2仅对Google的线上Schema移除长度／数量／数值边界及pattern，仍保留类型、枚举、必填项和禁止额外字段；提示中保留完整Schema，本地仍按完整原版Schema校验，1000字符上限没有放宽。此方法符合Google对复杂结构化输出的[限制说明](https://ai.google.dev/gemini-api/docs/structured-output)。
3. v2 Google Vertex EU返回429：上游共享池暂时限流。v3将同一模型固定到Google AI Studio路由，关闭回退；其他模型仍固定原路由。路由公开能力声明不代表冒烟成功。
4. 新反馈识别忽略自由文字改写、坐标微小变化和响应内实例名；使用命题、操作、问题类型及角色／本体边界区分，避免把同一图片上的坐标抖动当作新的修复依据。
5. v3四次HTTP／JSON格式检查通过，但GPT仅返回48项中的10项，完整覆盖失败，正式实验没有启动。v4将每次Judge Schema的数组数量设为实际命题数（minItems=maxItems）；Google线上仍使用已说明的简化Schema，本地覆盖检查继续对所有模型强制执行。此变更解决漏项工程问题，不代表视觉答案更准确。
6. v4正式panel三次请求在完整预留后并行执行；账本派发和结算加锁，测试覆盖并发费用不丢失。独立请求不共享回答，未改变投票规则。

v1至v3的原请求、错误和账本分别保存在 `artifacts/preflight/h0_prior_panel_openrouter_20260907_v1/` 至 `..._v3/`。首次完整预检与主运行位于 `artifacts/preflight/h0_prior_panel_openrouter_20260907_v4/`，保存所有前序账本哈希及累计费用负债。前序共发6次请求，v4最多还能发182次，包含4次冒烟；未因修订版本重置188次／15美元总额度。

v4的48项容量检查三位Judge均完整通过，Repair也通过；正式试验在VID31/73601先验臂遇到Gemini `blocked: OTHER` 后因费用明细缺失停止。v6仅续跑从未发送的步骤，无先验臂在同一目标再次被Gemini阻止；v7继续最后两个目标。所有已尝试的失败臂保留H0，没有重跑；8份H0各只调用一次。v5只是未完成的离线续跑预检，没有API调用：它发现无关Tracker训练代码变化，后续修正为锁定本实验实际依赖，保留那些本地修改。续跑均核对请求构造函数、模型、路由、本体和Schema未变，复用v4成功的冒烟证据。

**最终合并结果目录为 `artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/`**，保留v1/v2/v3/v4/v6原始调用目录与账本。最终预测完成时间为2026-09-07 07:58:23 UTC，完成SHA为 `69c5c6b5b4d88fe34643994fb72d11732caf4f6004d536d6860e1a54f3ac0a11`。直到该完成记录写入后才关联这8目标的GT算分。

## 样本与离线诊断

固定8个Training目标：VID103/18426、33676；VID23/13076、27726；VID31/40801、73601；VID96/14876、25876。全部通过官方适配器时间轴及图片哈希检查，未因缺失GT换样。VID23/27726仅Instrument和Phase有有效监督，因此IVT、Verb和Target各按7个有效目标评分，另外两头按8个评分。另核查25份历史plan，未发现与这批目标相距50原视频帧以内的目标。

先验候选诊断复用既有Training预测，15个有效IVT目标中H0漏检14个标签：

| 候选来源 | 新增IVT候选数 | 新增真IVT | 漏检覆盖率 | 新候选命中率 |
| --- | ---: | ---: | ---: | ---: |
| 无先验 | 0 | 0 | 0% | 无分母 |
| 全局先验 | 60 | 10 | 71.43% | 16.67% |
| 阶段＋全局 | 60 | 11 | 78.57% | 18.33% |

阶段条件额外命中只发生在一个视频，没有达到准备稿要求的至少两个视频；不据此声称阶段条件有可靠增益。上述结果只说明存在可审核的漏检候选，不能证明修复有效。新实验仍按原固定两臂执行，不根据GT重新挑候选、改阈值或选目标。

## 离线检查与复现

新增机制、接口适配和连续续跑测试16项通过；连同final-only边界、既有独立评分及冻结初始预测测试，共56项通过。改动Python文件Ruff通过。默认H0五个关键文件相对HEAD没有修改。独立审计重新计算各头集合TP/FP/FN，并验证每份H0恰好对应一份原生API回答，避免将修订稿冒充初始预测。

依赖：先按项目快速上手完成安装，再安装 `requirements-prior-panel.txt`。使用项目环境执行：

```powershell
.venv-p2/Scripts/python.exe -m pip install -r requirements-prior-panel.txt
.venv-p2/Scripts/python.exe -m pytest tests/unit/test_prior_panel.py tests/integration/test_prior_panel_budget.py -q
.venv-p2/Scripts/python.exe scripts/run_prior_panel_trial.py prepare --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe scripts/run_prior_panel_trial.py smoke --output artifacts/preflight/NEW_UNIQUE_RUN --api-key-file docs/API.txt
.venv-p2/Scripts/python.exe scripts/run_prior_panel_trial.py run --output artifacts/preflight/NEW_UNIQUE_RUN --api-key-file docs/API.txt
.venv-p2/Scripts/python.exe scripts/run_prior_panel_trial.py score --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe tools/audit/audit_prior_panel.py artifacts/preflight/NEW_UNIQUE_RUN
```

`prepare`不调用生成API，但会读取允许的Training监督来拟合先验；`smoke`和`run`会计费。旧目录禁止重复派发；继续明确修订版本时，用`--previous-run`列出所有属于同一次授权的先前版本，将已发请求和未结算负债计入总预算。调用数量上限与美元上限同时生效，不承诺一定跑满每个目标的三轮。

评分必须在全部预测完成并保存完成哈希后进行。D1、D2、D3是同一执行序列的相关前缀，不能当作三组独立实验或根据GT选最高的一轮。Testing未参与开发。

## 最终同批对照

每格为 **micro-F1 / 集合Accuracy（%）**；终态采用预定最多三轮策略，不按GT挑选轮次。

| 任务 | 有效GT数 | 固定H0 | 无先验＋三模型审核修复 | 先验＋三模型审核修复 |
| --- | ---: | ---: | ---: | ---: |
| Instrument | 8 | 96.30 / 87.50 | **100.00 / 100.00** | 96.30 / 87.50 |
| Verb | 7 | 58.33 / 42.86 | 58.33 / 42.86 | 58.33 / 42.86 |
| Target | 7 | 56.00 / 14.29 | 56.00 / 14.29 | 56.00 / 14.29 |
| IVT | 7 | **24.00 / 14.29** | **24.00 / 14.29** | **24.00 / 14.29** |
| Phase | 8 | 87.50 / 87.50 | 87.50 / 87.50 | 87.50 / 87.50 |

| 相对H0的最终修改 | 无先验臂 | 先验臂 |
| --- | ---: | ---: |
| 有益新增器械标签 | 1 | 0 |
| 有益IVT修改 | 0 | 0 |
| 有害标签修改／改坏目标 | 0 / 0 | 0 / 0 |
| 无收益替换 | 0 | 0 |
| 保持H0目标 | 7 | 8 |

改正的器械标签发生在只具备Instrument和Phase有效监督的目标，不能称为五头全部改对。无先验臂共有2次有效补丁提案和2次完整复核：按可用GT两次提案各有一个有益器械新增，最终接纳1次、拒绝1次，有益修改放行率50%；没有有害提案，错误修改放行率无分母，报告null而非0%。先验臂未触发补丁。实际没有第3轮调用，因此本轮未验证真实三轮反复修订的收益；其边界行为只有离线测试。

3/16个审核臂工程失败并回退H0：VID31/73601的两个Gemini分支被阻止；VID96/25876无先验臂的GPT到16384输出tokens后截断。所有目标保留在主对照中，缺失监督仅按任务mask排除。

## 具体瓶颈与判断

在7个有效IVT目标中，H0有9个假阳性、10个漏检。先验新增28个IVT候选，其中6个是真实漏检，覆盖率60%、命中率21.43%；仍有4个真实标签不在候选池中。它确实补进了部分正确候选，但没有转化为修复收益。

这6个真实新候选中，1个属于未完成审核的失败目标；另外5个完成了完整初审。**5个候选没有任何一位Judge给出原始PRESENT，其中3个被三位模型一致判为ABSENT。** 这些5项的原始判断与规范化判断相同，因此不能将这里的漏检归因于程序删掉了支持票。可用初审中也没有错误的H0 IVT获得多数ABSENT，IVT修复从触发阶段就没有启动。增加候选、解释或轮数本身不能纠正这些共同误判。

另外仍有工程层面的输出问题：先验臂GPT的完整初审回答中，19项出现目标／时序引用不符合输入的情况，被规范化为UNCLEAR；另有实例／交互见证缺失。它们会损失有效票，值得记录，但无法解释上述5个正确IVT的共同否定。JSON格式通过、证据字段存在和多模型一致，都没有证明视觉标签正确。

本批还观察到无先验臂跨有效轮次53个一致明确判断中有3个错误，先验臂60个中有6个错误；这些是相关的命题／轮次计数，不是独立样本准确率。完整成员检测精确率、召回率、弃权及修改接纳统计见独立审计。

**本轮不升级默认Pipeline，不进入Validation/Testing选方案，也不继续增加模型、记忆或轮数。** 已有的器械局部收益不足以达到预注册的IVT增益要求。优先完成冻结H0的正式实验和复现材料；若后续继续研究Verifier，应把当前原始误判与真实图像／类别定义逐项核对，明确区分视觉证据不足和类别理解偏差，再决定新的单变量实验。

## 费用与产物

累计74次POST：8次共享H0、32次无先验分支、24次先验分支、10次兼容性冒烟。正式两臂共64次请求，包含上述3个失败审核请求。没有Batch调用。

| 类别 | 已结算美元 |
| --- | ---: |
| 共享H0 | 0.09959400 |
| 无先验审核／修复 | 0.30416278 |
| 先验审核／修复 | 0.28433038 |
| 兼容性冒烟 | 0.15830270 |
| **合计** | **0.84638986** |

4次失败未返回逐笔成本，继续保留共2.54679680美元的保守预留；已结算加未结算预留共3.39318666美元，低于用户15美元上限。账户累计扣费增量与0.84638986美元已结算总额完全一致，未观察到额外扣费；账户对账不能替代缺失的逐笔账单，所以没有把预留强行清零。平均单调用耗时：H0约19.55秒、无先验分支23.96秒、先验分支26.79秒；审核请求并行，不能把这几个均值直接相加当作逐目标延迟。

- [最终预测及完整逐轮记录](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/predictions.json)
- [各头F1／Accuracy与修改统计](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/metrics.json)
- [独立审计](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/independent_audit.json)
- [正确候选的原始判断诊断](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/mechanism_diagnosis.json)
- [账户与源文件最终核对](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/final_account_reconciliation.json)
- [执行冻结清单及历史账本引用](../artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/plan.json)
