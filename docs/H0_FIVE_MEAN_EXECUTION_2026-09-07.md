# 五模型平均评分与三轮修复：开发小测试

状态：实现和离线检查完成，真实接口测试进行中。本文末尾以实际账本和评分补充结果，不将已实现的三轮上限当作已执行三轮。

## 核实后的问题

旧版 [prior_panel.py](../src/surgical_agent/research/verification/prior_panel.py) 的 `build_universe` 只包含 H0 标签、7类器械和少量先验候选；`issues_for` 只能发现池内问题。`panel_votes` 要求3人回答 PRESENT/ABSENT/UNCLEAR，`admit` 用新增2票、删除3票，完全没有1–5评分或取平均。`run_arm` 在无新问题、重复修订稿等条件下停止，之前真实实验没有R3。

旧8目标实验还有视觉层面的失败：有完整初审的5个真实新增IVT没有任何原始PRESENT，其中3个被三人一致判为ABSENT。另有时序引用规范化、输出截断、服务商拦截。候选覆盖、视觉判断、接纳规则和接口失败是不同问题，不能用JSON合法率代替语义效果。

## 本版如何工作

入口 [run_mean_panel_trial.py](../scripts/run_mean_panel_trial.py)，纯逻辑 [mean_panel.py](../src/surgical_agent/research/verification/mean_panel.py)。默认纯API入口和冻结H0保持原样。

1. 复用原8个Training目标的真实H0，不再调用初始预测。它们已被分析过，因此这是开发重放，不是独立验证集结果。
2. 五个轻量视觉模型各自看同一因果三帧，预测全部132个标签的存在程度：I7、V10、T15、IVT100。每项1–5整数；1明确不存在，2倾向不存在，3不确定，4倾向存在，5明确存在。每个模型评分4/5的标签即该模型的正类预测。Phase保持H0。
3. Python对每个标签取五人算术平均。均分≥4支持存在，≤2支持不存在，中间不产生修改。不是整帧总分，不是置信概率，也没有将三态投票伪装为数值平均。
4. 均分与当前已接纳预测冲突时，Qwen根据具体问题和图像提出最多16项局部补丁；没有独立候选H1/定位器调用。所有新增组件需要自己的问题项；删除IVT不自动删组件。
5. 修订稿必须再由五人完整审核。接纳只依据本轮均分，重新审核相对原H0的所有累计修改。新IVT的缺失组件也需支持；保留IVT需要的组件不能被删除。未确认修改回到H0，不用旧轮的高分兜底。
6. 最多3轮五模型审核、2次Qwen修复。初轮不向五人展示H0；后轮展示修订稿和该成员自己的前次回答，不展示其他成员分数。无可执行问题、空/非法补丁、失败审核或预算不足会提前停止；重复问题本身不再强制早停。中间分数代表未解决，不代表审核通过。

缺一位成员就不计算五人均分，保留最后完整审核过的状态。错误响应不自动重试，不增加第四轮。各轮前缀是同一次执行的相关结果，不能用GT选最好轮次。

## 医疗学术场景说明与接口检查

按用户要求，五位成员的系统提示和输入JSON第一项 `academic_context` 均说明：输入是用于学术医疗视频分析的腹腔镜手术帧，研究任务是标注器械、动作与交互组织；组织、血液和器械属于所研究的医疗过程。Repair也带相同说明。原H0提示不变。

这是真实场景说明，不代表服务商保证放行。v1中Flash-Lite返回完整132项评分，GPT-4.1 mini仍返回403 `violence/graphic`，响应明确注明未扣费。五人冒烟未通过，正式实验没有启动。原始响应与冻结源代码保存在 `artifacts/preflight/h0_five_mean_openrouter_20260907_v1/`。

正式预测前，v2将两个OpenAI评分席位换为Qwen3-VL8B和Gemma3-12B；Flash-Lite和Qwen-VL通过，但Claude Haiku也返回403 `violence/graphic`。v3将Claude席位替换为Mistral Small3.2-24B，固定图片、医疗说明、评分与循环规则，此后本任务不再继续换模型试探。最终五个候选成员：Gemini2.5 Flash-Lite、Qwen3-VL8B Instruct、Mistral Small3.2-24B、Gemini2.5 Flash、Gemma3-12B。Qwen3.8-max-0902仍只负责修复。模型端点兼容性通过OpenRouter公开端点元数据检查；元数据支持不等于真实冒烟通过。Qwen-VL不支持reasoning参数，因此不向它发送此参数；Qwen修复保留原low设置。所有旧版源代码与请求都在各自输出目录中冻结。

## 实验与费用边界

原8目标顺序不变：VID103/18426、33676；VID23/13076、27726；VID31/40801、73601；VID96/14876、25876。I/P8个有效GT，V/T/IVT7个；不因缺GT或接口失败换样。Testing不参与。

本次五模型方案总调用上限142（包含兼容性尝试）。每个目标最多15次评分+2次修复，兼容性冒烟6次；失败尝试也扣调用数。因此上限并不保证全部目标跑满三轮。原15美元累计预算继续扣除先前三模型实验费用与未知成本预留；v1新增两次尝试后，v2保守可用11.16002414美元、剩余140次请求；v2又尝试3次后，v3保守可用10.676120026美元、剩余137次。未知成本继续按完整预留计入，即使文字响应声称未收费，也不会悄悄当作费用凭证齐全。

这是同时更换评分方式、成员、候选范围和反馈机制的可行性测试，不能归因成仅增加两个模型的效果。暂不混入先验频率、Tracker、RAG或额外记忆。全本体覆盖解决候选池的程序限制，不保证模型能看对；自我复核重复使用相同图片，没有增加外部证据。

## 复现与验证

使用现有 `requirements-prior-panel.txt`。先prepare冻结代码、输入和价格；smoke通过后才能run；保存预测完成哈希后才能score。

```powershell
.venv-p2/Scripts/python.exe -m pytest tests/unit/test_mean_panel.py tests/unit/test_prior_panel.py tests/integration/test_prior_panel_budget.py -q
.venv-p2/Scripts/python.exe scripts/run_mean_panel_trial.py prepare --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe scripts/run_mean_panel_trial.py smoke --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe scripts/run_mean_panel_trial.py run --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe scripts/run_mean_panel_trial.py score --output artifacts/preflight/NEW_UNIQUE_RUN
.venv-p2/Scripts/python.exe tools/audit/audit_mean_panel.py artifacts/preflight/NEW_UNIQUE_RUN
```

新机制11项测试通过；与冻结H0、final-only、独立评分和原panel账本相关回归共67项通过。测试覆盖恰好五人、平均分边界、评分类型/长度、完整本体、共享组件、跨轮重新判断、第三轮失败回退，以及五模型实际请求的医疗说明位置。没有提交或推送Git。
