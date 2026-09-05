# Astra 六问题真实冒烟、修复尝试与失败边界

日期：2026-09-05。本文使用原始六问题编号：①H0/候选召回；②Verifier 视觉判断；③Repair 接纳依据；④hard-valid H0 绝对保护；⑤单 scope；⑥Phase–IVT 联合检查。不要与包含 GT mask、任务级可靠性的另一份“四问题清单”混用。

## 直白结论

已实际调用 OpenRouter `openai/gpt-6-astra`。接口返回的 model 字段与请求一致；不是把 Sol 改个显示名称。Astra 并没有解决当前语义识别问题。

按 [OpenAI 官方 Astra 文档](https://developers.openai.com/api/docs/models/gpt-6-astra) 核对参数后，新增配置使用 `reasoning.effort=low`，未沿用其他模型的 `none` 设置。

新实现的 V9 实验覆盖了更多任务、放开了候选截断，也取消了若干不合理的接纳路径，但两次同模型盲评会共同确认错误。本轮实验没有净收益，且丢失了 V8 的三次器械纠正。**V9 未替换默认/历史 V8 配置，不建议推广或开始正式 Gate 训练。** 新 V9 配置已明确标为实验失败、未晋升。

## 真实运行与评分口径

- 初次配置预检发现 H0 白名单不允许 Astra，已修复；该次没有到达 API，无费用。
- 修复前运行：`artifacts/preflight/smoke_astra_six_issues_before_ready_20260905/`。
- 修复后实验：`artifacts/preflight/smoke_astra_six_issues_after_v9_20260905/`。
- Validation VID110，连续目标时间点 4301、4326、4351、4376、4401、4426，按 25 帧编号一步推进，每次使用目标与前两帧。不是六次重复预测同一帧。
- 4426 的 H0 请求被服务商 `403 content_moderation` 拒绝；其余五帧作为配对样本。拒绝没有当成空预测正确，也没有绕过审核重新提交。
- V9 复用了这五帧的 H0 缓存，只重新调用 Verifier。Tracker 状态按连续帧更新；RULE Gate 阈值为 0，用于能力冒烟，不是训练过的 Gate。
- GT 仅由离线评分脚本读取，不进入 H0、Verifier 或接纳逻辑。五帧均为 I={0}、V={9}、T={14}、IVT={94}、Phase=0，五头 mask 全部有效。这是单一 null-interaction 段落，不代表复杂活动、其他阶段或完整数据集。

以下为逐头**标签集合完全匹配的帧数**，分母均为 5，不是 mAP：

| 输出 | Instrument | Verb | Target | IVT | Phase | 五头同时正确 |
|---|---:|---:|---:|---:|---:|---:|
| 相同 H0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 原 V8 + Astra | 3 | 0 | 0 | 0 | 0 | 0 |
| V9 双盲复核实验 + Astra | 0 | 0 | 0 | 0 | 0 | 0 |

| 帧 | V8 Instrument | V9 Instrument | 两版 IVT | 两版 Phase |
|---|---|---|---|---|
| 4301 | {0} | {0,6} | {13} | 4 |
| 4326 | {0,6} | {0,6} | {13} | 4 |
| 4351 | {0,6} | {0,6} | {13} | 4 |
| 4376 | {0} | {0,6} | {12,13} | 4 |
| 4401 | {0} | {0,6} | {12} | 4 |

V8 五帧均进入 instrument_presence，并使用本地 Tracker 路径，没有触发付费视觉 Verifier。V9 的已完成付费版本逐帧访问三个 scope，实际新增 23 次 provider 请求：19 次返回、4 次内容审核拒绝；首次复核不确定时不会再浪费第二次复核。

V9 没有接纳任何标签改动。相对 H0 的 rescue=0、harm=0，**不能把这个 harm=0 当作安全性证据**：没有发生修复，而且 H0 的五头 exact 全错。相对 V8，器械 exact 明确少了 3 帧。双盲复核曾把错误器械背书 3 次、错误阶段背书 3 次，证明“同模型两次一致”不等于独立可靠证据。

费用：V8 新增已报告 USD 0.233054；V9 新增已报告 USD 0.732309；合计 **USD 0.965363**。五次拒绝请求未返回费用字段，不能擅自当成账单已确认的零费用。后续检查均为本地测试/缓存回放，没有新增 API 调用。

## 六问题逐项判定与本轮代码动作

| 问题 | 本轮精确判断 | 实现及边界 |
|---|---|---|
| ①候选召回不足 | 这五帧不是主要瓶颈：H0 IVT top-8 已全部包含94；原候选池已全部容纳五头 GT。H0 Phase top-3 只有1/5含0，但候选扩充后5/5含0。 | V9 候选池覆盖完整固定 ontology：7/10/15/100/7，消除 H0 阶段和20候选截断造成的不可达；不表示模型能选对。另修 Training 先验漏读实例标签，见下节。 |
| ②Verifier 看错/排错 | 仍未解决。V9 既有一致看错，也有不确定；不是单纯门槛过高。 | 新增两个侧重点不同、不见 H0/Tracker/彼此答案的视觉复核；交互要求显式逐器械关联与 IVT 投影闭包，禁止匹配失败自动回填 null。机制更严，但真实实验无收益，因此不推广。 |
| ③Repair 接纳不代表语义更优 | 仍是能力瓶颈。结构合法、两个答案一致都不能证明 H1 比 H0 好。 | V9 hard-valid 与 hard-invalid 分支都要求 scope 对应证据；全量后检；证书绑定最终具体标签，失败/被后续改掉的证书不能跨任务背书。双盲一致仅是未校准代理，本轮证据已经证明它不足以解决该问题。 |
| ④hard-valid H0 被完全保护 | 当前 V8 已不再是“完全保护”：4301/4376/4401 接纳了有 Tracker 时间证据的正确删除。 | V9 可凭符合 scope 的证据修改结构合法的 I/Interaction/Phase；单元测试覆盖。真实五帧没有成功语义修复，不能称为全面解决。 |
| ⑤一次只检查一个 scope | 原路径确实五帧均止于 Instrument。 | 新增有预算的 scope 覆盖调度，局部成功后继续 Interaction/Workflow，上限三个 scope；不是把 max_attempts 从1改3。没有成功检查/没有证书的任务不冒充已验证。遇内容审核终止该帧；复杂多 scope 必须同时改动才能解开的硬冲突仍可能 Pending，未实现联合搜索保证。 |
| ⑥Phase–IVT 检查未接通 | 原正式入口没有传入严格表，关系 scope 又误要求一个专家同时拥有两端。 | 加入 Training 共现软检查及最终审计；CLI 接入明确的 reviewed 硬规则表 `--strict-phase-ivt-map`；修复“只改 Phase 或只改 IVT”也可处理关系冲突的路由。没有提供临床审定的硬表，默认 hard 仍关闭，不能声称与母体论文严格联合验证等同。 |

局部成功后的任务证书只对最终相同标签生效；若其他 scope 不确定/接口失败，最终 Accepted 标记 lower_reliability，不掩盖局部失败。这个审计/可靠性收尾在付费实验结束后修复，由真实缓存回放验证；不会追改原付费产物。

## 新定位的根因：训练先验只看 frame_supervision

数据适配器有两条合法监督路径：普通实例标注放在 `evaluation.instances`；其 `frame_supervision` 便利对象只带 Phase。旧先验函数却只检查后者的 `mask.ivt`，因此误把“此便利对象不提供 IVT”当成“本帧没有可用 IVT”。

实际漏读情况：

| Training 视频 | 旧帧级路径使用数 | 新补入有效实例聚合数 |
|---|---:|---:|
| VID31 | 3732 | 0 |
| VID103 | 0 | 1545 |
| VID23 | 0 | 1168 |
| VID96 | 0 | 1119 |
| 其他 Training 视频 | 0 | 0（对应联合 GT mask 不满足） |

已新增统一 Training Phase–IVT 目标枚举：优先有效帧级标签，否则对实例标签按完整任务 mask 聚合；每帧只计一次，忽略 Validation/Test。可用样本从3732增至7564，观察到的 Phase–IVT pair 从46增至87。未擅自启用缺少 GT 的其他六个 Training 视频。

新增 pair 仍不含 (Phase4, IVT13)，因此这个组合继续记作“Training 未观察”，而不是自动判为不可能；不能据此把模型的 Phase 改成验证集答案。V9 使用修正后的 Training 路径，历史 V7/V8 CLI 显式保留旧路线用于重现。

## 实现位置与复核

- Astra 请求配置：`configs/perception/joint_openrouter_astra_fixed3.yaml`。
- 旧版对照：`configs/perception/targeted_openrouter_astra_v8_fixed3.yaml`。
- 未推广实验：`configs/perception/targeted_openrouter_astra_v9_fixed3.yaml`、`configs/ablations/b_tracker_astra_coverage_smoke.yaml`。
- 候选、复核、覆盖、接纳：`src/surgical_agent/research/verification/{hypotheses,targeted_api,verifier,coverage,repair}.py`。
- 先验监督修复：`src/surgical_agent/research/signals/phase_graph.py`。
- 关系路由/软检查：`src/surgical_agent/research/safety.py`。
- 最终证书与失败可靠性：`src/surgical_agent/research/outcome.py`。
- CLI/装配：`scripts/run_final_dataset_pipeline.py`、`src/surgical_agent/systems/final_pipeline_factory.py`。
- 离线配对审计：`scripts/audit_paired_smoke.py`，只读已完成结果与独立 GT，无 API。
- 真实响应本地回放：`tests/integration/test_astra_cached_scope_smoke.py`；缓存缺失即测试失败，不回源、不读密钥、不发网络请求，原403按失败回放。

最终验证：全仓库 **1016 passed，14 skipped**；修改文件 Ruff 检查通过。独立真实缓存回放为5帧、21次缓存命中、3次原拒绝记录回放、0次网络调用，通过。新版拒绝停止规则使三个受阻帧的后续 scope 不再继续；回放标签仍无改善，未将工程测试通过写成识别问题解决。

离线复算命令（无 API 费用）：

```powershell
.venv-p2\Scripts\python.exe scripts/audit_paired_smoke.py --before artifacts/preflight/smoke_astra_six_issues_before_ready_20260905 --after artifacts/preflight/smoke_astra_six_issues_after_v9_20260905 --cache artifacts/final_pipeline_cache/astra_six_issues_20260905 --dataset D:\cholec_dataset --summary-only
.venv-p2\Scripts\python.exe -m pytest tests/integration/test_astra_cached_scope_smoke.py -q -s
```

## 是否需要训练

候选覆盖、监督路径、闭包、scope 调度、关系接线、可靠性和拒绝处理都不需要训练。修补先验需要重新统计 Training 数据，不是重新训练 Tracker。

②与③仍需要真实能力验证，不能再用“多轮/一致/合法”代替语义改善。下一步应在 Training-only、包含正确与错误 H0 的多类型片段上验证可区分证据及有害修复；这不必先训练一个新模型。如果无法取得净收益，再考虑领域监督或专门的接纳模型，而不是先训练 Gate 掩盖 Verifier 失败。若改变后的修复政策未来被采纳，Learned Gate 必须重新采集相应 counterfactual 标签并重拟合；当前 V9 已显式阻止沿用旧 Learned Gate。
