# 先验门控 + 联合 Phase 主线：T1 / T2 / T4 / T5

任务依据：`docs/HANDOFF_PRIOR_GATED_JOINT_MAINLINE_2026-09-11.md`。
目标版本：`prior-gated-joint-mainline-v1.0.0`。冻结候选，不替换默认，不采 Gate 数据。

## 主线及 T1

单一入口：`scripts/run_prior_gated_joint_mainline.py`。

H0（Gemini 3.8 Flash）→ 原图谱提示与补候选 → 两条并行分支：

- 原精简四头五席审核 → 原均分选择 → 原先验门控，使用 H0 Phase。
- 不带先验关系提示的 Gemini Phase 推荐 → 五席联合审核四头与七阶段，只采用其 Phase 决策。

最终合并四头与 Phase；审核后 Phase 不回流先验。仅移除盲投 Phase 面板。
每目标 `1 H0 + 1 proposal + 5 compact + 1 Phase recommendation + 5 joint = 13` 次调用。
模型、路由、提示、0.01/0.70 门控阈值、Phase 阈值 4 不变。

已零调用重放 `prior_gated_joint_vid110_confirm_20260911_v1_resume1`：
16 个目标，208 份实际请求逐项与历史 redacted request 比较一致；从原始 HTTP 回答按传输层解析器复原。
H0、gated_control、gated_control_jointphase 三臂全部匹配，四头与 gated_control 全部匹配。
字节比较指保持数组顺序的规范 JSON 序列化，不比较文件缩进。
证据：`artifacts/preflight/prior_gated_joint_mainline_replay_20260911_v1/replay.json`。

## T2a 单写入账本

`serialized_ledger.py` 提供单一队列写入线程。传输层持有 RLock 时复制完整账本快照，
交给唯一写入者；等待落盘成功后才允许请求派发。写线程不获取传输锁，避免互相等待。
结束时关闭并 join；写失败原样传播，没有本地写重试，也没有 API 重试。
旧实验入口和历史 frozen_source 保留；新主线使用此机制。

12 个线程、100 条更新及额外并发 persist：无记录丢失、余额一致，所有写入来自同一线程。
另测写异常只尝试一次并传播。包含原门控、Phase、默认 GLM 与确认相关回归共 238 项通过。
单一写入者解决同进程竞争；不能承诺外部进程持有 Windows 文件锁时仍不会失败。

## T2b 新增保险不采用

按交接的预定规则，先验加入须额外有精简审核均分 ≥3；缺失有效均分也不允许加入。
否决规则及其他选择不变。未改核心默认门控函数。

| 数据 | 原正确/错误先验加入 | 保险后正确/错误加入 | 平均 F1 原→新 | 总错漏原→新 |
|---|---:|---:|---:|---:|
| Training 136 去重目标 | 25 / 4 | 1 / 0 | 61.70 → 58.82 | 800 → 833 |
| 已归档 VID110 64 目标 | 9 / 4 | 1 / 2 | 64.72 → 62.62 | 325 → 336 |

两个集合都不满足“正确加入不减少、平均 F1 不降、错漏不增”；因此不采用。
表中统计只数 prior_added 日志，不把 H0/面板新增或重复目标混入。
证据：`artifacts/research/prior_gated_add_guard_20260911_v1.json`。
136 个 Training 与 64 个 Validation 目标均是已使用开发数据，不冒称独立验证。

## T4 冻结的前瞻 Testing 子集协议

选择 VID25、VID92、VID111，每视频 8 个时间分位目标，共 24 个；排除曾用于窗口开发的 VID06。
只按时间、完整因果窗口和五头标注可用性选样，目标图像间隔 >175 原始帧。
准备时访问 Testing 标注只提取有效性 mask；推理仅使用 InferenceSample 和图像；
全部预测和 HTTP 回答封存后，score 才关联 Testing 标签。先验单独由 Training 标注生成。
用户导入的 Grok/Qwen Testing 预测和指标不用于选样、改参数或模型选择。

协议先写 protocol.json，然后生成 selection 和 plan.json；无付费请求前冻结所有运行源码、
模型路由、阈值、图像及先验哈希。主线每目标最多 13 次，最多 312 次；本地额度为
OpenRouter $5、阿里云 ¥3。额度是保守占用上限，不是预期实扣。
不重试、不换模型、不替换或删除失败目标。H0 无有效结果时，不调用其依赖修复；
所有计划目标仍保留，无预测按空集合计分并单列失败数。失败计分约定不能当作有效输出。

主检验：完整目标均有预测，主线平均 F1 严格高于 H0、错漏严格低于 H0，IVT F1 不低于 H0，
不改坏任何 H0 正确的 Phase；相对 gated_control 四头一致、平均 F1 不降、错漏不增。
Phase 增量另报“至少改对 1 且改坏 0”，持平不称新增收益。
无论结果如何，不在此子集上继续调参，T5 只冻结候选，不根据分数挑另一版。

正式目录：`artifacts/preflight/prior_gated_joint_mainline_testing_20260911_v2`。
`v1` 仅做零调用准备和预检；其清单曾错误展示传输注册表中的历史 model 名称，
实际请求模型未变。v2 修正为 wire roster 的模型名，沿用相同选择规则，重新冻结，v1 未付费执行。

## T5 版本清单与局限

最终根目录 `BEST_PIPELINE_VERSION.json` 指向正式目录的 `release_manifest.json`。
清单含运行源码 SHA-256、提示版本、模型与路由、先验、阈值、证据哈希、默认文件原哈希。
旧 BEST 清单保存在正式目录 previous_best_manifest.json。DEFAULT_PIPELINE_VERSION.json 不改。

- 先验收益主要是 Phase 3 标注惯例对齐，不是新视觉证据。
- 联合 Phase 与 H0 存在同类定位错误，触发少，收益不稳定；历史合计改对 2、改坏 0。
  前两批采用旧的带先验提示 Phase 推荐，不能把这个合计当作精确同一新提示的三次确认。
- 旧 VID110 数据只来自同一视频，主要覆盖 Phase 1/2/3；Phase 4/6 可能回退全局。
- 已归档真值 Phase 与池外关系诊断也仅令新16帧四头错漏 74→73；该诊断不进入推理。
- 新 Testing 确认也是 24 目标子集，不是完整 Testing，也不构成统计显著性证明。
- 未生成 Gate 特征训练集、收益训练标签或 Gate 权重。

## T4 实际结果与费用

24 个计划目标全部保留：23 个有预测，1 个 H0 失败。共派发 300 次请求，
其中 297 次 JSON_PARSED、1 次围栏规范化后解析成功、1 次 SAFE_JSON_REJECTED、1 次 FAILED。
没有重试、没有替换目标，也没有发生账本 PermissionError。

| 臂 | IVT F1 | Phase F1 | 五头平均 F1 | 总错漏 |
|---|---:|---:|---:|---:|
| H0 | 32.00 | 55.32 | 56.96 | 143 |
| 原先验门控，保留 H0 Phase | 40.54 | 55.32 | 60.84 | 131 |
| 冻结主线：门控 + 联合 Phase | 40.54 | 55.32 | 60.84 | 131 |

以上按全部 24 个目标汇总 TP/FP/FN，失败目标预测为空集合；Phase F1 不是把失败目标剔除后的准确率。
联合 Phase 改对 0、改坏 0、错到错 0；主线与 gated_control 的预测完全一致。
相对 H0 平均 F1 增加 3.88 个百分点、IVT F1 增加 8.54 个百分点、总错漏减少 12。

但各视频并不一致：VID25 平均 F1 57.62→56.21、错漏 46→46；
VID92 为 46.29→62.00、错漏 53→41；VID111 为 65.03→64.29、错漏 44→44。
收益集中在 VID92，不能据此宣称在所有视频上改善，或已证明是全局最优版本。

预注册主检验 **未通过**：虽然汇总质量条件满足，但“全部目标都有预测”不满足。
联合 Phase 增量检验也未通过，因为没有改对目标。保持原冻结方案，不按 Testing 结果再调规则。

失败明细：

- `VID92_40601` 的 H0 请求记录为 `ProxyError`，15.67 秒后失败，没有服务端 response.json。
  仅凭该记录无法定位代理或上游的具体故障；跳过其依赖的后续 12 次调用，目标仍计分。
- `VID92_9401` 的 joint_r1 / gemini 审核席回答未通过安全 JSON 解析，状态为 SAFE_JSON_REJECTED；
  其余可用审核按原选择器规则处理，没有补调该席。

执行阶段墙钟耗时 **949.47 秒（15.82 分钟）**；目标之间顺序处理，目标内两个分支与五席审核并行。
不包含前面的离线预检、图像准备和后续评分时间。

| 账户与计费依据 | 金额 |
|---|---:|
| OpenRouter 接口返回 native cost 合计 | $0.64636862 |
| 阿里云按冻结费率和实际 token 的保守估算 | ¥0.739376 |
| H0 ProxyError 无法确认的预留占用 | $0.1710410 |

账本占用上限合计为 $0.81740962 + ¥0.739376；预留占用不能称作实际扣费，
阿里云金额也不是账单实扣核对结果。

评分前已从原始 HTTP 回答重建全部有效目标的预测；另用独立集合运算核对全部五头 TP/FP/FN。
确认 300 个 target/stage/seat 身份唯一、无盲投面板、请求模型全部匹配冻结名单、
539 份运行源码与封存证据 SHA-256 一致、DEFAULT 文件哈希不变。
独立审计：正式目录 `postrun_audit.json`；完整计分：`metrics.json`。

## 交付与复核入口

版本 `prior-gated-joint-mainline-v1.0.0` 冻结为 `frozen_candidate_not_default`；
根目录 `BEST_PIPELINE_VERSION.json` 保存同一清单，但文件名不代表本轮通过独立确认。
T1、T2a、T2b 的验证工作与 T4 实验完成；T2b 不采用，T4 通过标志为 false，T5 只冻结候选。

```powershell
.venv-p2/Scripts/python.exe -X utf8 scripts/run_prior_gated_joint_mainline.py --help
```

不要重新对正式目录运行 execute；它是已封存的单次实验目录。
本轮没有生成 Gate 训练数据、训练标签或权重，DEFAULT 未替换，代码未提交 git。
