# 连续视频窗口：论文设置核实与纯 API 对照

日期：2026-09-05。独立开发诊断；不替换默认 Pipeline，不改动冻结版本。

## 实验前冻结的方案

用户要求核实 StreamingVLM、Flash-VStream、M3-Agent、Vgent、TDC-Video、VideoARM 的数据规模、采样和窗口选择依据，并运行真实冒烟。

本地 Training / Validation 仅有 1 fps PNG，连续原视频可用来源是 Testing。固定 `Testing/VID06/vid06.mp4`，25 fps、68141 原帧、854×480。只根据标注索引确认目标存在，未根据标签内容筛选：`30001,30026,30051,30076,30101,30126`。这是连续六个 1 Hz 目标，首尾相隔 5 秒；不称为六个相邻原帧。索引预检与之后读取标签评分分开。该测试片段已用于开发诊断，不能作为未见测试的证据。

| 组别 | 目标 t 的实际图片 | 图片数 | 首尾时间跨度 | 目标步长 |
| --- | --- | ---: | ---: | ---: |
| A：三张相邻 | t-2,t-1,t | 3 | 0.08 秒 | 25 原帧 |
| B：三张覆盖一秒 | t-24,t-12,t | 3 | 0.96 秒 | 25 原帧 |
| C：完整连续窗口 | t-24 到 t，全部连续原帧 | 25 | 0.96 秒 | 25 原帧 |

A/B 对照检查时间跨度；B/C 对照检查相同跨度下的帧密度。所有组只预测最后一帧的五头标签，不输出整个窗口的标签并集。同一 Qwen 模型、最终标签 schema、全本体、目标 high / 历史 low 图片 detail；无 Tracker、Gate、Verifier、Repair、预测记忆或重试。各目标轮换组执行顺序，减少固定组顺序与服务时间混杂。

预算在执行前向用户说明：最多 18 次调用，规划 $1.50；下一次调用预留 $0.15，费用不明则停止。该预留是本地停止策略，不是提供商美元硬上限。全部预测落盘后再读取标签内容评分；缺失任务 GT 按 mask 排除，失败输出单独统计并保留有效 GT 分母。

本实验不直接比较步长 1 与 25 的识别精度：两者在相同目标 t、同一纯无状态窗口下请求相同，差别是额外推理哪些时间点。GT 只提供 1 fps，不能复制标签给其余原帧来制造密集准确率。

## 论文证据

详细核查笔记在 `artifacts/preflight/window_density_streaming_papers_20260905.md` 和 `artifacts/preflight/window_density_memory_papers_20260905.md`。区分原视频帧数、实际采样帧数、时间窗口、记忆容量和回答频率；论文未报告的值不能用猜测补齐。

CholecTrack20 官方说明：20 视频，约 35K 标注帧、65K 工具实例，标注 1 fps，原视频 25 fps；训练/验证/测试划分 10/2/8。来源：[官方仓库](https://github.com/CAMMA-public/cholectrack20)。本地审计记录的精确 canonical 标注帧数是 35009，不是原视频总帧数。

| 方法 | 数据规模口径 | 实际窗口/采样与步长 | 参数选择证据 |
| --- | --- | --- | --- |
| StreamingVLM | 原视频规范 24 fps；2449 场比赛，清洗后训练语料超过 4000 小时；525K 流式样本；Inf-Streams-Eval 为 20 场、平均 2.12 小时。未报告精确原始总帧数 | 训练 24 秒窗、12 秒重叠，起点步长 12 秒；推理视觉 KV 保留 16 秒。代码默认 2 fps，约 32 张历史采样图，每次新增 1 秒、约 2 张图；这不证明论文全部运行均为 2 fps | 训练重叠用于匹配流式推理注意模式；视觉窗 0/1/4/8/16/32 秒有消融，16 秒兼顾近期动作与效率，非所有指标严格最优。[论文](https://arxiv.org/html/2510.09608v1)、[固定版本代码](https://github.com/mit-han-lab/streaming-vlm/blob/a940d8cbfa14eefe651b7f4adeb81825c28cc6c6/streaming_vlm/inference/inference.py) |
| Flash-VStream | LLaVA-Video 9K 训练子集；公开 Video-MME 清单 900 视频/2700 QA，EgoSchema 清单 500/500；未报告统一原始总帧数 | 正文 1 fps 持续编码、流末尾提问；60 CSM + 30 DAM 是压缩特征记忆。当前官方默认评测脚本全视频至多 240 帧，超过后再次均匀抽样，不是最近 240 帧滑窗 | 通过聚类/细节记忆减少冗余；约 12K 视觉 token 以内满足其单 A100 首 token 延迟目标；CSM/DAM 预算分配有消融，1 fps 对手术动作没有最优性证明。[论文](https://arxiv.org/html/2506.23825v1)、[固定版本评测脚本](https://github.com/IVGSZ/Flash-VStream/blob/8f6bde2f397f4846505df4d03364d97617fc02ee/Flash-VStream-Qwen/scripts/eval.sh) |
| TDC-Video | 图文 3.2M 训练样本；7B 视频训练阶段 2M，音视频阶段 300K；MVBench/EgoSchema/Video-MME 平均时长约 16/180/1010 秒。上述是样本/时长而非原始帧总数 | 正文 1 fps、最多 24 语义片段；代码在各片段内每 8 张采样帧不重叠压缩。长输入存在 224 帧上限并可能再采样，故每 8 张不恒等于 8 秒，也不是预测步长 | 保留静态细节、压缩动态变化以控制 token；24/48 片段和 16/32 动态 token 有消融；8 帧块未找到独立选优证据。[论文](https://arxiv.org/html/2504.10443v1)、[固定版本分块代码](https://github.com/Hoar012/TDC-Video/blob/fb8333eca13aee442cfc441480e38d1bf3ab4999/tdc/cambrian_arch.py#L1603) |
| M3-Agent | 论文 v1：robot 100 视频，平均约 34 分钟；web 929 视频，平均约 27 分钟；当前 README web=920，存在版本差异 | 官方示例 30 秒非重叠分段。预处理默认 5 fps，约 150 帧/完整段；主记忆模型另读视频，最终视觉输入采样率未显式给出 | 分段形成情景/语义记忆；未找到证明 30 秒最优的消融。[论文](https://arxiv.org/html/2508.09736v1)、[官方配置](https://github.com/ByteDance-Seed/m3-agent/blob/master/configs/processing_config.json) |
| Vgent | Video-MME、MLVU、LongVideoBench 长视频问答；未报告统一原始总帧数 | 正文 1 fps、每片段 64 张采样帧，约 64 秒；代码非重叠分块。离线建图后按问题回答 | 减少上下文压力，跨段保留实体联系；未找到 64 帧/1 fps 本身的消融。[论文](https://arxiv.org/html/2510.14032v1)、[分块代码](https://github.com/xiaoqian-shen/Vgent/blob/main/utils/vgent.py) |
| VideoARM | Video-MME 900 视频/2700 QA；LongVideoBench 3763 视频/6678 QA，使用验证 1337 QA；EgoSchema 500 QA 子集 | 定位器按区间取 30–150 帧；局部检查代码默认最多 50 帧。按问题选择区间，没有统一滑窗步长 | Table 6：自适应平均 49.8 帧，Video-MME-long 76.5，对比固定 60 帧 74.0；LVB 均 70.5。作者依据是按区间复杂度分配采样。[论文](https://arxiv.org/html/2512.12360v2)、[代码默认](https://github.com/MILVLG/videoarm/blob/main/README.md) |

论文数据集的原视频总帧数经常没有报告，不能将平均时长乘假设 fps 得出的估算当成实测。M3 的 0.5 fps 是其 GPT-4o / Qwen-VL 对照设置；VideoARM 成本讨论中的 2 fps、10 秒段属于 DVD 对照，均不能移植为主方法配置。

## 执行状态

18 请求和 150 张原图已冻结；plan SHA-256：`87ea3f890fe182064cdef2b16a4eb4345c8f4bfdea101de80b9aac5043c8b7be`。离线审计实际 HTTP body 共 186 处图片引用，所有图片 SHA/顺序/数量与请求一致；独立 seek 检查 29977、30001、30126 像素与实际上传 PNG 一致。三组 system prompt 相同；专属测试 5 项 + 既有 pure-H0 测试 8 项，共 13 passed。正式 transport 仍拒绝超过 6 张图，仅独立实验子类允许本实验的 25 图合同。

真实调用完成：18/18 成功，缓存命中 0，未计价调用 0，合计 **$0.347988**。HTTP 原始响应均为 200，SSE 返回模型均为 `qwen/qwen3.8-max-0902`、提供商 Alibaba。全部 186 处实际 wire 图片 data URL 哈希与冻结 PNG 一致；预测文件在读取 GT 后未改变。

## 离线评分接口问题与纠正

最初的 `summary.json` 五头有效 GT 数均为 0，原因不是缺失原始 GT，而是 `CholecTrack20DatasetAdapter._iter_test` 刻意返回 `evaluation=None`、`frame_supervision=None`，防止推理读取答案。旧 `evaluate_offline(adapter, ...)` 不能直接从该 Testing 推理接口获取评分标签。

所有 18 份预测完成并落盘后，使用独立 `scripts/rescore_window_density_native_gt.py` 读取原始标注、按现有 canonical parser 和 task-wise aggregate mask 评分，结果输出到 **`artifacts/preflight/window_density_h0_qwen0902_vid06_20260905_scored/summary.json`**。原始 `plan.json`、`predictions.json`、`summary.json`、调用记录不覆盖。最初全零 mask 的汇总仅保留为接口问题的证据，不作为准确率结论。另有直接读取原生字段并独立手算 TP/FP/FN 的审计，结果完全一致。

## 真实结果

六个目标的全部五头 GT 均有效。每帧有三个工具实例（两个 grasper、一个 hook），集合评测去重后的 GT 均为：I=`[0,2]`、V=`[1,2]`、T=`[0,2]`、IVT=`[17,58]`、P=`[1]`。IVT17=`grasper–retract–gallbladder`；IVT58=`hook–dissect–cystic_duct`。这是按官方数值标注评分，不据此判定任何临床操作。

| 任务 | A：3 张相邻 F1 | B：3 张覆盖一秒 F1 | C：25 张连续 F1 | 整组正确 A / B / C |
| --- | ---: | ---: | ---: | --- |
| Instrument | 100% | 100% | 100% | 6/6、6/6、6/6 |
| Verb | 91.67% | 91.67% | 96% | 5/6、5/6、5/6 |
| Target | 50% | 50% | 56% | 0/6、0/6、0/6 |
| IVT | 50% | 50% | 48% | 0/6、0/6、0/6 |
| Phase | 100% | 100% | 100% | 6/6、6/6、6/6 |

I/V/T/IVT 为集合 micro-F1，Phase 为单标签 micro-F1（此处等于 accuracy）。不是 mAP。此处没有 null IVT，故纳入全部 100 类的 micro-F1 与只看非 null 类一致。整组正确要求集合完全一致，0/6 不表示没有部分命中。

| 组别 | 6 次调用费用 | 平均 API 延迟 | 延迟范围 |
| --- | ---: | ---: | ---: |
| A | $0.077394 | 26.40 秒 | 22.97–29.90 秒 |
| B | $0.076512 | 24.82 秒 | 22.90–28.60 秒 |
| C | $0.194082 | 32.09 秒 | 26.47–38.55 秒 |

完整 25 图成本约为 B 的 **2.54 倍**；API 延迟不含本地视频解码、PNG 编码和准备请求时间，不能称为每秒实时返回。

### 可定位的错误

- 三组所有目标都命中 IVT17，但**均未命中 hook 的正确 IVT58**。主要靶点输出反复为 `cystic_plate`（1），而 GT 是 `cystic_duct`（2）。完整 25 图没有解决这一错误。
- 30001：A/B 动作 `[retract,coagulate]`，C 为 `[retract,dissect]`，动作与 GT 相符；但 C 的 hook IVT59 仍为错误靶点 `cystic_plate`。
- 30101：C 的 Target 增加 `cystic_duct`，得到一个额外 TP，同时仍保留错误 `cystic_plate`；Verb 增加不属于末帧 GT 的 `coagulate`，IVT 增加错误 48=`hook–coagulate–cystic_duct`。因此 Target/Verb 的总体部分命中上升，不等于完整 IVT 改善。
- 同一预测中的头间数字映射也有不一致（例如输出 dissect，但 IVT 选择 coagulate 类；或 IVT60 的 gallbladder 与单独 target 的 cystic_plate 不一致）。这可以另行审计，但结构投影不会凭空提供正确的视觉靶点，不作为本轮窗口收益。

## 建议与证据边界

1. **当前纯无状态 API 用目标步长 25，即按 GT 的 1 Hz 输出。** 步长 1 在共同目标上不会增加上下文，只额外输出中间时间点；若以后加入状态更新，应单独比较，不把本次结果外推到有状态模型。
2. **完整 25 图不设为默认。** 本片段 Verb/Target 部分命中小幅提高，IVT F1 反而下降，费用明显增加。三帧输入保持基线；B 可以作为约一秒覆盖的低成本候选，但本次不能证明其优于 A 或旧的 `[t-50,t-25,t]` 两秒基线（旧基线未在这批目标上运行）。
3. 后续先在 Training / Validation 固定样本审计靶点和动作定义，再决定是否需要更长时间范围、局部空间细节或监督训练。密集窗口对照需补齐这两个 split 的原视频；现有稀疏 PNG 可做三帧较长跨度与空间靶点审计。不要反复用这个 Testing 片段挑参数。
4. 本次只有一个视频、连续六秒级目标、每组每目标一次采样，没有重复试验控制模型随机性；六帧标签相同且高度相关。不能证明任何窗口全局最优，也不能据此判定必须训练。若验证集持续存在同类错误，再考虑训练基于工具接触区域和全图上下文的 Target / Verb / IVT 识别，动机是弥补领域视觉区分能力，非让 JSON 更合法。

没有加入模型 Agent、异构模型、跨窗记忆或 Repair，没有额外付费重跑。所有先前 445 个代码/配置/报告文件哈希未变；冻结 V7 清单 SHA 仍为 `f63a3b301a8faf79f245a8517ded7bf84594bd14901f2ee4403dccd810cf9fdd`。

最终验证：新增原生离线评分测试后，相关测试共 **15 passed**；独立原生字段手算与 canonical parser / aggregate evaluator 的全部 TP/FP/FN、整组正确数和 F1 一致。原推理文件 SHA 与独立评分脚本 SHA 均匹配，新产物凭证泄漏检查通过。最终核验记录：`artifacts/preflight/window_density_final_validation_20260905.json`。本轮未重跑全仓库测试；前次的 1042 passed / 14 skipped 不冒充本轮验证。

## 产物

- 论文原文与固定版本代码核查：上述两份 `window_density_*_papers_20260905.md`。
- 原始实验：`artifacts/preflight/window_density_h0_qwen0902_vid06_20260905/`，包含 plan、requests、images、calls、原预测及推理适配器全零 mask 汇总。
- 正确原生 GT 评分：`artifacts/preflight/window_density_h0_qwen0902_vid06_20260905_scored/`。
- 独立 wire / 返回模型 / 手算指标核验：`artifacts/preflight/window_density_completion_audit_20260905.json`。
- 独立手算计数：`artifacts/preflight/window_density_manual_metrics_20260905.json`。
- 新入口：`scripts/run_window_density_h0_smoke.py`；独立后评分：`scripts/rescore_window_density_native_gt.py`。后者是本次 Testing 诊断所需步骤，原冻结入口保留以便复现调用。
