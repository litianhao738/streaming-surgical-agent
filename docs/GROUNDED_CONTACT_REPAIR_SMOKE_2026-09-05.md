# 接触区域证据驱动 Repair：实现与六帧冒烟测试

日期：2026-09-05。

## 结论

实验链路已经实现并通过真实 API 冒烟测试，但尚未解决核心视觉语义错误，不应替换默认 Pipeline。6 帧全部取得合法五头 H0；2 次接纳中，1 次减少 Verb 标签错误，1 次按当前指标无改善；Target、IVT 的整组正确率仍为 0/6。不能将“结构闭包改善”解释为“识别正确”。

## 本次实现范围

保留联合预测，不使用五个独立专家投票。独立实验链路为：

```text
三帧因果图像 → 联合五头 H0
同组三帧 → 不看 H0 的器械尖端/接触区域定位
原始三帧 + 目标帧区域裁剪 → 实例级 IVT 提案
实例级 IVT → 本地推导 Instrument / Verb / Target，Phase 保留 H0
若 H1 不同于 H0 → 隐藏来源的双假设视觉复核
有区分性证据才接纳；无有效证据则 KEEP H0
```

- 定位和提案不接收 H0、上游分数或 GT；提案可选择完整 100 类 IVT，不受 H0 top-k 限制。
- 裁剪来自原始目标图像像素，不用 GT 框，不用生成式图像增强。记录原图/裁剪哈希与像素坐标。
- 复核看到两个假设与视觉证据，但不看到“原答案/修复答案”标记、置信度或提案理由；假设位置按预定帧索引交替。
- 接纳要求实例覆盖、区域相关、器械/靶点/动作证据均满足，并明确偏好提案；不再一概保护结构合法的 H0。
- Phase 本轮不修复。最多定位 3 个器械实例；覆盖不足、提案不足或复核失败时保留 H0。
- 接纳证据仍由同一个模型判断，不能视为独立客观证据，也没有语义改善保证。
- 新实验请求显式传递图像 detail（历史 low、目标 high），纳入请求缓存标识；默认旧请求保持原行为，未批量改变缓存语义。
- 在 HTTP 响应解析之前保存脱敏响应与用量，方便定位非法 JSON/空正文等失败；不保存 API 授权头。
- H0 增加完整 schema 提醒，格式失败允许一次格式重试，不进行语义 best-of 选择。本次没有触发重试。

代码：

- `src/surgical_agent/research/verification/grounded_repair.py`
- `scripts/run_grounded_contact_smoke.py`
- `src/surgical_agent/api/providers/openrouter.py`（实验请求的显式 detail）
- `src/surgical_agent/api/schema.py`（新增实验响应契约）
- `tests/unit/test_grounded_contact_repair.py`

本次未将新实验模块接入默认 `final_pipeline`；原有其他修改和冻结版本均保留。

## 实验协议

- 模型：OpenRouter `qwen/qwen3.8-max-0902`，沿用之前单模型实验。
- Validation / VID110，目标帧：8601、8626、8651、8676、8701、8726。
- 每个目标使用 `[t-50, t-25, t]` 三帧；目标步长 25。这里的“连续六帧”是六个连续采样时间点，不是六个相邻原视频帧。
- 三路并发，窗口之间没有 Tracker、Gate 或跨窗口记忆。因此这是因果窗口级修复冒烟，不是带连续状态的完整 streaming 测试。
- 推理前冻结样本与方案；预测完成后读取 GT 离线评分。缺失 GT 按任务 mask 排除；本次六帧五项 GT 均有效。
- 20 次真实调用：6 H0 + 6 定位 + 6 提案 + 2 复核；缓存命中 0；响应均通过校验。
- API 返回费用合计 **USD 0.243062**，无未计价调用。

本次 H0 的 prompt/schema 提醒和图像传输与旧 A/B/C 不完全相同，且样本不同。修复收益只与本次同输入 H0 配对比较，不与旧报告直接归因比较。

## 结果

整组正确要求一个任务的预测标签集合与 GT 完全相等；micro-F1 统计部分标签命中。

| 任务 | H0 整组正确 | 修复后整组正确 | H0 micro-F1 | 修复后 micro-F1 |
| --- | --- | --- | --- | --- |
| Instrument | 5/6 | 5/6 | 0.9565 | 0.9565 |
| Verb | 0/6 | 0/6 | 0.3478 | 0.4348 |
| Target | 0/6 | 0/6 | 0.5217 | 0.5217 |
| IVT | 0/6 | 0/6 | 0.0000 | 0.0000 |
| Phase | 6/6 | 6/6 | 1.0000 | 1.0000 |

接纳详情：

- **8626**：IVT `[17,51] → [17,59]`，Verb `[1,3] → [1,2]`，GT Verb 为 `[0,2]`。Verb 集合对称差从 4 降至 2；仍未整组正确。Target 从 `[0,4]` 改为 `[0,1]`，GT 为 `[0,11]`，没有改善。IVT GT 为 `[7,62]`，修复前后均未命中。
- **8726**：IVT `[17,60] → [17,59]`，GT 为 `[7,62]`，没有改善。其余四头不变。
- 另外 4 帧提案等于 H0，直接 KEEP，没有调用对比复核。这是节约调用，不代表已经验证正确。
- “IVT 分量包含于组件预测”的闭包原本就为 6/6；更严格的“IVT 分量投影恰好等于组件集合”从 4/6 到 6/6。后者不是视觉正确性的证据，也不应无条件推广到所有标注情形。
- 按每项预测集合与 GT 的对称差，本次未观察到有害字段变化；仅 2 次接纳，不足以估计一般有害修复率。

## 当前根因与仍未解决的边界

1. **不是只因正确答案被候选池挡住。** 本轮提案能访问完整 IVT 本体，仍未预测出正确 IVT。候选可达性已放开，视觉选择仍失败。
2. **错误集中在动作和接触靶点。** 六帧 GT 均包含 `grasp`，H0/最终均将该器械预测为 `retract`；后五帧 GT 包含 `peritoneum`，预测则偏向其他邻近解剖类别。仅增加局部区域和 API 调用没有消除这些偏差。
3. **模型生成的定位/证据也会错。** 8601 GT 仅有 grasper，定位与提案仍给出两个实例，H0 也预测额外 hook。需进一步检查原图、标注边界和定位质量，不能把模型声称的“覆盖完整”当真值。
4. **新接纳机制仍能接受语义错误的提案。** 8726 的替换有模型支持，却没有 GT 收益；这说明去除 hard-valid 保护后仍需校准真实净收益，不能把布尔证据字段等同于可靠判断。
5. **本轮未验证 Phase 修复、Gate、Tracker 或全视频状态。** Phase 本来就是 6/6 并保持不变，不能归功于新 Repair。

本次无需训练即可运行；尚无证据表明必须新增一个训练模型。若要继续，应先在 Training-only 样本检验器械接触区域质量、grasp/retract 的标注语义和靶点混淆，必要时使用训练集限定的视觉示例/局部识别器。是否训练以该能力试验为依据，而不是先训练 Gate 或增加循环轮数。不要用这六帧验证 GT 写特判。

## 验证与复现

- 新增测试与 OpenRouter 单元测试：53 passed。
- 全量回归：1042 passed，14 skipped。
- 真实结果：`artifacts/preflight/grounded_contact_qwen0902_20260905/summary.json`
- 完整预测：同目录 `predictions.json`。
- 冻结方案：同目录 `plan.json`；完成状态：`run_status.json`。
- 请求、HTTP 原始响应、解析结果及用量：同目录 `calls/`。

```powershell
.venv-p2/Scripts/python.exe scripts/run_grounded_contact_smoke.py --dataset D:\cholec_dataset --config configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml --api-key-file docs/API.txt --cache artifacts/final_pipeline_cache/grounded_contact_pilot_20260905 --output artifacts/preflight/grounded_contact_qwen0902_20260905
```

保留原产物；如另做新实验，应使用新的 output 目录。
