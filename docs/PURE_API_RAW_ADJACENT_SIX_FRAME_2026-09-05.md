# 纯 API 联合预测：六个原视频相邻帧

日期：2026-09-05。六次真实调用完成；6/6 合法五头响应，费用 **$0.076242**，缓存命中 0，未计价调用 0。没有 Tracker、Gate、Verifier、Repair 或预测记忆。

## 输入与实际执行

用户要求原视频逐帧连续，而非按标注图片间隔 25 帧。VID110 本地只有采样 PNG，没有找到其原始视频，因此在调用前说明改用本地 Testing/VID06 原始 MP4，并固定目标 8601–8606。

- 模型：`qwen/qwen3.8-max-0902`；六个原始响应均回显该模型及 Alibaba 提供商。
- 原视频：`D:/cholec_dataset/Testing/VID06/vid06.mp4`，25 fps，854×480，共 68141 帧。
- 一基帧号，解码下标=帧号−1。目标步长 **1**；每目标输入 `[t-2,t-1,t]`，输入步长也为 **1**。
- 六个目标从第一帧到最后一帧覆盖 0.20 秒，每个输入窗口覆盖 0.08 秒。原实验的三帧窗口覆盖两秒，两者协议不同。
- 一次 API 联合输出 I/V/T/IVT/Phase，仅 selected 标签，无 top-k；每目标一次调用，无重试、无局部补答案。
- 沿用接触区域实验的联合 H0 本体/边界/schema 提醒；明确传递 low/low/high 图像 detail。加入真实采样 fps/步长说明并使用新 prompt 版本隔离缓存。
- 新入口 `scripts/run_raw_adjacent_h0_smoke.py` 先无凭证准备 plan、请求及图像，随后以同一冻结方案执行；默认 Pipeline 与旧脚本保持不变。
- 调用前规划预算 $0.15，限制最多六次调用；本地按已报告支出加 $0.04 预留判断是否继续，未知费用则停止。这不是提供商美元硬上限。本次六次正常完成，未触发停止。

## 六帧预测原样解码

| 目标帧 | 实际输入帧 | Instrument | Verb | Target | IVT | Phase |
| --- | --- | --- | --- | --- | --- | --- |
| 8601 | 8599,8600,8601 | bipolar (1) | null_verb (9) | null_target (14) | 95 | calot_triangle_dissection (1) |
| 8602 | 8600,8601,8602 | grasper (0) | retract (1) | gallbladder (0) | 17 | preparation (0) |
| 8603 | 8601,8602,8603 | grasper (0) | retract (1) | gallbladder (0) | 17 | calot_triangle_dissection (1) |
| 8604 | 8602,8603,8604 | 空集合 | 空集合 | gallbladder, liver (0,8) | 空集合 | preparation (0) |
| 8605 | 8603,8604,8605 | grasper (0) | retract (1) | gallbladder (0) | 17 | calot_triangle_dissection (1) |
| 8606 | 8604,8605,8606 | grasper (0) | retract (1) | gallbladder (0) | 17 | calot_triangle_dissection (1) |

IVT95=bipolar–null_verb–null_target；IVT17=grasper–retract–gallbladder。空集合是模型原始输出，不是调用失败；8604 的 Target 保留原样，没有通过闭包删除它。

五次相邻比较中，完整五头输出变化 **4/5**；Phase 变化 **4/5**；Instrument/Verb/Target/IVT 各变化 **3/5**。这只是输出稳定性统计，不是错误率。原图变化较小，但模型在器械身份和 Phase 上出现明显跳变；本次不做平滑或修复。

## GT 与结论边界

所有预测保存后才读取原始 `vid06.json`，发现其标注从 **13201** 开始。因此 **8601–8606 六帧均无精确 GT**，五头全部 mask 排除，准确率和 F1 均不可计算，不能把它们写成 0%。本次选段不适合做准确率评估；先前所说“六帧最多有一帧可评分”是上限，实际为零。

没有复制最近标注，没有插值，没有看结果后重选六帧重跑。即使选择带标注的时段，现有标注一般仍每 25 原帧一条，六个相邻原帧也不能自然获得六份独立 GT。要评价六帧准确率，需要它们的精确人工标注。

此次证明纯 API 相邻帧路径可执行，并展示了该选段的输出不稳定；无法证明准确率优于/劣于间隔 25 帧输入，也不能与不同视频 VID110 的旧实验做修复收益配对比较。本段被用作开发诊断，不作为未见的正式盲测结果。

## 核验与产物

- 预检：六目标均相差 1，六组三帧输入均相差 1，18 个图像 SHA 绑定通过，请求没有 GT、修复候选或跨窗口状态。
- 独立解码核验：一次 seek 后连续读取八个原帧，与实际上传图的 RGB 像素逐点一致；不是把稀疏 PNG 重新编号。
- 完成核验：全部 18 处实际 wire data-URL 哈希与保存图像匹配；脚本 SHA 与执行前计划相同。
- 相关既有测试：`8 passed`（`tests/unit/test_pure_h0_smoke.py`）。本轮没有再次运行全仓库测试。

产物目录：`artifacts/preflight/raw_adjacent_h0_qwen0902_vid06_20260905/`。

- `plan.json`：采样、模型、视频/代码/配置哈希、调用和预算方案。
- `requests/`、`images/`：六次完整请求、八张实际输入图。
- `calls/`：六次脱敏 wire、HTTP 原始响应、解析结果及费用。
- `predictions.json`：读取 GT 前冻结的原始预测。
- `predictions_with_evaluation.json`、`summary.json`：精确 GT 缺失状态和汇总。
- `input_alignment_audit.json`、`completion_audit.json`：连续解码及实际请求核验。

再次实验必须用新目录；默认只准备请求，显式 `--execute` 才会调用 API。
