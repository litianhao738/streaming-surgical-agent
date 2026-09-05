# H0 frame strategy Training pilot — 2026-09-05

当前建议：暂时保留现有 `[t-50,t-25,t]` 三帧初始预测基线，不把六帧输入升级为默认方案。此前六帧方案只是待验证假设，本轮没有观察到它对 IVT 的收益，也没有出现五头同时改善。

全部 32 个目标纳入评分时，B（三帧覆盖两秒）IVT micro-F1 为 32.52%，D（六帧覆盖五秒）为 28.80%。排除接口失败影响、只比较四组均成功的 28 个共同目标时，B 为 31.58%，D 为 30.09%，单图为 31.86%。因此也不能宣称三帧已经显著优于单图。

六帧的共同目标 Phase F1 较 B 提高 10.71 个百分点，但 Target 降低 1.50 个百分点，IVT 降低 1.49 个百分点。VID96 四种方案的 IVT F1 都只有约 6%–7%；增加帧数尚未解决该段的语义识别问题。后续优先检查真实图像、动作证据和靶点定义，不依据这些 GT 写特判。

这次受余额限制，只完成原计划前 32 个目标，覆盖四段 Training 视频的较早部分。不能据此宣布测试集性能、最佳窗口或滑动步长已确定。默认 Pipeline、Tracker 和冻结版本没有改动。

执行记录：4 次失败包括 1 次旧响应丢失、1 次不完整流式回复、2 次 OpenRouter 并发额度拒绝。另有 8 个完整回复曾被新增记录代码的响应头大小写问题误判为失败，已用原解析器从磁盘恢复；保留原记录，无重复推理调用。恢复与窗口逻辑的相关测试为 7 passed，新增脚本和测试的 Ruff 检查通过。

Model: `qwen/qwen3.8-max-0902` through OpenRouter. Pure joint five-head H0; no Tracker, repair or memory.
Original plan: 80 time-spread targets from four Training videos, four paired image strategies, no retries.
Account balance could not support the full plan. This report covers its first 32 targets (8 per video, 128 requests); later 48 targets were not run.
The smaller pilot covers earlier video portions and does not retain full-video temporal coverage. The original 80-target manifest and raw summary remain preserved.
A: current image; B: [-2,-1,0] seconds; C: [-5,-2,0]; D: [-5,-4,-3,-2,-1,0].
Only the final target is scored. History images use low detail; target uses high detail. Temperature 0.

Successful predictions: 124/128 in this pilot; shared successful targets: 28/32.
Confirmed cost: $1.481284, plus unknown cost for the prior lost response (VID103_5351_B).
The $3 unknown-cost reserve is a budget reserve, not a measured charge. The lost request was not repeated.
Known spend is a lower bound: it excludes the lost original response and unpriced gateway rejections; those are never relabeled as successful predictions.
Primary pilot metrics retain all 32 targets with valid GT; failed calls receive empty predictions. Secondary metrics use shared successful targets.
All 320 requests were frozen before scoring; selection used time bins and validity masks, not label identities.
Interrupted continuations triggered partial offline scoring. Remaining frozen requests were unchanged; final scoring followed completion of the affordable pilot.

## Primary micro-F1 (all 32 pilot targets per arm, masked by task)

| Task | A single | B 3/2s | C 3/5s | D 6/5s |
|---|---:|---:|---:|---:|
| instrument | 92.73% | 90.91% | 93.58% | 93.69% |
| verb | 58.41% | 63.16% | 63.72% | 65.52% |
| target | 43.48% | 42.86% | 45.05% | 40.35% |
| ivt | 28.80% | 32.52% | 29.03% | 28.80% |
| phase | 73.02% | 67.74% | 66.67% | 75.00% |

## Secondary micro-F1 (shared successful targets)

| Task | A single | B 3/2s | C 3/5s | D 6/5s |
|---|---:|---:|---:|---:|
| instrument | 92.93% | 93.07% | 94.95% | 92.93% |
| verb | 58.25% | 62.86% | 64.08% | 63.46% |
| target | 45.71% | 43.81% | 46.60% | 42.31% |
| ivt | 31.86% | 31.58% | 29.82% | 30.09% |
| phase | 78.57% | 71.43% | 75.00% | 82.14% |

## Per-video IVT micro-F1 (primary)

| Video | A | B | C | D |
|---|---:|---:|---:|---:|
| VID103 | 27.03% | 24.24% | 22.22% | 28.57% |
| VID23 | 33.33% | 38.46% | 30.77% | 32.00% |
| VID31 | 48.48% | 57.14% | 54.55% | 47.06% |
| VID96 | 6.45% | 6.90% | 6.90% | 6.45% |

## Limits

This is a small development pilot on Training videos, not held-out test performance or official mAP. IVT includes null classes 94–99.
Targets are spread over time, not a continuous streaming run; the study does not validate throughput, latency or cross-window state at 1 Hz.
A four-video result cannot establish that six images improve every action, target or phase. No default pipeline change follows automatically.
Wire image audit: PASS; 128 requests, 416 image references.
Original frozen scripts, sampling plan and prior artifacts were preserved. Continuations used eight concurrent calls, reduced to four for the balance limit, with no retries.
