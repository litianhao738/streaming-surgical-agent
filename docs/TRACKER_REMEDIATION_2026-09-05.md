# Tracker / OOF 工程修复与验证（2026-09-05）

本报告接续 [Tracker 训练与 OOF 正确性审计](TRACKER_TRAINING_AND_OOF_CORRECTNESS_2026-09-05.md)。本次处理越界框监督、断点续训、跨时间缺口关联、正式入口 OOF 路由四项工程问题。已有 full / 五折 OOF 权重继续作为 **legacy 框策略基线**保存；本次没有正式重训这些模型，没有付费 API 调用。

## 1. 框监督按版本修复

- `legacy_strict_v1` 保留历史行为，支持复算旧结果。
- `clip_to_frame_v2` 对有限且面积为正的框取图像范围内的交集；完全越界、退化或非有限框继续过滤，并记录原因。原始标注文件不修改。
- `build_detection_training_records` 的库默认值仍为 legacy；**新训练 CLI 默认显式传入 v2**，默认输出到新的 `artifacts/training/tracker_clip_v2`。
- OOF 评估通过 `--gt-bbox-policy` 显式选择 GT 口径，默认 legacy。报告写入策略和框质量统计，v2 使用独立默认报告名。

| 范围 | legacy 保留框 | v2 保留框 | 裁剪恢复 |
|---|---:|---:|---:|
| Training | 25,574 | 25,811 | 237 |
| Validation | 7,574 | 7,578 | 4 |
| Testing（仅标注质量统计） | 29,294 | 29,994 | 700 |

Training 仍过滤 1 个退化框。可用训练图片记录由 **14,207 增至 14,231**；VID31 没有实例框监督，仍不参加检测器训练，也未被生成为空框负样本。

现有 OOF 预测在两种 GT 口径下的分数如下，均评分九个视频的 14,231 帧：

| 指标 | legacy GT | clipped GT |
|---|---:|---:|
| macro AP50 | 58.5453% | 58.6843% |
| Precision | 83.3416% | 83.8825% |
| Recall | 84.9613% | 84.7274% |
| F1 | 84.1437% | 84.3028% |

**这里只改变 GT 框处理口径，不能称为模型精度改善。** 这是项目内部检测评估，不能当作 IVT、官方 COCO AP 或 MOT/IDF1/HOTA 分数。

实现：`src/surgical_agent/tracking/box_policy.py`、`training_data.py`、`oof_evaluation.py`、`scripts/evaluate_tracker_oof.py`。

## 2. 续训恢复同一模型和优化器结构

旧问题是恢复时关闭预训练下载，同时改变了 torchvision 的归一化及骨干冻结结构：原训练的 70 个可训练参数变成 192 个，优化器状态无法加载。

修复后，关闭预训练下载仍重建 COCO 初始化对应的 FrozenBatchNorm 和三阶段可训练骨干。模型契约记录结构与有序可训练参数名/形状。新的训练状态采用 v2，绑定框策略、标注文件哈希、实际监督记录指纹、模型契约及 RNG 状态。

旧 v1 状态只有在显式 `--allow-legacy-resume --bbox-policy legacy_strict_v1` 时才能恢复，且必须写入独立输出目录。旧状态不能伪装成 v2 框监督训练。新建训练默认拒绝覆盖已有权重或训练状态；`--resume-source-root` 支持读取原目录、写入新目录。

真实旧 fold_0 断点已恢复：46 个 FrozenBatchNorm、70 个可训练参数、70 份优化器状态，epoch 10 / 13,250 条 loss 记录；恢复后的模型张量与原状态一致。实际 FasterRCNN 的小图回归也验证了“更新一步 → 保存 → 恢复 → 再更新”与连续更新的第二步 loss 和全部模型状态张量一致。这是恢复正确性验证，不是正式训练或泛化性能评测。

本次恢复探针及更新回归在 CPU 上执行，未恢复原 CUDA RNG；不能将其描述为已在原 GPU 环境验证逐位重现。

实现：`src/surgical_agent/tracking/detector.py`、`training_checkpoint.py`、`scripts/train_tracker.py`。

## 3. 旧检测结果重新关联，保存为独立产物

读取旧的器械类别、框和置信度，仅重新计算 `track_id` / `age`；当相邻记录的源帧编号差大于 25 时清空活跃轨迹，不复用以前片段的 ID。没有再次运行检测器，也没有读取 GT 来决定关联。

新目录：`artifacts/preflight/tracker_remediation_20260905/reassociated/`。

- Training OOF 入口：`oof/index.json`。
- Validation / Testing full 入口：`full/predicted_tracks.json`。
- 每个预测文件有独立 `reassociation_manifest.json`，记录原文件/新文件哈希、关联参数、代码版本哈希、模型和数据来源。
- full 和五折共六个 checkpoint / training manifest 以独立文件复制保留来源，未改写旧文件。

独立审计覆盖 7 个预测文件、41,866 条帧记录、79,978 个检测、907 处时间缺口。这里 VID31 同时存在于 full 和 OOF 文件，计数包含重复记录。所有逐帧检测多重集合、帧编号和 split 保持不变；重关联后跨片段 ID 复用及缺口首帧 age 违规均为 0。

用相同 legacy GT 重算新 OOF：AP50 **0.58545327459572**，F1 **0.8414367315325781**，与旧结果完全相同。这符合“只修关联，不改检测”的预期，也不证明语义识别提高。

实现：`src/surgical_agent/tracking/reassociation.py`、`scripts/reassociate_tracker_artifacts.py`。

## 4. 正式 Pipeline 的路由和调用前检查

`scripts/run_final_dataset_pipeline.py` 新增 `--tracker-oof-index`，与 `--tracker-artifact` 互斥。Tracker 开关与输入源必须一致：

- Training 必须通过 OOF index，按视频选择未在该视频上训练的模型。VID31 使用从未消费其实例监督的 full 模型。
- Validation / Testing 使用 full 产物。兼容原生训练的 `root/predicted_tracks.json + root/full/` 布局和重关联文件的同目录布局。
- 在创建 API transport 之前，核对 canonical split、完整训练排除集合、repair manifest、checkpoint 实际文件 SHA、预测文件 SHA、OOF 策略一致性、目标帧及全部因果上下文覆盖、整个选中视频的 gap reset。
- 写入 `tracker_preflight.json`。同 run-id 恢复时要求该审计一致；不符合要求时直接报错，避免 B_tracker 因缺产物静默退化为无 Tracker。

运行时来源检查读取 checkpoint 文件 SHA 与 training manifest，**不声称重新核验 checkpoint 内置训练 metadata**；原权重的内置 metadata 已在前次独立 OOF 审计中检查。预检函数不读取目标 GT 类别值；此声明仅限该函数，正式 CLI 原有的 Training 先验构造仍会读取 Training 标注。

全部可用推理观察点的预检结果：

| split | 视频数 | 观察点数 | 结果 |
|---|---:|---:|---|
| Training | 10 | 18,248 | PASS |
| Validation | 2 | 4,433 | PASS |
| Testing | 8 | 15,282 | PASS |

这证明产物路由及覆盖，不代表在 Testing 上运行了 API 或评估模型。

正式入口另外以真实数据、mock API 跑 VID04 OOF 和 VID110 full 各两条输出，4 条 Tracker 状态全部为 `OK`、`tracker_available=1`。旧 OOF 的负例在 VID04 / 7976 因 gap reset 失败被拒绝，API transport 创建次数为 0。

实现：`src/surgical_agent/tracking/runtime_preflight.py`、`scripts/run_final_dataset_pipeline.py`。

## 5. 验证和保留证据

本轮新增证据均在 `artifacts/preflight/tracker_remediation_20260905/`：

- `preservation.json`、`before/`：修改前源码快照及历史产物哈希。
- `preservation_verification.json`：38 个既有 Tracker / OOF / 冻结基线文件 SHA 全部未变。
- `box_policy_full_annotation_audit.json`、`oof_legacy_gt_recheck.json`、`oof_clipped_gt_sensitivity.json`。
- `tracker_resume_fix_20260905.json`：真实旧断点恢复及合成两步更新回归。
- `reassociated/migration_manifest.json`、`reassociated/independent_audit.json`、`reassociated/oof/heldout_evaluation.json`。
- `runtime_training_preflight.json`、`runtime_validation_preflight.json`、`runtime_testing_preflight.json`。
- `mock_cli_validation.json`、`mock_runs/`、`negative_cli_preflight.json`。

最终全量 `.venv-p2/Scripts/python.exe -m pytest -q`：**1095 passed，14 skipped（45.50 秒）**。本轮修改的 Tracker、正式入口、训练/评估/迁移脚本及相关测试通过 Ruff；首次检查发现的五处 import 排序问题已修正。验证汇总见 `verification_summary.json`。

已有非本轮修改保留，未 reset 工作区，未改冻结 V7。

## 6. 后续模型状态与使用方法

**当前可以使用的是旧检测权重 + 修复后的关联 + 完整路由检查。** 新框策略尚未生成正式 full 和五折 OOF 权重。若要报告“修复框监督后的 Tracker”，需要以同一 v2 协议从 COCO 初始化重新训练 full 和五折；不能只换报告口径，也不能把旧优化器状态直接续成 v2。

待正式训练时使用独立目录，两步均显式采用相同五折配置（本轮未执行）：

```powershell
.venv-p2/Scripts/python.exe scripts/train_tracker.py --mode full --dataset-root D:/cholec_dataset --config configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml --bbox-policy clip_to_frame_v2 --output-root artifacts/training/tracker_clip_v2_oof5 --device cuda
.venv-p2/Scripts/python.exe scripts/train_tracker.py --mode oof --dataset-root D:/cholec_dataset --config configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml --bbox-policy clip_to_frame_v2 --output-root artifacts/training/tracker_clip_v2_oof5 --device cuda
```

计划范围是九个有实例框监督的 Training 视频，full 14,231 张采样图片，五折按视频排除；默认每模型 10 epochs。仍然是约 1 Hz 采样图片上的单帧器械检测训练，不是连续 25 原视频帧的时序训练。云 GPU 时长及费用需根据执行设备另行给出，不在本轮支出内。

本轮不改变联合五头 H0、三帧输入、Verifier/Repair、Gate 或冻结版本；也不能据此宣称 Verb、Target、IVT 语义问题已解决。
