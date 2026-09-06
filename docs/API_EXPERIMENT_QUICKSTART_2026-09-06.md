# 纯 API 基线：同学下载后运行

本页对应固定的同步 OpenRouter 版本。核心实现从提交 `7f69aad` 起已进入主分支。
使用 `qwen/qwen3.8-max-0902`，严格 Alibaba 路由；每个目标输入
`[t-50,t-25,t]` 三张因果图片，一次 API 联合输出 Instrument、Verb、Target、IVT、Phase。
目标每次前进 25 个原视频帧，即 1 秒；窗口重叠。视频开头或缺帧后只使用实际可用的历史。
这条入口不启用 Tracker、Gate、Verifier、Repair 或跨窗口记忆。

## 1. 下载和安装

Python 3.10–3.12，建议 3.11。纯 API 运行可以使用 CPU，不需要训练模型或下载 Tracker 权重。
下面是 PowerShell 命令；Linux 下载前用 `export GIT_LFS_SKIP_SMUDGE=1`，
虚拟环境激活改为 `source .venv/bin/activate`。

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/litianhao738/streaming-surgical-agent.git
cd streaming-surgical-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
git rev-parse HEAD
```

已下载的仓库先用 `git pull --ff-only` 更新。保留最后一行输出的提交号，
和结果一起交回；比较实验的同学应使用相同提交号。
依赖包含 PyTorch/torchvision，但这条入口不需要 GPU。

## 2. 准备数据和密钥

数据集与密钥由运行者在本地提供，不在 GitHub 中。将后续命令的
`D:/cholec_dataset`、`D:/private/openrouter-key.txt` 换成自己的路径。
密钥文件只放 OpenRouter key，不放进仓库；零费用检查不需要密钥。

使用项目已审计的完整 CholecTrack20 数据目录，保留 Training、Validation、Testing
以及下面的清单和修复标注；适配器会检查其来源和哈希。

- `repair_manifest.json`
- `Validation/VID30/vid30_repaired.json`
- `Training/VID31/vid31_phase_repaired.json`
- `Training/VID31/vid31_frame_ivt_repaired.json`

只有原始视频或原始下载包还不足以直接运行：需要上述项目配套数据准备产物。
Testing 原视频按标注帧号在线读取，不需要事先把整套视频全部拆帧。

## 3. 先做零费用检查

以下选择 Validation / VID110 的 8601、8626、8651 三个目标，
每个目标各有三张输入图片。“三个目标”意味着最多三次初始 API 调用，
不是一次调用同时预测三个目标。

```powershell
python scripts/run_dataset_api_pipeline.py --mode engineering --video-id VID110 --target-frame-id 8601 --max-frames 3 --dataset-root D:/cholec_dataset --preflight --run-id h0_preflight_3
python scripts/run_dataset_api_pipeline.py --mode engineering --video-id VID110 --target-frame-id 8601 --max-frames 3 --dataset-root D:/cholec_dataset --config configs/perception/joint_mock_h0.yaml --run-id h0_mock_3
```

第一条生成真实请求的安全元数据，`provider_calls` 应为 0、`planned_calls` 为 3。
第二条使用离线模拟响应验证整个输出链路；模拟结果的分数不代表模型效果。
每次运行换一个新的 `--run-id`，已有输出目录不会被覆盖。

## 4. 同步 API 三目标冒烟

```powershell
python scripts/run_dataset_api_pipeline.py --mode engineering --video-id VID110 --target-frame-id 8601 --max-frames 3 --dataset-root D:/cholec_dataset --api-key-file D:/private/openrouter-key.txt --authorize-data-upload --max-provider-calls 3 --run-id h0_real_3
```

该命令会产生实际 API 费用，最多三次提供商调用；缓存命中不重复计费。
上传授权参数表示允许将选中的数据集图片发送给 API。若本机必须使用代理，
可追加 `--proxy-url http://127.0.0.1:你的端口`。
预检通过不保证远端模型可用或账户余额充足。主版本不自动切换模型或提供商；
调用失败时保留失败记录并终止，先检查原因再重跑。

固定入口是 `scripts/run_dataset_api_pipeline.py`，默认配置是
`configs/perception/joint_openrouter_h0.yaml`。参数固定为 temperature 0、
reasoning low、最大输出 4096 tokens、历史图 low / 当前图 high。
使用原版已评估 prompt 和完整 JSON Schema。生产系统 prompt 的 UTF-8 SHA-256 为：

```text
c1009de8429158eab6b9c0f4ee39b12f51e5e970335f0c554866198191b6e3e5
```

这个哈希对应 `load_main_h0_prompt()` 渲染后的完整文本。
请保留主配置和模板；换模型或改 prompt 的结果需要单独命名和记录。
“固定基线”表示后续实验使用同一协议，不表示已经证明它在全部数据上最优。

## 5. 算分并交回结果

预测结束后，单独读取 GT 算分：

```powershell
python scripts/evaluate.py --run-dir artifacts/api_dataset/h0_real_3 --dataset-root D:/cholec_dataset
```

查看生成报告中的 `label_metrics`：各任务的 micro-Precision、micro-Recall、
micro-F1 和整组标签完全正确率。缺失 GT 按任务单独排除；例如缺失 IVT
不会减少有效 Phase 的分母。硬标签输出的 AP/mAP 不支持，不能当成有效排名分数。

把 Git 提交号、实际命令、完整的 `artifacts/api_dataset/<run-id>/`
以及评估命令输出的报告目录交回。保留 usage、请求来源信息、预测与运行清单，
这样可以核对效果、token 和费用；不要发送密钥文件。

## 6. 完整 Testing：先核对样本和预算

```powershell
python scripts/run_dataset_api_pipeline.py --mode paper --split testing --dataset-root D:/cholec_dataset --preflight --run-id h0_test_preflight
```

先查看 `planned_calls`，用同模型、同配置真实冒烟的费用预估整套开销。
核定预算后才执行下面的收费命令；`exact-selection` 将调用次数上限固定为所选目标数，
它不是金额上限。

```powershell
python scripts/run_dataset_api_pipeline.py --mode paper --split testing --dataset-root D:/cholec_dataset --api-key-file D:/private/openrouter-key.txt --authorize-data-upload --max-provider-calls exact-selection --run-id h0_test
python scripts/evaluate.py --run-dir artifacts/api_dataset/h0_test --dataset-root D:/cholec_dataset --authorize-test-gt-evaluation
```

Testing 结果只用于最终评价，不用于调 prompt、选择模型或训练。
完整 Validation 可将 `--split testing` 改成 `--split validation`，使用不同运行名，
评价时不需要 Testing GT 标志。

## 本次发布验证

在仅包含待提交文件的导出目录中验证：57 项 API 协议、入口和评分相关测试通过，
改动 Python 文件通过 Ruff；VID110 的上述三目标真实请求预检、离线模拟预测和
独立 GT 算分全部完成。预检产生 0 次提供商调用，本次验证没有付费 API 请求。
同时修复评分器对工程冒烟起始帧的处理：允许从指定中段开始，仍拒绝跳过可用帧、
乱序及把截断结果当作完整 split 评分。模型分数未在本次发布中重新测量。
