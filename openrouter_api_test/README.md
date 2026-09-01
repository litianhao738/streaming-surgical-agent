# OpenRouter 图片输入 API 测试

这个目录是独立的连通性测试工具，不依赖项目内部的 API 配置。它会把本地图片以 Base64 data URL 发到 OpenRouter 的 `chat/completions` 接口，并把输入副本、模型原始响应、提取文本和测试摘要都保存在本目录。

## 在 CMD 中运行

从项目根目录执行这一行，把模型号和 API Key 替换成你的值：

```bat
cd /d D:\PythonProject7
.venv-p2\Scripts\python.exe openrouter_api_test\test_openrouter.py --model "openai/gpt-5.6-sol" --api-key "在这里粘贴API_KEY" --image "openrouter_api_test\input.png"
```

也可以进入测试目录再执行：

```bat
cd /d D:\PythonProject7\openrouter_api_test
..\.venv-p2\Scripts\python.exe test_openrouter.py --model "openai/gpt-5.6-sol" --api-key "在这里粘贴API_KEY" --image "input.png"
```

当前临时 Key 文件的回归测试写法是：

```bat
cd /d D:\PythonProject7
.venv-p2\Scripts\python.exe openrouter_api_test\test_openrouter.py --model "openai/gpt-5.6-sol" --api-key-file "docs\API.txt" --image "openrouter_api_test\input.png"
```

## 在 CMD 中跑一个真实 Pipeline 时间点

下面这段会对 `VID110:901` 产生一个结构化预测。模型、临时 API Key、运行编号和并行数都从 CMD 传入：

```bat
cd /d D:\PythonProject7
set "MODEL=openai/gpt-5.6-sol"
set "OPENROUTER_API_KEY=在这里粘贴你自己的API_KEY"
set "PARALLELISM=1"
set "RUN_ID=openrouter_frame_test_001"

.venv-p2\Scripts\python.exe scripts\run_dataset_api_pipeline.py ^
  --mode engineering ^
  --video-id VID110 ^
  --target-frame-id 901 ^
  --max-frames 1 ^
  --pipeline-profile single_pass ^
  --dataset-root "D:\cholec_dataset" ^
  --config configs\perception\joint_openrouter_latency_dataset.yaml ^
  --model "%MODEL%" ^
  --api-key "%OPENROUTER_API_KEY%" ^
  --parallelism %PARALLELISM% ^
  --output-root artifacts\openrouter_pipeline_test ^
  --cache-root artifacts\openrouter_pipeline_test_cache\cmd ^
  --run-id "%RUN_ID%" ^
  --max-provider-calls exact-selection ^
  --authorize-data-upload
```

`VID110:901` 是该验证视频的首个可用时间点，所以本条测试实际只上传 1 张图。一般情况下，`--max-frames 1` 表示只预测一个目标时间点，但该时间点仍可从最多 6 帧的因果缓冲区中自适应选择最多 3 张图片。`--parallelism` 在同一视频内必须设为 `1`，因为后一时间点会读取前一时间点提交的状态；不同视频需要并行时，应分别启动独立 CMD 进程。

上面的命令使用效率版：OpenAI、Azure、Amazon Bedrock 三个上游按低延迟排序并允许 fallback。论文严格版把 `--config` 改为 `configs\perception\joint_openrouter_dataset.yaml`；该配置固定 OpenAI 上游且禁用 fallback。两套配置的路由策略进入请求哈希，不会误共用同一条 API cache。`--mode paper` 只接受论文严格版。

可继续替换的实验参数包括 `--pipeline-profile`、`--experiment-config`、`--evidence-threshold`、`--report-window-frames` 和 `--proxy-url`。同一个 `--run-id` 不能重复使用；缓存目录可以复用，以免重复请求相同输入。

## VPN / 代理

如果 VPN 是 TUN/全局模式，通常不需要额外参数。脚本会读取 CMD 中已有的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`。如果只开放了本地代理端口，可以显式追加：

```bat
--proxy "http://127.0.0.1:7890"
```

例如：

```bat
.venv-p2\Scripts\python.exe openrouter_api_test\test_openrouter.py --model "openai/gpt-5.6-sol" --api-key "在这里粘贴API_KEY" --image "openrouter_api_test\input.png" --proxy "http://127.0.0.1:7890"
```

## 输出文件

- `input.png`：随目录提供的测试图片，可替换为自己的图片。
- `input_used.png`：本次实际发送图片的副本。
- `model_info.json`：模型目录查询结果及图片输入能力。
- `response.json`：OpenRouter 原始 JSON 响应或原始错误。
- `response.txt`：从响应中提取的模型回答。
- `test_summary.json`：HTTP 状态、耗时、模型、token 用量和输入图片哈希。

重复运行会覆盖上一轮的同名输出。脚本不会把 API Key 写入任何输出文件。
