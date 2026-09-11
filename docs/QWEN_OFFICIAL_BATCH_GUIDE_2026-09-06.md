# Qwen 官方 Batch File 使用与费用估算

2026-09-06 核查。本文是操作说明和费用估算，没有提交付费任务。

## 模型、地域和价格

使用阿里云百炼华北2（北京）的官方 API key，环境变量名 `DASHSCOPE_API_KEY`。OpenAI SDK 的 base_url 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。

官方当前明确：`qwen3.8-max` 支持 Batch；固定快照 `qwen3.8-max-0902` 不支持 Batch。不能直接把现有快照名称原样用于批处理。使用别名时保存返回模型、日期与参数，并先做小样本对照。

北京人民币单价：Batch File 输入 6 元/百万 tokens、输出 18 元/百万 tokens；普通同步及 Batch Chat 分别为 12 元和 36 元。不要把 Batch Chat 当作半价 Batch File。

来源：https://help.aliyun.com/zh/model-studio/qwen3-8-max

## 本项目费用估算

来源是 `artifacts/preflight/h0_prompt_refinement_qwen0902_20260906/calls/*_baseline/api_usage.jsonl` 的 16 次 Qwen 调用，不是 Gemini，也没有混入 tuned/schema_only 实验。

- 平均输入 4028.25 tokens。
- 平均输出 831.1875 tokens，已包含平均 755.9375 个 reasoning tokens，不再重复相加。
- 测试集 8 视频，15,282 个目标，每目标一条请求；不因输入三张图而除以三。
- 平均用量外推：`15282 × (4028.25 × 6 + 831.1875 × 18) / 1000000 = 597.99803175 元`。
- 每条都按样本最贵一条外推：809.273592 元。
- 假设每条输入不超过 4029 且计费输出总数均为 4096：1496.138364 元。这是条件情景，不是对官方 thinking 限制的保证，也不是绝对上限。

可按约 600 元理解当前样本均值，先预留约 800–1000 元。该预留不保证封顶，也不保证一定用完。模型别名、官方 thinking 参数映射、图像处理和输出长度可能改变 token 数，应先小样本实测再更新估算。未计入 OSS 存储和流量、失败重提费用；这些也没有实际发生。

计算明细：`artifacts/preflight/qwen_official_batch_estimate_20260906/estimate.json`。

## 输入文件如何准备

每个目标一行 UTF-8 JSONL。每条仍使用原三帧 `[t-50,t-25,t]`，只预测 t 的五头；输入不含 GT、Tracker、修复或上一条预测。`custom_id` 使用 `VID01_51` 等唯一目标标识。

顶层结构为 `custom_id`、`method: POST`、`url: /v1/chat/completions`、`body`。`body` 中包含 `model: qwen3.8-max`、完整原 prompt/messages、三张图片、JSON schema、temperature 和已核实的官方思考设置。`enable_thinking` 直接放在 body，不能把 SDK 的 extra_body 包装原样写进 JSONL。当前 OpenRouter 的 reasoning.effort=low 不能不经核实就当作官方等价设置。

文件限制：每文件最多 50,000 条、500 MB；每行最多 1 MB。三张完整 PNG 的 Base64 很容易超出单行限制。推荐将图像放在私有 OSS 中，用服务端可访问、在整个排队执行期间有效的签名 URL；不需要把存储桶设为公开。不要在 JSONL 中写 D 盘本地路径。

按视频分文件便于核查，每文件仍须检查大小。不要为满足文件大小而无记录地压缩或降低输入图像质量。

## 提交、查询和下载示例

下面只展示 Batch File 生命周期，假设已经生成并校验 `requests.jsonl`。它不会替你抽帧或生成完整五头请求，也不是当前项目已经接通的命令。

安装 SDK：`python -m pip install openai`。将北京地域 key 配置到当前进程的 `DASHSCOPE_API_KEY`，不要写进输入文件。

只运行一次提交代码：

```python
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_retries=0,
)
uploaded = client.files.create(file=Path("requests.jsonl"), purpose="batch")
job = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
Path("batch_id.txt").write_text(job.id, encoding="utf-8")
print(job.id, job.status)
```

保存任务 ID，后续查询同一任务，不要重新提交原文件。若提交响应丢失，先去控制台核查是否已创建任务。

```python
job_id = Path("batch_id.txt").read_text(encoding="utf-8").strip()
job = client.batches.retrieve(job_id)
print(job.status, job.request_counts)
if job.output_file_id:
    client.files.content(job.output_file_id).write_to_file("results.jsonl")
if job.error_file_id:
    client.files.content(job.error_file_id).write_to_file("errors.jsonl")
```

这里的第二段复用第一段的 imports/client，但不重跑上传和创建部分。24h 是提交的执行窗口，不能保证每条请求都成功；下载成功和错误结果一起核查。

按 custom_id 对齐预测，拒绝重复目标，保留失败项；预测冻结后才读取 GT。按每头有效 GT mask 计算 micro-F1 与集合完全匹配 Accuracy，缺失 GT 不进入该任务分母，有 GT 的失败调用不能从分母中删除。JSON 合法率不代表识别分数。

官方接口说明：https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai

批处理与文件限制：https://www.alibabacloud.com/help/en/model-studio/batch-inference

## 项目当前状态

当前主入口是 OpenRouter 同步请求；改 key/base_url 不足以变成 Batch File。完整接入仍需要请求导出、文件分片与图片引用、官方参数对齐、持久化任务 ID、结果回收，以及接入现有 GT 评分器。建议先在 Training/Validation 固定少量目标，核验五头分数和计费后，再提交 Testing。


## 2026-09-06 后续核查更正

较新的 Batch File 接口参考（2026-09-04 更新）明确正式请求单行上限为 6 MB，并支持 Base64；旧概览仍写 1 MB，文档有差异。本次三图无损 PNG 请求约 1.6–1.7 MB，低于接口参考的 6 MB，可准备 Base64 输入，无需先上传 OSS。前文 1 MB 不应作为该新接口已确认的限制。来源：https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai


???????????? Qwen3.8 ? reasoning_effort=low????????? body.enable_thinking=true?body.reasoning_effort=low???? thinking_budget????????????????????????https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions
