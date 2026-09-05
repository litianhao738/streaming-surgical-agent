# 三帧联合 H0 主版本接入与验证

日期：2026-09-06。

## 结论与采用范围

主入口 `scripts/run_dataset_api_pipeline.py` 已改为默认加载
`configs/perception/joint_openrouter_h0.yaml`。它运行三帧因果输入、一次
OpenRouter API 联合预测目标帧 Instrument / Verb / Target / IVT / Phase。
这是初始预测 H0 的主版本接入，不代表旧完整 Tracker/Gate/Verifier 系统
已经适配新协议或获得更高准确率。

采用已有配对实测支持保留的原版基线：

- Qwen `qwen/qwen3.8-max-0902`，strict Alibaba 路由；temperature 0、low reasoning。
- 输入 `[t-50,t-25,t]`，真实相对时间 `[-2,-1,0]` 秒；输出目标前进一个
  1 Hz 采样点，即 25 个原视频帧。只输出目标帧标签。
- 历史图 low、当前图 high，实际 HTTP 图片字段显式携带 detail。
- 原版 prompt、完整 100 类 IVT 本体、重复完整 schema 均保留。
  没有采用 `schema_only`、`tuned`、六图、五次专家调用或未验证的领域示例。
- `joint_perception_final_only_v1` 返回五头最终标签，不返回 top-k 或置信分数。
- 每个未缓存目标最多一次 provider 尝试；不自动重试，不接 Verifier/Repair。

视频开始或缺帧后不足三帧时，只使用真实连续可用的 1–2 张因果图，明确记录
`partial_history`；不重复补图、不使用未来帧、不丢弃目标。这些边界窗口未包含
在此前完整三图准确率结论内。

## 补齐的工程缺口

原主入口仍限制旧模型、生成参数和 top-k 协议，实验版本不能直接替代默认运行。
现在正式包拥有 prompt、输入 JSON 和响应 schema，无需读取被 Git 忽略的
实验 artifacts，也不导入实验 runner 来拼装主请求。

解析后的 0/1 向量明确为 `hard_label_v1` 标签指示量，候选列表为空、置信度为空。
旧 Gate 不能把缺少分数解释为低置信，也不能由解析器虚构候选。旧完整研究入口
遇到新 final-only 协议时会在 API 调用前拒绝并说明兼容问题。

评估同步增加 `frame_recognition_metrics_v2`：每头独立使用有效 GT mask，输出
Precision、Recall、micro-F1、整组 Accuracy 和有效分母；null IVT 是有效类别。
有有效硬标签预测的任务 AP/mAP 明确标为 `null / unsupported`。没有有效 GT 的
任务标签指标为 null。原排名协议的 v1 评估保留。

## 验证证据

1. 接入后的首轮全量测试：1160 passed、14 skipped；随后又补齐了硬标签评估。
2. 六个真实图像窗口请求预检成功：Validation / VID103，目标
   `25101, 25126, 25151, 25176, 25201, 25226`，每个完整三图；实际 API 调用为 0。
3. 同六目标离线 mock 主 Pipeline 完成，覆盖上下文、解析、结果持久化及缓存。
4. 保存的真实 OpenRouter SSE 响应完成主入口离线回放：真实请求的文字、schema、
   参数、路由及三张图片 data URL 哈希与已有实测基线完全一致。
   此回放使用注入的本地 sender，网络调用为 0，新增费用为 0；输出中的 usage/cost
   仅为历史响应元数据，不能计为新实验。
5. Git index 导出的干净源码已完成同六目标真实请求预检，不依赖本机未跟踪代码
   或历史实验 artifacts。该干净目录完整测试：**1162 passed、16 skipped**，
   55.30 秒。与本机初轮相比，新增评估测试全部通过；另有两个依赖未发布本地
   Astra/Synapse artifacts 的既有测试按原规则跳过。已确认导入包来自干净目录。
6. 发布前索引审计：全部项目静态导入依赖齐全，未加入密钥、原始图像、base64
   图像或新增大型模型。已有 Tracker full/OOF 权重没有修改。
7. 主入口、请求协议、解析、兼容保护和评估相关改动的 Ruff 检查通过，
   暂存补丁格式检查通过。

完整生产 system prompt 的 UTF-8 SHA-256：

`c1009de8429158eab6b9c0f4ee39b12f51e5e970335f0c554866198191b6e3e5`

本地运行证据位于 `artifacts/preflight/main_h0_promotion_20260906/`，不上传原始图片
或本次运行 artifacts。此处工程验证没有重新估计识别准确率，也没有发起付费 API 实验。

## 清理与发布范围

默认配置与入口说明统一到主 H0。历史 top-k、fixed6、五专家和修复配置标为研究
版本；部分字节相同的配置也在冻结哈希清单中，因此保留用于复现。没有为“清理”
删除冻结版本。此次发布同时保存此前会话已存在的源码、测试、配置和报告修改，
避免只提交本轮入口文件而漏掉它们依赖的新模块。

运行步骤见 [主 API 使用说明](MAIN_API_PIPELINE.md)。本地工程链路已具备运行条件；
真正联机时仍需要可用的外置凭据和服务额度，是否成功返回由当次服务响应决定。
