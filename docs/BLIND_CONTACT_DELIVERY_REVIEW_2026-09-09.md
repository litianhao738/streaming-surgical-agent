# 同学 Blind Contact Audit 交付包审查

日期：2026-09-09。输入压缩包 `D:/delivery_blind_contact_audit_20260909.zip`，SHA256 为 `d6b70b3d59e9668bc214ef16c32ebff8a807189596313df4b9a02e5550196b65`。

已解压到独立目录 `artifacts/external_reviews/delivery_blind_contact_audit_20260909_v1/`。四个原脚本和 README 原样保存，未复制覆盖当前 `scripts/`，未运行该包的付费入口或 Batch 提交。审查仅调用了已读过的纯解析函数，使用合成输入验证边界；原文件哈希不变。

## 判断

**视觉检查思路值得保留，但这个包目前是 H0 提示词对照实验，不是已经验证可接入当前 Pipeline 的 Verifier/Repair。** 它在已有 H0 prompt 后追加一段要求：内部先清点每个器械尖端、接触区域和目标帧前的可见运动，再逐项反证器械—动作—组织是否属于同一真实关系。模型不看 H0 答案、五模型分数、图谱先验或 GT，直接输出另一份完整五头预测。

优点是针对接触错认和关系混配，避免先看到旧答案后跟着确认；只增加一段内部观察步骤，不要求额外五模型审核或多轮。局限是“默默检查”没有可观察的中间记录，不能证明模型实际完成了器械清点、定位或反证。单次新回答仍可能漏检、误认或与 H0 共同出错。

离线 Phase 脚本把这份新预测的 I/V/T/IVT 全部保留，再把 Phase 替换为 H0 Phase。它没有逐项 Verifier，也没有有益修改接纳判断；因此更准确地说是“新四头预测＋H0 Phase”的拼合，不是当前的候选审核与局部修改。

## 版本与成绩证据

README 声称基于旧提交 `517a89a`，交付提交 `1216280`，分支 `codex/blind-contact-audit-experiment`。本地可确认 `517a89a` 是已有历史提交，但本地没有 `1216280` Git 对象；压缩包没有完整仓库或补丁，无法独立证明这五个文件对应那个提交。

README 声称下面结果来自一个 VID31 的八目标 Training 切片。这是同学提供的数字，**本次没有独立重算**：

| 任务 | 其 H0 F1 | 其 Audit＋冻结 Phase F1 |
|---|---:|---:|
| Instrument | 73.3% | 78.6% |
| Verb | 42.9% | 55.2% |
| Target | 42.9% | 57.1% |
| IVT | 30.3% | 37.5% |
| Phase | 100.0% | 100.0% |

包中没有该实验的具体八个帧号、manifest、请求、原始响应、inference、评分计数、费用或运行时间。不能核实抽样独立性、哪种传输产生这些成绩、错误修复/误改数量，也不能与我们的旧八目标或新二十四目标直接排名。确认该成绩至少需要这次运行的 manifest、requests、batch_output、inference、summary，以及 Phase replay 的源哈希。

## 已确认的实现问题

1. **非法标签异常没有被正确捕获。** `run_gemini_blind_audit_ablation.extract` 和 Batch `decode` 捕获的是 `ValueError` 等；项目 `validate_final_only` 抛出的 `ApiSchemaError` 继承 `RuntimeError`。合成输入 `Verb=[999]` 在两者都直接抛异常，而非记为失败预测；可能中止同步实验或结果收集。应显式捕获该异常，逐目标保留失败状态与完整分母，禁止替换样本。

2. **只读取 `parts[0].text`。** 合成的“前一 part 为思考文本、后一 part 为合法 JSON”会被误判失败。应按实际响应协议提取最终非思考文本，并检查 finish reason；不能仅假定第一段就是答案。

3. **模型身份字段读取错误。** 当前代码读 `response.get("model")`，合成的标准 `modelVersion` 响应得到 `returned_model=null`。Google 的 GenerateContentResponse 明确使用 `modelVersion`，还提供 `usageMetadata`；应记录并校验实际模型，同时整理 token 和费用数据。[官方响应定义](https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse)

4. **Batch 的请求绑定需要验证。** 文件提交只写 `metadata.target_key/arm`，收集也只认同样 metadata；当前官方 File Batch 文档示例用顶层 `key` 与 `request`。这是与文档不一致的兼容风险，包中没有成功原始响应证明该 metadata 方言可用，不能直接断言一定失败。建议使用稳定 `key`，在本地 manifest 映射回目标/组别，并保留缺失、重复、未知 key 的拒绝检查。[官方 File Batch 格式](https://ai.google.dev/gemini-api/docs/batch-api#input-file)

5. **三张图片被拼成一张，但文字仍描述三张独立输入。** Batch `request` 把三张图横向拼接为一幅宽图，沿用 `main_h0_input` 原帧描述，没有在输入文字说明左/中/右对应历史/目标。虽然两组使用同一拼图，组内提示词对照仍有价值，但不能说与我们固定三张独立图、历史 low/目标 high 的输入协议完全一致。JPEG quality=95 也不是像素无损。官方说明 File Batch 使用完整 GenerateContentRequest，并支持多模态配置，没有在所查文档中找到“Batch File 必须只有一个 image part”的规定；同学可能碰到特定接口问题，需要其错误记录才能判断。[官方 Batch 配置](https://ai.google.dev/gemini-api/docs/batch-api#request-configuration)

6. **默认样本不能验证本次关心的三个交互头。** 默认 VID02/VID11/VID17/VID37 各两帧，本地核实八帧均只有 Instrument/Phase mask 有效，Verb/Target/IVT mask 全为 false。按现有评分器会排除这些缺失 GT，并非算作零分；但该默认命令无法验证 Verb/IVT 改善。加 `--require-all-task-gt` 将拒绝这些目标。README 的 VID31 结果显然来自另外的 CLI 配置，确切参数缺失。

7. **封存与回放检查不足。** Batch collector 没有复核提交时 request SHA、模型实际版本和所有输入源码；重复运行会覆盖下载和评分文件。Phase replay 只检查成功状态和各头是 list，不查 ID 类型/范围/去重/Phase 单选；合成的各头 `[999,999]` 会通过其 `require_labels`。它也没有检查输入是否与完整 manifest 一一对应，或在回放时强制 Training split。当前 scorer 会对类型/标签等作进一步检查，但不能把这个浅检查称为完整的原始回答重放。

8. **缺少可比较的成本与时间记录。** 同步版没有固定 thinking 配置，也没有我们当前每张图的 low/high 参数；Batch 转换还替换了 schema 方言。这些在同学自己的两组之间大体固定，却不等于当前 OpenRouter H0 协议。需单独记录推理配置、解析失败、实际返回模型、用量、调用时长和新增费用。

纯函数核验记录在解压目录的 `LOCAL_REVIEW_PROBES.json`；默认样本 mask 核验在 `LOCAL_DEFAULT_MASK_CHECK.json`。对这些默认视频只读取了任务可用性，未把 GT 标签用于争议复核实验的推理。

## 建议如何借鉴

本轮已先完成独立的“争议 IVT 一次视觉复核”试验，结果未改善，见 [新 24 目标实验记录](DISPUTED_RELATION_TRIAL_2026-09-09.md)。同学方案未混入其中，因此没有同时改变 H0 和接纳逻辑。

如果后续验证它，优先做一个**固定 H0 的盲补候选对照**：借用这段先查器械接触与同一关系的观察要求，让现有单模型补候选分支盲看三张原图，保留原候选上限、原五模型审核和原接纳规则。这样每目标仍是一份候选调用，不需要再增加五席轮次；同时可以直接测候选覆盖是否改善。该适配尚未实现或验证，不能把包内 H0 改写成绩等同于盲候选成绩。

另一种合法实验是完全照其科学问题，比较固定 H0 prompt 与 H0＋Audit 附加段，但必须作为独立 H0 研究分支，使用相同 OpenRouter 模型、原始三图、reasoning 和 JSON Schema，冻结新 Training 样本；不能覆盖朋友正在跑的冻结 H0。

**结论：值得借鉴其视觉观察要求，暂不直接合并整包；先补原始证据、修正上述接入边界，再独立验证。** 本次没有启动同学包的付费实验，也没有将其 README 成绩写入我们的已验证结果表。
