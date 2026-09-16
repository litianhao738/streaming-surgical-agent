# Qwen 候选提案续跑

用户要求暂停原实验并把后续提案切换到基座 Qwen Max，同时保留已完成结果。
原目录：`artifacts/experiments/qwen38_half_probe_complete_20260916_r2`。
续跑目录：`artifacts/experiments/qwen38_half_probe_qwen_proposer_continue_20260916_r3`。

暂停采用暂时禁止新预算预留、等待在途请求完成的方式；未强杀网络请求。
原实验记录保留，续跑继承已完成的帧、费用和请求日志。
已发送过 Gemini 提案的目标继续复用原提案；其余目标使用 `qwen3.8-max` 生成提案。
结果逐帧记录 `proposal_model`，计划保留 `legacy_proposal_targets` 和原计划哈希。
这仍是 Qwen H0 实验，但提案来源是混合版本，不能作为全量纯 Qwen 提案成绩发表。

新建立的 Qwen pipeline 计划默认由 Qwen 同时负责 H0 和提案；评审和 Report/Judge 路由不变。
复用 Gemini 拟合的新 54 维 Gate，未利用 Qwen Testing 标签重训。

阿里云保留原 ¥200 上限，除非用户明确选择提高。Qwen Max 提案会显著增加阿里云消耗，原上限预计无法覆盖全程。

查看进度：

```powershell
.venv-tracker-gpu/Scripts/python.exe -X utf8 scripts/run_pipeline.py testing-status --output artifacts/experiments/qwen38_half_probe_qwen_proposer_continue_20260916_r3
```

只在进程停止且明确处理失败原因后使用 resume；不要对运行中的目录重复启动。
