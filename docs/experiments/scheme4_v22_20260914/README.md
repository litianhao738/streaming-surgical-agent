# Current scheme 4 v2.2 evidence

Default: original proposer prompt, fixed scheme-4 Gate, Tracker M1–M3, v2.2 output cleanup.

- `pipeline_*`: 6,059 Training rows, full-fit F1 67.0209, 30,774 errors; not independent validation.
- `ambiguity_*`: four zero-call research alternatives, all rejected; per-frame predictions and stage traces contain no GT.
- `two_target_v21_*`: historical original two-target live run. The v2.2 post-processing calculation is separately reported in `ambiguity_recall_report.json`.
- `manifest.json`: exact source paths, statuses, sizes and byte hashes. Receipts preserve their historical filenames/hashes; map them to published names through the manifest.

Prompt experiments, raw images, provider request/response logs, credentials and per-frame GT are excluded. Original v2.1 files remain in the sibling `scheme4_pipeline_20260914` directory as historical evidence.
