# Qwen OpenAI-Compatible Engineering Run

The Qwen gateway is an engineering substitute for API testing. It uses the
same compact joint prediction prompt and runtime JSON validator, but it does
not change the frozen paper backbone automatically. No API key is stored in
this repository.

From Windows CMD, provide the model, gateway, and temporary key at runtime:

```bat
cd /d D:\PythonProject7
set "CHOLECTRACK20_ROOT=D:\cholec_dataset"
set "RUN_ID=qwen_probe_001"
python scripts\run_dataset_api_pipeline.py ^
  --mode engineering ^
  --video-id VID30 ^
  --max-frames 1 ^
  --pipeline-profile single_pass ^
  --dataset-root "%CHOLECTRACK20_ROOT%" ^
  --config configs\perception\joint_qwen_compatible_dataset.yaml ^
  --model "qwen3.8-max" ^
  --base-url "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" ^
  --api-key "PASTE_TEMPORARY_KEY_HERE" ^
  --output-root artifacts\api_dataset ^
  --cache-root artifacts\api_dataset_cache\qwen_compatible ^
  --run-id "%RUN_ID%" ^
  --max-provider-calls exact-selection ^
  --authorize-data-upload
```

`--base-url` accepts either the base ending in `/v1` or the full endpoint
ending in `/chat/completions`. The CLI normalizes the former. Start with one
frame; increase `--max-frames` only after the response schema is confirmed.
