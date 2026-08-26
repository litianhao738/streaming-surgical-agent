# AutoDL Quickstart

P2 remains the recommended first upload checkpoint. The repository contains the
portable DatasetAdapter, local model, masked loss, checkpoint path, canonical
pipeline, writer, evaluator, tests, and smoke entry point. The OpenRouter P3 API
transport is also implemented and has completed a real synthetic-image smoke
with `openai/gpt-5.6-sol`; P4 and later research modules remain phase-gated.

## Upload

Upload the repository and the complete external CholecTrack20 directory. The
dataset directory must still contain the official 10/2/8 split plus:

- `repair_manifest.json`
- `Validation/VID30/vid30_repaired.json`
- `Training/VID31/vid31_phase_repaired.json`
- `Training/VID31/vid31_frame_ivt_repaired.json`

The adapter checks these files and their SHA-256 values before sampling.

## Environment

Choose an AutoDL image with Python 3.10 or 3.11 and a working CUDA PyTorch. Keep
the image's matching PyTorch build, then install the remaining dependencies:

```bash
python -m pip install -r requirements-autodl.txt
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Set the external data path at runtime:

```bash
export CHOLECTRACK20_ROOT=/root/autodl-tmp/cholec_dataset
```

Do not upload separate Cholec80 or CholecT50 roots for normal training,
validation, or testing. They are needed only when independently auditing or
regenerating the provenance-bearing sidecars already contained in the complete
CholecTrack20 root.

## Verify Before Training

```bash
python -m ruff check .
python -m compileall -q src tools tests
python -m pytest -q
python scripts/verify_autodl_bundle.py \
  --dataset-root "$CHOLECTRACK20_ROOT"
python scripts/run_local_smoke.py \
  --config configs/experiments/local_smoke.yaml \
  --device cuda
```

Do not start a long training run unless all commands pass. P2 artifacts are
engineering evidence only. The real OpenRouter P3 transport is verified, while
exact immutable backend identity remains partial. Paper evaluation granularity
must still be frozen from the P2 audit before P4.

## Optional Real P3 Smoke

Provide the OpenRouter credential separately in an ignored local file. The
current local bundle format places the raw `sk-or-v1-...` credential on its
first non-empty line; never paste the key directly into shared shell history.

```bash
python scripts/smoke_api.py \
  --config configs/api/openrouter.yaml \
  --real \
  --api-key-file docs/API.txt
```

This smoke uploads only the generated synthetic blue square. It does not upload
any CholecTrack20 frame.
