# AutoDL Quickstart

P2 is the recommended first upload checkpoint. The repository now contains the
portable DatasetAdapter, local model, masked loss, checkpoint path, canonical
pipeline, writer, evaluator, tests, and smoke entry point. P3 and later research
modules are intentionally not implemented yet.

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

## Verify Before Training

```bash
python -m ruff check .
python -m compileall -q src tools tests
python -m pytest -q
python scripts/run_local_smoke.py \
  --config configs/experiments/local_smoke.yaml \
  --device cuda
```

Do not start a long training run unless all four commands pass. P2 artifacts are
engineering evidence only. The exact API identity is still a P3 blocker, and
paper evaluation granularity must be frozen from the P2 audit before P4.
