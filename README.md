# Streaming SurgicalAgent: Phase-aware five-head probe Gate

The default pipeline uses `five-head-probe-gate-20260916` (54 features including Phase probe evidence).

H0 -> prior-guided proposal -> Qwen probe -> Gate -> one conditional five-head
verification panel -> repair -> Tracker M1/M2 and v2.2 label cleanup -> causal
phase smoothing -> H0/FULL reports and Rule v2 / Judge v2-low evaluation.

The post-Gate panel shares responses across instrument, verb, target, IVT and
phase. It stops only when both interaction and phase decisions are settled.
The five-head Qwen probe runs before the Gate and is reused by the panel. RAM is disabled.

Git contains source code; the current Gate and Tracker inference weights are
available separately as [GitHub Release assets](https://github.com/litianhao738/streaming-surgical-agent/releases/tag/tracker-phase-gate-weights-20260916).
See [weight download and restoration](docs/MODEL_WEIGHTS.md).
Datasets, images, H0/OOF caches, experiment outputs and credentials remain local.
A fresh clone still requires the relevant inputs and credentials before inference.

- [Source-only setup and entry points](docs/SOURCE_ONLY_PIPELINE.md)
- [New Gate commands, Gemini input provenance and validation limits](docs/NEW_GATE_PIPELINE_2026-09-16.md)
- [Default pipeline configuration](DEFAULT_PIPELINE_VERSION.json)
- [Default Gate configuration](DEFAULT_PGP_GATE_VERSION.json)

Install dependencies with `python -m pip install -r requirements.txt` using a
supported Python 3.10-3.12 environment. GPU execution requires a matching CUDA
PyTorch/torchvision installation. Display the selected pipeline without loading
private artifacts with `python scripts/run_pipeline.py info`.

Use a new experiment directory when switching Gate versions. Never resume old
predictions under a different verification policy. Independent Testing quality
or wall-time improvements are not implied by this code release.

## Gate routing baselines

Random, Uncertainty-only and Rule-based Gate source, formulas and pseudocode are available in [mechanism_internal_ablation](mechanism_internal_ablation/README.md). These fixed-threshold baselines share the existing pipeline evidence; generated data and synthetic score tables are not part of this source release.
