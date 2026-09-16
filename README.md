# Streaming SurgicalAgent: Phase-aware five-head probe Gate

The default pipeline uses `five-head-probe-gate-20260916` (54 features including Phase probe evidence).

H0 -> prior-guided proposal -> Qwen probe -> Gate -> one conditional five-head
verification panel -> repair -> Tracker M1/M2 and v2.2 label cleanup -> causal
phase smoothing -> H0/FULL reports and Rule v2 / Judge v2-low evaluation.

The post-Gate panel shares responses across instrument, verb, target, IVT and
phase. It stops only when both interaction and phase decisions are settled.
The five-head Qwen probe runs before the Gate and is reused by the panel. RAM is disabled.

This is a **source-only repository**. Datasets, images, H0 caches, trained Gate
and Tracker weights, experiment outputs, reports and credentials are not
distributed. Default manifests reference local artifacts; a fresh clone cannot
run paid inference until those private prerequisites have been supplied.

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
