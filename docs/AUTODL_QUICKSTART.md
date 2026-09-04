# AutoDL Quickstart

The repository now contains the portable DatasetAdapter, canonical pipeline,
writer, evaluator, OpenRouter `openai/gpt-5.6-sol` dataset rollout, Always
Verify, rule Gate, candidate-bounded joint verification, deterministic
coordination, finalized-only state stores, and a strict learned-Gate artifact
loader. Tracker training/export is implemented; trained Tracker/Gate artifacts
are still generated after upload. See
[`AUTODL_TRAINING_CHECKLIST.md`](AUTODL_TRAINING_CHECKLIST.md) for the exact
post-upload order.

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
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
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
engineering evidence only, and current API rollout artifacts remain explicitly
paper-ineligible until the full training/experimental protocol is frozen.

## Train and Export the Predicted Tracker

First prove the entire GPU path with one bounded optimizer batch and two
Validation frames:

```bash
python scripts/train_tracker.py \
  --mode smoke \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --device cuda \
  --max-train-batches 1 \
  --max-prediction-frames 2
```

Then train the paper support model. This uses only supervision-qualified
Training boxes, excludes VID31's disabled instance supervision, and exports
predicted tracks for Validation, Testing, and VID31:

```bash
python scripts/train_tracker.py \
  --mode full \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --device cuda

cat artifacts/training/tracker/full/validation_detection_metrics.json
```

Long tracker stages display dynamic `tqdm` bars for data preparation,
optimizer batches, prediction frames, and Validation GT loading. The current
epoch and loss stay on the same terminal line. Add `--no-progress` only for
non-interactive background logging.

Keep the full checkpoint. Before building the second module's D0/D1 Gate data,
generate five video-level out-of-fold Training artifacts. The OOF output root
must contain the full checkpoint because VID31 is predicted by that checkpoint
without entering any Tracker optimizer:

```bash
mkdir -p artifacts/training/tracker_oof5
cp -a artifacts/training/tracker/full artifacts/training/tracker_oof5/

python scripts/train_tracker.py \
  --mode oof \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml \
  --output-root artifacts/training/tracker_oof5 \
  --device cuda
```

The runtime artifact used by Validation/Test is
`artifacts/training/tracker/predicted_tracks.json`. OOF files remain separate
under `artifacts/training/tracker_oof5/oof/`; never replace them with in-sample
Training predictions when constructing Gate data. After transferring the OOF
bundle back beside the local dataset, run the CPU-only held-out audit and
detection evaluation before collecting Gate counterfactuals:

```powershell
.\.venv-p2\Scripts\python.exe scripts\evaluate_tracker_oof.py `
  --dataset-root D:\cholec_dataset `
  --tracker-oof-index artifacts\training\tracker_oof5\oof\index.json `
  --tracker-config configs\tracker\fasterrcnn_mobilenet_v3_5090_oof5.yaml
```

The command fails closed on fold leakage, config/repair/checkpoint hash drift,
or incomplete frame coverage. It scores the nine videos with valid instance
boxes and records VID31 coverage without treating its frame-level labels as
instance-level bounding-box ground truth.

## Optional Real P3 Smoke

Provide the OpenRouter credential separately in an ignored local file. The
current local bundle format places the raw `sk-or-v1-...` credential on its
first non-empty line; never paste the key directly into shared shell history.

```bash
python scripts/run_api_single_pass.py \
  --config configs/perception/joint_openrouter_gate_owned_smoke.yaml \
  --real \
  --api-key-file docs/API.txt
```

This smoke uploads only three generated 32x32 RGB frames, validates the current
`joint_perception_gate_owned_compact_v1` response, and immediately replays the
identical request from cache. It does not upload any CholecTrack20 frame. The
legacy `scripts/smoke_api.py` transport probe remains available as historical
P3 coverage.

## CholecTrack20 Dataset API Rollout

Start with the non-paid mock transport over one real local frame. This decodes
CholecTrack20 media locally, writes the paired prediction/evidence artifacts,
and exercises the persistent cache without making a network call:

```bash
python scripts/run_dataset_api_pipeline.py \
  --mode engineering \
  --video-id VID30 \
  --max-frames 1 \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/perception/joint_mock_dataset.yaml \
  --output-root artifacts/api_dataset \
  --cache-root artifacts/api_dataset_cache/mock_validation \
  --run-id local_real_frame_mock_001 \
  --max-provider-calls exact-selection \
  --authorize-data-upload
```

Only after that command and its artifacts pass inspection, opt in to one real
OpenRouter call. The configuration authorization and the literal CLI flag are
both required:

The dataset configurations use `joint_perception_gate_owned_compact_v1`: the provider
returns 3/4/5/8/3 ranked candidates, while local parsing restores the full
7/10/15/100/7 evaluator vector shapes by zero-filling omitted IDs.

```bash
python scripts/run_dataset_api_pipeline.py \
  --mode engineering \
  --video-id VID30 \
  --max-frames 1 \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/perception/joint_openrouter_dataset.yaml \
  --api-key-file docs/API.txt \
  --output-root artifacts/api_dataset \
  --cache-root artifacts/api_dataset_cache/openrouter_validation \
  --run-id openrouter_real_frame_001 \
  --max-provider-calls exact-selection \
  --authorize-data-upload
```

Inspect `api_usage.jsonl` and `dataset_rollout_artifact.json`, including the
reported token usage, provider cost, and `verification_summary`, before raising
the frame limit. Every run needs a fresh run ID; the cache directory may be
reused. Paper mode must run complete Validation or Test videos and therefore
forbids `--max-frames`.

Dataset API rollout, `infer_v3.py`, and `run_always_verify.py` share one
frame-level `API inference` progress bar. It shows the current video/frame on
one dynamically refreshed terminal line; add `--no-progress` to disable it.

For the causal no-training V3 runtime, use `scripts/infer_v3.py` with the same
selection, dataset, config, credential, output, cache, and authorization
arguments. It selects the rule Gate by default. After training G1, add
`--gate-artifact /path/to/final_g1.json` to activate the learned Gate. Use
`scripts/run_always_verify.py` for the fixed Always-Verify ablation; it budgets
up to two calls per frame.

The first context contribution is selected independently with an executable
experiment YAML (or the equivalent explicit CLI switches). Use the
`workflow_only` config with the Training-only graph built on this instance:

```bash
python scripts/infer_v3.py \
  --mode engineering \
  --video-id VID30 \
  --max-frames 1 \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/perception/joint_mock_dataset.yaml \
  --experiment-config configs/ablations/workflow_only.yaml \
  --output-root artifacts/api_dataset \
  --cache-root artifacts/api_dataset_cache/workflow_mock \
  --run-id workflow_mock_001 \
  --max-provider-calls exact-selection \
  --authorize-data-upload
```

`configs/experiments/v3_track_workflow.yaml` additionally requires the strict
artifact at `artifacts/training/tracker/predicted_tracks.json`; it rejects unknown or
annotation/GT fields and cannot be used until controlled causal tracker
inference has produced that artifact. See `docs/TRACK_WORKFLOW_CONTEXT.md`.

## First-module four-way ablation

Run all four settings on the complete Validation split with the same API model,
selection, evaluator, and single-pass policy. The four YAML files change only
the admitted Track/Workflow context:

```bash
for experiment in \
  configs/ablations/no_workflow.yaml \
  configs/ablations/track_only.yaml \
  configs/ablations/workflow_only.yaml \
  configs/ablations/no_event_memory.yaml
do
  name=$(basename "$experiment" .yaml)
  python scripts/run_dataset_api_pipeline.py \
    --mode paper \
    --split validation \
    --pipeline-profile single_pass \
    --dataset-root "$CHOLECTRACK20_ROOT" \
    --config configs/perception/joint_openrouter_dataset.yaml \
    --experiment-config "$experiment" \
    --api-key-file docs/API.txt \
    --output-root artifacts/context_ablation \
    --cache-root artifacts/context_ablation_cache \
    --run-id "validation_${name}" \
    --max-provider-calls exact-selection \
    --authorize-data-upload
  python scripts/evaluate.py \
    --run-dir "artifacts/context_ablation/validation_${name}" \
    --dataset-root "$CHOLECTRACK20_ROOT"
done
```

The comparison rows are Frames, Frames+Track, Frames+Workflow, and
Frames+Track+Workflow. `no_event_memory.yaml` is intentionally used for the
fourth row so the first-module table does not accidentally attribute Event
Memory effects to Track/Workflow context.

## Offline Frame Evaluation

After a rollout reaches `COMPLETE`, align its predictions with local GT and
calculate Instrument/Verb/Target/IVT video-wise mAP plus Phase Accuracy and
Macro-F1:

```bash
python scripts/evaluate.py \
  --run-dir artifacts/api_dataset/<run-id> \
  --dataset-root "$CHOLECTRACK20_ROOT"
```

The report is written beside the run as `<run-id>__evaluation`. Use a fresh
`--output-dir` to override that location. Test evaluation additionally needs
`--authorize-test-gt-evaluation`; engineering/truncated Test runs are rejected.
Current rollout-v1 reports, including Validation results using the candidate
VID30 repair, remain explicitly paper-ineligible even though their engineering
metrics are usable.

### Testing GT coverage and missing-label policy

The audited local Synapse release contains JSON GT for all eight Testing
videos. Its usable supervision is not identical across tasks:

| Task or field | Usable Testing coverage | Reporting rule |
| --- | ---: | --- |
| Instrument | 15,282 fully supervised frames | Report multi-label metrics on all eight videos |
| Phase | 15,282 fully supervised frames | Report Accuracy and Macro-F1 on all eight videos |
| Bounding box and track ID | 29,994 annotated instances | May support a separate detection/tracking evaluation |
| Verb and Target | 6,631 fully supervised frames | Report only the labelled-frame subset |
| IVT/Triplet | 5,953 fully supervised frames | Report only the labelled-frame subset |

Verb, Target, and IVT supervision is concentrated in VID06, VID25, VID92, and
VID111. A value of `-1` means that task is unavailable for that instance. If
any annotated instance in a frame lacks a task label, the existing evaluator
masks the entire frame for that task. Missing or partial labels are therefore
ignored, not treated as negative classes, and do not enter that task's metric
denominator.

After a complete paper-mode rollout of all eight Testing videos, explicitly
authorize offline GT access:

```bash
python scripts/evaluate.py \
  --run-dir artifacts/api_dataset/<testing-run-id> \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --authorize-test-gt-evaluation
```

This authorization applies only to post-hoc offline scoring. Testing labels
must not enter training, threshold calibration, candidate generation, prompts,
causal memory, or API requests. In the paper, label Verb/Target/IVT results as
partial-GT subset results rather than full eight-video Testing metrics.
