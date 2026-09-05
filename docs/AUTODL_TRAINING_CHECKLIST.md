# AutoDL Training Checklist

Upload the current repository first; train, collect or fit the items below one
at a time. A script being executable does not mean its scientific prerequisite
has passed.

## Ready now — no local training

- GPT-5.6 Sol joint perception through OpenRouter.
- Always Verify and the deterministic evidence-threshold rule Gate.
- Factorized candidate generation and joint API verification.
- Candidate-bounded deterministic KEEP/REPAIR coordination.
- Finalized-only workflow state and bounded event memory.
- Prediction/evidence writing and offline GT evaluation.
- Frozen `final_g1` Gate artifact loading at inference time.
- Finalized Workflow context, Training-only phase-graph loading, and executable
  `frames_only|track_only|workflow|track_workflow` context profiles.
- Strict predicted-track artifact loading, causal prompt serialization, real
  Faster R-CNN training, online Hungarian association, and artifact export.

The external GPT backbone and verifier are API inference components; this
project does not train them. Candidate generation, coordination, memory, and
the rule Gate are deterministic and also require no gradient training.

## Train or fit after upload, in this order

1. **Predicted tracker — ready.** Run `scripts/train_tracker.py --mode smoke`,
   then `--mode full`. Preserve its checkpoint, training manifest, Validation
   metrics, and `predicted_tracks.json`. Run `--mode oof` before D0/D1 so every
   Training video uses a checkpoint that excluded it.
2. **Local Joint Perception — ready for AutoDL training.**
   `scripts/train_perception.py` now trains one causal three-frame model with
   Instrument/Verb/Target/IVT multi-label heads and a Phase single-label head.
   It trains only on Training, calibrates thresholds on Validation, never opens
   Testing, and exports a loadable checkpoint plus validation metrics.
3. **Workflow transition artifact — deterministic fit.** Build the train-only
   phase transition graph with `scripts/build_phase_transition_graph.py`.
4. **Surgical priors — deterministic fit.** Build train-only IVT/workflow
   priors after their exact semantics are frozen. The current
   `scripts/build_surgical_priors.py` is blocked.
5. **Gate dataset D0 — API data-generation code ready.** Run
   `scripts/collect_formal_gate_counterfactuals.py` on Training with the
   five-fold Tracker OOF index. It calls JointPerception once and every legal
   Verify/Repair scope, joins GT only afterward, and is cache/resume safe. The
   three independent same-snapshot scopes run concurrently, so the provider
   account must permit up to three in-flight verification requests per frame;
   causal frames within one video remain sequential.
   `--allow-api-failures` is restricted to coverage/debug pilots: it records
   terminal provider rejections and continues, but must not be used to silently
   construct the formal Gate training dataset.
   The active Repair-development path is now the conservative same-model V7
   contract: `--api-config` supplies Joint H0 and `--verification-api-config`
   supplies the same GPT model with
   `targeted_openrouter_gpt56sol_constrained_fixed3.yaml`. Interaction verification
   selects IVT only and derives I/V/T deterministically; Verified cannot coexist
   with uncertainty; and hard-valid H0 cannot be replaced automatically. The
   two-frame VID13 endpoint probe on 2026-09-04 had 6/6 successful fresh Verifier
   calls and no accepted Repair. It prevented harm, but it also blocked a correct
   phase 0→1 proposal on VID13:1 because H0 was structurally valid. Therefore V7
   is a safety baseline, not evidence of positive Repair capability. GPT+Grok,
   Gemini, and Claude remain historical capability probes rather than active
   defaults. Do not create D0 until a separately frozen admission rule shows
   positive rescue with controlled harm on a predeclared Training-only probe.
6. **Bootstrap Gate G0 — code ready after D0 passes.** Build paired examples
   with `scripts/build_gate_oof_dataset.py`, then run `scripts/train_gate.py`.
   It emits one video-cross-fitted bootstrap artifact per fold; these artifacts
   are non-deployable and may be used only for the D1 Training rollout.
7. **Policy-matched rollout D1 — data generation.** Run G0 on Training folds and
   collect every encountered state plus realized bounded-verification benefit.
8. **Final Gate G1 — training.** Train on D1 and export a strict
   `benefit_gate_linear_v1` artifact with `gate_stage=final_g1`,
   `source_split=training`, raw evidence-frame-v1 features, provenance, and the
   exact rollout recipe.
9. **Operating threshold — validation calibration.** Select and freeze the Gate
   threshold on Validation only. Never refresh, retrain, or calibrate on Test.
10. **Optional reliability calibration.** Fit only if reliability-weighted
   retrieval remains in the final method; it is not needed by the current
   bounded event-memory runtime.

## Local Joint Perception commands

Run smoke first after uploading the new code:

```bash
python scripts/train_perception.py --mode smoke \
  --dataset-root "$CHOLECTRACK20_ROOT" --device cuda
```

Only after smoke passes, start the full run:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/train_perception.py --mode full \
  --dataset-root "$CHOLECTRACK20_ROOT" \
  --config configs/perception/local_joint_mobilenet_v3.yaml \
  --device cuda
```

The full output is written below `artifacts/training/perception/full/`.

## Upload-ready environment check

```bash
python -m pip install -r requirements-autodl.txt
python -m pip install -e . --no-deps
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m ruff check .
python -m pytest -q
python scripts/verify_autodl_bundle.py --dataset-root "$CHOLECTRACK20_ROOT"
```

The AutoDL image should supply its CUDA-matched PyTorch. The requirements file
therefore intentionally does not reinstall `torch`.

## Current runnable inference profiles

- `scripts/run_dataset_api_pipeline.py --pipeline-profile single_pass`: one API
  call per frame and no verification.
- `scripts/run_always_verify.py`: up to two API calls per frame, including one
  joint verification call.
- `scripts/infer_v3.py`: rule Gate by default; add
  `--gate-artifact /path/to/final_g1.json` after G1 is trained to activate the
  learned Gate.

All current rollout artifacts remain `paper_metric_eligible=false` until the
full experimental protocol and required training artifacts are frozen.

## Testing evaluation rule

The local Synapse Testing bundle does contain GT, but supervision is
task-dependent. Instrument, Phase, bounding boxes, and track IDs cover all
eight Testing videos. Verb, Target, and IVT are only partially available,
primarily in VID06, VID25, VID92, and VID111.

- Score a task only on frames for which every annotated instance has a valid
  label for that task.
- Ignore `-1` and partially labelled frames for that task. Do not convert them
  to negative labels or include them in the metric denominator.
- Run Test evaluation only after Training and Validation decisions are frozen,
  using a complete eight-video paper-mode rollout and
  `--authorize-test-gt-evaluation`.
- Keep Testing GT strictly outside training, Gate datasets, calibration,
  prompts, causal state, and API requests.
- Report Verb/Target/IVT as partial-GT subset results; do not describe them as
  complete Testing-set metrics.
