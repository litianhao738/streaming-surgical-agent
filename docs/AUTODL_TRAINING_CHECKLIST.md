# AutoDL Training Checklist

Upload the current repository first; train or fit the items below one at a time
afterward. Do not run the blocked training scripts as if they were finished.

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
2. **Workflow transition artifact — deterministic fit.** Build the train-only
   phase transition graph with `scripts/build_phase_transition_graph.py`.
3. **Surgical priors — deterministic fit.** Build train-only IVT/workflow
   priors after their exact semantics are frozen. The current
   `scripts/build_surgical_priors.py` is blocked.
4. **Gate dataset D0 — data generation.** Produce out-of-fold, GT-aligned
   state/action benefit examples from Training only. Validation and Test must
   not enter D0. The current `scripts/build_gate_oof_dataset.py` is blocked.
5. **Bootstrap Gate G0 — training.** Train the first small benefit predictor on
   D0. G0 is a rollout policy, not a deployable final Gate.
6. **Policy-matched rollout D1 — data generation.** Run G0 on Training folds and
   collect every encountered state plus realized bounded-verification benefit.
7. **Final Gate G1 — training.** Train on D1 and export a strict
   `benefit_gate_linear_v1` artifact with `gate_stage=final_g1`,
   `source_split=training`, raw evidence-frame-v1 features, provenance, and the
   exact rollout recipe.
8. **Operating threshold — validation calibration.** Select and freeze the Gate
   threshold on Validation only. Never refresh, retrain, or calibrate on Test.
9. **Optional reliability calibration.** Fit only if reliability-weighted
   retrieval remains in the final method; it is not needed by the current
   bounded event-memory runtime.

## Not a paper model

`scripts/train_perception.py` currently aliases the one-step P2 local smoke. It
checks CUDA, masking, checkpoint save/load, and data plumbing, but it is not the
main GPT perception model and must not be reported as a trained paper baseline.

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
