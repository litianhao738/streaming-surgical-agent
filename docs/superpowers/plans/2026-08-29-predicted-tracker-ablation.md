# Predicted Tracker and Context Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a provenance-recorded CholecTrack20 instrument detector, export causal predicted tracks, and run frames/track/workflow/track+workflow ablations.

**Architecture:** A Torchvision Faster R-CNN MobileNetV3-FPN detector consumes only supervision-qualified Training boxes. A deterministic online Hungarian associator turns per-frame detections into local tracks, and an artifact writer emits the existing strict runtime schema. Runtime configuration gains a `track_only` profile without changing the prediction or evaluation contracts.

**Tech Stack:** Python 3.10–3.12, PyTorch/Torchvision, SciPy, Pillow, PyYAML, Pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-predicted-tracker-ablation-design.md`

## Global Constraints

- CholecTrack20 instrument IDs `0..6` map to detector labels `1..7`; label `0` is background.
- Runtime tracks are predicted, online-forward-only, and reset at video boundaries.
- VID31 is predicted and used by downstream rollout, but its disabled instance/bbox/track fields never supervise the detector.
- Validation and Testing never enter optimizer updates.
- OOF Training predictions come only from checkpoints that excluded the predicted video.
- OracleTrack is diagnostic-only.

---

### Task 1: Training samples and detector configuration

**Files:**
- Create: `src/surgical_agent/tracking/training_data.py`
- Create: `src/surgical_agent/tracking/config.py`
- Create: `configs/tracker/fasterrcnn_mobilenet_v3.yaml`
- Test: `tests/unit/test_tracker_training_data.py`

**Interfaces:**
- Consumes: `CholecTrack20DatasetAdapter`, canonical instance targets, and repair-manifest field decisions.
- Produces: `TrackerTrainingConfig`, `DetectionTrainingSample`, `InstrumentDetectionDataset`, `instrument_to_model_label()`, and `model_to_instrument_label()`.

- [ ] **Step 1: Write failing tests** for the 0..6 ↔ 1..7 mapping, pixel `xyxy` conversion, rejection of disabled VID31 instance supervision, and deterministic video selection.
- [ ] **Step 2: Run** `pytest tests/unit/test_tracker_training_data.py -q` and confirm failures are caused by missing modules.
- [ ] **Step 3: Implement** strict YAML parsing and a read-only dataset that returns `image: Tensor` and Torchvision targets `{boxes, labels, image_id, area, iscrowd}`.
- [ ] **Step 4: Re-run** the focused tests and confirm they pass.

### Task 2: Online association and artifact export

**Files:**
- Modify: `src/surgical_agent/tracking/associator.py`
- Create: `src/surgical_agent/tracking/artifact_writer.py`
- Modify: `src/surgical_agent/tracking/__init__.py`
- Test: `tests/unit/test_predicted_tracker_runtime.py`

**Interfaces:**
- Consumes: normalized detections `(instrument_id, bbox_tlwh, score)` and checkpoint/config/repair hashes.
- Produces: `CausalHungarianAssociator.update(frame_id, detections) -> tuple[PredictedTrack, ...]` and `write_predicted_track_artifact(...) -> Path`.

- [ ] **Step 1: Write failing tests** proving same-class IoU matching preserves an ID, cross-class detections do not match, state expires, frame order is strict, and reset prevents cross-video IDs.
- [ ] **Step 2: Run** `pytest tests/unit/test_predicted_tracker_runtime.py -q` and verify expected failures.
- [ ] **Step 3: Implement** Hungarian matching with forbidden cross-class costs, bounded age, monotonic IDs, atomic canonical JSON writing, and immediate provider reload validation.
- [ ] **Step 4: Re-run** the focused tests and confirm they pass.

### Task 3: Detector, training, evaluation, and export CLI

**Files:**
- Modify: `src/surgical_agent/tracking/detector.py`
- Replace: `scripts/train_tracker.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_tracker_detector.py`
- Test: `tests/integration/test_tracker_training_cli.py`

**Interfaces:**
- Consumes: `TrackerTrainingConfig`, qualified samples, output video IDs, `--mode smoke|full|oof`, dataset root, and device.
- Produces: checkpoint, training manifest, Validation AP@0.50/precision/recall/F1, full predicted-track artifact, and per-fold OOF artifacts.

- [ ] **Step 1: Write failing tests** for model-label decoding, AP@0.50 calculation, checkpoint provenance, split-safe selection, and a synthetic no-download smoke export.
- [ ] **Step 2: Run** the focused tests and verify expected failures.
- [ ] **Step 3: Implement** the Faster R-CNN factory, optimizer loop, checkpoint save/load, deterministic three-fold assignment, Validation evaluation, and online artifact export.
- [ ] **Step 4: Re-run** focused tests and confirm they pass.

### Task 4: Independent track-only context ablation

**Files:**
- Modify: `src/surgical_agent/config/context_experiment.py`
- Modify: `src/surgical_agent/systems/api_dataset_system.py`
- Modify: `scripts/run_dataset_api_pipeline.py`
- Create: `configs/ablations/track_only.yaml`
- Modify: `tests/unit/test_context_experiment_config.py`
- Modify: `tests/integration/test_api_dataset_cli.py`
- Modify: `tests/integration/test_api_dataset_pipeline.py`

**Interfaces:**
- Consumes: `context_profile=track_only` plus one predicted-track artifact.
- Produces: a pipeline with predicted tracks enabled, workflow disabled, phase graph absent, event memory disabled.

- [ ] **Step 1: Add failing tests** for accepted `track_only`, required artifact, forbidden phase graph, and a no-op workflow store.
- [ ] **Step 2: Run** the focused context tests and verify expected failures.
- [ ] **Step 3: Implement** profile parsing, CLI validation, track-provider activation, workflow-store selection, and the YAML config.
- [ ] **Step 4: Re-run** the focused tests and confirm all pass.

### Task 5: AutoDL documentation, bundle, and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/AUTODL_QUICKSTART.md`
- Modify: `docs/AUTODL_TRAINING_CHECKLIST.md`
- Modify: `requirements-autodl.txt`
- Modify: `scripts/verify_autodl_bundle.py`

**Interfaces:**
- Consumes: completed training CLI and four experiment profiles.
- Produces: exact AutoDL commands and a source-only ZIP that excludes data, secrets, checkpoints, caches, and runtime outputs.

- [ ] **Step 1: Add or update documentation-contract tests** for the full/smoke/OOF commands and all four ablations.
- [ ] **Step 2: Run focused documentation and bundle tests** and verify failures precede edits.
- [ ] **Step 3: Document commands**, verify Torchvision availability, and update the deployment verifier.
- [ ] **Step 4: Run** Ruff, compilation, full Pytest, bundle verification, and secret scan.
- [ ] **Step 5: Build** `D:\PythonProject7_autodl_code_20260829_tracker.zip`, inspect its contents, and report its SHA-256.
