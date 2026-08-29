# Predicted Tracker and Context Ablation Design

## Purpose

Complete the trainable part of the Track–Workflow Structured Causal Context
module so that AutoDL can train a predicted instrument detector, export strictly
causal local tracks, and run a fair context ablation. The detector/tracker is an
auditable support component; the paper contribution remains the use of predicted
track and historical workflow evidence in the streaming agent.

## Frozen research choices

- Detector: Torchvision
  `fasterrcnn_mobilenet_v3_large_fpn` initialized from
  `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`.
- Output classes: model label `0` is background; CholecTrack20 instrument IDs
  `0..6` map bijectively to model labels `1..7`.
- Association: deterministic online Hungarian matching using same-instrument
  IoU cost, followed by bounded unmatched-track aging. Association consumes only
  the current detector result and state committed at earlier frames.
- Main training source: CholecTrack20 official Training videos whose field-level
  audit permits instrument, bounding-box, and track supervision.
- VID31 participates in prediction export and later causal rollout. Its audited
  frame-level I/V/T/IVT and Phase labels remain available to their existing
  downstream branches, but its disabled instance/bbox/track fields are not used
  as detector or association supervision.
- Oracle tracks remain diagnostic-only and are never accepted by the main
  `track_only` or `track_workflow` profiles.
- No API call is needed to train or run the tracker.

## Training and fold safety

The training entry point supports three explicit modes:

1. `smoke`: run a bounded optimizer/inference/export cycle for deployment
   validation. It is not a paper result.
2. `full`: train one detector on every supervision-qualified Training video.
   Use this checkpoint to predict Validation, Testing, and VID31. Validation and
   Testing are never used by the optimizer or threshold selection inside this
   command.
3. `oof`: create three deterministic video-level folds over the
   supervision-qualified Training videos. Each fold checkpoint predicts only
   its held-out Training videos. VID31 is predicted by the full checkpoint,
   which has never consumed VID31 instance supervision. These per-fold artifacts
   are the only track inputs permitted when later building D0/D1 Gate data from
   Training videos.

Each artifact contains predictions from exactly one checkpoint so the existing
`predicted_track_context_v1.checkpoint_sha256` field remains truthful. OOF
artifacts are kept separate and are combined only at the later Gate-data level.

## Components

### Detector training dataset

A focused tracking dataset adapter reuses the canonical CholecTrack20 parser and
the existing repair/supervision manifest. It returns the current RGB frame plus
pixel-space `xyxy` boxes and labels. It drops invalid or disabled instances,
never mutates the dataset root, and records the exact selected video IDs.

### Detector

The detector module builds the frozen Faster R-CNN architecture, replaces its
classification head with eight outputs, performs a standard Torchvision
detection loss update, and converts inference outputs back to normalized TLWH.
Score threshold, NMS threshold, seed, image resizing, optimizer, and epoch count
come from one versioned YAML configuration and are copied into the training
manifest.

### Causal associator

The associator is reset at each video boundary. At frame `t`, predictions are
matched only to active state from frames `<t`. Matching is class constrained and
uses Hungarian assignment over `1 - IoU`; matches below the configured IoU
threshold are rejected. Unmatched detections receive monotonic local IDs, while
unmatched tracks expire after `max_age`. Exported `age` counts observed causal
updates and never uses an official GT track ID.

### Artifact writer

The writer emits the already enforced `predicted_track_context_v1` schema with:

- provider and source-model identifier;
- checkpoint SHA-256;
- `causal=true` and `inference_mode=online_forward_only`;
- producer version;
- repair-manifest and inference-config SHA-256 values;
- per-video split, increasing frame IDs, normalized boxes, detector score, and
  local predicted track age.

The completed artifact is immediately reloaded through
`PrecomputedPredictedTrackProvider`; an artifact that fails the runtime contract
is not reported as complete.

### Context profiles

The runtime exposes four independent context settings:

| Profile | Predicted track | Historical workflow | Event memory |
|---|---:|---:|---:|
| `frames_only` | off | off | off |
| `track_only` | on | off | off |
| `workflow` | off | on | off |
| `track_workflow` | on | on | configurable |

`track_only` requires a predicted-track artifact and forbids a phase-transition
graph. `workflow` requires the train-derived phase-transition graph and forbids
a predicted-track artifact. `track_workflow` requires both. All four use the
same API backbone, causal visual window, output schema, and evaluator.

## Outputs

The default full run writes beneath `artifacts/training/tracker/`:

- `full/checkpoint.pt`;
- `full/training_manifest.json`;
- `full/validation_detection_metrics.json`;
- `predicted_tracks.json` for Validation, Testing, and VID31;
- `oof/fold_<n>/checkpoint.pt`, manifest, and held-out Training artifact when
  OOF mode is requested;
- a concise run summary containing durations and selected video IDs.

The checkpoint and generated predictions remain runtime artifacts and are not
added to Git or the source archive.

## Metrics and paper use

Tracker validation reports instrument detection AP@0.50, macro AP@0.50, recall,
precision, and F1 at the frozen score/IoU operating point. These diagnose the
support model; they do not establish the main contribution.

The first contribution table uses the existing offline frame evaluator and the
same cached API backbone outputs where possible:

1. Frames only.
2. Frames + predicted Track.
3. Frames + historical Workflow.
4. Frames + predicted Track + historical Workflow.

Report Instrument, Verb, Target, IVT, and Phase metrics plus the aggregate and
per-video confidence intervals already defined by the evaluation pipeline. The
effect attributed to the first innovation is the downstream difference between
these controlled settings, not detector AP alone.

## Failure behavior

The training/export command stops on an unavailable CUDA request, missing
Torchvision detection operators, empty qualified training data, a checkpoint
whose metadata does not match the requested model/class mapping, a video split
violation, or an artifact rejected by the strict provider. A smoke flag may
bound batches and frames, but its outputs are labeled non-paper.

## Testing

- Unit tests cover class mapping, supervision qualification, box conversion,
  causal matching, video reset, track expiry, and artifact provenance.
- Configuration tests cover the new `track_only` profile and mutually exclusive
  workflow/track inputs.
- A lightweight integration test runs a synthetic optimizer/export path without
  downloading weights.
- Existing API mock tests prove all four context configurations can reach the
  same prediction/evaluation path.
- Final verification runs focused tests, full Pytest, Ruff, compilation, an
  AutoDL bundle verifier, and a secret scan before producing a new source ZIP.

## Out of scope

- Using current or future GT phase/track state at inference.
- Treating VID31's disabled instance fields as valid supervision.
- Training the API-VLM.
- Implementing the Specialist/Gate module in this change.
- Claiming real-time performance before measuring it on AutoDL.
