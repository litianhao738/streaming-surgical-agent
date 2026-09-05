# Track–Workflow Structured Causal Context

This module is the first context contribution of the V3.1 pipeline. It is an
orthogonal runtime choice, not a second pipeline implementation.

## Executable profiles

The dataset API entry points accept one of three `--context-profile` values:

| Profile | Predicted tracks | Finalized workflow | Phase graph |
|---|---:|---:|---:|
| `frames_only` | no | no | forbidden |
| `workflow` | no | yes | optional, recommended |
| `track_workflow` | required artifact | yes | optional, recommended |

`auto` preserves the historical behavior: `single_pass` resolves to
`frames_only`; verification profiles resolve to `workflow`.

The same images, API model, output schema, candidates, Gate, verifier, cache,
and evaluation stay fixed across profiles. This makes the context contribution
independently ablatable without duplicating pipeline code.

## Workflow contract

`FinalizedWorkflowStateStore` is snapshotted before frame `t` and updated only
after prediction/evidence persistence succeeds. The API receives only:

- finalized predicted phase history;
- phase stability;
- observed phase transitions;
- `source_max_frame_id < t`.

When a phase-transition graph is supplied, the loader verifies its schema and
hash, rebuilds the expected graph from the current dataset root's official
Training phase labels, and requires exact equality. The runtime therefore does
not trust self-declared source video IDs. The accepted graph activates
`phase_change_anomaly` in `evidence_frame_v1`.

## Predicted-track artifact contract

`track_workflow` never reads CholecTrack20 annotation tracks. It requires an
external prediction artifact with schema `predicted_track_context_v1`:

```json
{
  "schema_version": "predicted_track_context_v1",
  "provider": "predicted_tracker_v1",
  "source_model_identifier": "tool-detector-fold0",
  "checkpoint_sha256": "<64 lowercase hex characters>",
  "causal": true,
  "inference_mode": "online_forward_only",
  "producer_version": "predicted_track_producer_v1",
  "dataset_repair_manifest_sha256": "<64 lowercase hex characters>",
  "inference_config_sha256": "<64 lowercase hex characters>",
  "videos": {
    "VID30": {
      "source_split": "validation",
      "frames": [
        {
          "frame_id": 1,
          "tracks": [
            {
              "track_id": "tool-1",
              "instrument_id": 0,
              "bbox_tlwh": [0.1, 0.2, 0.3, 0.4],
              "score": 0.9,
              "age": 1
            }
          ]
        }
      ]
    }
  }
}
```

Runtime validation rejects unknown fields, GT-bearing fields, bad ontology
IDs, invalid boxes, duplicate tracks, non-increasing frames, split mismatch,
missing selected frames, and future-frame context. Only the selected causal
window is serialized into a request. Every accepted artifact must declare the
fixed `online_forward_only` inference mode, the fixed producer version, the
dataset repair-manifest SHA-256, and the inference-config SHA-256; the provider
and its snapshot expose all four values together with checkpoint and artifact
hashes. Missing, unknown, or unsupported values fail closed.

These provenance fields strengthen auditability, but cannot mathematically
prove that an arbitrary JSON artifact was generated without GT. Formal
predicted-track artifacts must be produced by the controlled, causal AutoDL
generator; hand-authored or externally generated files are not sufficient for
a GT-free research claim. Until that generator and tracker are trained, use
`workflow`; do not report `track_workflow` results.

## AutoDL commands

Workflow context with the already generated Training-only graph:

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

After controlled causal tracker training/export, select the full context with:

```text
--experiment-config configs/experiments/v3_track_workflow.yaml
```

Every rollout artifact records the experiment-config hash, resolved context
profile, actual event-memory switch, phase-graph version/hash/source videos,
and predicted-track artifact hash plus provenance fields.

## Fixed three-frame causal window

The current dataset API path uses `[t-2, t-1, t]` as its maximum causal
observation buffer. All available frames in that window are uploaded in
chronological order, the target frame is always last, and Tracker never selects
the visual input. At a video boundary, the window grows naturally from one to
three frames rather than duplicating the first image. The earlier adaptive
six-frame-buffer/three-image policy is historical and is not valid for the
primary factorial.

For CholecTrack20, `[t-2, t-1, t]` means frame IDs `[t-50, t-25, t]` at the
verified 1 FPS annotation clock. A frame-ID increment greater than 25 resets
the causal window and immediate temporal comparison; history before the gap is
not presented as adjacent motion. Tracker association generation uses the same
maximum link gap. Detector weights are unchanged, but association artifacts
created before this policy must be regenerated for strict formal use.

Every run persists `causal_window_audit.jsonl`. It records candidate frame IDs,
selected image IDs, tracker availability, track persistence, visual change,
displacement, and the motion-state summary. This is the audit record used for
fixed-window checks and any separately declared supplementary ablation.
