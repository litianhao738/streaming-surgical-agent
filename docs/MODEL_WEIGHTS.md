# Gate and Tracker inference weights

[Download release: tracker-phase-gate-weights-20260916](https://github.com/litianhao738/streaming-surgical-agent/releases/tag/tracker-phase-gate-weights-20260916)

The release ZIP contains the exact artifacts used by the current default Phase Gate and the Testing Tracker:

| Artifact | Runtime path |
|---|---|
| Gate configuration | `artifacts/training/gate/five_head_probe_gate_20260916_r2_harm_overall/model.json` |
| Gate estimator | `artifacts/training/gate/five_head_probe_gate_20260916_r2_harm_overall/estimator.joblib` |
| Tracker checkpoint | `artifacts/training/tracker_clip_v2_oof5_20260906/full/checkpoint.pt` |
| Tracker configuration | `artifacts/training/tracker_clip_v2_oof5_20260906/full/training_manifest.json` |

Download the ZIP, `weights_manifest.json`, and `SHA256SUMS.txt`. Check the ZIP hash against SHA256SUMS, then extract into the repository root:

```powershell
Get-FileHash .\tracker-phase-gate-weights.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\tracker-phase-gate-weights.zip -DestinationPath .
```

Individual file hashes and sizes are listed in `weights_manifest.json` and match the existing runtime manifests. No runtime configuration changes are needed.

The Gate has 54 features (42 historical and 12 Phase probe features), uses the harm label with the overall threshold rule, and has threshold 0.011637720680921813. It is a Training research model: `deployable=false`, with no independent validation claim.

The Tracker checkpoint is the full Training fit Faster R-CNN MobileNetV3 Large FPN used for Testing inference. The association algorithm uses parameters from the training manifest. The release does not contain OOF fold models/caches, optimizer state, or training data. Training replay requires its separate held-out prediction caches. Dataset inputs, prior artifacts and API credentials must still be supplied independently.

Weights are distributed as Release attachments; large binary files are not added to Git history.
