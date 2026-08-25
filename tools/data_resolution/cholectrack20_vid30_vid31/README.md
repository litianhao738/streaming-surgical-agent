# CholecTrack20 VID30/VID31 identity resolution

This directory contains a read-only resolution workflow for the anomalous
`VID30` and `VID31` release assets. It never renames, moves, overwrites, or
deletes source dataset files.

## What works without Cholec80

The local stage:

- records JSON/PNG frame-set relations and source hashes;
- confirms the VID17/VID30 core-annotation duplicate from the existing audit;
- renders deterministic contact sheets for all four JSON-to-PNG pairings;
- emits a field-level supervision manifest that keeps raw assets but disables
  unresolved supervision;
- audits negative labels and IVT component consistency with task-wise masks.

```powershell
.\.venv\Scripts\python.exe `
  tools\data_resolution\cholectrack20_vid30_vid31\resolve_identity.py `
  --dataset-root D:\cholec_dataset `
  --output-dir artifacts\data_resolution\cholectrack20_vid30_vid31
```

## Cholec80 files required for pixel identity

Provide either the original Cholec80 `video30` and `video31` MP4 files or
complete extracted frame directories whose original numeric filenames are
preserved. A complete Cholec80 download is not required for this step.

```powershell
.\.venv\Scripts\python.exe `
  tools\data_resolution\cholectrack20_vid30_vid31\resolve_identity.py `
  --dataset-root D:\cholec_dataset `
  --output-dir artifacts\data_resolution\cholectrack20_vid30_vid31 `
  --cholec80-video30 D:\cholec80\videos\video30.mp4 `
  --cholec80-video31 D:\cholec80\videos\video31.mp4
```

For a complete frame export, the tool uses global nearest-frame retrieval
before reporting observed frame-number offsets. It does not assume that a
Track20 frame filename and a Cholec80 filename share the same convention.

```powershell
.\.venv\Scripts\python.exe `
  tools\data_resolution\cholectrack20_vid30_vid31\resolve_identity.py `
  --dataset-root D:\cholec_dataset `
  --output-dir artifacts\data_resolution\cholectrack20_vid30_vid31 `
  --cholec80-frames30 D:\cholec80_30_31\30 `
  --cholec80-frames31 D:\cholec80_30_31\31 `
  --cholec80-phase30 D:\cholec80_30_31\video30-phase.txt `
  --cholec80-phase31 D:\cholec80_30_31\video31-phase.txt
```

The upstream stage records perceptual-hash, correlation, and absolute-error
evidence for the full 2x2 media identity matrix. Its raw-JSON phase projection
is audit-only. Run `build_derived_supervision.py` to create image-aligned VID31
phase supervision. Neither tool silently authorizes a repaired box, IVT, or
tracking pairing.

## Evidence boundary

Cholec80 can establish media identity and provide phase evidence. It cannot
reconstruct missing CholecTrack20 bounding boxes or three-perspective track
IDs. CholecT50 labels, if separately obtained for VID31, can provide
I/V/T/Triplet evidence but cannot replace Track20 tracking ground truth.
