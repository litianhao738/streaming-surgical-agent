# P2 Local Smoke Baseline Report

## Decision

`P2_STATUS: PASS`

P2 validates the engineering path only. It does not establish paper performance,
API-backbone quality, Gate benefit, Specialist value, or memory effectiveness.

## Implemented Path

```text
external read-only CholecTrack20
  -> manifest/hash/split validation
  -> DatasetAdapter and strict causal sample
  -> lightweight local frame model
  -> task-wise masked BCE/CE
  -> real backward and optimizer step
  -> atomic checkpoint and exact reload
  -> canonical NeverVerify pipeline
  -> Gold-free PredictionRecord
  -> offline EvaluationEngine
  -> atomic JSONL, manifest, metrics, and provenance
```

The canonical runner exposes component slots for context, perception, candidate
generation, Gate, Specialist, coordinator, workflow state, and EventMemory. P2
uses explicit traceable no-op/disabled implementations where research modules
do not yet belong. Later experiments must replace components without copying the
rollout loop.

## Data Boundaries Verified

- Official partition: 10 training, 2 validation, 8 testing videos.
- VID02: official raw route; frame-level smoke supervision is phase only.
- VID31: CholecT50 frame-level Instrument/Verb/Target/Triplet presence plus Cholec80 phase; all instance,
  bbox, operator, and tracking supervision remains disabled.
- VID30: explicit repaired validation sidecar; frame-level smoke evaluation is
  phase only, while instance annotations remain a separate target granularity.
- Test: only Gold-free `InferenceSample`; no optimizer, selection, or smoke metric use.
- Repair manifest, original VID30/VID31 JSON, and all materialized sidecars are
  checked by SHA-256 before sampling.

The full eligibility audit covered 20 videos and 37,675 operational samples.
This operational count includes the explicit VID30/VID31 sidecar frame sets and
must not be reported as the raw-release annotation count (35,009 frames).

## Executed Evidence

- Real CPU masked optimizer step: PASS.
- Finite total loss and gradients: PASS.
- Valid task counts for the two-sample VID02/VID31 batch:
  `I=1, V=1, T=1, IVT=1, Phase=2`.
- Exact model/optimizer checkpoint round-trip: PASS.
- VID02, VID31, and VID30 canonical rollout: PASS.
- Atomic prediction JSONL and hash-verified completion manifest: PASS.
- Full target-granularity audit: PASS.
- Historical full environment (2026-08-24): `79 passed`.
- Independent current environment (2026-08-25, `.venv-p2`, `CHOLECTRACK20_ROOT`
  configured, optional raw `CHOLEC80_30_31_ROOT` absent): `78 passed, 1 skipped`.
  The skipped test only rechecks optional upstream Cholec80 phase-file hashes;
  materialized VID30/VID31 runtime sidecar and hash tests passed.

Evidence run:
`artifacts/p2/p2_20260824T095905Z_9f862ff9/`.

## Remaining Gates

- P3 is next. Exact API provider, endpoint provenance, provider-returned model
  identifier, multimodal capability, cache, and a small real API smoke remain required.
- Before P4, instance-level versus frame-level prediction/evaluation and matching
  rules must be frozen from the P2 eligibility audit.
- VID30 must be disclosed as a candidate reconstruction and excluded in a paper
  sensitivity analysis.
- Canonical per-sample task error remains blocked until P9 and must not be inferred
  from the P2 smoke loss.

Passing P2 means the repository is ready to upload to AutoDL for environment
verification and continued staged implementation. It does not authorize long
paper training or P3+ experiments before their own gates pass.
