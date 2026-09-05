# Frozen Verifier baseline — V7 — 2026-09-04

This file anchors the machine-readable baseline manifest created before any
new Verifier or Repair-admission design work.

- Git baseline: `main@772cbd47556a48a2096e87851f12622bacc07ec8`
- Worktree before creating the freeze manifests: clean
- Architecture: `CONSERVATIVE_SAME_MODEL_V7`
- H0 model: `openai/gpt-5.6-sol`
- Verifier model: `openai/gpt-5.6-sol`
- Verify attempts: `1`
- RAG: disabled
- Hard-valid H0 semantic changes: rejected with
  `FALLBACK_KEEP/HARD_VALID_H0_PROTECTED`
- Machine-readable manifest:
  `artifacts/preflight/frozen_verifier_baseline_v7_2026-09-04.json`
- Machine-readable manifest SHA-256:
  `f63a3b301a8faf79f245a8517ded7bf84594bd14901f2ee4403dccd810cf9fdd`

Locked evidence:

- V6 20-frame pilot manifest SHA-256:
  `7cc6c226f37afdb22d2799934d1843312ea10fefe0d2b085834181da068b1b32`
- V6 counterfactuals SHA-256:
  `2d20a6953ddab83747cf6c8f18f41643b6e53ebc3f50075404004634a409a407`
- V7 endpoint manifest SHA-256:
  `8b06dc66611441f97b887999bc839df6e878ff8beb5fb57063f58a3be1b1d5a7`
- V7 endpoint counterfactuals SHA-256:
  `23110de8f2ea9b9797580865b2b6230fc7b8acea77addcb9e44bf9e9ef4ec6be`

This baseline is development evidence. New rules must use a new version and
new artifact path; `VID13:1`, `VID13:24376`, and the V6 pilot may not be reused
as confirmation data after they influence the design.
