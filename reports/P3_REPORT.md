# P3 API Client + Cache Report

## Stage

`P3_STATUS: PARTIAL`

Provider-neutral API infrastructure and deterministic mock validation are
implemented. `REAL_API_SMOKE: BLOCKED` because provider, endpoint, credential,
requested model alias, and provider-returned model identity are not approved or
verified. P4 is not authorized.

## P2 Baseline

- First Git baseline commit: `be99b96dc1d4cb9933c1142aa6e1921e0d0c40bc`.
- Commit identity was repository-local: `Codex <codex@localhost>`.
- API keys, caches, checkpoints, `tmp/`, logs, and generated runtime artifacts
  were excluded from the commit.

## Implemented

- immutable credential-free `ApiRequest` and in-memory `ApiImageInput`;
- canonical SHA-256 over provider/model/endpoint, prompt/schema versions,
  structured payload, generation parameters, and actual image content hash;
- typed transport/schema/cache/retry error taxonomy;
- atomic file cache with fail-closed corruption and collision checks;
- bounded retry for explicitly retryable transport errors only;
- JSONL usage ledger separating logical calls, provider attempts, and cache hits;
- config-driven mock provider adapter and returned-model-field capture;
- P3-only structured transport-probe schema, independent of P4 surgical output;
- synthetic non-sensitive image smoke and explicit real-call blocker.

## Commands Executed

```powershell
.venv-p2\Scripts\python.exe -m ruff check .
.venv-p2\Scripts\python.exe -m compileall -q src tools tests
$env:CHOLECTRACK20_ROOT='D:\cholec_dataset'
.venv-p2\Scripts\python.exe -m pytest -q -ra
.venv-p2\Scripts\python.exe scripts\smoke_api.py `
  --config configs/api/mock.yaml `
  --output-root artifacts/p3 `
  --run-id p3_mock_release_20260825
.venv-p2\Scripts\python.exe scripts\run_local_smoke.py `
  --config configs/experiments/local_smoke.yaml `
  --dataset-root D:\cholec_dataset `
  --output-root tmp/p2_after_p3 `
  --run-id p2_regression_20260825 `
  --device cpu `
  --skip-full-granularity-audit
```

## Observed Mock Smoke

- input: generated 8x8 PNG; no CholecTrack20 image;
- first logical call: cache miss, one injected retry, two mock provider attempts;
- second logical call: cache hit, zero provider attempts;
- structured response validation: PASS;
- returned mock model field captured separately from requested mock alias;
- usage summary: 2 logical calls, 2 provider attempts, 1 cache hit;
- clean-tree artifact: `artifacts/p3/p3_mock_release_20260825/p3_mock_smoke.json`
  (generated and ignored); it records the exact clean Git commit used by the run.
- frozen P2 smoke after the base status advanced to P3: PASS; the P2 experiment
  override remains reproducible and does not inherit P3 runtime semantics.

## Tests

- P3 targeted tests: `17 passed` before the final full-suite run.
- Repository-wide regression (2026-08-25, `.venv-p2`, `CHOLECTRACK20_ROOT`
  configured, optional raw `CHOLEC80_30_31_ROOT` absent): `95 passed, 1 skipped`.
  The skip remains the optional upstream Cholec80 provenance hash check.

## Explicit Boundaries

- `GPT-5.6 Terra` remains a declared planned backbone name, not a verified exact
  API `model_identifier`.
- No provider, endpoint, credential variable, or real model ID was inferred.
- If a future real provider returns only a floating alias, P3 remains `PARTIAL`.
- No CholecTrack20 image may be uploaded before data-use authorization.
- P3 does not freeze `InitialPrediction`, instance/frame matching, P4 evaluation
  granularity, or the P9 canonical per-sample Gate error.

## Next Stage Readiness

`P4_READY: NO`

P3 can become PASS only after an approved provider adapter performs a sanitized
real multimodal smoke and the response provenance verifies provider, endpoint,
returned model field, schema behavior, cache, retry/error handling, and usage.

## OpenRouter Continuation (2026-08-26)

The former blocked provider candidate was replaced with OpenRouter at the
user's direction; Requesty runtime source, configuration, routing, and tests
were removed. The active real configuration is:

- provider: `openrouter`;
- endpoint: `https://openrouter.ai/api/v1/chat/completions`;
- requested model: `openai/gpt-5.6-sol`;
- environment: `.venv-p2`, Python 3.11.7;
- evidence revision before the working-tree implementation:
  `d0da8af7e06f74d869f47ada007293a2035f65b6`.

The formal smoke used one deterministic synthetic 8x8 PNG and no dataset
frame. The initial request returned a sanitized `invalid_json_schema` failure;
two strict-schema constant properties were missing explicit type declarations.
After that contract fix, the real multimodal structured-response smoke
completed successfully. The sanitized evidence records:

- returned model: `openai/gpt-5.6-sol`;
- response ID, token usage, and provider cost are present;
- first logical call is a cache miss with one provider call;
- the identical second call is a cache hit with zero provider calls and zero
  replay cost;
- both logical calls share the same canonical request hash;
- no raw response, parsed model message, image bytes, or credential is stored.

The exact immutable backend identifier and an independent exact-identity
evidence source were not returned. The honest phase verdict is therefore
`P3_STATUS: PARTIAL`, while the real OpenRouter transport itself is operational.
Targeted OpenRouter/credential regression verification reports `38 passed`.
The final repository-wide run with `CHOLECTRACK20_ROOT=D:\cholec_dataset`
reports `348 passed, 3 skipped`; Ruff, compileall, and the single-root AutoDL
verifier also pass.
The earlier `79 passed`, `78 passed, 1 skipped`, and `95 passed, 1 skipped`
records above remain historical evidence rather than being overwritten.

The runtime dataset check continues to use only
`CHOLECTRACK20_ROOT=D:\cholec_dataset`; the optional external
`CHOLEC80_30_31_ROOT` is absent. No sidecar, manifest, split, label semantics,
or research-task definition changed. Formal performance and paper experiments
did not run, and no P4 or P9 contract was frozen.
