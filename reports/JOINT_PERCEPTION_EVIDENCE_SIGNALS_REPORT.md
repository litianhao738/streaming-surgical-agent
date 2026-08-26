# Joint Perception and Evidence Signals Completion Report

Status: `CORE_COMPLETE_REAL_SMOKE_FAILED_PROVIDER_404`

## Delivered

- Added exact real and mock Joint Perception configurations plus the named
  `api_single_pass` experiment configuration.
- Added a gold-free synthetic runner that executes the existing
  `CanonicalStreamingPipeline` once with three ordered 32x32 RGB frames.
- The assembly uses `CausalPerceptionContextBuilder`, one shared
  `CachedMultimodalApiClient`, `JointApiVlm`,
  `FrameEvidenceSignalExtractor`, disabled candidates/Specialists,
  `NeverVerify`, `NoOpCoordinator`, no-op causal stores,
  `PredictionFinalizer`, and a fresh `FrameResultWriter`.
- The same client replays a rebuilt no-prior request from an ephemeral cache;
  the pipeline is not run twice and no parsed provider payload is retained in
  the run directory.
- Added explicit three-image cached-client accounting coverage, strict config
  bool/int edge coverage, and the repository-wide Ruff annotation fix in
  `perception/contracts.py`.

## TDD Evidence

Initial integration RED:

```text
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest tests/integration/test_api_single_pass.py -q
ModuleNotFoundError: No module named 'scripts.run_api_single_pass'
1 error in 0.23s
```

Initial integration GREEN:

```text
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest tests/integration/test_api_single_pass.py -q
17 passed in 9.54s
```

The single-provider-attempt hardening had its own RED before the production
change:

```text
test_real_single_pass_never_retries_the_authorized_provider_call
AssertionError: assert 3 == 1
1 failed in 2.63s
```

After setting the authorized smoke policy to one attempt:

```text
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest tests/integration/test_api_single_pass.py -q
18 passed in 8.83s
```

The inherited three-image cache/config characterization tests passed before
runner implementation: `12 passed in 0.56s`.

## Local Verification

- Starting commit: `500dbf946672e4c2802e43816f7bf02a52cf47d9`.
- Task commit subject: `feat: complete joint perception single pass` (the final
  hash is reported in the handoff because a commit cannot contain its own
  hash).
- Python: `3.11.7`.
- Key packages: Torch `2.13.0+cpu`, Pillow `12.3.0`, PyYAML `6.0.3`, pytest
  `9.1.1`, Ruff `0.16.4`.
- Post-review focused integration: `18 passed in 9.03s`.
- Post-review repository suite: `549 passed, 13 skipped in 20.21s`.
- Ruff after the final behavior change: `All checks passed!`.
- Full-token tracked-source secret scan returned no match (`git grep` exit 1).
  The literal prefix command in the plan also matches inherited safe docs,
  regex source, and split fixtures; those are not credential values.

## Authorized Real Synthetic Smoke

UTC attempt timestamp from the sanitized ledger:
`2026-08-26T18:59:59.983037+00:00`.

Exactly one authorized CLI invocation used the ignored absolute credential
file and no dataset image. It made one provider attempt and failed safely:

```text
JOINT_SINGLE_PASS_FAILED category=provider_4xx
```

Sanitized evidence:

- HTTP status: `404`;
- requested model: `openai/gpt-5.6-sol`;
- returned model: unavailable;
- response ID: unavailable;
- request hash:
  `13086b8dda61a5a2a9c4a91a0dc8601efb2985ca9e48e4a8114590212be28e2b`;
- provider calls: `1`; retries: `0`;
- token usage and provider cost: unavailable;
- ordered image hashes:
  `f72ef93fe46f9e67f2a9cb65217bfa3e436f481516626d53d98523367dd3bb14`,
  `e6e5416e5a29857028468fcd5bb64eafc9865fa8a9d9a85c243109a7ab8c0b93`,
  `d6e61eba692a158a889ab50388ad05853708a20b160850b10c029b1956dfc870`;
- paired prediction/evidence files: not written because strict provider success
  and response validation precede pipeline persistence;
- cache replay: not reached, so no cache-hit claim is made;
- runtime persisted only the sanitized usage ledger; no raw body, parsed
  payload, prompt, image bytes, headers, argument vector, or credential.

OpenRouter's public model catalog listed the requested model and advertised
image/structured-output support at review time. Without retaining or exposing
the provider body, the strongest safe diagnosis is that no eligible route or
resource was available for this exact request/account at that moment. No retry
or second paid smoke was attempted.

## Boundaries and Concerns

- No CholecTrack20 image was uploaded.
- No paper-performance experiment ran; all generated results are explicitly
  `paper_metric_eligible: false`.
- Instance detection, predicted tracking, the learned Gate, Specialist calls,
  repair coordination, and EventMemory remain outside this slice.
- The runnable mock/core path is complete. Real-provider evidence remains a
  failed smoke, not a success and not an exact-backend-identity claim.
