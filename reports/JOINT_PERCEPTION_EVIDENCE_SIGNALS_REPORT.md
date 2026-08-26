# Joint Perception and Evidence Signals Completion Report

Status: `CORE_COMPLETE_REAL_SMOKE_INCOMPLETE_PARSE_FAILURE`

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
- The exact OpenRouter generation contract emits only `max_tokens: 4096`,
  retains `provider.require_parameters: true`, and omits unsupported
  `temperature`/`top_p`. Provider-facing schema output omits `uniqueItems`;
  the local semantic parser still rejects duplicate selected IDs.
- A real origin response must include exact non-null input/output/total token
  counts and finite non-negative cost before parsing or paired persistence.

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

This HTTP 404 attempt is incomplete evidence. The sanitized record does not
establish whether the cause was routing, account state, endpoint behavior,
model availability, or request compatibility, so no causal or transient-error
diagnosis is claimed. No retry or second paid smoke was attempted in that run.

### Fix Round 1 authorized attempt

After the unsupported generation/schema keywords were removed and the real
accounting completion gate was added, exactly one fresh authorized invocation
ran under ID `task10_fix1_real_20260827`. It ended with the safe category
`parse_failure`, one provider call, zero retries, and request hash
`602ff5f3af06cc40d0dedb4ed314cf05947fbc3a64b9d1a59939a2f63a31362d`.
The request retained the same three ordered image hashes listed above.

The transport only reports `parse_failure` after a success-class HTTP response
cannot be normalized into the required completion/usage contract. The raw body
is intentionally discarded, so the evidence cannot distinguish response
shape, finish state, structured content, or usage-field causes. Returned model,
response ID, input/output/total tokens, and provider cost are unavailable. The
completion gate therefore did not permit a success artifact, manifest, paired
prediction/evidence, or cache replay. No retry was made. Runtime allowlist and
full-token tracked/runtime secret scans were clean.

## Boundaries and Concerns

- No CholecTrack20 image was uploaded.
- No paper-performance experiment ran; all generated results are explicitly
  `paper_metric_eligible: false`.
- Instance detection, predicted tracking, the learned Gate, Specialist calls,
  repair coordination, and EventMemory remain outside this slice.
- The runnable mock/core path is complete. Both real-provider attempts remain
  incomplete: the earlier HTTP 404 and Fix Round 1 parse failure are not
  successes, not transient-failure diagnoses, and not exact-backend-identity
  claims.
