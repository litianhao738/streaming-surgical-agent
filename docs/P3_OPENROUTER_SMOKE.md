# P3 OpenRouter Real API Evidence

## Verdict

P3 is `PARTIAL`. The real multimodal structured API path is operational, but
the response does not provide an immutable backend-model identifier or a
separate exact-identity evidence source.

## Effective API Boundary

- Provider: `openrouter`
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Requested model: `openai/gpt-5.6-sol`
- Returned model: `openai/gpt-5.6-sol`
- Exact backend model: unavailable
- Input: one deterministic, synthetic 8x8 PNG; no CholecTrack20 image uploaded
- Output: strict `p3_multimodal_smoke_v1` JSON Schema

OpenRouter's public model catalog reported one exact model match with image
input, structured outputs, `response_format`, and reasoning support. The first
real request exposed an `invalid_json_schema` diagnostic because two constant
properties lacked explicit JSON Schema `type` declarations. After adding the
types, the formal real smoke completed.

## Runtime Evidence

Artifact:
`artifacts/p3/p3_openrouter_gpt56sol_real_20260826/p3_api_smoke.json`

Validated invariants:

- first logical call: cache miss and exactly one provider call;
- second logical call: cache hit, zero provider calls, and zero replay cost;
- both calls share the canonical request hash;
- response ID and returned model identifier are present;
- first-call provider cost is recorded;
- verdict remains `PARTIAL` because exact backend identity fields are empty.

The earlier schema failure is retained recoverably under
`artifacts/p3/p3_openrouter_gpt56sol_real_20260826_failed_schema_400/` and
contains only a sanitized usage record.

## Credential Safety

`docs/API.txt` is ignored and untracked. The loader reads the OpenRouter key
from the first non-empty line of the local bundle, rejects multiple OpenRouter
credentials, and never serializes the credential. The smoke artifact excludes
the parsed model message, raw provider response, image bytes, and credential.

## Reproduction

```powershell
.venv-p2\Scripts\python.exe scripts/smoke_api.py `
  --config configs/api/openrouter.yaml `
  --real `
  --api-key-file docs/API.txt `
  --run-id p3_openrouter_gpt56sol_real_20260826
```

Use a new run ID when repeating the command because output directories are
created exclusively.

## Current Gate-owned contract continuation (2026-09-02)

The active dataset wire contract was subsequently changed to
`joint_perception_gate_owned_compact_v1`. A separate synthetic smoke profile
now exercises that exact prompt/schema without uploading surgical data:

```powershell
.\.venv-p2\Scripts\python.exe scripts\run_api_single_pass.py `
  --real `
  --config configs\perception\joint_openrouter_gate_owned_smoke.yaml `
  --api-key-file docs\API.txt `
  --output-root artifacts\p3_current_schema `
  --run-id p3_gate_owned_openrouter_synthetic_20260902
```

Sanitized evidence from that run:

- three deterministic 32x32 synthetic RGB frames; no Track20 image uploaded;
- requested and returned model strings: `openai/gpt-5.6-sol`;
- first call: cache miss, one provider call, 2,178 input tokens, 319 output
  tokens, 2,497 total tokens, 20,279.42 ms, and no retry;
- second identical call: cache hit, zero provider calls, zero current cost, and
  the same canonical request hash;
- strict Gate-owned JSON validation, parser, prediction/evidence persistence,
  response ID capture, and post-run credential scan all passed.

The ignored local evidence artifact is
`artifacts/p3_current_schema/p3_gate_owned_openrouter_synthetic_20260902/single_pass_artifact.json`.
Its SHA-256 is
`f9688b17985f2bdacbd1e7293887e801540caae3c3d35a059ef02c26970d6de3`.
An official OpenAI probe using the separately supplied local key failed with
the sanitized category `authentication`. A real VID30 OpenRouter probe reached
the provider but was rejected as `content_moderation`; the successful synthetic
run proves that this was not a URL, request-shape, schema, or parser failure.
The verdict remains `P3 PARTIAL` only because the returned model string is a
routing alias rather than independently verifiable immutable backend identity.
