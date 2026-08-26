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
