# CholecTrack20 Single-Root Portability and Requesty P3 Design

> Provider amendment: the user explicitly replaced the Requesty boundary with
> OpenRouter and required removal of the old Requesty runtime code. The active
> implementation uses `https://openrouter.ai/api/v1/chat/completions` with
> `openai/gpt-5.6-sol`. Requesty-specific sections below are retained only as
> historical design context and are not an active runtime contract.

**Status:** Approved in chat on 2026-08-26; implementation has not started.

## Purpose

Complete the CholecTrack20 single-data-root AutoDL portability evidence and the
provider-neutral P3 Requesty API infrastructure. The implementation must run a
real multimodal structured-response smoke, preserve research definitions, and
stop before P4.

The current repository already has a passing P2 engineering baseline and a
mock-only P3 API path. This design extends those contracts incrementally. It
does not redesign the research pipeline or create a Requesty-only bypass.

## Current Evidence and Selected Approach

The read-only audit established the following baseline:

- `CHOLECTRACK20_ROOT=D:\cholec_dataset` supplies every runtime/training input.
- VID30 and VID31 sidecars are already materialized under that root, and their
  hashes match `repair_manifest.json`.
- Manifest path rebasing contains runtime paths within the supplied root.
- With that root configured, the current suite reports `95 passed, 1 skipped`;
  the skip is limited to optional upstream Cholec80 provenance sources.
- The current P3 implementation is mock-only and its `PARTIAL` report is honest.
- `docs/API.txt` is untracked but not yet ignored and has not been read.

This satisfies objective condition A: do not copy, rebuild, or rewrite the
dataset. Add verification, isolation tests, a bundle manifest, and deployment
documentation only.

Three approaches were considered:

1. Incrementally extend the existing provider-neutral P3 stack. **Selected.**
   This keeps Requesty under the same request hash, cache, retry, usage, and
   error contracts as the mock provider.
2. Add a standalone Requesty smoke script. Rejected because it would duplicate
   and bypass the provider-neutral contracts that P3 is required to prove.
3. Rewrite the API subsystem. Rejected because it expands risk without serving
   a P3 requirement.

## Frozen Boundaries

The implementation must preserve all of these invariants:

- P2 remains an engineering smoke, not a performance or paper claim.
- VID31 uses CholecT50 frame-level Instrument/Verb/Target/Triplet presence and
  Cholec80 phase only. It does not gain instance, bounding-box, operator, or
  track supervision.
- VID30 remains a candidate repaired validation source. Its disclosure and the
  future exclusion sensitivity experiment remain required.
- Split membership, label meaning, masks, research tasks, and paper
  contributions do not change.
- CholecT50/Cholec80 sidecars remain derived provenance, never described as
  native official CholecTrack20 annotations.
- No P4 instance-aware prediction schema, matching rule, or granularity choice
  is introduced. P9 gate-error semantics remain deferred.
- Historical reports and test results are append-only evidence.

## 1. Single-Root Portability Architecture

### Declarative bundle contract

Add a committed, non-secret AutoDL bundle manifest. It declares:

- schema version and dataset identity;
- `CHOLECTRACK20_ROOT` as the only runtime data-root environment variable;
- the official `Training`, `Validation`, and `Testing` layout and expected
  10/2/8 video counts;
- the relative paths of `repair_manifest.json` and the three materialized
  VID30/VID31 sidecars;
- `repair_manifest.json` as the authority for raw and materialized SHA-256
  values;
- Cholec80 and CholecT50 roots as optional provenance/regeneration inputs that
  are forbidden as runtime/training dependencies.

The manifest must not duplicate the dataset, contain machine-specific absolute
paths, or claim redistribution rights.

### Read-only verifier

Expose one focused verifier callable from Python and a CLI. Given a dataset
root, it will:

1. resolve the root and bundle manifest;
2. discover the official split exactly as the runtime adapter does;
3. load and rebase the repair manifest against the supplied root;
4. enumerate every runtime-required media, annotation, phase, IVT, and manifest
   path;
5. reject missing, ambiguous, case-mismatched, or out-of-root paths;
6. verify the 10/2/8 split, 20 unique videos, materialized sidecar hashes,
   original VID30/VID31 JSON hashes, and repair-manifest hash;
7. classify any external references as provenance/regeneration-only; and
8. return a deterministic, JSON-serializable report containing relative paths,
   hashes, roles, and PASS/FAIL evidence but no dataset contents.

Containment is checked on resolved paths using `Path.relative_to`; string-prefix
checks are not sufficient. Recorded Windows paths in the repair manifest remain
historical evidence and continue to be rebased rather than deleted.

### Runtime proof

Integration verification will remove Cholec80/CholecT50-related environment
variables, set only `CHOLECTRACK20_ROOT`, load VID30 and VID31, and run the P2
training/runtime smoke. This proves that AutoDL requires the project, one
complete CholecTrack20 root, and a separately supplied Requesty key only.

## 2. Credential and Persistence Boundary

Before the key is first read, implementation must:

1. add the exact path `docs/API.txt` to `.gitignore`;
2. prove `git ls-files --error-unmatch docs/API.txt` fails; and
3. prove `git check-ignore docs/API.txt` succeeds.

The file remains in place and unchanged. A small credential loader supports two
mutually exclusive CLI inputs:

- `--api-key-file <path>` reads one non-empty `NAME=value` line, splitting only
  at the first `=` so the remainder is preserved verbatim;
- `--api-key <value>` accepts an explicit value for compatibility.

Evidence runs use `--api-key-file`; they never place the real value in a command
line. The credential is held only by the constructed Requesty transport and is
used only to form the in-memory authorization header. Its representation is
redacted, and it never enters `ApiRequest`, canonical request metadata, request
hashes, configuration dumps, cache files, usage records, artifacts, reports,
exception messages, tracebacks, or Git.

Persistence uses allowlists, not retrospective string replacement:

- successful cache/usage records persist normalized response fields and
  whitelisted Requesty metadata only;
- raw provider responses and response bodies are never cached or ledgered;
- failure records persist category, retryability, HTTP status, sanitized
  provider code, and attempt counts, never raw bodies or arbitrary exception
  text;
- smoke artifacts contain only the sanitized contract record.

After the real run, an in-memory exact-value scan checks tracked text and all
new cache, usage, artifact, and report text. Failure output may name affected
paths but may not print the matching value. The key file itself is excluded
from this scan target and separately verified as ignored and untracked.

## 3. Effective API Configuration

Use one validated configuration vocabulary for mock and Requesty:

- `enabled`
- `mode` (`mock` or `real`)
- `provider`
- `endpoint_identifier`
- `requested_model_identifier`
- `prompt_version`
- `response_schema_version`
- `generation_parameters`
- provider-specific non-secret options

The Requesty candidate is:

- endpoint: `https://router.requesty.ai/v1/responses`
- requested model: `openai-responses/gpt-5.6-sol`

These are request values, not proof of exact backend identity. No API key or
key-file path is stored in committed provider configuration.

Configuration is validated before transport construction. Disabled configs,
mode/provider mismatch, missing required values, an unknown provider, a schema
without a registered validator, and unsafe Requesty endpoints all fail closed
before a provider call. The registry constructs the transport and binds the
schema validator selected by `response_schema_version`; `smoke_api.py` does not
hard-code either.

## 4. Provider-Neutral Request, Cache, Retry, and Usage Contracts

### Canonical request metadata

Derive a versioned, immutable metadata record from every `ApiRequest`:

- provider and endpoint;
- requested model;
- prompt and response-schema versions;
- canonical generation parameters;
- canonical payload digest;
- for each image: identifier, media type, byte length, and SHA-256 digest; and
- the final canonical request hash.

Credentials and authorization headers cannot be represented by `ApiRequest`, so
they cannot affect or leak through the hash. The metadata is sufficient to
audit/reconstruct the logical request contract without storing image bytes.

### Cache

A versioned cache envelope stores both canonical request metadata and a
sanitized response record. A hit is accepted only if the filename/key, embedded
request hash, provider, endpoint, requested model, prompt/schema versions,
generation parameters, payload digest, and image metadata exactly match the
current request. Any mismatch or malformed entry raises a typed cache error; it
is never treated as a valid hit.

The first successful provider result retains its provider cost as origin
provenance. A replay is a new logical call with:

- `cache_hit=true`;
- `provider_call_count=0`;
- `retry_count=0`;
- current-call provider cost `0.0`; and
- the requested and returned model identifiers retained.

Origin cost, if exposed for provenance, is a separately named field and is not
added to current-call usage.

### Retry and terminal outcomes

Retry execution returns or raises a terminal outcome carrying total attempts,
provider call count, and retry count. This context is preserved for success,
retry exhaustion, immediate non-retryable failure, and the important sequence
"retryable failure followed by non-retryable failure."

Only rate limits, timeouts, connection failures, and retryable 5xx responses are
retryable. Authentication, other 4xx responses, parse failures, schema failures,
configuration failures, and cache failures are non-retryable. Tests inject mock
failures; the real API is not intentionally rate-limited.

### Usage ledger

Each logical call appends a versioned sanitized record containing:

- request hash and canonical request metadata;
- cache status and actual provider call/retry counts;
- prompt/schema versions and generation parameters;
- provider, endpoint, provider request ID, and safe gateway metadata;
- requested model, provider-returned model, and nullable exact backend/model
  identity with its evidence source;
- input/output/total tokens, latency, and current-call provider cost; and
- either validated P3 payload metadata or a sanitized error classification.

The ledger never contains raw response bodies or arbitrary error payloads.

## 5. Strict P3 Schema

`p3_multimodal_smoke_v1` remains a transport probe with exactly four fields:

- `schema_version`: exactly `p3_multimodal_smoke_v1`;
- `message`: non-empty string;
- `image_observed`: boolean; and
- `structured`: exactly `true`.

The JSON Schema sets `additionalProperties: false`, and the runtime validator
enforces the same exact key and type contract. Instance predictions, boxes,
tracks, operators, surgical labels, and any other P4-looking fields are
rejected. Provider responses are type-checked before schema validation so all
invalid transport outputs become typed parse/schema failures and are ledgered.

## 6. Requesty Responses Transport

The Requesty adapter uses the documented non-streaming Responses endpoint. It
is constructed with a secret value and an injectable HTTP sender so unit tests
never use real credentials or the network.

The request body contains:

- the configured requested model;
- versioned instructions and an `input` user message;
- `input_text` plus one base64 data-URL `input_image` generated from a synthetic
  non-sensitive PNG;
- `text.format` with the strict P3 JSON Schema; and
- only configured generation parameters supported by the Responses endpoint.

The adapter requires a completed response and extracts:

- response `id`;
- response-body `model` as the provider-returned model;
- output text parsed as JSON and validated by the registered schema;
- input, output, and total tokens;
- Requesty's `usage.cost` when present;
- measured end-to-end latency; and
- whitelisted headers such as `x-requesty-provider`,
  `x-requesty-request-id`, `x-requesty-latency-ms`, and
  `x-requesty-cache` when present.

HTTP authentication failure, 429, timeout, connection, retryable 5xx,
non-retryable 4xx, malformed JSON, incomplete responses, missing output text,
and structured-payload failures map to the provider-neutral error taxonomy.
Response bodies are used only in memory for parsing and are not included in
raised messages or persisted records.

## 7. Identity Evidence and P3 Verdict

The implementation maintains three distinct fields:

1. requested model: the configured alias sent to Requesty;
2. provider-returned model: the response-body `model` value; and
3. exact backend/model identity: nullable, with an explicit evidence source.

Neither the requested alias, `x-requesty-provider`, nor a repeated response-body
alias is promoted to exact identity. No naming-pattern heuristic is allowed.
Exact identity is populated only from explicit, auditable response evidence
that identifies the resolved immutable backend/model. Otherwise it remains
unknown.

`P3_STATUS: PASS` requires all P3 contract/transport gates plus explicit exact
identity evidence. If the real response exposes only a floating alias, the real
transport may pass while the overall status remains `PARTIAL`. The report must
not guess or silently equate the three fields.

## 8. Smoke Flow and Evidence

The real smoke performs one logical request twice using the same synthetic PNG,
text, versions, schema, and generation parameters:

1. start with an isolated empty local cache;
2. first invocation: cache miss and one or more accurately counted provider
   attempts, normally one;
3. persist the validated sanitized response;
4. second invocation: exact cache hit with zero provider calls and zero
   current-call provider cost; and
5. write a sanitized summary that proves multimodal input, structured output,
   request hash, response ID, usage, cache behavior, and identity fields.

The artifact may include hashes and relative evidence paths but not the image
bytes, full raw response, key, key-file contents, or a command containing the
key. Runtime cache, usage, and smoke artifacts remain Git-ignored and are not
committed.

## 9. Testing and Verification

Implementation follows test-driven development. Required automated coverage
includes:

- dataset-root precedence, Windows-to-Linux manifest rebasing, resolved-path
  containment, missing paths, traversal, case/drive differences, split counts,
  bundle roles, and all required hashes;
- VID30/VID31 loading and P2 runtime smoke with external upstream roots unset;
- credential file parsing at the first `=`, mutually exclusive CLI inputs,
  redacted representation/errors, and key absence from every persistence path;
- canonical request sensitivity and credential exclusion;
- cache metadata tampering for every bound field and replay cost/accounting;
- retryable-to-non-retryable attempt accounting and every error category;
- enabled/provider/schema configuration routing and fail-closed cases;
- strict schema rejection of extra and P4-looking fields;
- Requesty request construction, multimodal data URL, response parsing, safe
  headers, usage/cost, identity separation, and HTTP/error mapping using an
  injected fake sender;
- mock smoke regression and a fake-network Requesty integration test; and
- final exact-value secret scans.

Final evidence runs use the supported Python 3.11 environment and include:

- targeted P3 and portability tests;
- Ruff;
- full pytest with `CHOLECTRACK20_ROOT=D:\cholec_dataset`;
- a complete P2 runtime smoke with external Cholec80/CholecT50 roots absent;
- the real synthetic-image Requesty miss/hit smoke;
- secret scan; and
- Git diff, status, tracked-file, and ignore checks.

New results append the date, relevant environment settings, pre-evidence commit
SHA, runtime, and result to the historical 79-pass, 78-pass/1-skip, and
95-pass/1-skip evidence. Existing historical statements are not rewritten as
if they were contemporaneous.

## 10. Documentation, Git, and Completion

Update the following without overwriting history:

- `reports/P3_REPORT.md`;
- `docs/V3_1_API_IMPLEMENTATION_AUDIT.md`; and
- `docs/AUTODL_QUICKSTART.md`.

The AutoDL guide will state exactly what to upload, how to set
`CHOLECTRACK20_ROOT`, how to supply the Requesty key separately, how to run the
single-root verifier, and that Cholec80/CholecT50 directories are not required
unless regenerating/auditing the already materialized sidecars.

Git commits include code, tests, non-secret configuration, the bundle manifest,
and reports. They exclude `docs/API.txt`, dataset contents, cache, checkpoints,
temporary files, complete provider responses, and runtime artifacts. Existing
user changes are preserved.

Completion requires requirement-by-requirement evidence that the dataset root
is self-contained, no implicit external data is opened, research semantics are
unchanged, no credential leaked, the real multimodal structured Requesty smoke
worked, contract/error tests passed, the P3 verdict matches actual identity
evidence, Git/report evidence is complete, and no P4/P9 contract was frozen.
