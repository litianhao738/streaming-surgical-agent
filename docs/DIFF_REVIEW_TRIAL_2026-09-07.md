# Verifier / Repair diff-review trial — 2026-09-07

Implemented independently of the frozen Qwen H0 and historical grounded runner:

- `src/surgical_agent/research/verification/diff_review.py`: versioned
  `frame_label_diff_review_v1` schema, exact change-list binding and two policies.
- `scripts/run_diff_review_trial.py`: offline preflight, synthetic mock, and an
  explicitly dispatched synchronous Gemini replay using the original images and
  candidates. It does not enable Tracker, Gate, memory or additional rounds.

## What changes

Each assessment judges a frame-level label's presence, not whether a proposed
edit sounds reasonable. ADD requires SUPPORTED plus a reference to the target
original image. REMOVE requires CONTRADICTED, FRAME scope, a declared full-frame
review and that target-image reference. Valid citations bind real inputs but do
not establish that the model's observations are visually correct.

Policy A accepts the whole candidate only when every actual change passes.
Policy B starts from H0 and applies approved changes with explicit dependencies:
new IVTs need their necessary component labels; removing an IVT never deletes
components automatically; a component required by a remaining IVT cannot be
deleted. Unaffected independent heads and instrument 6 remain representable.
Phase stays at H0. Capacity overflow or malformed optional data keeps H0.

The API package now lazily imports its client/registry exports to fix the
pre-existing cold-import cycle through `grounded_repair`. Public API names and
the frozen H0 contract are retained.

## Frozen diagnostic

Source: `artifacts/preflight/grounded_gemini_semantic_supplement_20260906/`.
All four source targets remain in the masked evaluation. Only three have a
changed valid H1 and receive one new review each:

| Target | Changed-label assessments |
| --- | ---: |
| VID103 / 18501 | 3 |
| VID103 / 33751 | 3 |
| VID96 / 14951 | 5 |
| VID96 / 25951 | No candidate; keep H0, zero calls |

The plan freezes source PNGs, their original API encodings, crops, historical
requests and responses, source code and the current pricing snapshot. H1 and the
old final are recomputed from their original records before dispatch. GT values
are read only after new predictions have been persisted; masks remain per task.

Maximum new requests: 3. Budget: USD 0.20, with USD 0.05 reserved before each
request, zero retries. This is a local stopping budget, not a provider-enforced
hard cap. Model and route remain `google/gemini-3.8-flash` / strict Google AI Studio.

This is a diagnostic on previously inspected samples. It cannot establish
generalization or justify tuning against these four labels. A larger fixed
Training comparison is not included in this run; Testing is untouched.

## Validation and commands

The new logic has 37 passing unit tests and the replay runner has 16 passing
integration tests. Tests cover claim completeness, invalid citations, target-only
evidence requirements, shared-component deletion, IVT dependencies, independent
instrument labels, output limits, cold imports, audited transport and the
four-target denominator. Related API and H0 regression checks also pass.
The runner tests include timeout and unpriced-response stopping: only the first
request is dispatched, all four H0 predictions remain, and total cost is unknown.

Real-source offline mock:
`artifacts/preflight/diff_review_agent_mock_20260907/summary.json`.
It makes zero API calls and uses synthetic INSUFFICIENT assessments; its scores
are not model results.

Paid diagnostic output:
`artifacts/preflight/diff_review_gemini_replay_20260907/`.

## Actual paid attempt

The first review request (VID103 / 18501) timed out. One provider attempt was
recorded, with no valid review and no returned price. The accounting guard
stopped immediately with `UNPRICED_CALL_STOP`; the other two planned requests
were not dispatched, and no retry was made.

All four targets retain H0 in both A and B. The stored comparison tables therefore
describe failure fallback, not successfully reviewed model predictions. There
is no new semantic result and no basis to claim either improvement or failure of
the new visual-review method.

`accounting_note.json` explicitly records total added cost as null/unknown.
The original `summary.json` field `added_cost_usd=0.0` is only the sum of known
prices, accompanied by `unpriced_calls=1`; it must not be reported as a free
request. The original run and frozen source are preserved. The current runner's
subsequent accounting fix distinguishes known cost from unknown total cost.

The recommended next experiment remains A as the primary intervention, with B
derived from the same responses as an offline comparison. Resolve the incomplete
provider accounting before any further paid run. Tracker remains a
separate localization ablation, not a prerequisite for this test.

For a fresh output directory, omit `--execute` to inspect the frozen plan:

```powershell
.venv-p2/Scripts/python.exe scripts/run_diff_review_trial.py `
  --source artifacts/preflight/grounded_gemini_semantic_supplement_20260906 `
  --output artifacts/preflight/diff_review_reproduction_new `
  --pricing-snapshot artifacts/preflight/diff_review_readiness_20260907/endpoints.json `
  --max-targets 3 --budget-usd 0.20 --reserve-usd 0.05
```

`--mock` runs the synthetic offline path. Real execution requires `--execute`
and an external `--api-key-file`; existing dispatch locks prevent redispatch.
Refresh prices before a later paid run. Do not commit credentials or silently
repeat a failed or unpriced call.

## Subsequent user-requested small test

A separately authorized small test was executed in a fresh directory after the
user requested another run. Its first review also timed out; remaining requests
were stopped and both unpriced-call records remain distinct. See
[the small-test record](DIFF_REVIEW_SMALL_TEST_2026-09-07.md) for the exact scope,
fallback-only metrics, unknown costs and successful non-generating authentication
check. Neither attempt supplies semantic evidence for the new verifier.
