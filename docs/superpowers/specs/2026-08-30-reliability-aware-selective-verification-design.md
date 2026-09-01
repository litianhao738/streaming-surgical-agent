# Reliability-Aware Selective Verification Design

## Goal

Reduce average API latency and make every prediction auditable without changing
the paper comparison backbone. One initial VLM call emits a compact structured
hypothesis; deterministic local checks decide whether only specific fields need
verification; reports are generated at event boundaries rather than per frame.

## Initial wire contract

The provider-facing contract is versioned as
`joint_perception_reliability_compact_v2`. It does not mutate compact v1.

- Instrument, verb, target and IVT remain multi-label; phase remains single-label.
- Every returned top-k label has `{id, confidence}`. Confidence is a VLM
  self-report used for ranking and routing, not a calibrated probability.
- `uncertainty` is a bounded list of `{path, reason, alternative_ids}`. Paths are
  JSON-pointer-like task paths and each path may occur once.
- The response contains no status, evidence prose, chain of thought, report,
  clinical significance or next step.
- The orchestrator assigns `Candidate` after strict parsing. Status is never
  supplied by the model.

## API observability

Every provider response and usage row records prompt tokens, completion tokens,
reasoning tokens, locally derived visible-output tokens, gateway provider,
time-to-first-token and total latency. OpenRouter uses SSE streaming so TTFT is
the elapsed time to the first non-empty content delta; cache replays use
`time_to_first_token_ms=null` and zero current-call latency. Compatibility aliases
for the existing input/output/latency fields remain available during migration.

The initial GPT-5.6 Sol configuration uses the lowest model-supported effort,
`reasoning.effort=none`, and a compact output cap. OpenRouter's public model
metadata is the source of support; a real one-frame smoke validates usability.

## Local Gate and targeted verification

The Gate receives the parsed perception result, causal context and deterministic
evidence. It produces ordered findings and flagged fields for:

1. ontology validity;
2. IVT component closure;
3. selected triplet compatibility;
4. phase-triplet compatibility when a train-derived compatibility artifact is
   available;
5. temporal jumps against the prior committed state and phase graph;
6. low per-label confidence; and
7. VLM-declared field uncertainty.

No findings transitions Candidate to Accepted. Findings request verification for
only the flagged fields. A targeted verifier receives only those paths and their
bounded candidates, returns only those paths, and the deterministic coordinator
rejects out-of-scope or out-of-pool changes. Always-Verify flags all five fields;
Selective-Verify uses Gate findings; Single-Pass makes no verifier call. All three
main profiles reuse the same requested and returned backbone identity. A separate
cascade configuration may use a faster initial model and stronger verifier and is
reported only as an efficiency variant.

## Reliability state and memory

Each persisted prediction carries `initial_state`, `gate_reasons`,
`flagged_fields`, `repaired_fields`, `final_status` and `memory_action`.

- Accepted: Gate accepted the initial hypothesis.
- Verified: all requested fields were confirmed or safely repaired.
- Pending: verification was incomplete, unavailable or explicitly unresolved.
- Rejected: verification rejected the candidate without a safe admitted repair.

Verified events enter reliable memory. Accepted events enter reliable memory only
above the high-confidence threshold; otherwise they enter short-term memory.
Pending events enter a pending buffer. Candidate and Rejected events are not
written. Workflow state is updated only from Accepted or Verified events.

## Event reports

Reports are separate event artifacts, not frame-level prose. A report manager
flushes at phase change/event-segment end, fixed frame windows and video end.
`template_report` is deterministic and default. `llm_report` is an injectable
event-level generator retained for paper-side quality evaluation and never runs
inside the initial perception call.

Template language is status-aware: Verified is stated as confirmed, Accepted as
a high-confidence observation, and Pending uses explicit uncertainty language.
Candidate and Rejected events are excluded from reports.

## Compatibility and evaluation

Compact v1 remains registered for old artifacts. V2 requests use a distinct
prompt/schema version and therefore a distinct request hash. Existing evaluator
dense vectors remain full-size and zero-fill only absent top-k scores; paper text
must not call those zero-filled vectors full model logits.

Primary comparison profiles freeze one backbone. Report latency, TTFT, reasoning
tokens, visible tokens, provider calls, verification rate, final status counts and
repair fields alongside the existing frame metrics.

