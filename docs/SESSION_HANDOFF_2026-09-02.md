# Session handoff — verified project state

Date: 2026-09-02 (Asia/Shanghai)

This file is the starting point for the next Codex session. It intentionally
separates verified implementation facts from proposed research designs. Do not
infer that a proposed ablation or diagram is already implemented.

## Start the next session with this request

> Read `D:\PythonProject7\docs\SESSION_HANDOFF_2026-09-02.md`, then inspect the
> referenced code. Treat the document as a factual handoff, not as permission
> to implement a new ablation design. First report discrepancies and propose
> one corrected module-ablation specification for approval.

## Repository and verification baseline

- Branch before this cleanup: `main`.
- Last implementation commit before this cleanup:
  `08ddcfc2f6d7f07f9bc3d91e194ac66a3836a0a6`.
- Dataset root used by local verification: `D:\cholec_dataset`.
- Full regression before cleanup: `911 passed, 14 skipped`.
- Ruff, compileall, single-root portability, wheel-resource, and exact-secret
  scans passed.
- `docs/API.txt` and `docs/openaiAPI.txt` are ignored and untracked. Never print
  or commit their contents.
- User-owned `docs/openrouter.pdf` and `openrouter_api_test/input.png` are not
  project cleanup targets.

## What is actually trained or fitted

1. **Tracker: formally trained.** Keep:
   - `artifacts/training/tracker/full/checkpoint.pt`;
   - `artifacts/training/tracker/predicted_tracks.json`;
   - Tracker Validation metrics and manifests.
2. **Phase transition graph: deterministically built**, not a neural model:
   `artifacts/training/phase_transition_graph.json`.
3. **Local five-head Joint Perception: code exists but no formal full
   checkpoint has been completed.**
4. **Learned Benefit Gate: no formal final G1 model exists.** Existing Gate
   artifacts are pilot/probe outputs and are not paper-final models.
5. **Specialists: not three independently implemented experts.** Current
   scope names are routed through one generic targeted verifier.
6. **Memory: bounded storage exists, but reliability-weighted retrieval is not
   implemented as a validated contribution.**

## Active perception API contract

- Active wire version: `joint_perception_gate_owned_compact_v1`.
- Active prompt:
  `src/surgical_agent/perception/prompts/perception_prompt_gate_owned_compact.txt`.
- Active schema:
  `src/surgical_agent/perception/prompts/perception_schema_gate_owned_compact.json`.
- Initial VLM output contains sparse selected IDs and top-k scores for
  Instrument, Verb, Target, IVT, and Phase. It does not own reliability status,
  uncertainty routing, report generation, or chain-of-thought.
- The prior `joint_perception_reliability_compact_v2` prompt performed better
  than a rejected temporary v3 candidate on a small diagnostic. It is retained
  only as rollback/provenance support; it is not the active contract.
- Legacy `joint_perception_frame_v1` and compact-v1 resources remain for old
  cache/artifact compatibility. Do not choose them for a new run without an
  explicit migration decision.

## Verified API facts

- A real OpenRouter request with three synthetic images and the active schema
  succeeded. The identical replay was a cache hit with zero new provider calls.
- The returned model string was `openai/gpt-5.6-sol`, but OpenRouter did not
  expose an independently verifiable immutable backend identity. P3 therefore
  remains `PARTIAL`.
- A real VID30 request reached OpenRouter but returned `content_moderation`.
  This is not evidence of a URL, JSON Schema, or parser defect.
- The local official OpenAI credential probe returned `authentication` at the
  time of testing.
- No full real-dataset API rollout with the active contract has been proved.

## Current executable pipeline order

The implementation in `src/surgical_agent/systems/pipeline.py` executes:

1. reset/continue the video boundary;
2. snapshot Workflow, Memory, and predicted-track state;
3. build the causal context;
4. run Joint Perception;
5. build Evidence Signals;
6. build candidates;
7. run the Gate;
8. optionally call the verifier/Specialist;
9. run the post-verification Coordinator;
10. finalize and persist the prediction/evidence pair;
11. commit Workflow/Memory state.

The post-verification Coordinator location is real and appropriate for merge,
admission, and fallback. However, its current implementation checks scope,
requested fields, candidate admission, touched tasks, and score semantics but
does **not** rerun complete post-merge IVT closure, phase–IVT consistency, and
temporal hard invariants. That is an open implementation gap.

The current `ReliabilityGatePolicy` combines deterministic findings with the
ACCEPT/VERIFY decision. Therefore a separately reusable pre-Gate checker and a
learned Gate are not yet cleanly separated in code.

## Current profile behavior — do not rename conceptually

- `single_pass`: no candidate generator, no verification, no repairing
  Coordinator.
- `always_verify`: flags all five tasks, but the current system assembly still
  wires the generic `TargetedApiVerifier`; it is not yet a faithful independent
  reproduction of the mother repository's complete reflection pipeline.
- `selective_verify`: deterministic `ReliabilityGatePolicy` plus the generic
  targeted verifier.
- `learned_gate`: loads a frozen linear artifact, currently constrained to a
  single `joint` scope, plus the same generic targeted verifier.

## Invalid former 80-frame A/B/C/D interpretation

The removed `run_learned_gate_200_demo.py` must not be treated as a valid formal
ablation runner:

- `A_base` and `B_tracker` used `selective_verify`, not a faithful previous
  method with unconditional Verification/Reflection and All-task Rewrite;
- `C_gate` and `D_full` used the current single-scope learned Gate;
- all verification-enabled groups shared the generic targeted verifier;
- the labels therefore did not isolate the mechanisms shown in the motivation
  figure.

The real 80-frame outputs are preserved under
`artifacts/archive/invalid_protocol_learned_gate_80_demo_20260831` only to
salvage paid API/cache evidence. They are not paper results.

## Mother repository facts

The referenced SurgReflect repository performs base prediction, then phase/IVT
candidate and deterministic reflection operations, optional model-based repair,
and deterministic constraint enforcement again after model repair. It does not
implement the current project's Benefit Gate plus scoped post-verification
Coordinator architecture. Its reported pipeline must not be paraphrased as if
it were already reproduced here.

Its global statistical RAG is described as being built from benchmark gold
labels. This project must not copy that evaluation behavior: any phase/IVT
prior used for formal experiments must be built from Training only.

## Research decisions still requiring a fresh specification

Do not implement these from prior chat prose alone:

1. the exact faithful/casualized definition of `A_base`;
2. whether a pre-Gate deterministic checker is a shared input transform or part
   of the Evidence innovation;
3. whether `C_gate` and `D_full` share selected frames and verification budget;
4. the three Specialist scopes and their distinct prompts/candidate spaces;
5. whether ordinary Memory is log-only or fed back into causal context;
6. exact post-merge hard invariants owned by the Coordinator;
7. the final module-ablation table and attribution rules.

## Recommended immediate next action

Write and approve one short experiment contract before changing runtime code.
It should define every row by executable switches, hold the backbone, samples,
initial cache, ontology, evaluator, and accounting fixed, and state exactly one
mechanism changed per comparison. Only then replace the removed pilot runner.

## Cleanup performed with this handoff

- Removed the downloaded-skill planning/specification remnants under
  `.superpowers/` and `docs/superpowers/`.
- Removed the misleading old 80-frame demo runner.
- Removed the rejected temporary Prompt-v3 diagnostic report after preserving
  its factual conclusion above.
- Removed the non-measured `learned_gate_500_projection` artifact.
- Removed rejected temporary Prompt-v3 and null-fix runtime outputs/caches.
- Preserved the real 80-frame paid API evidence in an explicitly invalid-protocol
  archive.
- Preserved Tracker artifacts, the active prompt/schema, rollback-compatible
  legacy schemas, real API evidence, credentials, dataset files, the user PDF,
  and the user test image.
