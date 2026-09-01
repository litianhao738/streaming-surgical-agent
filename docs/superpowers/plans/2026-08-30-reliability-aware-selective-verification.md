# Reliability-Aware Selective Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compact per-label confidence, field-level uncertainty, API telemetry, local reliability gating, targeted verification, state-aware memory and event reports.

**Architecture:** Preserve the canonical streaming pipeline and introduce versioned contracts at its existing perception, Gate, verifier, coordinator and artifact boundaries. Keep the initial VLM call minimal; all state transitions, checks, memory actions and report triggers are deterministic local operations.

**Tech Stack:** Python 3.10-3.12, dataclasses, urllib/OpenRouter SSE, JSON Schema, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-reliability-aware-selective-verification-design.md`

## Global Constraints

- Initial wire schema is `joint_perception_reliability_compact_v2` and emits no status or report.
- Instrument/verb/target/IVT are multi-label; phase is single-label.
- Main Single-Pass, Always-Verify and Selective-Verify profiles use one shared backbone.
- Verification may return only Gate-flagged fields.
- Reports are event-level and `template_report` is the default.
- No GT enters runtime inference, Gate, verifier, memory or report generation.

---

### Task 1: Versioned perception contract

**Files:**
- Create: `src/surgical_agent/perception/prompts/perception_schema_reliability_compact.json`
- Create: `src/surgical_agent/perception/prompts/perception_prompt_reliability_compact.txt`
- Modify: `src/surgical_agent/perception/schema.py`
- Modify: `src/surgical_agent/perception/contracts.py`
- Modify: `src/surgical_agent/perception/parser.py`
- Modify: `src/surgical_agent/api/schema.py`
- Modify: `src/surgical_agent/api/providers/mock.py`
- Test: `tests/unit/test_joint_perception_parser.py`
- Test: `tests/unit/test_joint_api_vlm.py`

**Interfaces:**
- Consumes: strict `{task.topk[].id, task.topk[].confidence, uncertainty[]}` JSON.
- Produces: `PerceptionEvidence.ranked_candidates` plus typed field uncertainties and a full-size dense score vector.

- [ ] Write tests that fail because v2 is unregistered and per-label confidence/uncertainty cannot parse.
- [ ] Run the focused parser/request tests and confirm failure names the missing v2 contract.
- [ ] Add the versioned schema, prompt, validator, parser and deterministic mock payload.
- [ ] Run the focused tests and confirm v1 remains compatible.

### Task 2: Provider telemetry and lowest reasoning effort

**Files:**
- Modify: `src/surgical_agent/api/contracts.py`
- Modify: `src/surgical_agent/api/providers/openrouter.py`
- Modify: `src/surgical_agent/api/client.py`
- Modify: `src/surgical_agent/api/usage.py`
- Modify: `src/surgical_agent/api/cache.py`
- Modify: `src/surgical_agent/api/accounting.py`
- Modify: `configs/perception/joint_openrouter_dataset.yaml`
- Test: `tests/unit/test_p3_openrouter.py`
- Test: `tests/unit/test_p3_api_client.py`
- Test: `tests/unit/test_p3_api_usage.py`

**Interfaces:**
- Consumes: OpenRouter streaming chat-completion chunks and usage details.
- Produces: prompt/completion/reasoning/visible token counts, provider, TTFT and total latency on response and ledger records.

- [ ] Write failing SSE and usage tests with literal chunk fixtures and cache replay expectations.
- [ ] Run focused API tests and verify the missing telemetry fails.
- [ ] Add streaming aggregation, canonical telemetry fields and compatibility aliases.
- [ ] Set initial reasoning effort to `none` and lower the compact output cap.
- [ ] Run focused API tests.

### Task 3: Reliability state and local Gate

**Files:**
- Create: `src/surgical_agent/research/reliability/state.py`
- Create: `src/surgical_agent/research/gate/reliability.py`
- Modify: `src/surgical_agent/systems/pipeline.py`
- Modify: `src/surgical_agent/inference/schemas.py`
- Modify: `src/surgical_agent/inference/frame_result_writer.py`
- Modify: `src/surgical_agent/research/memory/store.py`
- Modify: `src/surgical_agent/workflow/state_store.py`
- Test: `tests/unit/test_reliability_gate.py`
- Test: `tests/unit/test_committed_state_stores.py`
- Test: `tests/integration/test_api_dataset_pipeline.py`

**Interfaces:**
- Consumes: context, perception result, evidence and optional train-derived phase-triplet compatibility.
- Produces: ordered Gate findings, Candidate transition, final status and memory action.

- [ ] Write failing tests for each required local check and status/memory transition.
- [ ] Run focused tests and verify failures are caused by absent reliability behavior.
- [ ] Implement the Gate findings and state transition contracts.
- [ ] Persist state audit fields and route memory/workflow writes by final status.
- [ ] Run focused state and pipeline tests.

### Task 4: Field-targeted verification

**Files:**
- Create: `src/surgical_agent/research/verification/prompts/targeted_verification_prompt.txt`
- Create: `src/surgical_agent/research/verification/targeted_api.py`
- Modify: `src/surgical_agent/research/verification/contracts.py`
- Modify: `src/surgical_agent/research/verification/coordinator.py`
- Modify: `src/surgical_agent/systems/api_dataset_system.py`
- Test: `tests/unit/test_targeted_api_verifier.py`
- Test: `tests/unit/test_verification_runtime.py`
- Test: `tests/integration/test_verification_pipeline_runtime.py`

**Interfaces:**
- Consumes: `GateDecision.flagged_fields` and bounded candidate IDs.
- Produces: a full prediction patched only at requested paths plus per-field verification outcomes.

- [ ] Write failing request/parser/coordinator tests proving unflagged tasks cannot change.
- [ ] Run focused verification tests and observe the missing targeted path.
- [ ] Implement the targeted request, response contract and deterministic coordination.
- [ ] Wire Always-Verify to all fields and Selective-Verify to findings only.
- [ ] Run focused verification tests.

### Task 5: Event-level reports

**Files:**
- Create: `src/surgical_agent/research/reporting/__init__.py`
- Create: `src/surgical_agent/research/reporting/contracts.py`
- Create: `src/surgical_agent/research/reporting/template.py`
- Create: `src/surgical_agent/research/reporting/manager.py`
- Create: `src/surgical_agent/research/reporting/writer.py`
- Modify: `src/surgical_agent/systems/api_dataset_system.py`
- Test: `tests/unit/test_event_reporting.py`

**Interfaces:**
- Consumes: finalized events and trigger configuration.
- Produces: atomic event report JSONL and summary counts; supports local template or injected event-level LLM generator.

- [ ] Write failing tests for phase change, window, video-end and status-aware language.
- [ ] Run tests and confirm no current report subsystem satisfies them.
- [ ] Implement contracts, deterministic templates, trigger manager and writer.
- [ ] Integrate default template reports into dataset runs.
- [ ] Run report tests.

### Task 6: Fair experiment profiles and verification

**Files:**
- Modify: `scripts/run_dataset_api_pipeline.py`
- Modify: `configs/experiments/api_single_pass.yaml`
- Modify: `configs/experiments/v3_always_verify.yaml`
- Create: `configs/experiments/v3_selective_verify.yaml`
- Create: `configs/experiments/v3_cascade_efficiency.yaml`
- Modify: `docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md`
- Test: `tests/integration/test_api_dataset_cli.py`
- Test: `tests/unit/test_documentation_contracts.py`

**Interfaces:**
- Consumes: one primary backbone config and optional cascade verifier config.
- Produces: artifacts that explicitly declare backbone equality or efficiency-cascade status.

- [ ] Write failing CLI/config tests for selective profile, reporting options and backbone identity.
- [ ] Implement the minimal CLI/config wiring and artifact metadata.
- [ ] Run focused CLI/config tests.
- [ ] Run all unit and integration tests, then a one-frame mock smoke.
- [ ] If credentials and local data are available, run one authorized real frame with reasoning effort `none` and report measured TTFT/latency without exposing secrets.

