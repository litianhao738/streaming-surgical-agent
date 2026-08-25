# Document Responsibilities

## Research Design Source of Truth

[Streaming SurgicalAgent V3.1-API academic architecture revision](architecture/Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md)

This document is authoritative for:

- research problem definition and paper claims;
- academic architecture and online pipeline semantics;
- module responsibilities and methodological boundaries;
- experiment rationale and interpretation.

## Code Implementation Contract

[Streaming SurgicalAgent V3.1-API Codex implementation specification](architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md)

This document is authoritative for:

- repository structure and implementation scope;
- typed interfaces and engineering contracts;
- phase order, hard gates, and acceptance criteria;
- required tests, commands, artifacts, and reports.

## Precedence

The user's latest explicit instruction takes precedence over both documents.
For research-design questions, use the academic architecture revision. For code
construction and verification, use the Codex implementation specification.

## Current Checkpoint

- P0: `PASS`.
- P1: `PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION`.
- P2: `PASS`; local masked train/infer/checkpoint/artifact smoke completed.
- P3: `PARTIAL`; provider-neutral client/cache/hash/retry/usage and mock
  synthetic-image smoke completed, real API identity smoke remains `BLOCKED`.
- Canonical implementation package: `src/surgical_agent/`.
- Compatibility-only namespace: `src/streaming_surgical_agent/`.
- Active data contract: `<resolved dataset root>/repair_manifest.json`.
- Deferred hard gates: exact API identity at P3, prediction/evaluation granularity before P4, and
  canonical per-sample Gate error at P9.
