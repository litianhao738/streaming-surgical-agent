# Session handoff — Verifier capability pilot and current blockers

Date: 2026-09-04 (Asia/Shanghai)

## 1. Start the next session with this request

> 完整阅读 `D:\PythonProject7\docs\SESSION_HANDOFF_2026-09-04.md`，然后逐一检查其中引用的实际代码、配置和 artifact。不要继承旧会话对 Tracker、Gate、Repair 或模型效果的口头结论。先复核 20 帧 pilot 的事实与代码边界，再提出 Verifier 修复方案；在 Repair capability 通过以前，不要开始正式 D0 或 Gate 训练。

This is a dated factual handoff, not a replacement for the user-supplied
Pipeline source of truth. Verify live code before changing anything.

## 2. Documentation authority

Authority order:

1. The user's latest explicit instruction.
2. `docs/architecture/Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md`
   — sole authority for online order, Guard/Gate branches, bounded Repair,
   finalization and causal Memory.
3. `docs/architecture/CANONICAL_PIPELINE_TRACKER_GATE_SPEC_2026-09-03.md`
   — authority for the Tracker × Gate factorial and academic protocol. It may
   not redefine the runtime Pipeline.
4. Live code and generated artifacts — evidence of what is actually implemented
   or executed.
5. Older handoffs and specifications are historical context only.

Do not infer implementation completion from a design document.

## 3. Current research decision

**Do not train the formal Gate yet.**

The 20-frame Training-only capability pilot found zero positive Instrument
repairs, zero positive Interaction repairs, only two positive Workflow repairs,
four harmful verified changes, and incomplete provider coverage. This identifies
a blocker but is too small and coverage-biased for paper-level performance.

The next task is to improve or replace the fixed Verifier and repeat a
predeclared Training-only capability probe. Do not use Gate to hide an absence
of underlying Repair value.

## 4. Frozen Pipeline boundaries

The intended online path remains:

```text
StreamController
→ TrackerModule
→ CausalContextBuilder
→ JointPerception (H0)
→ DecisionSupportBuilder / SafetyValidator
→ RoutingController
     ├─ MandatorySafetyGuard          fixed
     ├─ GatePolicy / deterministic G0 Gate ablation point
     └─ VerificationBudgetController fixed
→ 0/1 bounded TargetedVerifyAndRepair episode
→ DeterministicCoordinator / postcheck
→ OutcomeFinalizer
→ atomic FinalizationTransaction
→ Output_t
→ admitted Event_t is visible no earlier than t+1
```

Non-negotiable meanings:

- Gate runs only on hard-valid H0 and never commits output.
- Online inference executes at most one Gate-selected scope.
- Repair is bounded by the original candidate pool, scope and frozen `N_max`;
  every proposed result is deterministically postchecked.
- Accepted/Verified become final only after atomic finalization.
- Pending does not enter trusted Memory.
- Tracker and Gate are the requested factorial ablations. Perception, Verifier,
  Safety, budget, Coordinator and finalization stay shared across cells.
- Formal JointPerception and Verify omit Tracker state from semantic API input,
  preserving the Tracker ablation boundary.
- Training-only forced counterfactual branches are not online routing. They may
  run concurrently because they share one immutable pre-decision snapshot.

## 5. Live implementation checkpoint

### 5.1 Active API contracts

- Joint H0 version: `joint_perception_gate_owned_compact_v1`.
- Targeted Verify version: `targeted_verification_prompt_v6`.
- Candidate version: `factorized_closure_topk_phase_complete_v3`.
- Verify is blind to H0 selections and upstream scores.
- Phase exposes all seven IDs.
- I/V/T retain components required by frozen selected IVTs.
- IVT retains up to eight perception candidates.
- V6 ontology is scope-local: Instrument gets Instrument names, Workflow gets
  Phase names, and Interaction gets component names plus candidate IVT rows.
- `required_fields` preserves frozen unrequested-IVT closure.
- Earlier images are causal context; labels describe only `target_frame_id`.

Inspect:

```text
src/surgical_agent/research/verification/targeted_api.py
src/surgical_agent/research/verification/prompts/targeted_verification_prompt_v6.txt
src/surgical_agent/perception/ontology_prompt.py
src/surgical_agent/research/verification/hypotheses.py
```

### 5.2 Counterfactual latency optimization

`FormalGateCounterfactualCollector.collect()` builds all legal scope loops from
the same H0/candidate/context/evidence snapshot, runs them in a bounded
`ThreadPoolExecutor`, then serializes proposals in canonical order:

```text
instrument_presence → interaction → workflow
```

This changes wall-clock scheduling only. Online Gate still selects at most one
scope. Concurrency prerequisites were fixed:

- `ProviderCallBudget` uses an `RLock` and counts exact transport attempts;
- `UsageLedger` serializes read-modify-atomic-replace transactions;
- `FileApiCache` retains immutable non-overwriting entries.

Inspect:

```text
src/surgical_agent/research/gate/counterfactual.py
src/surgical_agent/api/budget.py
src/surgical_agent/api/usage.py
src/surgical_agent/api/cache.py
```

### 5.3 Pilot-only API failure continuation

`scripts/collect_formal_gate_counterfactuals.py` remains fail-fast by default.
The explicit `--allow-api-failures` flag is for coverage/debug pilots only:

- terminal H0 failures are recorded in `api_failures.json`;
- later frames continue;
- failed H0 frames do not become Gate records;
- Verify failures remain missing route labels/bounded fallback outcomes;
- manifest status becomes `COMPLETE_WITH_API_FAILURES`.

Do not use this flag to silently construct formal D0.

### 5.4 Tracker state

- Do not retrain the Tracker detector because this Verifier pilot failed or
  because the VLM visual window changed.
- Five-fold OOF index:
  `artifacts/training/tracker_oof5/oof/index.json`.
- The index covers ten Training videos and passed artifact-hash validation.
- Transport archive: `tracker_oof5_complete_20260903.tar.gz`.
- VID31 has OOF prediction coverage and frame labels but no instance boxes; it
  is excluded from box metrics, not from all Gate supervision.
- The primary visual contract is now three contiguous 1 FPS observations. The
  causal buffer resets when the CholecTrack20 frame-ID increment exceeds 25,
  and Tracker association generation now uses the same gap bound. The existing
  detector checkpoints are reusable, but the dated OOF association artifacts
  above predate this rule and are historical until regenerated.
- Six-frame H0/Verifier caches, counterfactual records and learned-Gate examples
  cannot be mixed with the three-frame protocol. Recollect Gate data only after
  Repair admission is fixed and a Training-only capability pilot demonstrates
  positive benefit under controlled harm.

## 6. Verified regression state

Latest checks after V6, concurrency and pilot-resilience changes:

```text
Ruff: all checks passed
pytest: 955 passed, 14 skipped in 30.71s
```

Use `.\.venv-p2\Scripts\python.exe` for tests/runtime and
`.\.venv\Scripts\python.exe -m ruff` for Ruff.

The worktree is intentionally dirty and contains extensive uncommitted work.
Do not reset, checkout, delete or overwrite unrelated changes.

## 7. Real single-frame timing evidence

Controlled fixed-H0 artifact:
`artifacts/preflight/openrouter_real_vid02_one_frame_controlled_v6_parallel`

- H0 cache hit; three V6 Verify calls fresh and concurrent.
- Wall time `31.326 s`.
- Verify input/output `13,143 / 549` tokens.
- Old V5 input on the fixed H0 was `16,013` tokens: V6 reduced input about
  `17.9%`.
- Old V5 sequential provider latencies summed to about `64.427 s`.

Fully cold artifact:
`artifacts/preflight/openrouter_real_vid02_one_frame_v6_parallel_cold`

- One fresh JointPerception plus three fresh concurrent Verify calls.
- Wall time `56.702 s`.
- H0 latency `23.849 s`.
- Verify latencies `21.648`, `21.719`, `22.826 s`, overlapped.
- Input/output `18,060 / 872` tokens.
- Provider-reported cost `$0.0442638`.
- Zero retry and zero parse/schema failure.

Observed TTFT was about 19–22 seconds, so provider inference/queueing dominated;
the HTTP layer was not replaced merely to claim connection-pooling speedup.

## 8. Twenty-frame Training-only capability pilot

### 8.1 Protocol

- Ten Tracker-OOF Training videos, two timeline-spread samples per video.
- Twenty attempted frame identities.
- One Joint H0 plus three same-snapshot scope counterfactuals per covered frame.
- GT joined only after all provider calls for each video.
- OpenRouter requested `openai/gpt-5.6-sol`, strict OpenAI routing.
- Fixed causal images: maximum six, low-detail history, auto-detail target.
- `N_max=1` in `configs/experiments/final_pipeline_shared.yaml`; this pilot does
  not establish it as academically optimal.
- `--allow-api-failures` was used because this was a coverage pilot.

Command used:

```powershell
.\.venv-p2\Scripts\python.exe scripts/collect_formal_gate_counterfactuals.py `
  --dataset-root D:\cholec_dataset `
  --tracker-oof-index artifacts/training/tracker_oof5/oof/index.json `
  --api-config configs/perception/joint_openrouter_final_fixed6.yaml `
  --api-key-file docs/API.txt `
  --max-frames-per-video 2 `
  --max-provider-calls 400 `
  --allow-api-failures `
  --output-dir artifacts/preflight/openrouter_real_gate_pilot20_v6_resilient `
  --cache-root artifacts/final_pipeline_cache/openrouter_real_gate_pilot20_v6_resilient
```

Do not rerun into those same paths: successful responses are cached but terminal
failures are not, and another run would mix usage accounting across attempts.

### 8.2 Coverage, time and accounting

```text
attempted frames:               20
completed observations:        17
H0 terminal moderation fails:   3
logical/provider calls:         71 / 71
successful calls:              62
failed calls:                   9
retries:                        0
wall time:                      811.203 s (13 min 31 s)
input tokens:                   219,829
output tokens:                   14,211
total tokens:                   234,040
provider-reported cost:          $0.6201929
```

All nine failures were non-retryable `403 content_moderation`:

- H0 failures: `VID02:76376`, `VID103:4126`, `VID11:676`;
- six additional scope-call failures among otherwise evaluable frames.

H0 coverage was 17/20 (`85%`). Metrics below are conditional on provider
acceptance and may be selection-biased.

### 8.3 H0 results on 17 evaluable frames

Mean H0 masked utility was `0.696078`; three frames had utility zero.

| Task | Correct | Exact accuracy | GT representable by candidate pool |
|---|---:|---:|---:|
| Instrument | 12/17 | 70.6% | 14/17 (82.4%) |
| Verb | 3/6 | 50.0% | 6/6 (100%) |
| Target | 3/6 | 50.0% | 5/6 (83.3%) |
| IVT | 3/5 | 60.0% | 4/5 (80.0%) |
| Phase | 11/17 | 64.7% | 17/17 (100%) |

Different denominators are expected because sparse frame labels do not provide
valid supervision for every task at every frame.

### 8.4 Scope benefit and harm

`benefit_label=1` means strictly positive masked utility delta. Zero includes
equal and harmful verified outcomes. Null means API failure or unresolved status.

| Scope | Observed | Improved | Equal | Harmed | Missing |
|---|---:|---:|---:|---:|---:|
| Instrument presence | 14 | 0 | 12 | 2 | 3 |
| Interaction | 14 | 0 | 13 | 1 | 3 |
| Workflow | 13 | 2 | 10 | 1 | 4 |

Missing routes:

- Instrument: 2 API failures + 1 semantic unresolved;
- Interaction: 3 semantic unresolved;
- Workflow: 4 API failures.

Positive Workflow repairs:

- `VID13:1`: utility `+0.50`;
- `VID96:38026`: utility `+0.25`.

Harmful verified changes:

- Workflow on `VID02:6701`: `-0.50`;
- Interaction on `VID13:1`: about `-0.1667`;
- Instrument on `VID13:24376`: about `-0.1667`;
- Instrument on `VID17:34026`: about `-0.1667`.

A GT-aware offline oracle choosing only the two beneficial Workflow branches
would raise mean utility on the 17 covered frames from about `0.696` to `0.740`.
This is an undeployable upper bound, not Gate performance.

### 8.5 Interpretation

The dominant issue is Verifier selection, not only candidate recall:

- Phase GT was always in the pool, yet Workflow repaired only two H0 Phase
  errors and corrupted one correct Phase.
- Instrument GT was reachable in 14/17 frames, but Instrument Verify produced
  no positive repair and harmed two frames.
- Interaction produced no positive repair despite good candidate reachability.
- JointPerception and Verify use the same requested model. Blind V6 reduced
  anchoring inputs but did not eliminate same-model self-confirmation.

Returned `Verified` is not GT correctness. It means a decisive schema-valid
bounded selection survived deterministic admission.

## 9. Artifact locations and checksums

Primary pilot directory:

```text
artifacts/preflight/openrouter_real_gate_pilot20_v6_resilient/
```

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `7cc6c226f37afdb22d2799934d1843312ea10fefe0d2b085834181da068b1b32` |
| `counterfactuals.jsonl` | `2d20a6953ddab83747cf6c8f18f41643b6e53ebc3f50075404004634a409a407` |
| `api_failures.json` | `1801af1910fffe6cf725c1801aea28b9a2510fb0c74bff2e2972a9193195e0b9` |
| `api_usage.jsonl` | `4683ad51df13aba556aecceed36626fb1322db17ae9e7a020ed147c393c5b762` |

Successful response cache:
`artifacts/final_pipeline_cache/openrouter_real_gate_pilot20_v6_resilient/`.
Do not edit cache envelopes manually.

## 10. Open blockers

### A. Repair capability

Instrument and Interaction have no observed positive benefit. Workflow benefit
is sparse and accompanied by harm. There is insufficient evidence for a useful
three-scope Gate.

### B. Provider coverage

OpenRouter strict OpenAI routing rejected 15% of H0 requests in this sample and
six Verify requests. Silently dropping them would bias D0 and evaluation. Use a
provider/model with a suitable de-identified medical-image policy or predeclare
a transparent failure/fallback policy. Do not evade moderation through image
distortion or misleading prompts.

### C. Cost identity

OpenRouter provides cost but no immutable backend identity. In an earlier repeat,
the identical request hash
`367caa894a7d4cfdb205be937de9c66f243f56528f08a06a2854d893e3be1c04`
returned `$0.011291` and `$0.0016908` on fresh calls with the same token counts.
Treat cost as provider-reported audit metadata, not a stable price estimate.

### D. Statistical sufficiency

Twenty attempts, 17 covered frames and two positives cannot support Gate fitting,
threshold selection, confidence intervals or Test claims.

## 11. Recommended next-session sequence

1. Re-audit live code and recompute pilot tables from cache plus Training GT.
2. Freeze capability criteria before new paid calls: rescue, harm, missingness,
   candidate recall, net utility, token/cost and provider coverage by scope.
3. Run a paired Training-only comparison of V6 against a fixed heterogeneous or
   demonstrably stronger visual Verifier using identical images, candidates and
   schema. Model selection is capability development, not a Tracker/Gate
   ablation; once selected it stays shared across all four cells.
4. Preserve the factorial boundary. Do not feed Tracker only to one Verifier or
   cell. Tracker-conditioned Repair is a separate design change requiring user
   approval and a revised factorial interpretation.
5. Resolve provider coverage and pin routing/model identity as tightly as the API
   permits. Any fallback must be predeclared and reported, not chosen after GT.
6. Do not tune on these 20 and report the same 20 as confirmation. Use a separate
   Training-only confirmation subset.
7. Only after positive Repair value with controlled harm: freeze `N_max`, create
   formal D0, train video-cross-fitted bootstrap Gate, then proceed through
   D1/G1/Validation under the canonical protocol.

## 12. Commands for local inspection

Full validation:

```powershell
cd D:\PythonProject7
.\.venv\Scripts\python.exe -m ruff check .
.\.venv-p2\Scripts\python.exe -m pytest -q
```

Inspect existing pilot without provider calls:

```powershell
Get-Content artifacts\preflight\openrouter_real_gate_pilot20_v6_resilient\manifest.json
Get-Content artifacts\preflight\openrouter_real_gate_pilot20_v6_resilient\api_failures.json
Get-Content artifacts\preflight\openrouter_real_gate_pilot20_v6_resilient\counterfactuals.jsonl
```

Validate Tracker OOF artifacts:

```powershell
.\.venv-p2\Scripts\python.exe -c "from pathlib import Path; from surgical_agent.tracking.oof_index import load_tracker_oof_index; i=load_tracker_oof_index(Path(r'artifacts/training/tracker_oof5/oof/index.json')); [i.provider_for(v) for v in sorted(i.video_to_artifact)]; print('OOF INDEX AND ARTIFACT HASHES: PASS')"
```

Credential file: `docs/API.txt`. Never print or commit it. The user authorized
the completed pilot; announce any materially larger paid run before starting it.

## 13. Latest files to inspect

```text
src/surgical_agent/api/budget.py
src/surgical_agent/api/usage.py
src/surgical_agent/perception/ontology_prompt.py
src/surgical_agent/research/gate/counterfactual.py
src/surgical_agent/research/verification/targeted_api.py
src/surgical_agent/research/verification/prompts/targeted_verification_prompt_v6.txt
scripts/collect_formal_gate_counterfactuals.py
tests/unit/test_p3_api_client.py
tests/unit/test_p3_api_usage.py
tests/unit/test_targeted_api_verifier.py
tests/unit/test_final_pipeline_contracts.py
docs/AUTODL_TRAINING_CHECKLIST.md
docs/architecture/Streaming_Surgical_Final_Pipeline_Architecture_and_Pseudocode.md
docs/architecture/CANONICAL_PIPELINE_TRACKER_GATE_SPEC_2026-09-03.md
```

No commit was created. Preserve unrelated worktree changes.

## 14. One-sentence handoff

The Pipeline and concurrent Training-only collector are consistent and fully
regression-tested, but the same-model V6 Verifier has inadequate Instrument and
Interaction value, sparse Workflow benefit, nontrivial harm and incomplete
provider coverage; fix and revalidate Repair before training Gate.
