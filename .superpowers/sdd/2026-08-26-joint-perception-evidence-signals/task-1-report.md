# Task 1 Report: Freeze Joint Perception Contracts and Score Semantics

## Status

DONE

## Implementation

- Added the `surgical_agent.perception` package and immutable joint-perception contracts:
  `RankedCandidate`, `EvidenceReference`, `PerceptionEvidence`,
  `ApiCallProvenance`, `JointPerceptionResult`, and `PerceptionBackend`.
- Enforced the five task heads from `TASK_CLASS_COUNTS`, ontology ID bounds,
  unique descending candidate rankings, finite `[0, 1]` scores, exact confidence
  keys, causal non-negative evidence frame IDs, and the closed evidence-reference
  vocabulary.
- Added `PerceptionEvidence.local_unavailable(frame_id)` with explicit empty
  evidence and null confidence values for all five heads.
- Added `ApiCallProvenance.from_response()` with only allowlisted API identity and
  accounting fields, plus `ApiCallProvenance.local()` for non-API runs.
- Added `score_semantics` to `InitialPrediction` and `PredictionRecord`, defaulting
  to `probability_v1`, restricted to `probability_v1` and
  `uncalibrated_rank_v1`, and propagated it through `PredictionFinalizer`.
- Added contract tests covering safe fields, validation boundaries, local
  unavailable evidence, and finalizer propagation.

## Files changed

- `src/surgical_agent/perception/__init__.py`
- `src/surgical_agent/perception/contracts.py`
- `src/surgical_agent/inference/schemas.py`
- `src/surgical_agent/systems/pipeline.py`
- `tests/unit/test_joint_perception_contracts.py`

## Test commands and results

Interpreter used for every command:
`D:\PythonProject7\.venv-p2\Scripts\python.exe`

Focused and legacy tests:

```text
python -m pytest tests/unit/test_joint_perception_contracts.py tests/unit/test_p2_pipeline_artifacts.py -q
...........                                                              [100%]
11 passed in 2.60s
```

Complete suite:

```text
python -m pytest -q
345 passed, 13 skipped in 9.81s
```

## RED evidence

Before implementation:

```text
python -m pytest tests/unit/test_joint_perception_contracts.py -q
ModuleNotFoundError: No module named 'surgical_agent.perception'
1 error during collection
```

The expected failure was the missing new perception contract package.

For the finalizer propagation behavior, the implementation line was temporarily
removed after the regression test was added:

```text
python -m pytest tests/unit/test_joint_perception_contracts.py::test_score_semantics_is_restricted_and_finalizer_copies_it -q
F                                                                        [100%]
AssertionError: assert 'probability_v1' == 'uncalibrated_rank_v1'
1 failed
```

## GREEN evidence

After implementing the contracts and restoring finalizer propagation:

```text
python -m pytest tests/unit/test_joint_perception_contracts.py::test_score_semantics_is_restricted_and_finalizer_copies_it -q
.                                                                        [100%]
1 passed in 2.35s
```

The focused and complete suite results above are the final passing results.

## Self-review

- Confirmed all new dataclasses are frozen and nested evidence mappings are
  normalized to immutable mapping proxies/tuples.
- Confirmed no contract dataclass admits credential, raw-response, GT, or
  free-form-reasoning fields.
- Confirmed API provenance excludes parsed payload, provider request IDs,
  costs, latency, and other non-allowlisted fields.
- Confirmed legacy predictions retain the `probability_v1` default.
- `git diff --cached --check` completed without whitespace errors.

## Concerns

None. Backend migration and evidence persistence are intentionally deferred to
the later tasks in the approved plan.
