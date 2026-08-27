# Task 1: Gold-Free Dataset Selection Report

## Implementation

- Added `CholecTrack20DatasetAdapter.iter_inference_video()` to produce
  `InferenceSample` objects from media/frame identity only. It uses test JSON
  solely to read test-frame identities and otherwise resolves available media
  frames directly; it does not construct evaluation or supervision targets.
- Added immutable `RolloutSelection` and `resolve_rollout_selection()` for
  bounded engineering selection and complete, sorted paper-split selection.
  The selector normalizes video IDs, validates mode/bounds, rejects paper
  truncation, checks video-major increasing sample order, and exposes frozen
  frame counts.
- Added unit coverage for engineering bounds, case normalization, complete
  sorted paper selection, immutable frame counts, provider-call count, and
  noncanonical frame ordering. Added local-data integration coverage for PNG
  and MP4 gold-free inference samples.

## Files Changed

- `src/surgical_agent/data/dataset.py`
- `src/surgical_agent/data/api_rollout_selection.py`
- `tests/unit/test_api_rollout_selection.py`
- `tests/integration/test_p2_local_pipeline.py`

## TDD Evidence

### RED

Command:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py
```

Result: collection failed as expected with
`ModuleNotFoundError: No module named 'surgical_agent.data.api_rollout_selection'`.

### GREEN

Command:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py
```

Result: `7 passed in 0.16s`.

## Verification

Focused tests and lint:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py tests\integration\test_p2_local_pipeline.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src\surgical_agent\data tests\unit\test_api_rollout_selection.py tests\integration\test_p2_local_pipeline.py
```

Results:

- `7 passed, 3 skipped in 4.82s`
- `All checks passed!`

Full suite:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q
```

Result: `570 passed, 14 skipped in 24.43s`.

## Self-Review

- Confirmed the new iterator does not call annotation parsing or either
  supervision loader, and does not produce evaluation objects.
- Confirmed engineering selection is explicitly bounded and paper selection
  cannot be truncated.
- Confirmed samples are checked against the selected video sequence and frame
  IDs are strictly increasing within each video.
- Confirmed `frame_counts` is a read-only mapping and `RolloutSelection` is a
  frozen dataclass.
- Ran `git diff --check`; it produced no whitespace errors.

## Concerns

No implementation concerns. The local-data integration checks are skipped in
this environment because `CHOLECTRACK20_ROOT` is not configured; this is the
test module's intended skip condition.
