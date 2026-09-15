# Source-only pipeline setup

The current default uses a retrained 42-feature Gate and a shared five-head
post-Gate panel. `DEFAULT_PGP_GATE_VERSION.json` selects the model version and
local model manifest. `DEFAULT_PIPELINE_VERSION.json` describes the pipeline.
All model artifacts and experiment datasets stay outside version control.

## Local prerequisites

- A locally provisioned Gate model manifest and estimator at the location named
  by `DEFAULT_PGP_GATE_VERSION.json`. Both file hashes are checked at load time.
- The trained Tracker checkpoint, manifest and appropriate CUDA environment.
- The licensed local dataset, prepared selection/scope, causal input images,
  H0 cache and query-video-excluded Training priors expected by the runner.
- Local provider credentials at the configured paths. Never commit API files.
- Report model/request parameters are included under `configs/reporting/`;
  generated reports and calibration outputs remain local.

The selected testing scope lives locally at
`artifacts/evaluation/testing_half_unified_gate_tracker_plan_20260916`.
Scope and source hashes prevent mixing unrelated inputs and model versions.
The repository does not download private data or weights automatically.

## Entry points

`scripts/run_testing_half_complete.py` is the complete Testing entry point:
on-demand Tracker/cache reuse, bounded concurrent inference, chronological
phase finalization, H0/FULL reports and evaluation. It supports `run`, `prepare`,
`execute`, `resume` and `status`. Paid execution requires `--allow-paid` and
remains subject to configured account budgets. Use a fresh `--output` directory
for this Gate version. `resume` applies to an interrupted core inference stage;
report recovery is handled separately by `scripts/retry_testing_half_reports.py`.

`scripts/run_tracker_scheme4_pipeline.py` supplies the Training/replay entry
point. `scripts/run_pipeline.py info` displays the default selection without
requiring private model files.

To rebuild from locally available, sealed Training response caches:

```powershell
python scripts/build_unified_gate_cache.py --output artifacts/training/gate/unified_replay_new --workers 4
python scripts/train_gate_with_phase.py --unified-cache artifacts/training/gate/unified_replay_new --output artifacts/training/gate/unified_gate_new
```

Both steps are offline. Retraining writes a separate model; it does not silently
replace default model pointers or bypass the model/source integrity checks.

## Runtime behavior

Gate keeps the original H0/proposal/Qwen-probe feature prefix. If it opens,
Gemini proposes a phase and a single joint panel verifies all five tasks.
Four-head repair and phase admission use that same panel, retaining prior rules,
ambiguity protection and invalid-response fallback. Maximum logical calls are
nine per routed frame including H0. The pre-Gate Qwen probe is separate from
the post-Gate panel. This maximum is not a wall-time or total-cost guarantee.

Tracker detections and snapshots are cached locally for reuse across experiments.
M3 uses the repaired phase before smoothing, with causal nonrecursive 60-second
votes and chronological finalization after parallel inference. H0-only warmup
frames initialize the stream. FULL reports consume finalized predictions; the
paired H0 reports preserve the baseline. RAM remains disabled.

Core implementation: `src/surgical_agent/research/gate/unified_review.py`,
`tracker_pipeline_v2.py`, `scripts/run_testing_half_pipeline.py`, and
`scripts/run_testing_half_complete.py`.

## Local tests without datasets or paid requests

```powershell
python -m pytest tests/unit/test_unified_review.py tests/unit/test_testing_half_pipeline.py tests/unit/test_testing_half_complete.py tests/unit/test_testing_half_http_policy.py -q
```

Data and model files previously tracked by older revisions are removed from the
current source tree. Existing Git history is not rewritten by this update.
