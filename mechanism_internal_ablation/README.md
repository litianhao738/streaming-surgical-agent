# Gate mechanism baselines

Source implementation of Random, Uncertainty-only, Rule-based and the existing
Learned PG Gate comparator. Definitions, formulas, missing-score behavior and
pseudocode are documented in [GATE_BASELINES_DEFINITIONS.md](GATE_BASELINES_DEFINITIONS.md).

The complete pipeline is described in [SOURCE_ONLY_PIPELINE.md](../docs/SOURCE_ONLY_PIPELINE.md).
`python scripts/run_pipeline.py info` prints the selected full pipeline without
loading local weights or making API calls. The full pipeline retains the
five-head Probe, MAR, Tracker, output cleanup and causal phase smoothing.

## Implementations

- `src/fixed_threshold_gate.py`: independent Bernoulli Random (p=0.5, seeds
  0–19), uncertainty threshold 0.5, and rule threshold 0.5. No top-K matching.
- `src/mechanisms.py`: rule score and shared repair mechanisms. Its historical
  `selections` function is the separate matched-budget experiment.
- `src/metrics.py`: masked exact-set accuracy, micro precision/F1 and repair counts.
- `src/run_mechanism_internal_ablation.py`: prepare/collect/replay/score for the
  shared fixed-five-seat small-batch experiment. Only `collect --allow-paid`
  dispatches new model requests, subject to its frozen plan and account budgets.

The three heuristic baselines are offline routing policies over shared snapshots;
this release does not add a `--gate random` option to the complete pipeline CLI.
The small-batch ablation scores before Tracker/cleanup/phase smoothing. Apply the
same downstream processing to every arm when comparing complete-pipeline results.

## Offline replay

With a locally completed, compatible small-batch source run:

```powershell
python -B mechanism_internal_ablation/src/fixed_threshold_gate.py --source mechanism_internal_ablation/outputs/testing480_20260920_r3 --output mechanism_internal_ablation/outputs/fixed_threshold_new
```

This verifies frozen sources and saved predictions, recomputes the routing
decisions, seals predictions, and then scores against local GT masks. Use a new
output directory. Raw data, saved responses, priors, trained Gate/Tracker weights,
credentials and generated results are not included. See `protocol.json` for the
expected local artifact references; published parameters are not a claim that a
fresh clone contains completed experiments. Historical sealed caches may require
the exact source revision recorded by their manifests.

## Offline checks

```powershell
python -B -m pytest mechanism_internal_ablation/tests tests/unit/test_unified_review.py tests/unit/test_testing_half_pipeline.py tests/unit/test_testing_half_complete.py tests/unit/test_testing_half_resume.py tests/unit/test_qwen_half_pipeline.py -q
```

Random score perturbations, synthetic tables and full-population projections are
not included as experimental results in this source release.
