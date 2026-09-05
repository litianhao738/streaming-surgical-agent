# Configuration entry points

For the main one-call API predictor, use only
`perception/joint_openrouter_h0.yaml`; offline checks use
`perception/joint_mock_h0.yaml`. See [the run guide](../docs/MAIN_API_PIPELINE.md).

Other `joint_*` and `targeted_*` presets belong to named historical experiments
or the separate Tracker/Gate/Verifier research Pipeline. They are not competing
defaults. In particular, `*fixed6*` presets retain frozen six-frame experiments;
some byte-identical files are separately recorded in preservation manifests.
Do not change those files to make an old experiment look like the new H0.

The main prompt and JSON schema live in the installable package, not in generated
artifacts. Model, sampling, reasoning or prompt changes should use a new versioned
experiment until a paired comparison supports promotion.
