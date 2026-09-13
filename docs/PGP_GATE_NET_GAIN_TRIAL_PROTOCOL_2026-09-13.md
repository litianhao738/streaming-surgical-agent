# PGP net-gain offline trial (fixed before fitting)

Current dual-threshold PGP remains the default selected in DEFAULT_PGP_GATE_VERSION.json. This experiment cannot promote itself. No Tracker inputs, paid API, Testing, VID110, deployment, deletion or replacement of existing results.

Reuse the hash-verified 6,059 original-attempt Training replay rows from pgp_gate_no_tracker_20260913_r1. Keep all 866 features, prior, two Qwen probes, four branch actions and causal exact stopping unchanged. Fit fresh video-balanced, class-balanced logistic heads with the existing estimator configuration.

The single experimental change is coherent net-gain branch routing: help/harm are positive/negative merged-view branch frame-F1 changes, respectively. Interaction averages four heads; phase uses its own head. Route by help score minus harm score, one threshold per branch. Scores are uncalibrated rankings, not expected F1 or probabilities of correctness. The fixed 17-value grid in pgp_net_gain.py produces 289 joint policies.

Use four-video outer LOVO and three-video inner LOVO. Select only from inner out-of-fold scores under unchanged original-label constraints: pooled F1 at least both cheap and full, errors no greater than either, every video no worse than cheap on both metrics, actual gain, fewer logical calls than full continuation with the same probes/stopping. Minimize calls, then maximize worst-video F1 delta and pooled F1, minimize errors, fixed policy index. If infeasible, skip both and record failure while charging all five prefix calls.

Report outer results separately from thresholds selected on all Training OOF scores. The latter is tuning-set performance. Prior inputs are not fully rebuilt within outer folds, and these four videos have been repeatedly used for development; neither result establishes independent generalization. Preserve failed results without retuning against outer labels. Compare the pinned default, cheap, full, old whole-frame Gate and a fixed 300-draw same-video call-cap block-random control.
