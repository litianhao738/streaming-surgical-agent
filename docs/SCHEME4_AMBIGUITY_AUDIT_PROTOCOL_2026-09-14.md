# Scheme 4 ambiguity admission and IVT recall audit

This protocol is written before running the new comparison. Scope: the sealed 6,059 Training response caches, fixed v2.2 Gate routes, and the two separately archived live targets. No API calls, Testing data, new model fitting, or default changes.

Compare v2.2 against four fixed research policies:

- `qwen4_add`: admit only missing grasp/retract IVTs from the existing candidate pool when the already-consumed, valid Qwen IVT rating and all three component ratings are at least 4. The instrument and target must already be in the final v2.2 output; the phase-conditioned leave-video-out IVT prior must be at least 0.01. Add the corresponding verb. Do not delete or replace existing active IVTs.
- `qwen5_add`: same, but require the IVT rating to be 5 (component ratings remain at least 4).
- `consumed_panel_add`: only when the unchanged route actually consumes all five reviewers, restore new ambiguous IVTs admitted by the existing panel/phase-prior selector before ambiguity repair. Require the final instrument and target to be present; add their verbs. No access to unconsumed reviewers.
- `consumed_panel_release`: only on those five-reviewer routes, remove ambiguity protection for verb and IVT, then apply the same M1/M2 and v2.2 cleanup. This measures the cost of the broad release; it is not presumed safe.

For additions exceeding any final schema capacity, keep the entire original output (no GT-based truncation). Every policy receives predictions, ratings and leave-video-out priors only; GT is loaded only for subsequent scoring and diagnosis. All policies preserve call counts and Gate decisions.

Report pooled/per-video five-head F1, IVT TP/FP/FN, corrected versus newly introduced errors, changed-frame gains/losses, and schema fallbacks. A development candidate must improve IVT F1 and pooled mean F1, reduce total errors, and avoid deterioration in IVT F1/mean F1/errors on every video. Even a passing candidate requires independent validation; Training screening does not authorize an automatic default promotion.

Diagnose IVT 19 (grasper/retract/liver), plus all IVTs, through H0, candidate pool, cheap output, actual consumed-review output before protection, after protection and final v2.2. Separate missing candidates, Gate skips, early stopping, panel/prior selection, ambiguity rollback and output filtering. Stage counts need not be monotonic because later stages can add labels. Missing candidates cannot be assigned invented reviewer judgments.

The archived two-target run is analyzed separately and never mixed with older full-cache responses for the same frame IDs. Its hypothetical policies use only its own Qwen answer and priors; no unavailable paid reviewers are simulated.
