# Joint Perception and Evidence Signals Design

Status: awaiting written-spec review
Date: 2026-08-26
Target repository: `D:\PythonProject7`

## 1. Decision Summary

The next implementation slice replaces the local smoke perception slot with a
frame-level, causal, OpenRouter-backed Joint Perception adapter using
`openai/gpt-5.6-sol`. The adapter produces one structured multi-task
hypothesis plus raw, explicitly non-calibrated evidence. A separate,
deterministic extractor converts that raw evidence and finalized causal history
into an auditable `EvidenceProfile`.

This slice does not implement the learned Benefit Gate, Specialist calls,
deterministic repair coordination, instance detection, or the full predicted
tracker. The existing `NeverVerify` path remains active so that the new
perception and evidence contracts can be evaluated independently.

The approved evaluation hierarchy is:

1. frame-level video-wise mAP for Instrument, Verb, Target, and IVT;
2. video-wise Phase macro-F1 plus Accuracy;
3. instance detection and Tracking metrics in separate future tables, never
   averaged into the frame-level score;
4. Gate and repair metrics only after those components exist.

## 2. Research Purpose

The subsystem must implement the left side of the proposed method:

```text
causal stream (frames <= t)
    -> one Joint Perception call
    -> structured InitialPrediction + PerceptionEvidence
    -> deterministic EvidenceSignalExtractor
    -> EvidenceProfile
    -> existing NeverVerify policy
```

The design separates hypothesis generation from evidence assessment. The VLM
may report candidates and self-assessment, but it must not decide that its own
prediction is correct, decide whether verification runs, or modify a finalized
prediction. Evidence values are inputs to a future Benefit Gate, not correctness
probabilities.

## 3. Frozen Boundaries

- Inference accepts no ground-truth object, label mask, evaluation target, or
  future frame.
- The target frame is the greatest frame ID in every request.
- All state read by a request was finalized at a frame ID strictly smaller than
  the target frame ID.
- One state produces at most one base Joint Perception API call.
- The main backbone is `openai/gpt-5.6-sol` through OpenRouter.
- API keys remain outside request payloads, hashes, cache entries, predictions,
  evidence records, logs, and Git.
- Dataset images are not sent to an external API while
  `data_upload_authorized` is false. The default remains false; synthetic
  fixtures exercise the real transport boundary until separate authorization.
- Missing evidence is represented as unavailable, never as numeric zero.
- Model-reported scores are ranking scores in `[0, 1]`, not calibrated
  probabilities and not correctness claims.
- Frame-level, instance-level, and Tracking metrics remain separate.
- VID31 frame-level supervision is never converted into instance supervision.

## 4. Scope

### Included

- a provider-neutral Joint Perception protocol;
- an OpenRouter-backed implementation using the existing cached API client;
- a compact causal request builder for one to three ordered frames;
- a strict frame-level response schema and parser;
- dense per-task ranking-score reconstruction from bounded ranked-candidate
  output;
- deterministic static and temporal evidence signals;
- immutable evidence provenance and availability masks;
- pipeline integration with the existing `NeverVerify` branch;
- video-wise frame recognition metrics;
- synthetic/mock integration tests and an opt-in real synthetic smoke.

### Excluded

- learned routing or threshold calibration;
- Specialist prompts and additional verification calls;
- candidate-constrained repair and coordinator logic;
- bbox or instance prediction;
- predicted track training and HOTA evaluation;
- EventMemory retrieval;
- real CholecTrack20 image upload without a later explicit authorization;
- paper-performance claims from synthetic or smoke runs.

## 5. Architecture

### 5.1 Components

```text
CausalPerceptionContextBuilder
    -> JointPerceptionRequestBuilder
    -> CachedMultimodalApiClient
    -> JointPerceptionResponseParser
    -> JointPerceptionResult
    -> FrameEvidenceSignalExtractor
    -> EvidenceProfile
    -> FrameResultSink(prediction, evidence)
    -> CanonicalStreamingPipeline
```

`CausalPerceptionContextBuilder` owns ordering and causality. The request
builder owns serialization and prompt/schema versions. The API client owns
transport, cache, retry, usage, and provider provenance. The response parser
owns strict ontology and score validation. The signal extractor owns only
deterministic transformations; it does not call an LLM.
`FrameResultSink` owns the paired prediction/evidence write boundary and returns
only after both records are durable. A partial write marks the run incomplete
and prevents causal-state commit.

### 5.2 Result contract

```python
@dataclass(frozen=True)
class JointPerceptionResult:
    prediction: InitialPrediction
    raw_evidence: PerceptionEvidence
    api_provenance: ApiCallProvenance
```

The existing local backend must return the same wrapper with
`raw_evidence.source="local_smoke"` and explicit unavailable fields. This keeps
the canonical pipeline backend-neutral.

## 6. Input Contract

The request payload contains only compact causal context:

```json
{
  "video_id": "VIDxx",
  "target_frame_id": 123,
  "causal_frame_ids": [121, 122, 123],
  "prior_finalized_prediction": {
    "source_frame_id": 122,
    "instrument_ids": [],
    "verb_ids": [],
    "target_ids": [],
    "ivt_ids": [],
    "phase_id": null
  },
  "workflow_summary": {
    "source_max_frame_id": 122,
    "recent_finalized_phases": []
  },
  "ontology_version": "cholectrack20_v1"
}
```

Rules:

- `causal_frame_ids` are strictly increasing and end at
  `target_frame_id`;
- the window contains at most three frames and uses only available history at
  the beginning of a video;
- images and frame identifiers use the same ordering;
- prior state is optional for the first frame and otherwise must have
  `source_frame_id < target_frame_id`;
- request construction rejects any key whose name or type can carry targets,
  labels, masks, or future-state data;
- prompt version, response schema version, context payload, ordered image
  hashes, provider, endpoint, model, and generation settings participate in the
  canonical request hash.

## 7. Joint Perception Output

The first implementation remains frame-level because that is the approved main
recognition granularity and matches the current `InitialPrediction` contract.
The following JSON is a shortened conceptual example; an actual wire response
must contain the exact ranked-candidate counts frozen below.

```json
{
  "schema_version": "joint_perception_frame_v1",
  "instrument": {
    "selected_ids": [0],
    "topk": [{"id": 0, "score": 0.84}]
  },
  "verb": {
    "selected_ids": [1],
    "topk": [{"id": 1, "score": 0.66}]
  },
  "target": {
    "selected_ids": [2],
    "topk": [{"id": 2, "score": 0.63}]
  },
  "ivt": {
    "selected_ids": [12],
    "topk": [{"id": 12, "score": 0.59}]
  },
  "phase": {
    "selected_id": 3,
    "topk": [{"id": 3, "score": 0.78}]
  },
  "evidence_refs": [
    {"frame_id": 123, "code": "CURRENT_VISUAL_SUPPORT"}
  ],
  "self_reported_confidence": {
    "instrument": 0.84,
    "verb": 0.66,
    "target": 0.63,
    "ivt": 0.59,
    "phase": 0.78
  }
}
```

Ranked-candidate limits are fixed for the first implementation:

- Instrument: all 7 ontology classes;
- Verb: all 10 ontology classes;
- Target: all 15 ontology classes;
- IVT: top 20 of 100 ontology classes;
- Phase: all 7 ontology classes.

Every top-k list is score-descending, contains unique ontology-valid IDs, and
uses finite scores in `[0, 1]`. Every selected ID must appear in its task's
top-k list. Phase has exactly one selected ID. Unreported classes receive rank
score zero when reconstructing the dense vectors needed for AP computation.
These zeros mean "not returned in the bounded ranked list", not calibrated
absence probability. Scores need not sum to one.

For backward compatibility, the first slice writes these dense ranking scores
into the existing `InitialPrediction.probabilities` field. The field name does
not change their semantics: run provenance records
`score_semantics="uncalibrated_rank_v1"`, and no component may interpret them
as calibrated probabilities. Renaming the shared field is outside this slice.

`evidence_refs` use a closed code vocabulary and causal frame IDs. Free-form
model reasoning and chain-of-thought are neither requested nor persisted.
The first schema permits only `CURRENT_VISUAL_SUPPORT`,
`CAUSAL_VISUAL_TREND`, `PRIOR_STATE_SUPPORT`, and
`AMBIGUOUS_VISUAL_SUPPORT`.

## 8. Evidence Signals

### 8.1 Contract

```python
@dataclass(frozen=True)
class EvidenceValue:
    value: float | None
    available: bool
    source: str
    source_max_frame_id: int

@dataclass(frozen=True)
class EvidenceProfile:
    video_id: str
    frame_id: int
    task_values: Mapping[str, Mapping[str, EvidenceValue]]
    global_values: Mapping[str, EvidenceValue]
    evidence_version: str
```

All numeric evidence values are normalized to `[0, 1]`, where larger values
mean more uncertainty, conflict, or anomaly. The meaning is fixed per signal;
the future Gate may not reverse conventions silently.

### 8.2 Signals implemented in this slice

1. `candidate_ambiguity`
   - per task;
   - `1 - clamp(top1_score - top2_score, 0, 1)`;
   - if no second candidate exists, unavailable rather than zero.

2. `ivt_internal_conflict`
   - global and attributable to I/V/T/IVT;
   - fraction of selected IVT IDs whose ontology components are inconsistent
     with the selected Instrument, Verb, or Target sets;
   - zero is valid only when at least one selected IVT can be checked.

3. `temporal_set_change`
   - per I/V/T/IVT task;
   - Jaccard distance between the current selected set and the immediately
     preceding finalized selected set;
   - both empty sets produce zero; missing prior state produces unavailable.

4. `phase_change_anomaly`
   - Phase task;
   - one when the current phase transition is outside the frozen allowed
     transition graph, zero when allowed;
   - missing prior finalized phase produces unavailable.

The allowed phase-transition graph consists of self-transitions plus directed
transitions observed in the training split only. A deterministic builder writes
the graph, source-video list, version, and hash before validation/test. The
graph is an offline training artifact and is never rebuilt inside inference.
It is immutable during validation/test; when no frozen graph is configured,
`phase_change_anomaly` is unavailable rather than guessed from an assumed
clinical order.

5. `self_reported_uncertainty`
   - per task;
   - `1 - self_reported_confidence`;
   - retained as an auxiliary model-reported feature, never treated as a
     calibrated probability.

### 8.3 Deferred signals

The following names are reserved but absent from `evidence_frame_v1` rather
than emitted as fake zeros:

- `track_continuity`;
- `track_semantic_conflict`;
- `spatial_instability`;
- `phase_ivt_compatibility` from a train-only frozen prior;
- `memory_disagreement`;
- `similar_event_support`.

They enter later schema versions only when their producer exists and has an
independent causality/provenance contract.

## 9. Pipeline Integration

The canonical execution order remains one runner:

```text
video reset/continue
-> gold-free sample resolution
-> finalized prior snapshot
-> causal context construction
-> Joint Perception
-> strict response and ontology validation
-> deterministic EvidenceProfile
-> DisabledCandidateGenerator
-> NeverVerify ACCEPT
-> NoOpCoordinator KEEP
-> paired prediction/evidence persistence
-> finalized prior-state commit
```

`PipelineComponents.perception` changes from a concrete
`LocalSmokePerception` type to a protocol returning
`JointPerceptionResult`. The signal extractor receives the causal context and
Joint Perception result. The local smoke assembly remains reproducible through
an adapter that emits unavailable evidence.

Finalized prior state stores only normalized prediction fields and provenance;
it never stores evaluation targets. State resets at every video boundary and
updates only after `FrameResultSink` confirms that both prediction and evidence
records are durable. If one side fails, the run is incomplete and state is not
advanced.

## 10. Failure Handling

- malformed structured output: explicit schema failure, no fabricated
  prediction;
- unknown ontology ID or inconsistent top-k: explicit ontology failure;
- future or misordered frame: causal contract failure before the API call;
- unavailable prior state: temporal signals unavailable, current perception
  remains valid;
- API timeout/rate limit/provider failure: existing typed retry behavior;
- exhausted retry or nonretryable failure: write a sanitized call-failure
  record and stop that sample;
- no automatic fallback from GPT to the local model inside the same experiment;
  a local comparison is a separately named configuration;
- no raw key, free-form model reasoning, or unredacted provider error is
  persisted.

## 11. Metrics

### 11.1 Main frame-level recognition

For Instrument, Verb, Target, and IVT, compute per-class AP from ranking scores,
then video-wise mAP using only task-valid labels. Unsupported classes and
masked tasks are excluded, not converted to negatives. Use the official
`ivtmetrics` semantics where compatible and pin the library/version in the run
manifest.

For Phase, compute per-video Accuracy and per-video macro-F1 by averaging
class-wise F1 over annotated phases, then average across videos. A class with no
positive ground truth in a video is undefined and excluded from that video's
macro average; a class present in ground truth but never predicted receives F1
zero. Per-video class support and excluded classes are recorded. Frame-level AP
uses the official `ivtmetrics` video-wise behavior: classes with no positive
ground truth in that video are excluded and null triplets are ignored.

Current exact-set and Phase-accuracy smoke metrics remain explicitly labeled
engineering diagnostics and are not renamed as paper metrics.

### 11.2 Separate future metrics

- instance output: mAP at IoU 0.5 and optional mAP over IoU 0.5:0.95;
- Tracking: HOTA primary, with DetA, AssA, IDF1, and ID switches;
- no aggregate combines frame recognition, instance detection, and Tracking.

### 11.3 Future Gate labels and metrics

The future Gate uses bounded per-sample task error, not AP, because AP is not
sample-decomposable:

```text
e_frame_multilabel = 1 - sample_set_F1
e_phase = 1[predicted_phase != true_phase]
raw_delta(t, r, k) = e_before(t, k) - e_after(t, r, k)
benefit(t, r) = weighted masked mean_k(raw_delta(t, r, k))
```

Missing task labels are excluded from both numerator and denominator. The first
version uses equal task weights because every error lies in `[0, 1]`. For
`sample_set_F1`, both prediction and target empty means F1 one; one empty and
the other non-empty means F1 zero.

Once a Gate exists, required metrics are benefit MAE, per-state route Spearman
correlation, positive-benefit AUPRC, positive-benefit route capture rate, route
regret, verification rate, and final task performance at matched API budget.
Once repairs exist, required metrics are repair success rate, field-level
repair precision, harm rate, CandidateRecall@K, and zero scope violations.

### 11.4 Framework and statistics

Framework gain is computed only between the full method and single-pass
baseline using the same exact model identifier:

```text
gain(task) = full_method_metric(task) - single_pass_metric(task)
```

Report absolute percentage-point differences and cost deltas. Statistical
resampling uses the video as the paired bootstrap unit; continuous frames are
not treated as independent observations.

## 12. Persistence and Reproducibility

Each run persists separate allowlisted records:

- `predictions/<video_id>.jsonl`;
- `evidence/<video_id>.jsonl`;
- `api_usage.jsonl`;
- cached parsed response envelopes;
- resolved prompt/schema/config versions;
- Git, environment, dataset-manifest, and model-identity provenance.

Prediction and evidence records share `run_id`, `video_id`, `frame_id`, and a
prediction hash. Evidence records never include image bytes, prompts containing
secrets, model free-form text, or evaluation targets.

## 13. Configuration

Add a named single-pass experiment whose meaningful settings are:

```yaml
pipeline:
  context: causal_frames_v1
  perception: joint_openrouter_gpt56sol
  evidence_signals: frame_evidence_v1
  gate_policy: never_verify
  verifier: disabled_traceable
  coordinator: keep_only

perception:
  max_causal_frames: 3
  data_upload_authorized: false
  prompt_version: joint_perception_frame_v1
  response_schema_version: joint_perception_frame_v1
```

Changing `data_upload_authorized` is an explicit experiment decision and is
never inferred from the presence of an API key.

## 14. Planned File Ownership

New focused modules:

- `src/surgical_agent/perception/contracts.py`;
- `src/surgical_agent/perception/context_builder.py`;
- `src/surgical_agent/perception/joint_api_vlm.py`;
- `src/surgical_agent/perception/parser.py`;
- `src/surgical_agent/perception/prompts/perception_prompt.txt`;
- `src/surgical_agent/perception/prompts/perception_schema.json`;
- `src/surgical_agent/research/signals/contracts.py`;
- `src/surgical_agent/research/signals/frame_evidence.py`;
- `src/surgical_agent/evaluation/frame_metrics.py`;
- `src/surgical_agent/inference/frame_result_writer.py`;
- `scripts/build_phase_transition_graph.py`;
- `scripts/run_api_single_pass.py`;
- `configs/experiments/api_single_pass.yaml`;
- `configs/perception/joint_openrouter.yaml`.

Existing integration points:

- `src/surgical_agent/inference/schemas.py`;
- `src/surgical_agent/systems/pipeline.py`;
- `src/surgical_agent/api/providers/openrouter.py`;
- `src/surgical_agent/evaluation/evaluator.py`;
- `src/surgical_agent/inference/writer.py`.

## 15. Testing Strategy

Tests must cover:

- causal frame ordering and no-future rejection;
- target-bearing key/type rejection;
- exact request hash sensitivity to ordered frame content and context;
- strict response fields, top-k order, uniqueness, ranges, and ontology IDs;
- dense score reconstruction and explicit score semantics;
- each evidence formula, availability rule, and `[0, 1]` range;
- train-only phase-transition graph construction, hashing, and validation/test
  immutability;
- video reset and finalized-history-only temporal signals;
- local backend compatibility wrapper;
- NeverVerify pipeline integration and evidence persistence;
- paired-write failure preventing causal-state commit;
- masked task exclusion from video-wise metrics;
- per-video rather than pooled Phase aggregation;
- cache miss/hit accounting with multi-image synthetic fixtures;
- real transport only with synthetic images while authorization is false;
- credential and raw-response absence from persisted artifacts.

Paid API calls are never part of routine unit or CI tests.

## 16. Acceptance Criteria for This Slice

The slice is complete when:

1. the same canonical pipeline runs with either local smoke or Joint
   Perception assembly;
2. one to three causal images produce a strictly validated frame-level
   `InitialPrediction` and `EvidenceProfile`;
3. the evidence extractor has no API or GT dependency;
4. `NeverVerify` preserves the initial prediction unchanged;
5. prediction, evidence, cache, and usage artifacts are reconstructable and
   credential-free;
6. video-wise mAP and Phase macro-F1/Accuracy operate with task masks;
7. no instance, Tracking, Gate, Specialist, Memory, or paper-performance claim
   is implied by this completion;
8. the complete local regression suite remains green.

## 17. Reference Anchors

- CAMMA `ivtmetrics`: https://github.com/CAMMA-public/ivtmetrics
- CholecTriplet2021 benchmark: https://hal.science/hal-03938852v1/file/2204.04746v2.pdf
- CholecTrack20 CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Nwoye_CholecTrack20_A_Multi-Perspective_Tracking_Dataset_for_Surgical_Tools_CVPR_2025_paper.pdf
- Metrics Matter in Surgical Phase Recognition: https://arxiv.org/abs/2305.13961
- SelectiveNet risk-coverage framing: https://arxiv.org/abs/1901.09192
- Multi-expert deferral: https://proceedings.mlr.press/v235/mao24d.html
