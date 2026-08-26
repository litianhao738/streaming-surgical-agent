# Joint Perception and Evidence Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved causal OpenRouter Joint Perception slice with deterministic evidence, paired result persistence, and formal frame-level evaluation while keeping `NeverVerify` active.

**Architecture:** The canonical pipeline receives a gold-free causal context, calls a provider-neutral perception backend once, deterministically derives an `EvidenceProfile`, keeps the prediction unchanged through `NeverVerify`, and durably writes prediction/evidence as one logical result before causal state advances. The API backend reuses the existing cache/retry/usage client and OpenRouter transport; offline evaluation remains the only branch allowed to receive labels.

**Tech Stack:** Python 3.10–3.12, dataclasses and protocols, PyTorch tensors, Pillow PNG encoding, OpenRouter Chat Completions, existing `CachedMultimodalApiClient`, NumPy/scikit-learn, `ivtmetrics==0.1.5`, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-joint-perception-evidence-signals-design.md`

## Global Constraints

- Inference accepts no ground-truth object, label mask, evaluation target, or future frame.
- The target frame is the greatest frame ID in every request, and all prior state has a strictly smaller frame ID.
- One state produces at most one base Joint Perception API call.
- The main backbone is `openai/gpt-5.6-sol` through OpenRouter.
- API keys never enter payloads, hashes, cache entries, predictions, evidence, logs, reports, or Git.
- `data_upload_authorized` defaults to `false`; only `synthetic:` images may cross the real API boundary while false.
- Missing evidence is unavailable (`value=None`, `available=False`), never a fabricated zero.
- Model scores use `score_semantics="uncalibrated_rank_v1"`; they are not calibrated probabilities and need not sum to one.
- Frame, instance, and tracking metrics stay separate; this plan creates only frame-level metrics.
- VID31 remains frame-level I/V/T/IVT plus phase supervision and is never converted to instance supervision.
- `NeverVerify`, disabled Specialists, and KEEP-only coordination remain active throughout this slice.
- Routine unit and CI tests make no paid API calls.

---

### Task 1: Freeze Joint Perception Contracts and Score Semantics

**Files:**
- Create: `src/surgical_agent/perception/__init__.py`
- Create: `src/surgical_agent/perception/contracts.py`
- Modify: `src/surgical_agent/inference/schemas.py`
- Test: `tests/unit/test_joint_perception_contracts.py`

**Interfaces:**
- Produces: `RankedCandidate`, `EvidenceReference`, `PerceptionEvidence`, `ApiCallProvenance`, `JointPerceptionResult`, and `PerceptionBackend.predict(context) -> JointPerceptionResult`.
- Produces: `PerceptionEvidence.local_unavailable(frame_id)` for the local backend, with five empty ranked lists, five null confidences, and no evidence references.
- Produces: `InitialPrediction.score_semantics` and `PredictionRecord.score_semantics`, restricted to `probability_v1` or `uncalibrated_rank_v1`.
- Consumes: existing `InitialPrediction`, `ApiResponseRecord`, and the five task names in `TASK_CLASS_COUNTS`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_joint_result_carries_prediction_evidence_and_safe_api_provenance():
    result = JointPerceptionResult(
        prediction=prediction(score_semantics="uncalibrated_rank_v1"),
        raw_evidence=perception_evidence(),
        api_provenance=ApiCallProvenance.from_response(api_record()),
    )
    assert result.prediction.score_semantics == "uncalibrated_rank_v1"
    assert result.api_provenance.request_hash == "a" * 64


def test_evidence_contract_rejects_missing_task_head():
    values = valid_ranked_candidates()
    values.pop("phase")
    with pytest.raises(ValueError, match="five task heads"):
        PerceptionEvidence(
            source="joint_openrouter_gpt56sol",
            ranked_candidates=values,
            self_reported_confidence=valid_confidences(),
            evidence_refs=(),
            source_max_frame_id=2,
        )
```

Also assert finite `[0,1]` scores, unique task candidate IDs, causal non-negative evidence frame IDs, the closed evidence-reference vocabulary, exact five-head confidence keys, and the absence of credential/raw-response/free-form-reasoning fields from all dataclasses.

- [ ] **Step 2: Run the tests and verify the expected import/field failures**

Run: `python -m pytest tests/unit/test_joint_perception_contracts.py -q`

Expected: FAIL because `surgical_agent.perception.contracts` and `score_semantics` do not exist.

- [ ] **Step 3: Implement the immutable contracts**

```python
TASK_NAMES = tuple(TASK_CLASS_COUNTS)
EVIDENCE_REF_CODES = frozenset({
    "CURRENT_VISUAL_SUPPORT",
    "CAUSAL_VISUAL_TREND",
    "PRIOR_STATE_SUPPORT",
    "AMBIGUOUS_VISUAL_SUPPORT",
})

@dataclass(frozen=True)
class RankedCandidate:
    class_id: int
    score: float

@dataclass(frozen=True)
class EvidenceReference:
    frame_id: int
    code: str

@dataclass(frozen=True)
class PerceptionEvidence:
    source: str
    ranked_candidates: Mapping[str, tuple[RankedCandidate, ...]]
    self_reported_confidence: Mapping[str, float | None]
    evidence_refs: tuple[EvidenceReference, ...]
    source_max_frame_id: int

@dataclass(frozen=True)
class ApiCallProvenance:
    source: str
    provider: str
    endpoint_identifier: str | None
    request_hash: str | None
    requested_model_identifier: str | None
    returned_model_identifier: str | None
    cache_hit: bool | None
    provider_call_count: int

@dataclass(frozen=True)
class JointPerceptionResult:
    prediction: InitialPrediction
    raw_evidence: PerceptionEvidence
    api_provenance: ApiCallProvenance

class PerceptionBackend(Protocol):
    def predict(self, context: "PerceptionContext") -> JointPerceptionResult:
        raise NotImplementedError
```

`ApiCallProvenance.from_response()` copies only allowlisted accounting/identity fields. `ApiCallProvenance.local()` produces a non-API record with `source="local_smoke"`, null request/model fields, and `provider_call_count=0`.

Add `score_semantics: str = "probability_v1"` to `InitialPrediction` and `PredictionRecord`, validate the two allowed values, and copy it in `PredictionFinalizer.finalize()`.

- [ ] **Step 4: Run focused and legacy schema tests**

Run: `python -m pytest tests/unit/test_joint_perception_contracts.py tests/unit/test_p2_pipeline_artifacts.py -q`

Expected: PASS; legacy local predictions default to `probability_v1`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/surgical_agent/perception src/surgical_agent/inference/schemas.py src/surgical_agent/systems/pipeline.py tests/unit/test_joint_perception_contracts.py
git commit -m "feat: add joint perception contracts"
```

### Task 2: Build and Validate Gold-Free Causal Context

**Files:**
- Create: `src/surgical_agent/perception/context_builder.py`
- Test: `tests/unit/test_causal_perception_context.py`

**Interfaces:**
- Consumes: `InferenceSample`, a `[T,C,H,W]` or `[1,T,C,H,W]` tensor, causal workflow/memory snapshots, and `PredictionRecord | None`.
- Produces: `PerceptionContext(sample, frames, images, workflow_snapshot, memory_snapshot, prior_finalized_prediction)`.
- Produces: `PerceptionContextError` and `CausalPerceptionContextBuilder.build() -> PerceptionContext`.

- [ ] **Step 1: Write failing causality and image-order tests**

```python
def test_builder_binds_three_ordered_frames_to_three_image_hashes():
    context = CausalPerceptionContextBuilder(max_frames=3).build(
        sample=sample(causal_frame_ids=(10, 11, 12)),
        frames=torch.stack((red_tensor(), green_tensor(), blue_tensor())),
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=prediction_record(frame_id=11),
    )
    assert [image.identifier for image in context.images] == [
        "synthetic:10", "synthetic:11", "synthetic:12"
    ]
    assert len({image.sha256 for image in context.images}) == 3


def test_builder_rejects_noncausal_prior_before_encoding():
    with pytest.raises(PerceptionContextError, match="strictly earlier"):
        builder.build(
            sample=sample(causal_frame_ids=(10, 11, 12)),
            frames=three_frames(),
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=prediction_record(frame_id=12),
        )
```

Cover frame-count mismatch, non-finite/out-of-range tensors, non-RGB tensors, windows longer than three, GT-bearing mapping keys (`ground_truth`, `evaluation_target`, `frame_supervision`, `label_mask`, `frame_task_mask`, `labels`, `future_state`), and GT-bearing dataclass instances nested inside snapshots.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest tests/unit/test_causal_perception_context.py -q`

Expected: FAIL because the builder and context do not exist.

- [ ] **Step 3: Implement deterministic context construction**

```python
@dataclass(frozen=True)
class PerceptionContext:
    sample: InferenceSample
    frames: Tensor
    images: tuple[ApiImageInput, ...]
    workflow_snapshot: Mapping[str, Any]
    memory_snapshot: Mapping[str, Any]
    prior_finalized_prediction: PredictionRecord | None

class CausalPerceptionContextBuilder:
    def __init__(self, *, max_frames: int = 3) -> None:
        if not 1 <= max_frames <= 3:
            raise ValueError("max_frames must be in 1..3")
        self.max_frames = max_frames

    def build(
        self,
        sample: InferenceSample,
        frames: Tensor,
        *,
        workflow_snapshot: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        prior_finalized_prediction: PredictionRecord | None,
    ) -> PerceptionContext:
        require_gold_free(workflow_snapshot)
        require_gold_free(memory_snapshot)
        ordered_frames = normalize_ordered_frames(
            frames, expected_count=len(sample.causal_frame_ids)
        )
        validate_prior(sample, prior_finalized_prediction)
        images = tuple(
            ApiImageInput(identifier, "image/png", encode_rgb_png(frame))
            for identifier, frame in zip(sample.media_refs, ordered_frames)
        )
        return PerceptionContext(
            sample=sample,
            frames=ordered_frames,
            images=images,
            workflow_snapshot=dict(workflow_snapshot),
            memory_snapshot=dict(memory_snapshot),
            prior_finalized_prediction=prior_finalized_prediction,
        )
```

Normalize `[1,T,C,H,W]` to `[T,C,H,W]`; require `T == len(sample.causal_frame_ids) <= max_frames`, `C == 3`, finite values in `[0,1]`, and prior identity matching the same video with `prior.frame_id < sample.target_frame_id`. Encode each tensor deterministically as RGB PNG through Pillow and preserve the `InferenceSample.media_refs` value as the `ApiImageInput.identifier`; tests use `synthetic:<frame_id>` identifiers.

Implement a recursive gold-free guard with an exact forbidden-key set so legitimate predicted `target_ids` and `target_frame_id` remain allowed. Reject `EvaluationTarget`, `FrameSupervisionTarget`, and `LabelMask` objects by type.

- [ ] **Step 4: Run context and existing causal-window tests**

Run: `python -m pytest tests/unit/test_causal_perception_context.py tests/unit/test_p1_data_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/surgical_agent/perception/context_builder.py tests/unit/test_causal_perception_context.py
git commit -m "feat: build causal perception context"
```

### Task 3: Add the Strict Joint Perception Schema and Parser

**Files:**
- Create: `src/surgical_agent/perception/schema.py`
- Create: `src/surgical_agent/perception/parser.py`
- Create: `src/surgical_agent/perception/prompts/__init__.py`
- Create: `src/surgical_agent/perception/prompts/perception_prompt.txt`
- Create: `src/surgical_agent/perception/prompts/perception_schema.json`
- Modify: `src/surgical_agent/api/schema.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_joint_perception_parser.py`

**Interfaces:**
- Produces: `JOINT_PERCEPTION_SCHEMA_VERSION = "joint_perception_frame_v1"`.
- Produces: `joint_perception_schema() -> dict[str, Any]` and `validate_joint_perception_payload(payload) -> None` registered through existing `schema_for()` / `validator_for()`.
- Produces: `parse_joint_perception_response(response, *, video_id, frame_id, backend) -> JointPerceptionResult`.

- [ ] **Step 1: Write strict parser tests**

```python
def test_parser_reconstructs_dense_scores_and_selected_sets():
    result = parse_joint_perception_response(
        api_record(parsed_payload=valid_joint_payload()),
        video_id="VID02",
        frame_id=12,
        backend="joint_openrouter_gpt56sol",
    )
    assert len(result.prediction.probabilities["ivt"]) == 100
    assert sum(score > 0 for score in result.prediction.probabilities["ivt"]) == 20
    assert result.prediction.triplet_ids == (12,)
    assert result.prediction.score_semantics == "uncalibrated_rank_v1"


@pytest.mark.parametrize("mutation", [
    "unknown_field", "wrong_count", "unsorted_scores", "duplicate_id",
    "selected_not_ranked", "unknown_id", "bad_evidence_code", "future_ref",
])
def test_parser_fails_closed_on_invalid_payload(mutation):
    with pytest.raises(ApiSchemaError):
        parse_joint_perception_response(
            api_record(parsed_payload=mutate(valid_joint_payload(), mutation)),
            video_id="VID02",
            frame_id=12,
            backend="joint_openrouter_gpt56sol",
        )
```

The exact candidate counts are I=7, V=10, T=15, IVT=20, Phase=7. Phase has exactly one selected ID. Dense IVT positions outside the top 20 are exactly `0.0`.

- [ ] **Step 2: Run parser tests and verify failure**

Run: `python -m pytest tests/unit/test_joint_perception_parser.py -q`

Expected: FAIL because the schema registry and parser are absent.

- [ ] **Step 3: Add the packaged JSON Schema and prompt**

The schema uses `additionalProperties: false` at every object level. Each task object requires its selected field and `topk`; candidate items require exactly `id` and `score`; `minItems` and `maxItems` enforce the fixed counts. The phase object requires `selected_id` rather than `selected_ids`. The prompt instructs the model to use only frames at or before the target, emit no prose, preserve ontology numeric IDs, and treat scores as ranking scores.

Register package resources:

```toml
[tool.setuptools.package-data]
"surgical_agent.perception.prompts" = ["*.txt", "*.json"]
```

- [ ] **Step 4: Implement manual semantic validation and parsing**

```python
def parse_joint_perception_response(
    response: ApiResponseRecord,
    *,
    video_id: str,
    frame_id: int,
    backend: str,
) -> JointPerceptionResult:
    validate_joint_perception_payload(response.parsed_payload)
    dense = {
        task: dense_scores(response.parsed_payload[wire_name]["topk"], class_count)
        for task, wire_name, class_count in TASK_LAYOUT
    }
    prediction = InitialPrediction(
        instrument_ids=selected("instrument"),
        verb_ids=selected("verb"),
        target_ids=selected("target"),
        triplet_ids=selected("ivt"),
        phase_id=phase_selected(),
        probabilities=dense,
        score_semantics="uncalibrated_rank_v1",
        backend=backend,
    )
    return JointPerceptionResult(
        prediction=prediction,
        raw_evidence=PerceptionEvidence(
            source=backend,
            ranked_candidates=ranked,
            self_reported_confidence=confidences,
            evidence_refs=references,
            source_max_frame_id=frame_id,
        ),
        api_provenance=ApiCallProvenance.from_response(response),
    )
```

Convert all schema/semantic failures to a sanitized `ApiSchemaError` without embedding payload content. Register the schema version in `SCHEMAS` without changing the P3 smoke schema behavior.

- [ ] **Step 5: Run schema, parser, and P3 regression tests**

Run: `python -m pytest tests/unit/test_joint_perception_parser.py tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_openrouter.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add pyproject.toml src/surgical_agent/perception src/surgical_agent/api/schema.py tests/unit/test_joint_perception_parser.py
git commit -m "feat: parse strict joint perception output"
```

### Task 4: Build Multi-Image OpenRouter Requests and the API Backend

**Files:**
- Create: `src/surgical_agent/perception/joint_api_vlm.py`
- Modify: `src/surgical_agent/api/providers/openrouter.py`
- Modify: `src/surgical_agent/api/providers/mock.py`
- Modify: `src/surgical_agent/config/schema.py`
- Test: `tests/unit/test_joint_api_vlm.py`
- Test: `tests/unit/test_p3_openrouter.py`
- Test: `tests/unit/test_p3_api_config_schema.py`

**Interfaces:**
- Consumes: `PerceptionContext`, `ApiConfig`, and `CachedMultimodalApiClient`.
- Produces: `JointPerceptionRequestBuilder.build(context) -> ApiRequest`.
- Produces: `JointApiVlm.predict(context) -> JointPerceptionResult`.
- Changes: OpenRouter accepts one to three ordered images and sets `ProviderResponse.image_count` to the actual count.
- Changes: `ApiConfig` validates `data_upload_authorized: bool = false` and `max_causal_frames: int = 3` so these settings survive typed config loading.

- [ ] **Step 1: Write failing request-hash, upload-policy, and transport tests**

```python
def test_three_image_request_hash_depends_on_order_and_context():
    first = builder.build(context(image_order=(0, 1, 2)))
    reordered = builder.build(context(image_order=(1, 0, 2)))
    assert canonical_request_metadata(first).request_hash != (
        canonical_request_metadata(reordered).request_hash
    )


def test_false_upload_authorization_rejects_dataset_identifier_before_call():
    backend = JointApiVlm(client=spy_client(), request_builder=builder,
                          data_upload_authorized=False)
    with pytest.raises(ApiContractError, match="synthetic"):
        backend.predict(context(image_ids=("D:/dataset/frame.png",)))
    assert spy_client.call_count == 0
```

Extend the OpenRouter transport test to assert three `image_url` content items appear after the text item in the same order, while the original one-image P3 smoke request remains valid.

- [ ] **Step 2: Run focused API tests and confirm failure**

Run: `python -m pytest tests/unit/test_joint_api_vlm.py tests/unit/test_p3_openrouter.py -q`

Expected: FAIL on the one-image transport restriction and missing backend.

- [ ] **Step 3: Generalize the OpenRouter transport without weakening validation**

Require `1 <= len(request.images) <= 3`. Build user content as one text item followed by every image in tuple order. Accept optional `payload.system_text`; otherwise use the existing strict JSON-only system instruction. Derive a JSON-Schema name from the registered response-schema version using only letters, digits, and underscores. Keep authentication, response parsing, error taxonomy, cache hashing, and P3 smoke behavior unchanged.

```python
def image_content(image: ApiImageInput) -> dict[str, object]:
    encoded = base64.b64encode(image.content).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
    }

content = [{"type": "text", "text": input_text}]
content.extend(image_content(image) for image in request.images)
body["messages"][1]["content"] = content
```

- [ ] **Step 4: Implement the request builder and API backend**

`JointPerceptionRequestBuilder` serializes exactly and selects backend name `joint_openrouter_gpt56sol` for OpenRouter or `joint_mock` for the deterministic mock:

```python
payload = {
    "system_text": load_prompt_text(),
    "input_text": json.dumps({
        "video_id": context.sample.video_id,
        "target_frame_id": context.sample.target_frame_id,
        "causal_frame_ids": list(context.sample.causal_frame_ids),
        "prior_finalized_prediction": safe_prior_mapping(context.prior_finalized_prediction),
        "workflow_summary": safe_workflow_summary(context.workflow_snapshot),
        "ontology_version": "cholectrack20_v1",
    }, sort_keys=True, separators=(",", ":")),
}
```

The `ApiRequest` uses the exact provider, endpoint, model, prompt/schema version, generation parameters, and ordered `context.images`. `JointApiVlm.predict()` checks the synthetic-only policy, calls the cached client exactly once, and passes the returned record to `parse_joint_perception_response()`.

Extend `MockProviderTransport` to return a deterministic valid Joint Perception payload when `response_schema_version == "joint_perception_frame_v1"`; preserve the original P3 payload for `p3_multimodal_smoke_v1`.

- [ ] **Step 5: Run API regression tests**

Run: `python -m pytest tests/unit/test_joint_api_vlm.py tests/unit/test_p3_openrouter.py tests/unit/test_p3_api_client.py -q`

Expected: PASS, including ordered multi-image cache miss/hit accounting.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/surgical_agent/perception/joint_api_vlm.py src/surgical_agent/api/providers/openrouter.py src/surgical_agent/api/providers/mock.py src/surgical_agent/config/schema.py tests/unit/test_joint_api_vlm.py tests/unit/test_p3_openrouter.py tests/unit/test_p3_api_config_schema.py
git commit -m "feat: call joint perception through OpenRouter"
```

### Task 5: Implement Deterministic Evidence Signals with the Official IVT Map

**Files:**
- Create: `src/surgical_agent/research/signals/__init__.py`
- Create: `src/surgical_agent/research/signals/contracts.py`
- Create: `src/surgical_agent/research/signals/frame_evidence.py`
- Create: `src/surgical_agent/research/signals/resources/__init__.py`
- Create: `src/surgical_agent/research/signals/resources/ivt_components_v1.csv`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_frame_evidence_signals.py`

**Interfaces:**
- Produces: `EvidenceValue(value, available, source, source_max_frame_id)`.
- Produces: `EvidenceProfile(video_id, frame_id, task_values, global_values, evidence_version="evidence_frame_v1")`.
- Produces: immutable `PhaseTransitionGraph(transitions, source_video_ids, version, sha256)` consumed optionally by the extractor.
- Produces: `load_ivt_components() -> Mapping[int, tuple[int, int, int]]`.
- Produces: `FrameEvidenceSignalExtractor.extract(context, result) -> EvidenceProfile`.

- [ ] **Step 1: Vendor and verify the authoritative IVT component map**

Derive the packaged four-column `ivt,instrument,verb,target` table from CAMMA `ivtmetrics/ivtmetrics/maps.txt` at commit `c0a565cef6e09fc090b2b68c3b967d4140d3b91b` using `https://raw.githubusercontent.com/CAMMA-public/ivtmetrics/c0a565cef6e09fc090b2b68c3b967d4140d3b91b/ivtmetrics/maps.txt`. The authoritative raw file SHA-256 is `e031ce8646491ddabfd37a23804e46cf4775d257b569deb7702eafd7e9119092`. Preserve the first four numeric values of all 100 source rows exactly and record source commit, URL, and raw SHA in `frame_evidence.py`; require IVT IDs 0–99 exactly once and validate component ranges.

Add package data:

```toml
"surgical_agent.research.signals.resources" = ["*.csv"]
```

- [ ] **Step 2: Write failing formula and availability tests**

```python
def test_candidate_ambiguity_is_one_minus_top_two_margin():
    profile = extractor().extract(context(), result(scores=(0.7, 0.4)))
    value = profile.task_values["instrument"]["candidate_ambiguity"]
    assert value == EvidenceValue(0.7, True, "joint_rank_margin", 12)


def test_missing_prior_makes_temporal_and_phase_signals_unavailable():
    profile = extractor().extract(context(prior=None), result())
    assert profile.task_values["ivt"]["temporal_set_change"].available is False
    assert profile.task_values["phase"]["phase_change_anomaly"].value is None
```

Cover IVT component conflict fractions, both-empty Jaccard distance `0.0`, one-empty Jaccard distance `1.0`, phase allowed/disallowed transitions, self-reported uncertainty, local-smoke missing ranking evidence, range checks, and `source_max_frame_id <= frame_id`.

- [ ] **Step 3: Run signal tests and confirm failure**

Run: `python -m pytest tests/unit/test_frame_evidence_signals.py -q`

Expected: FAIL because signal contracts and extractor are absent.

- [ ] **Step 4: Implement immutable values and all five approved signals**

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
    evidence_version: str = "evidence_frame_v1"

def unavailable(source: str, source_max_frame_id: int) -> EvidenceValue:
    return EvidenceValue(None, False, source, source_max_frame_id)

@dataclass(frozen=True)
class PhaseTransitionGraph:
    transitions: tuple[tuple[int, int], ...]
    source_video_ids: tuple[str, ...]
    version: str
    sha256: str

    def allows(self, previous: int, current: int) -> bool:
        return (previous, current) in self.transitions

def ambiguity(candidates: tuple[RankedCandidate, ...], frame_id: int) -> EvidenceValue:
    if len(candidates) < 2:
        return unavailable("joint_rank_margin", frame_id)
    return EvidenceValue(
        1.0 - min(max(candidates[0].score - candidates[1].score, 0.0), 1.0),
        True,
        "joint_rank_margin",
        frame_id,
    )
```

Implement `candidate_ambiguity`, `ivt_internal_conflict`, `temporal_set_change`, `phase_change_anomaly`, and `self_reported_uncertainty` exactly as approved. If any selected IVT lacks a component-map entry, fail the extractor contract; if no IVT is selected, IVT conflict is unavailable. Put the IVT conflict value in `global_values` and in the I/V/T/IVT task maps. Do not emit reserved deferred signal names.

- [ ] **Step 5: Run signal and contract tests**

Run: `python -m pytest tests/unit/test_frame_evidence_signals.py tests/unit/test_joint_perception_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add pyproject.toml src/surgical_agent/research/signals tests/unit/test_frame_evidence_signals.py
git commit -m "feat: derive deterministic perception evidence"
```

### Task 6: Build and Freeze the Train-Only Phase Transition Graph

**Files:**
- Create: `src/surgical_agent/research/signals/phase_graph.py`
- Create: `scripts/build_phase_transition_graph.py`
- Test: `tests/unit/test_phase_transition_graph.py`

**Interfaces:**
- Consumes: the `PhaseTransitionGraph` contract created in Task 5.
- Produces: `PhaseObservation(video_id, frame_id, phase_id, split)`.
- Produces: `build_phase_transition_graph(observations) -> PhaseTransitionGraph`.
- Produces: `write_phase_transition_graph(graph, path) -> Path` and `load_phase_transition_graph(path) -> PhaseTransitionGraph`.
- Consumes: training-only `ResolvedSample.frame_supervision` from `CholecTrack20DatasetAdapter` in the CLI.

- [ ] **Step 1: Write failing construction, hash, and split-isolation tests**

```python
def test_graph_contains_self_edges_and_observed_directed_edges_only():
    graph = build_phase_transition_graph((
        obs("VID01", 1, 0), obs("VID01", 2, 0), obs("VID01", 3, 2),
    ))
    assert graph.transitions == ((0, 0), (0, 2), (2, 2))
    assert graph.source_video_ids == ("VID01",)


def test_graph_rejects_validation_observation():
    with pytest.raises(ValueError, match="training"):
        build_phase_transition_graph((obs("VID30", 1, 0, DatasetSplit.VALIDATION),))
```

Also test unordered/duplicate frame rejection, missing phase exclusion by the CLI adapter, deterministic hash equality under identical ordered input, stored-hash tampering rejection, and immutable loaded transitions.

- [ ] **Step 2: Run phase-graph tests and confirm failure**

Run: `python -m pytest tests/unit/test_phase_transition_graph.py -q`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement deterministic graph construction around the frozen contract**

```python
def canonical_graph_payload(
    transitions: tuple[tuple[int, int], ...],
    source_video_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": "phase_transition_graph_v1",
        "version": "phase_transition_train_v1",
        "source_video_ids": list(source_video_ids),
        "transitions": [list(edge) for edge in transitions],
    }
```

Group observations by video, require strictly increasing frame IDs, add self-transitions for every observed phase, add each consecutive directed transition, sort all content, and hash the canonical JSON without the `sha256` field. Loading recomputes and compares the hash.

- [ ] **Step 4: Implement the CLI over the single CholecTrack20 root**

`scripts/build_phase_transition_graph.py` accepts `--dataset-root`, `--output`, and optional `--derived-manifest`. Instantiate `CholecTrack20DatasetAdapter`, enumerate only entries whose split is `DatasetSplit.TRAINING`, convert only phase-masked `frame_supervision` records to observations, and atomically write the artifact. It never reads validation/test labels to build transitions.

- [ ] **Step 5: Run tests and CLI help**

Run: `python -m pytest tests/unit/test_phase_transition_graph.py -q`

Run: `python scripts/build_phase_transition_graph.py --help`

Expected: tests PASS and CLI exits 0 without touching the dataset.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/surgical_agent/research/signals/phase_graph.py scripts/build_phase_transition_graph.py tests/unit/test_phase_transition_graph.py
git commit -m "feat: freeze train-only phase transitions"
```

### Task 7: Persist Prediction and Evidence as One Logical Frame Result

**Files:**
- Create: `src/surgical_agent/inference/frame_result_writer.py`
- Modify: `src/surgical_agent/inference/writer.py`
- Modify: `src/surgical_agent/research/signals/contracts.py`
- Test: `tests/unit/test_frame_result_writer.py`

**Interfaces:**
- Produces: `EvidenceRecord` with `run_id`, `video_id`, `frame_id`, `prediction_sha256`, task/global values, evidence version, and schema version.
- Produces: `FrameResultSink.write(prediction, evidence) -> EvidenceRecord` protocol.
- Produces: `FrameResultWriter.write(prediction, evidence) -> EvidenceRecord` and `FrameResultWriter.finalize(metadata) -> Path`.

- [ ] **Step 1: Write failing paired-write and failure-state tests**

```python
def test_writer_persists_matching_per_video_records_and_hash(tmp_path):
    writer = FrameResultWriter(tmp_path, run_id="run")
    evidence_record = writer.write(prediction_record(), evidence_profile())
    assert evidence_record.prediction_sha256 == prediction_record_sha256(
        writer.predictions[0]
    )
    assert json_line(tmp_path / "predictions/VID02.jsonl")["frame_id"] == 12
    assert json_line(tmp_path / "evidence/VID02.jsonl")["frame_id"] == 12


def test_evidence_write_failure_keeps_run_incomplete_and_pipeline_can_refuse_commit(
    tmp_path, monkeypatch
):
    writer = FrameResultWriter(tmp_path, run_id="run")
    inject_second_atomic_write_failure(monkeypatch)
    with pytest.raises(ArtifactWriteError):
        writer.write(prediction_record(), evidence_profile())
    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "INCOMPLETE"
```

Cover identity mismatch, run mismatch, duplicate frame, evidence values containing secrets/free-form text, finalization before any record, finalization after partial failure, per-video SHA-256s, and completion manifest written last.

- [ ] **Step 2: Run writer tests and confirm failure**

Run: `python -m pytest tests/unit/test_frame_result_writer.py -q`

Expected: FAIL because the paired writer is absent.

- [ ] **Step 3: Implement canonical prediction hashing and paired records**

```python
@dataclass(frozen=True)
class EvidenceRecord:
    run_id: str
    video_id: str
    frame_id: int
    prediction_sha256: str
    task_values: Mapping[str, Mapping[str, EvidenceValue]]
    global_values: Mapping[str, EvidenceValue]
    evidence_version: str
    schema_version: str = "evidence_record_v1"

def prediction_record_sha256(record: PredictionRecord) -> str:
    payload = json.dumps(
        asdict(record), default=_json_default, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

Before either data file is changed, atomically write `run_status.json` with `status="INCOMPLETE"` and the current sample identity. Atomically rewrite the candidate prediction JSONL, then the candidate evidence JSONL. Update in-memory records only after both writes return. If the second write fails, retain the incomplete marker and reject `finalize()`.

Promote the existing private writer helpers to `json_default()` and `atomic_write_text()` in `inference/writer.py`; update `PredictionWriter` to use the public names and have `FrameResultWriter` reuse them. Keep compatibility aliases only if an existing test imports the private names.

`finalize()` verifies identical prediction/evidence sample sets and hashes, writes an allowlisted manifest containing per-video file hashes and `status="COMPLETE"`, then atomically replaces `run_status.json` with the completed state.

- [ ] **Step 4: Run writer and legacy artifact tests**

Run: `python -m pytest tests/unit/test_frame_result_writer.py tests/unit/test_p2_pipeline_artifacts.py -q`

Expected: PASS; the legacy `PredictionWriter` stays usable by legacy-only callers until Task 8 migrates the canonical pipeline.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/surgical_agent/inference/frame_result_writer.py src/surgical_agent/inference/writer.py src/surgical_agent/research/signals/contracts.py tests/unit/test_frame_result_writer.py
git commit -m "feat: persist paired frame results"
```

### Task 8: Integrate Joint Results, Evidence, and Finalized History into the Canonical Pipeline

**Files:**
- Modify: `src/surgical_agent/systems/pipeline.py`
- Modify: `src/surgical_agent/systems/baseline_system.py`
- Modify: `scripts/run_local_smoke.py`
- Modify: `tests/unit/test_p2_pipeline_artifacts.py`
- Modify: `tests/integration/test_p2_local_pipeline.py`
- Create: `tests/integration/test_joint_perception_pipeline.py`

**Interfaces:**
- Consumes: `PerceptionBackend`, `FrameEvidenceSignalExtractor`, and `FrameResultSink`.
- Produces: `PipelineRunResult(prediction, evidence, event, runtime_trace)`.
- Maintains: `_prior_finalized_prediction: PredictionRecord | None`, reset per video and updated only after paired persistence succeeds.

- [ ] **Step 1: Write failing canonical-order and state-commit tests**

```python
def test_pipeline_extracts_evidence_before_never_verify_and_persists_pair():
    result = pipeline().run(sample(), three_frames(), run_id="run")
    assert result.prediction.gate_action == "ACCEPT"
    assert result.evidence.evidence_version == "evidence_frame_v1"
    assert result.runtime_trace[4:8] == (
        "05_perception_validated",
        "06_evidence_profile_built",
        "07_candidates_built",
        "08_gate_accept",
    )


def test_failed_pair_write_does_not_advance_prior_state():
    pipeline = pipeline(result_sink=failing_sink())
    with pytest.raises(ArtifactWriteError):
        pipeline.run(sample(frame_id=12), frame(), run_id="run")
    assert pipeline.prior_finalized_prediction is None
```

Also test local backend wrapper provenance/evidence, prior frame visibility on the second same-video call, reset on video change, `NeverVerify` object identity preservation, and no GT-bearing parameter on `CanonicalStreamingPipeline.run()`.

- [ ] **Step 2: Run integration tests and confirm failure**

Run: `python -m pytest tests/integration/test_joint_perception_pipeline.py tests/unit/test_p2_pipeline_artifacts.py -q`

Expected: FAIL because perception still returns `InitialPrediction`, signals are no-op mappings, and persistence is prediction-only.

- [ ] **Step 3: Generalize component protocols and migrate execution order**

Change `LocalSmokePerception.predict()` to wrap its decoded prediction in `JointPerceptionResult` with `PerceptionEvidence.local_unavailable(frame_id)` and `ApiCallProvenance.local()`. Replace concrete component annotations with protocols:

```python
class EvidenceSignalExtractor(Protocol):
    def extract(
        self, context: PerceptionContext, result: JointPerceptionResult
    ) -> EvidenceProfile:
        raise NotImplementedError

class FrameResultSink(Protocol):
    def write(
        self, prediction: PredictionRecord, evidence: EvidenceProfile
    ) -> EvidenceRecord:
        raise NotImplementedError
```

The runner order becomes context → Joint Perception → evidence → candidates → `NeverVerify` → KEEP → finalize → paired write → state stores → internal prior prediction. The result sink must return before any store or internal prior state is updated.

- [ ] **Step 4: Migrate the local assembly to real evidence persistence**

Use `CausalPerceptionContextBuilder`, `FrameEvidenceSignalExtractor`, and `FrameResultWriter` in `P2BaselineSystem` and `run_local_smoke.py`. Keep the local model and its `probability_v1` score semantics. Update the local manifest path and tests to assert both prediction and evidence outputs exist; retain the existing engineering-smoke metric labels.

- [ ] **Step 5: Run canonical local and mock Joint Pipeline tests**

Run: `python -m pytest tests/unit/test_p2_pipeline_artifacts.py tests/integration/test_p2_local_pipeline.py tests/integration/test_joint_perception_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/surgical_agent/systems/pipeline.py src/surgical_agent/systems/baseline_system.py scripts/run_local_smoke.py tests/unit/test_p2_pipeline_artifacts.py tests/integration/test_p2_local_pipeline.py tests/integration/test_joint_perception_pipeline.py
git commit -m "feat: integrate evidence into canonical pipeline"
```

### Task 9: Add Formal Video-Wise Frame Recognition Metrics

**Files:**
- Create: `src/surgical_agent/evaluation/frame_metrics.py`
- Modify: `src/surgical_agent/evaluation/evaluator.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_frame_recognition_metrics.py`

**Interfaces:**
- Produces: `FrameMetricAccumulator.update(prediction, target) -> None`.
- Produces: `FrameMetricAccumulator.compute() -> FrameMetricReport`.
- Produces: frame video-wise mAP for I/V/T/IVT and video-wise Phase macro-F1/Accuracy with supports and excluded classes.
- Produces: `TaskMapReport`, `PhaseVideoReport`, `PhaseMetricReport`, and `FrameMetricReport` dataclasses with only JSON-serializable mappings/tuples/scalars.
- Produces: `framework_gain(full, baseline) -> float` and `paired_video_bootstrap(full_by_video, baseline_by_video, *, seed, resamples) -> BootstrapInterval`, with video IDs as the only resampling unit.

- [ ] **Step 1: Pin the benchmark dependency**

Add `ivtmetrics==0.1.5` to project dependencies. Its official `Recognition.compute_video_AP` behavior is the parity reference, and IVT classes 94–99 are the six null triplets excluded by `ignore_null=True`.

- [ ] **Step 2: Write failing metric-semantics tests**

```python
def test_video_wise_map_is_not_pooled_frame_ap():
    report = accumulator(two_unequal_videos_fixture()).compute()
    assert report.tasks["instrument"].video_wise_map == expected_video_average
    assert report.tasks["instrument"].video_wise_map != pooled_average


def test_phase_macro_f1_excludes_absent_classes_and_scores_missed_present_class_zero():
    report = accumulator(phase_fixture()).compute()
    video = report.phase.per_video["VID02"]
    assert video.excluded_classes == (2, 3, 4, 5, 6)
    assert video.class_f1[1] == 0.0
```

Cover task masks, no-positive AP exclusion, IVT null-class exclusion, all-zero returned IVT tail scores, per-video supports, identity mismatches, score semantics in the report, and a fixture whose IVT result matches `ivtmetrics.Recognition(num_class=100).compute_video_AP("ivt", ignore_null=True)`.

Add a bootstrap test with three named videos and a fixed seed; assert every resampled unit is a complete video, the point estimate equals the paired mean difference, and two repeated calls return the identical interval.

- [ ] **Step 3: Run metric tests and verify failure**

Run: `python -m pytest tests/unit/test_frame_recognition_metrics.py -q`

Expected: FAIL because formal frame metrics do not exist.

- [ ] **Step 4: Implement AP and Phase aggregation**

For each task, group only mask-eligible frames by video. For each video/class, use `sklearn.metrics.average_precision_score` when the class has at least one positive; otherwise record `None`. Average each class over videos ignoring `None`, then average defined classes for video-wise mAP. For IVT, remove classes 94–99 from the final AP/mAP aggregation while retaining their support metadata.

For Phase, compute per-video Accuracy and per-class F1 over ground-truth-present classes. A present class with no predictions has F1 `0.0`; a ground-truth-absent class is `None` and excluded. Average video macro-F1 and Accuracy equally across videos.

```python
@dataclass(frozen=True)
class TaskMapReport:
    class_ap: Mapping[int, float | None]
    class_video_support: Mapping[int, int]
    video_wise_map: float | None

@dataclass(frozen=True)
class PhaseVideoReport:
    accuracy: float
    macro_f1: float
    class_f1: Mapping[int, float | None]
    class_support: Mapping[int, int]
    excluded_classes: tuple[int, ...]

@dataclass(frozen=True)
class PhaseMetricReport:
    video_wise_accuracy: float | None
    video_wise_macro_f1: float | None
    per_video: Mapping[str, PhaseVideoReport]

@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resampling_unit: str = "video"

@dataclass(frozen=True)
class FrameMetricReport:
    tasks: Mapping[str, TaskMapReport]
    phase: PhaseMetricReport
    score_semantics: tuple[str, ...]
    schema_version: str = "frame_recognition_metrics_v1"

class FrameMetricAccumulator:
    def __init__(self) -> None:
        self._records: list[
            tuple[PredictionRecord, FrameSupervisionTarget]
        ] = []

    def update(
        self, prediction: PredictionRecord, target: FrameSupervisionTarget
    ) -> None:
        if (prediction.video_id, prediction.frame_id) != (
            target.video_id, target.frame_id
        ):
            raise ValueError("prediction and target identity mismatch")
        self._records.append((prediction, target))

    def compute(self) -> FrameMetricReport:
        return compute_frame_metric_report(tuple(self._records))
```

Expose this through `EvaluationEngine.summarize_frame_recognition(predictions, targets) -> FrameMetricReport`; do not rename or remove the existing engineering smoke metrics.

Implement framework gain as `full - baseline` and paired bootstrap over the sorted intersection of video IDs. Reject missing/mismatched video sets, non-finite values, non-positive resample counts, and boolean/non-integer seeds. Return the observed gain plus percentile 2.5/97.5 bounds from a local NumPy generator initialized by the explicit seed; never resample frames.

- [ ] **Step 5: Run formal, smoke, and parity tests**

Run: `python -m pytest tests/unit/test_frame_recognition_metrics.py tests/integration/test_p2_local_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 9**

```bash
git add pyproject.toml src/surgical_agent/evaluation/frame_metrics.py src/surgical_agent/evaluation/evaluator.py tests/unit/test_frame_recognition_metrics.py
git commit -m "feat: compute video-wise frame metrics"
```

### Task 10: Add Configured Single-Pass CLI, Real Synthetic Smoke, and Completion Evidence

**Files:**
- Create: `configs/perception/joint_openrouter.yaml`
- Create: `configs/perception/joint_mock.yaml`
- Create: `configs/experiments/api_single_pass.yaml`
- Create: `scripts/run_api_single_pass.py`
- Create: `tests/integration/test_api_single_pass.py`
- Create: `reports/JOINT_PERCEPTION_EVIDENCE_SIGNALS_REPORT.md`
- Modify: `docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md`

**Interfaces:**
- Produces: a CLI that runs one synthetic causal state through the canonical Joint Perception Pipeline and writes paired artifacts plus safe API usage/cache provenance.
- Consumes: `--config`, `--output-root`, `--run-id`, and exactly one of `--api-key-file` / `--api-key` for real mode.
- Preserves: `data_upload_authorized: false` and `openai/gpt-5.6-sol` defaults.

- [ ] **Step 1: Write failing mock end-to-end CLI test**

```python
def test_single_pass_mock_writes_prediction_evidence_and_safe_usage(tmp_path):
    artifact = run_single_pass(
        config=load_api_config(PROJECT_ROOT / "configs/perception/joint_mock.yaml"),
        output_dir=tmp_path / "run",
        api_key=None,
    )
    assert artifact["model_requested"] == "mock-joint-perception-v1"
    assert artifact["causal_frame_ids"] == [0, 1, 2]
    assert (tmp_path / "run/predictions/SYNTHETIC01.jsonl").is_file()
    assert (tmp_path / "run/evidence/SYNTHETIC01.jsonl").is_file()
    assert_no_secret_or_raw_response(tmp_path / "run")
```

Also assert `NeverVerify`/KEEP, one provider call on the first request, a cache hit with zero provider calls/cost when the identical request is replayed by the smoke probe, three ordered image hashes, and rejection of real mode without a credential.

- [ ] **Step 2: Run the CLI integration test and confirm failure**

Run: `python -m pytest tests/integration/test_api_single_pass.py -q`

Expected: FAIL because the configs and CLI do not exist.

- [ ] **Step 3: Add exact configurations**

`joint_openrouter.yaml` contains:

```yaml
enabled: true
mode: real
provider: openrouter
endpoint_identifier: https://openrouter.ai/api/v1/chat/completions
requested_model_identifier: openai/gpt-5.6-sol
prompt_version: joint_perception_frame_v1
response_schema_version: joint_perception_frame_v1
generation_parameters:
  max_output_tokens: 4096
  temperature: 0.0
provider_options:
  timeout_seconds: 120.0
synthetic_input_required: true
cache_required: true
data_upload_authorized: false
max_causal_frames: 3
```

The mock config changes only provider/mode/endpoint/model and retains the same prompt/schema/policy. The experiment config names `causal_frames_v1`, `joint_openrouter_gpt56sol`, `frame_evidence_v1`, `never_verify`, `disabled_traceable`, and `keep_only` exactly.

- [ ] **Step 4: Implement the safe synthetic runner**

Create three deterministic 32×32 RGB tensors with identifiers `synthetic:joint:0`, `synthetic:joint:1`, and `synthetic:joint:2`. Construct a gold-free `InferenceSample` for `SYNTHETIC01`, assemble cache/usage/client/backend/extractor/writer/pipeline from config, and run target frame 2. For the cache probe, build the identical request and call the same client again outside the stateful runner; assert `cache_hit=True`, `provider_call_count=0`, and `provider_cost=0.0` without writing a duplicate frame result.

The CLI prints only the artifact path and a safe status category. It uses the existing `resolve_api_key()` and credential scanner patterns from `scripts/smoke_api.py` and never prints a credential-bearing command.

- [ ] **Step 5: Run mock integration, Ruff, and the complete local suite**

Run: `python -m pytest tests/integration/test_api_single_pass.py -q`

Run: `python -m ruff check src tests scripts`

Run: `python -m pytest -q`

Expected: all commands exit 0; no paid API call occurs.

- [ ] **Step 6: Run the authorized real synthetic OpenRouter smoke**

Use `docs/API.txt` through `--api-key-file` without echoing its contents:

```text
python scripts/run_api_single_pass.py --real --config configs/perception/joint_openrouter.yaml --api-key-file docs/API.txt --output-root artifacts/joint_perception
```

Expected evidence: a three-image structured response, requested and returned model IDs, response ID, request hash, first-call usage/cost, second-call cache hit with zero current cost/provider calls, paired prediction/evidence files, and a successful secret scan. Do not commit runtime output.

- [ ] **Step 7: Write the completion report and update architecture status**

Record exact commands, UTC timestamp, Python/dependency versions, starting/final commit, mock/full-test counts, real returned model string, cache evidence, credential scan result, and the explicit statement that no CholecTrack20 image or paper-performance experiment ran. Document that instance detection, tracking, Gate, Specialists, coordination repairs, and EventMemory remain outside this slice.

- [ ] **Step 8: Run final secret/Git verification**

Run: `git grep -n -I -E "sk-or-v1-|Bearer [A-Za-z0-9_-]{20,}" -- . ":(exclude)docs/API.txt"`

Expected: no credential match.

Run: `git status --short`

Expected: only intended source/config/test/report changes before the final commit; no cache, artifact, dataset, or `docs/API.txt` entry.

- [ ] **Step 9: Commit Task 10**

```bash
git add configs/perception configs/experiments/api_single_pass.yaml scripts/run_api_single_pass.py tests/integration/test_api_single_pass.py reports/JOINT_PERCEPTION_EVIDENCE_SIGNALS_REPORT.md docs/architecture/Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md
git commit -m "feat: complete joint perception single pass"
```

- [ ] **Step 10: Apply completion skills and preserve evidence**

Invoke `superpowers:requesting-code-review`, address findings, then invoke `superpowers:verification-before-completion`. Re-run the exact focused, full-suite, Ruff, secret-scan, and Git-status commands after the last fix. The goal remains active unless the broader objective—not only this slice—passes its requirement-by-requirement completion audit.
