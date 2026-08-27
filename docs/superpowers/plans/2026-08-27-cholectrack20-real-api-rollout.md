# CholecTrack20 Real API Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-aligned CLI that sends real CholecTrack20 causal frame windows through the existing GPT-5.6 Sol single-pass pipeline with Gold-free selection, bounded provider calls, reusable cache, and durable per-video outputs.

**Architecture:** Extend the read-only dataset adapter with a label-free inference iterator, load exact PNG or MP4 causal frames into the existing canonical pipeline, and keep API budgeting inside the cache-aware client so cache hits never consume the provider-call limit. A thin CLI resolves engineering or complete-split selection, assembles the existing Joint Perception pipeline, and writes one fresh result tree while optionally reusing an external parsed-response cache.

**Tech Stack:** Python 3.10-3.12, PyTorch, Pillow, OpenCV headless, existing OpenRouter transport/cache/usage contracts, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-cholectrack20-real-api-rollout-design.md`

## Global Constraints

- Use `openai/gpt-5.6-sol` through `https://openrouter.ai/api/v1/chat/completions` with the existing `joint_perception_frame_v1` prompt/schema and `{max_output_tokens: 4096, reasoning: {effort: low}}`.
- At most three ordered causal frames may enter a request, and the target frame must be the maximum causal frame ID.
- API execution consumes only `InferenceSample`; no `EvaluationTarget`, `FrameSupervisionTarget`, task labels, or test GT may enter runtime requests or causal state.
- Engineering mode requires exactly one video plus a positive frame limit and is always `paper_metric_eligible: false`.
- Paper mode accepts only complete Validation or Test selection and rejects truncation.
- Real data upload requires both `data_upload_authorized: true` in the exact named config and the literal CLI flag `--authorize-data-upload`.
- Every invocation has a hard provider-call limit; a cache hit consumes zero provider calls.
- Result directories are fresh. Reruns use a new run ID and may reuse an external cache root.
- Automated tests and implementation verification make no real provider calls.
- No API key, raw provider body, prompt text, base64 image, image bytes, GT, or absolute dataset path may be persisted in output/cache provenance.
- Keep the existing P3 exact immutable backend identity status `PARTIAL` and mark rollout outputs `paper_metric_eligible: false`.

---

### Task 1: Gold-Free Dataset Selection

**Files:**
- Modify: `src/surgical_agent/data/dataset.py`
- Create: `src/surgical_agent/data/api_rollout_selection.py`
- Create: `tests/unit/test_api_rollout_selection.py`
- Modify: `tests/integration/test_p2_local_pipeline.py`

**Interfaces:**
- Produces: `CholecTrack20DatasetAdapter.iter_inference_video(video_id: str, *, max_samples: int | None = None) -> Iterator[InferenceSample]`
- Produces: `RolloutSelection(mode: str, split: DatasetSplit | None, video_ids: tuple[str, ...], samples: tuple[InferenceSample, ...], frame_counts: Mapping[str, int])`
- Produces: `resolve_rollout_selection(adapter, *, mode: str, video_id: str | None, max_frames: int | None, split: str | None) -> RolloutSelection`
- Consumes later: Tasks 4-5 use the ordered `samples`, `video_ids`, and `frame_counts` without touching evaluation objects.

- [ ] **Step 1: Write failing selection-mode tests**

Add tests that define a tiny duck-typed inference source and demonstrate the required API:

```python
from types import SimpleNamespace


class FakeInferenceSource:
    def __init__(
        self,
        splits: dict[str, DatasetSplit],
        samples: dict[str, tuple[InferenceSample, ...]],
    ) -> None:
        self.entries = {
            video_id: SimpleNamespace(split=split)
            for video_id, split in splits.items()
        }
        self._samples = samples

    def iter_inference_video(
        self,
        video_id: str,
        *,
        max_samples: int | None = None,
    ) -> Iterator[InferenceSample]:
        selected = self._samples[video_id]
        yield from selected if max_samples is None else selected[:max_samples]


def _sample(
    video_id: str,
    frame_id: int,
    split: DatasetSplit = DatasetSplit.VALIDATION,
) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_id,
        causal_frame_ids=(frame_id,),
        media_refs=(f"{frame_id}.png",),
        source_split=split,
        alignment_version=PNG_ALIGNMENT_VERSION,
    )


def test_engineering_selection_requires_one_video_and_positive_limit() -> None:
    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {"VID30": (_sample("VID30", 1), _sample("VID30", 2))},
    )

    selected = resolve_rollout_selection(
        source,
        mode="engineering",
        video_id="vid30",
        max_frames=1,
        split=None,
    )

    assert selected.video_ids == ("VID30",)
    assert [sample.target_frame_id for sample in selected.samples] == [1]
    assert dict(selected.frame_counts) == {"VID30": 1}


@pytest.mark.parametrize(
    ("video_id", "max_frames"),
    [(None, 1), ("VID30", None), ("VID30", 0), ("VID30", True)],
)
def test_engineering_selection_rejects_incomplete_bounds(
    video_id: str | None,
    max_frames: int | None,
) -> None:
    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION},
        {"VID30": (_sample("VID30", 1),)},
    )
    with pytest.raises((TypeError, ValueError)):
        resolve_rollout_selection(
            source,
            mode="engineering",
            video_id=video_id,
            max_frames=max_frames,
            split=None,
        )


def test_paper_selection_uses_every_sorted_video_and_rejects_truncation() -> None:
    source = FakeInferenceSource(
        {"VID30": DatasetSplit.VALIDATION, "VID11": DatasetSplit.VALIDATION},
        {
            "VID11": (_sample("VID11", 1), _sample("VID11", 2)),
            "VID30": (_sample("VID30", 1), _sample("VID30", 2)),
        },
    )
    selected = resolve_rollout_selection(
        source,
        mode="paper",
        video_id=None,
        max_frames=None,
        split="validation",
    )
    assert selected.video_ids == ("VID11", "VID30")
    assert [sample.video_id for sample in selected.samples] == [
        "VID11", "VID11", "VID30", "VID30"
    ]
    with pytest.raises(ValueError, match="paper mode forbids truncation"):
        resolve_rollout_selection(
            source,
            mode="paper",
            video_id=None,
            max_frames=1,
            split="validation",
        )
```

- [ ] **Step 2: Run the new unit test and verify RED**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py
```

Expected: collection fails because `surgical_agent.data.api_rollout_selection` and `iter_inference_video` do not exist.

- [ ] **Step 3: Implement the label-free adapter iterator**

In `dataset.py`, add an iterator that uses only media/frame identity:

```python
def iter_inference_video(
    self,
    video_id: str,
    *,
    max_samples: int | None = None,
) -> Iterator[InferenceSample]:
    entry = self._entry(video_id)
    if max_samples is not None and (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples < 0
    ):
        raise ValueError("max_samples must be a non-negative integer")
    if entry.split is DatasetSplit.TESTING:
        frame_ids = _read_test_frame_ids(Path(entry.annotation_file), entry.video_id)
        resolver = Mp4FrameResolver(
            video_id=entry.video_id,
            split=entry.split,
            media_path=entry.media_source,
            frame_count=max(frame_ids),
            decoder_index_offset=-1,
            alignment_version=MP4_ALIGNMENT_VERSION,
        )
    else:
        resolver = self._png_resolver(entry, self._derived(entry.video_id))
        frame_ids = resolver.available_frame_ids
    selected = frame_ids if max_samples is None else frame_ids[:max_samples]
    for frame_id in selected:
        causal_ids = _causal_from_sorted(
            frame_ids,
            target_frame_id=frame_id,
            max_frames=self.causal_window_size,
        )
        yield InferenceSample(
            video_id=entry.video_id,
            target_frame_id=frame_id,
            causal_frame_ids=causal_ids,
            media_refs=tuple(resolver.resolve(value).media_path for value in causal_ids),
            source_split=entry.split,
            alignment_version=(
                MP4_ALIGNMENT_VERSION
                if entry.split is DatasetSplit.TESTING
                else PNG_ALIGNMENT_VERSION
            ),
        )
```

Do not call `parse_annotation_file`, `load_frame_level_ivt_supervision`, or `load_image_phase_supervision` in this method.

- [ ] **Step 4: Implement immutable mode selection**

Create `api_rollout_selection.py` with a frozen dataclass, strict `engineering|paper` validation, case-normalized video IDs, sorted official split selection, sample-order checks, and frozen frame counts. Use the adapter entry split only to select videos; never inspect a target object.

```python
@dataclass(frozen=True)
class RolloutSelection:
    mode: str
    split: DatasetSplit | None
    video_ids: tuple[str, ...]
    samples: tuple[InferenceSample, ...]
    frame_counts: Mapping[str, int]

    @property
    def expected_provider_calls(self) -> int:
        return len(self.samples)
```

Validate that samples are video-major, strictly increasing within a video, and match `video_ids` exactly.

- [ ] **Step 5: Add local-data Gold-free integration coverage**

Extend `test_p2_local_pipeline.py`:

```python
def test_api_inference_iterator_returns_only_gold_free_samples() -> None:
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    validation = tuple(adapter.iter_inference_video("VID30", max_samples=2))
    testing = tuple(adapter.iter_inference_video("VID01", max_samples=2))
    assert all(isinstance(sample, InferenceSample) for sample in validation + testing)
    assert all(len(sample.causal_frame_ids) <= 3 for sample in validation + testing)
    assert not hasattr(validation[0], "evaluation")
    assert testing[0].alignment_version == MP4_ALIGNMENT_VERSION
```

- [ ] **Step 6: Run Task 1 tests and commit**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py tests\integration\test_p2_local_pipeline.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src\surgical_agent\data tests\unit\test_api_rollout_selection.py tests\integration\test_p2_local_pipeline.py
```

Expected: PASS, with the local-data test skipped only when `CHOLECTRACK20_ROOT` is absent.

Commit:

```powershell
git add src/surgical_agent/data/dataset.py src/surgical_agent/data/api_rollout_selection.py tests/unit/test_api_rollout_selection.py tests/integration/test_p2_local_pipeline.py
git commit -m "feat: select gold-free API rollout samples"
```

---

### Task 2: Exact PNG and MP4 Causal Media Loading

**Files:**
- Create: `src/surgical_agent/data/api_media.py`
- Create: `tests/unit/test_api_causal_media.py`
- Modify: `tests/integration/test_p2_local_pipeline.py`
- Modify: `pyproject.toml`
- Modify: `requirements-autodl.txt`

**Interfaces:**
- Produces: `VideoFrameReader.read_many_rgb(path: Path, decoder_indices: tuple[int, ...]) -> tuple[numpy.ndarray, ...]`
- Produces: `OpenCvVideoFrameReader`
- Produces: `LoadedApiWindow(runtime_sample: InferenceSample, frames: torch.Tensor)`
- Produces: `CausalApiMediaLoader.load(sample: InferenceSample) -> LoadedApiWindow`
- Consumes later: Task 4 passes `runtime_sample` and `frames` directly to `CanonicalStreamingPipeline.run`.

- [ ] **Step 1: Write failing PNG, MP4, and path-sanitization tests**

Create tests with two real temporary PNGs and a recording decoder at the external codec boundary:

```python
def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (6, 4), color).save(path)


class RecordingVideoFrameReader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[int, ...]]] = []

    def read_many_rgb(
        self,
        path: Path,
        decoder_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]:
        self.calls.append((path, decoder_indices))
        return tuple(
            np.full((4, 6, 3), index % 255, dtype=np.uint8)
            for index in decoder_indices
        )


def _sample(
    *,
    video_id: str,
    frame_ids: tuple[int, ...],
    media_refs: tuple[str, ...],
    split: DatasetSplit,
    alignment: str,
) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_ids[-1],
        causal_frame_ids=frame_ids,
        media_refs=media_refs,
        source_split=split,
        alignment_version=alignment,
    )


def test_png_window_loads_native_rgb_and_replaces_paths_with_safe_ids(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "1.png", (255, 0, 0))
    _write_rgb(tmp_path / "2.png", (0, 255, 0))
    sample = _sample(
        video_id="VID30",
        frame_ids=(1, 2),
        media_refs=(str(tmp_path / "1.png"), str(tmp_path / "2.png")),
        split=DatasetSplit.VALIDATION,
        alignment=PNG_ALIGNMENT_VERSION,
    )

    loaded = CausalApiMediaLoader().load(sample)

    assert loaded.frames.shape == (2, 3, 4, 6)
    assert loaded.runtime_sample.media_refs == (
        "cholectrack20:VID30:frame:1",
        "cholectrack20:VID30:frame:2",
    )
    assert str(tmp_path) not in repr(loaded.runtime_sample)


def test_test_mp4_uses_annotation_id_minus_one_indices(tmp_path: Path) -> None:
    video = tmp_path / "vid01.mp4"
    video.touch()
    reader = RecordingVideoFrameReader()
    sample = _sample(
        video_id="VID01",
        frame_ids=(1, 26, 51),
        media_refs=(str(video), str(video), str(video)),
        split=DatasetSplit.TESTING,
        alignment=MP4_ALIGNMENT_VERSION,
    )

    loaded = CausalApiMediaLoader(video_reader=reader).load(sample)

    assert reader.calls == [(video.resolve(), (0, 25, 50))]
    assert loaded.frames.shape[0] == 3
```

Also test mismatched PNG geometry, a non-MP4 test path, wrong alignment, and negative/zero test frame IDs.

- [ ] **Step 2: Run the media tests and verify RED**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_causal_media.py
```

Expected: collection fails because `surgical_agent.data.api_media` does not exist.

- [ ] **Step 3: Implement the loader and safe runtime sample**

Create the frozen result and reader protocol:

```python
@dataclass(frozen=True)
class LoadedApiWindow:
    runtime_sample: InferenceSample
    frames: Tensor


class VideoFrameReader(Protocol):
    def read_many_rgb(
        self,
        path: Path,
        decoder_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]: ...
```

`CausalApiMediaLoader.load` must validate the input sample, load every causal frame in order, stack finite RGB tensors in `[0, 1]`, and return a copied `InferenceSample` with path-free canonical identifiers. PNG uses Pillow. Test MP4 requires one common path and calls `read_many_rgb(path, tuple(frame_id - 1 for frame_id in causal_frame_ids))`.

Implement `OpenCvVideoFrameReader` with a lazy `import cv2`, one `VideoCapture` per causal window, `CAP_PROP_POS_FRAMES`, `read()`, `release()` in `finally`, and BGR-to-RGB conversion. Raise `DatasetContractError` without embedding decoder payloads.

- [ ] **Step 4: Declare and install the MP4 dependency**

Add the same compatible requirement to `pyproject.toml` and `requirements-autodl.txt`:

```text
opencv-python-headless>=4.8,<5
```

Install it into the project environment for verification:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pip install "opencv-python-headless>=4.8,<5"
```

- [ ] **Step 5: Add one optional official MP4 decode integration test**

Use `next(adapter.iter_inference_video("VID01", max_samples=1))`, load it with the real OpenCV reader, and assert `[T,3,H,W]`, finite values in `[0,1]`, and path-free runtime identifiers. Keep the existing local-data skip marker.

- [ ] **Step 6: Run Task 2 tests and commit**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_causal_media.py tests\integration\test_p2_local_pipeline.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src\surgical_agent\data tests\unit\test_api_causal_media.py tests\integration\test_p2_local_pipeline.py
```

Commit:

```powershell
git add pyproject.toml requirements-autodl.txt src/surgical_agent/data/api_media.py tests/unit/test_api_causal_media.py tests/integration/test_p2_local_pipeline.py
git commit -m "feat: load causal API media windows"
```

---

### Task 3: Provider-Call Budget and Shared Real Accounting

**Files:**
- Create: `src/surgical_agent/api/budget.py`
- Create: `src/surgical_agent/api/accounting.py`
- Modify: `src/surgical_agent/api/client.py`
- Modify: `src/surgical_agent/api/errors.py`
- Modify: `scripts/run_api_single_pass.py`
- Modify: `tests/unit/test_p3_api_client.py`
- Modify: `tests/integration/test_api_single_pass.py`

**Interfaces:**
- Produces: `ProviderCallBudget(limit: int)` with `consume() -> None`, `used`, and `remaining`.
- Produces: `ApiProviderCallBudgetError(code="provider_call_budget_exhausted")`.
- Extends: `CachedMultimodalApiClient(..., provider_call_budget: ProviderCallBudget | None = None)`.
- Produces: `CompleteAccountingTransport(transport: ProviderTransport)` shared by synthetic and dataset real runners.
- Consumes later: Task 5 constructs one budget per CLI invocation.

- [ ] **Step 1: Write the failing cache-aware budget test**

Add a test proving the limit is checked only after a cache miss:

```python
def test_provider_budget_counts_misses_and_rejects_without_transport_call(tmp_path: Path) -> None:
    transport = MockProviderTransport()
    usage = UsageLedger(tmp_path / "api_usage.jsonl")
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(tmp_path / "cache"),
        usage=usage,
        validator=validate_p3_smoke_payload,
        retry_policy=RetryPolicy(max_attempts=1),
        provider_call_budget=ProviderCallBudget(1),
    )

    client.call(_request())
    client.call(_request())
    uncached = replace(
        _request(),
        images=(ApiImageInput("synthetic:uncached", "image/png", b"uncached"),),
    )
    with pytest.raises(ApiProviderCallBudgetError):
        client.call(uncached)

    assert transport.provider_call_count == 1
    assert usage.records()[-1]["provider_call_count"] == 0
    assert usage.records()[-1]["error_code"] == "provider_call_budget_exhausted"
```

Add regression tests showing `CompleteAccountingTransport` preserves the Fix Round 3 behavior for missing input/output/total tokens or cost.

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_p3_api_client.py -k "provider_budget or complete_accounting"
```

Expected: import/signature failures because the new budget/accounting modules do not exist.

- [ ] **Step 3: Implement the budget before the retry/transport boundary**

Add a dedicated error:

```python
class ApiProviderCallBudgetError(ApiError):
    code = "provider_call_budget_exhausted"
```

Implement a small mutable budget object with strict positive-int validation. In `CachedMultimodalApiClient.call`, after a cache miss and before `RetryPolicy.execute`, call `consume()`. When it raises, log one failure row with `provider_call_count=0`, `retry_count=0`, and re-raise. Existing callers pass no budget and retain current behavior.

- [ ] **Step 4: Extract complete-accounting transport without behavior change**

Move `_require_complete_real_accounting` and `_RealAccountingTransport` from `run_api_single_pass.py` into `api/accounting.py` as:

```python
class CompleteAccountingTransport:
    def __init__(self, transport: ProviderTransport) -> None: ...
    def send(self, request: ApiRequest) -> ProviderResponse: ...
```

Update the existing single-pass runner and tests to import it. Do not change request generation, provider routing, or real smoke evidence.

- [ ] **Step 5: Run focused and full API tests, then commit**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_p3_api_client.py tests\integration\test_api_single_pass.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src\surgical_agent\api scripts\run_api_single_pass.py tests\unit\test_p3_api_client.py tests\integration\test_api_single_pass.py
```

Commit:

```powershell
git add src/surgical_agent/api/budget.py src/surgical_agent/api/accounting.py src/surgical_agent/api/client.py src/surgical_agent/api/errors.py scripts/run_api_single_pass.py tests/unit/test_p3_api_client.py tests/integration/test_api_single_pass.py
git commit -m "feat: bound cache-aware provider calls"
```

---

### Task 4: Dataset API Rollout Application

**Files:**
- Create: `src/surgical_agent/systems/api_dataset_system.py`
- Create: `tests/integration/test_api_dataset_pipeline.py`

**Interfaces:**
- Produces: `DatasetApiRunResult(predictions, manifest_path, frame_counts, usage_summary)`.
- Produces: `DatasetApiPipelineSystem(client, config, writer, media_loader)`.
- Produces: `DatasetApiPipelineSystem.run(selection: RolloutSelection, *, run_id: str) -> DatasetApiRunResult`.
- Consumes: Task 1 selection, Task 2 media loader, existing pipeline components, client, writer, and Evidence Signals.
- Consumed later: Task 5 CLI assembles dependencies and serializes the safe result artifact.

- [ ] **Step 1: Write the failing two-video canonical pipeline test**

Build three temporary PNG samples across two videos, a real `MockProviderTransport`, cache/client, writer, and selection:

`_selection_with_pngs` writes native RGB PNGs, creates matching
`InferenceSample` values, and constructs the frozen `RolloutSelection` directly.
`_mock_system` derives a dataset-safe mock config with
`replace(load_api_config("configs/perception/joint_mock.yaml"),
synthetic_input_required=False, data_upload_authorized=True)`, then assembles
`MockProviderTransport`, `FileApiCache`, `UsageLedger`,
`CachedMultimodalApiClient(provider_call_budget=ProviderCallBudget(limit))`,
`FrameResultWriter`, and `CausalApiMediaLoader`. It returns the system and the
transport so the assertion observes real cache/client behavior rather than a
mocked system.

```python
def test_dataset_api_system_runs_ordered_samples_and_resets_each_video(tmp_path: Path) -> None:
    selection = _selection_with_pngs(tmp_path, (("VID11", 1), ("VID11", 2), ("VID30", 1)))
    system, transport = _mock_system(tmp_path, provider_call_limit=3)

    result = system.run(selection, run_id="dataset-mock")

    assert [(p.video_id, p.frame_id) for p in result.predictions] == [
        ("VID11", 1), ("VID11", 2), ("VID30", 1)
    ]
    assert "01_video_boundary_reset" in result.predictions[0].trace
    assert "01_video_boundary_continue" in result.predictions[1].trace
    assert "01_video_boundary_reset" in result.predictions[2].trace
    assert transport.provider_call_count == 3
    assert result.manifest_path.is_file()
```

Add a second test that runs the same selection in a fresh output directory with the same `FileApiCache` and a fresh mock transport, expecting zero provider calls and three cache hits.

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\integration\test_api_dataset_pipeline.py
```

Expected: collection fails because `surgical_agent.systems.api_dataset_system` does not exist.

- [ ] **Step 3: Implement the thin application layer**

Create:

```python
@dataclass(frozen=True)
class DatasetApiRunResult:
    predictions: tuple[PredictionRecord, ...]
    manifest_path: Path
    frame_counts: Mapping[str, int]
    usage_summary: Mapping[str, object]
```

The system constructor assembles one `CanonicalStreamingPipeline` using the exact existing Joint Perception/Evidence/NeverVerify/KEEP components. `run` loops over `selection.samples`, calls `media_loader.load`, passes only `loaded.runtime_sample` and `loaded.frames`, collects predictions, and finalizes the writer with `{"paper_metric_eligible": False}`. It asserts result identities and counts match selection before finalization.

Do not add evaluation, thresholding, verifier, specialist, or learned state behavior.

- [ ] **Step 4: Verify first pass and cache replay, then commit**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\integration\test_api_dataset_pipeline.py tests\integration\test_joint_perception_pipeline.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src\surgical_agent\systems\api_dataset_system.py tests\integration\test_api_dataset_pipeline.py
```

Commit:

```powershell
git add src/surgical_agent/systems/api_dataset_system.py tests/integration/test_api_dataset_pipeline.py
git commit -m "feat: run dataset frames through API pipeline"
```

---

### Task 5: Real-Data Config, CLI, Artifacts, and AutoDL Handoff

**Files:**
- Create: `configs/perception/joint_mock_dataset.yaml`
- Create: `configs/perception/joint_openrouter_dataset.yaml`
- Create: `scripts/run_dataset_api_pipeline.py`
- Create: `tests/integration/test_api_dataset_cli.py`
- Modify: `docs/AUTODL_QUICKSTART.md`

**Interfaces:**
- Produces CLI: `scripts/run_dataset_api_pipeline.py`.
- Produces programmatic entry: `run_dataset_api_rollout(..., transport: ProviderTransport | None = None) -> dict[str, object]` for mock tests.
- Consumes Tasks 1-4 and existing credential/config/transport/cache/usage/artifact utilities.

- [ ] **Step 1: Write failing CLI contract tests**

Test parser and preflight behavior without reading a real key or media:

```python
def test_real_dataset_cli_requires_double_upload_authorization(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "--mode", "engineering",
        "--video-id", "VID30",
        "--max-frames", "1",
        "--dataset-root", str(tmp_path),
        "--config", str(REAL_DATASET_CONFIG),
        "--api-key-file", str(tmp_path / "API.txt"),
        "--output-root", str(tmp_path / "out"),
        "--cache-root", str(tmp_path / "cache"),
        "--run-id", "real-one",
        "--max-provider-calls", "exact-selection",
    ])
    with pytest.raises(ApiContractError, match="authorize-data-upload"):
        run(args)


def test_paper_cli_rejects_max_frames() -> None:
    args = build_parser().parse_args([
        "--mode", "paper", "--split", "validation", "--max-frames", "1",
    ])
    with pytest.raises(ApiContractError, match="paper mode forbids truncation"):
        validate_cli_selection(args)
```

Add an injected mock end-to-end test over two temporary PNG samples. Assert:

- one paired prediction/evidence record per sample;
- exact selection and completed counts;
- one usage row per logical call;
- first run provider calls equal sample count;
- second fresh run with the same cache has zero provider calls;
- `track20_image_uploaded` is false for mock and true only for real mode;
- `paper_metric_eligible` is false;
- no absolute media path, API key, raw response, prompt, base64, image bytes, or GT field is persisted.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\integration\test_api_dataset_cli.py
```

Expected: collection fails because the CLI and configs do not exist.

- [ ] **Step 3: Add exact mock and real-data configs**

Both configs use the existing joint prompt/schema, three causal frames, cache, and low reasoning effort. The real config is:

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
  reasoning:
    effort: low
provider_options:
  timeout_seconds: 120.0
synthetic_input_required: false
cache_required: true
data_upload_authorized: true
max_causal_frames: 3
```

The mock config uses the existing mock endpoint/model, empty provider options, and the same three boolean policy fields.

- [ ] **Step 4: Implement strict CLI preflight and rollout assembly**

The parser exposes:

```text
--mode engineering|paper
--video-id VID30
--max-frames 3
--split validation|testing
--dataset-root PATH
--config PATH
--api-key-file PATH
--output-root PATH
--cache-root PATH
--run-id SAFE_ID
--max-provider-calls POSITIVE_INT|exact-selection
--authorize-data-upload
```

Resolve and validate mode, selection, config identity, upload authorization, run/cache paths, and call limit before reading a credential, decoding media, or building a real transport. Mock mode rejects credentials; real mode requires exactly one credential. The output directory must be fresh and must not resolve inside the dataset root. Cache may preexist but must not resolve inside the dataset root or output directory.

Before any provider call, stream-hash `<dataset_root>/repair_manifest.json` with SHA-256 and retain only its digest for the artifact; never persist the manifest contents or its absolute path.

Assemble `UsageLedger`, `FileApiCache`, optional `CompleteAccountingTransport`, `ProviderCallBudget`, `CachedMultimodalApiClient(max_attempts=1)`, `FrameResultWriter`, media loader, and `DatasetApiPipelineSystem`.

Write `dataset_rollout_artifact.json` with only:

```python
{
    "schema_version": "cholectrack20_api_rollout_v1",
    "status": (
        "REAL_RESPONSE_RECEIVED" if config.mode == "real" else "MOCK_COMPLETE"
    ),
    "run_id": run_id,
    "mode": selection.mode,
    "split": selection.split.value if selection.split else None,
    "video_ids": list(selection.video_ids),
    "expected_frame_counts": dict(selection.frame_counts),
    "completed_frame_counts": dict(result.frame_counts),
    "provider": config.provider,
    "model_requested": config.requested_model_identifier,
    "models_returned": sorted({
        str(row["returned_model_identifier"])
        for row in usage.records()
        if row.get("returned_model_identifier") is not None
    }),
    "prompt_version": config.prompt_version,
    "response_schema_version": config.response_schema_version,
    "repair_manifest_sha256": repair_manifest_sha256,
    "alignment_versions": sorted({sample.alignment_version for sample in selection.samples}),
    "usage": result.usage_summary,
    "cache_entry_count": len(tuple(cache_root.glob("*.json"))),
    "track20_image_uploaded": config.mode == "real",
    "paper_metric_eligible": False,
    "manifest_file": "manifest.json",
}
```

Use `atomic_write_json`, run the exact-secret scan over result and cache files in real mode, and print only `DATASET_API_ROLLOUT_MOCK_COMPLETE artifact=...`, `DATASET_API_ROLLOUT_REAL_RESPONSE_RECEIVED artifact=...`, or a sanitized failure category.

- [ ] **Step 5: Update AutoDL commands**

Add the non-paid mock engineering command and the opt-in real one-frame command to `docs/AUTODL_QUICKSTART.md`. State that users must inspect usage/cost before raising the frame limit and that paper mode must run complete videos.

- [ ] **Step 6: Run focused, local, and full verification**

Run without a real key/network call:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q tests\unit\test_api_rollout_selection.py tests\unit\test_api_causal_media.py tests\unit\test_p3_api_client.py tests\integration\test_api_dataset_pipeline.py tests\integration\test_api_dataset_cli.py tests\integration\test_api_single_pass.py
D:\PythonProject7\.venv-p2\Scripts\python.exe -m pytest -q
D:\PythonProject7\.venv-p2\Scripts\python.exe -m ruff check src tests scripts
D:\PythonProject7\.venv-p2\Scripts\python.exe -m compileall -q src scripts
```

With `CHOLECTRACK20_ROOT=D:\cholec_dataset`, run:

```powershell
D:\PythonProject7\.venv-p2\Scripts\python.exe scripts\run_dataset_api_pipeline.py --mode engineering --video-id VID30 --max-frames 1 --dataset-root D:\cholec_dataset --config configs\perception\joint_mock_dataset.yaml --output-root artifacts\api_dataset --cache-root artifacts\api_dataset_cache\mock_validation --run-id local_real_frame_mock_001 --max-provider-calls exact-selection --authorize-data-upload
```

Expected: mock completion, one real CholecTrack20 frame processed locally, no network call, one paired result, one cache entry, clean secret/path scan, and `paper_metric_eligible: false`.

- [ ] **Step 7: Commit the runnable handoff**

```powershell
git add configs/perception/joint_mock_dataset.yaml configs/perception/joint_openrouter_dataset.yaml scripts/run_dataset_api_pipeline.py tests/integration/test_api_dataset_cli.py docs/AUTODL_QUICKSTART.md
git commit -m "feat: add real-data API rollout CLI"
```

Do not run the real OpenRouter command during implementation. Hand the user the exact one-frame real command only after all mock/local/full checks and code review pass.
