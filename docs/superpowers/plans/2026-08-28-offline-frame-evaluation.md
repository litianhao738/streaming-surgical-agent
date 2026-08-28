# Offline Frame Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocked offline evaluator with a working CholecTrack20 Validation/Test entry point that verifies completed predictions, aligns authorized frame GT, computes the existing formal metrics, and writes deterministic result artifacts.

**Architecture:** A strict artifact reader reconstructs `PredictionRecord` values without dataset access. A split-local data loader then verifies only the selected split, constructs frame targets, and separates scored GT identities from canonical media-only identities. A coordinator reuses `FrameMetricAccumulator`, derives eligibility/provenance, and atomically writes the report; `scripts/evaluate.py` is only a CLI adapter.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, hashlib/json, existing CholecTrack20 parser/masks/media resolvers, NumPy/scikit-learn through the existing metrics package, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-offline-frame-evaluation-design.md`

## Global Constraints

- Evaluation is offline-only: no credential resolution, API client, provider transport, media decoding, model update, or write beneath the input run/dataset roots.
- Instrument/Verb/Target/IVT remain frame-level multi-label; Phase remains frame-level single-label.
- Reuse `frame_recognition_metrics_v1`; do not introduce a second formula.
- Validation access is split-local and must never touch the Testing directory.
- Test access requires `--authorize-test-gt-evaluation`, paper mode, eight videos, and complete reconstructed prediction counts before dataset access.
- Current `cholectrack20_api_rollout_v1` results remain `paper_metric_eligible=false`.
- VID30 remains `candidate_repaired_validation`; its derived field qualification is applied before instance masks.
- Canonical runtime frames without authoritative GT are reported as unscored; missing predictions for GT-bearing frames fail.
- Input/output JSON rejects duplicate keys and non-finite values; persisted output uses sorted, indented, ASCII-safe UTF-8 JSON plus one trailing newline.

---

### Task 1: Strict completed-run artifact reader

**Files:**
- Create: `src/surgical_agent/evaluation/offline_artifacts.py`
- Create: `tests/unit/test_offline_artifacts.py`
- Modify: `src/surgical_agent/evaluation/__init__.py`

**Interfaces:**
- Consumes: `PredictionRecord`, `DatasetSplit`, `prediction_record_sha256`, the v1 frame manifest and rollout formats.
- Produces: `CompletedRun`, `ArtifactFile`, `OfflineEvaluationError`, `load_completed_run(run_dir: str | Path) -> CompletedRun`, and `identity_sha256(identities) -> str`.

- [ ] **Step 1: Write strict reader tests and artifact fixture builder**

```python
def test_load_completed_run_reconstructs_verified_predictions(tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path, split="validation")
    loaded = load_completed_run(run_dir)
    assert loaded.run_id == "eval-unit"
    assert loaded.mode == "engineering"
    assert loaded.predictions[0].source_split is DatasetSplit.VALIDATION
    assert loaded.prediction_identities == (("VID110", 1),)


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_json_key", "nonfinite_score", "bool_frame_id", "bad_file_hash",
     "duplicate_identity", "mixed_score_semantics", "eligibility_disagreement"),
)
def test_load_completed_run_rejects_malformed_or_tampered_input(
    tmp_path: Path, mutation: str
) -> None:
    run_dir = write_completed_run_fixture(tmp_path, mutation=mutation)
    with pytest.raises(OfflineEvaluationError):
        load_completed_run(run_dir)
```

- [ ] **Step 2: Run the tests and observe the missing-module failure**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_offline_artifacts.py -q`

Expected: FAIL during import because `offline_artifacts.py` does not exist.

- [ ] **Step 3: Implement immutable reader contracts and duplicate-safe JSON**

```python
@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class CompletedRun:
    run_dir: Path
    run_id: str
    mode: str
    declared_split: DatasetSplit | None
    effective_split: DatasetSplit
    video_ids: tuple[str, ...]
    frame_counts: Mapping[str, int]
    predictions: tuple[PredictionRecord, ...]
    prediction_files: Mapping[str, ArtifactFile]
    evidence_files: Mapping[str, ArtifactFile]
    input_hashes: Mapping[str, str]
    provider: str
    model_requested: str
    models_returned: tuple[str, ...]
    prompt_version: str
    response_schema_version: str
    repair_manifest_sha256: str
    alignment_versions: tuple[str, ...]
    paper_metric_eligible: bool

    @property
    def prediction_identities(self) -> tuple[tuple[str, int], ...]:
        return tuple((row.video_id, row.frame_id) for row in self.predictions)
```

Implement `_load_json()` with `object_pairs_hook` that rejects duplicate keys,
`parse_constant` that rejects non-finite values, exact key-set validators, safe
relative path resolution, byte SHA-256, exact non-boolean integer/number
guards, prediction/evidence JSONL pairing, and strict JSON-to-dataclass
normalization. Treat nested evidence values and rollout `usage` as opaque JSON
objects after their enclosing hashes/types pass.

- [ ] **Step 4: Run reader tests and existing writer tests**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_offline_artifacts.py tests\unit\test_frame_result_writer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/surgical_agent/evaluation/offline_artifacts.py src/surgical_agent/evaluation/__init__.py tests/unit/test_offline_artifacts.py
git commit -m "feat: load verified offline prediction runs"
```

---

### Task 2: Split-local selection and frame GT aggregation

**Files:**
- Create: `src/surgical_agent/evaluation/frame_ground_truth.py`
- Create: `tests/unit/test_frame_ground_truth.py`
- Modify: `src/surgical_agent/evaluation/__init__.py`

**Interfaces:**
- Consumes: `CompletedRun`, `parse_annotation_file`, `ExactFrameFolderResolver`, `load_derived_supervision_manifest`, canonical per-instance masks.
- Produces: `GroundTruthSource`, `EvaluationData`, `load_evaluation_data(run, dataset_root, *, authorize_test_gt_evaluation) -> EvaluationData`, and `aggregate_frame_target(frame, *, allowed_tasks, source) -> FrameSupervisionTarget`.

- [ ] **Step 1: Write aggregation, selection, and Test-lock tests**

```python
def test_aggregate_frame_target_masks_partial_task_without_making_negative() -> None:
    frame = canonical_frame_with_two_instances(second_verb=-1)
    target = aggregate_frame_target(
        frame,
        allowed_tasks=frozenset({"instrument", "verb", "target", "ivt", "phase"}),
        source="Validation/VID110/vid110.json",
    )
    assert target.instrument_ids == (0, 1)
    assert target.mask.instrument is True
    assert target.verb_ids == ()
    assert target.mask.verb is False


def test_test_run_is_rejected_before_dataset_access_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = completed_test_run_fixture()
    monkeypatch.setattr(Path, "exists", fail_if_called)
    with pytest.raises(OfflineEvaluationError, match="authorization"):
        load_evaluation_data(run, tmp_path / "dataset", authorize_test_gt_evaluation=False)


def test_validation_loader_reports_media_only_identity(tmp_path: Path) -> None:
    dataset = write_validation_dataset_fixture(tmp_path, media_ids=(1, 2), gt_ids=(1,))
    data = load_evaluation_data(
        completed_validation_run_fixture(frame_ids=(1, 2)),
        dataset,
        authorize_test_gt_evaluation=False,
    )
    assert data.runtime_identities == (("VID110", 1), ("VID110", 2))
    assert tuple(data.targets) == (("VID110", 1),)
```

- [ ] **Step 2: Run the tests and observe the missing-module failure**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_frame_ground_truth.py -q`

Expected: FAIL during import because `frame_ground_truth.py` does not exist.

- [ ] **Step 3: Implement split-local resolution and GT aggregation**

```python
@dataclass(frozen=True)
class GroundTruthSource:
    relative_path: str
    sha256: str
    provenance_status: str


@dataclass(frozen=True)
class EvaluationData:
    split: DatasetSplit
    runtime_identities: tuple[tuple[str, int], ...]
    targets: Mapping[tuple[str, int], FrameSupervisionTarget]
    sources: Mapping[str, GroundTruthSource]
    provenance_by_video: Mapping[str, str]
    repair_manifest_sha256: str
```

Implementation order:

1. validate `dataset_root` only after the artifact/Test gates;
2. hash `repair_manifest.json` and compare it with the run;
3. for Validation, enumerate only `Validation/<video>` and numeric PNG stems;
4. for VID30, use its manifest `media_source`, `annotation_source`, and
   `field_supervision`, require contained paths, and record
   `candidate_repaired_validation`;
5. for ordinary Validation, use `Frames` and the raw video JSON;
6. for authorized Testing, enumerate only `Testing`, require exactly eight
   canonical videos, and use the annotation frame keys after parsing;
7. compare reconstructed runtime identities with prediction identities;
8. aggregate sorted task unions with mask-all-instances-valid semantics;
9. return only selected targets and reject a target outside a paper runtime
   selection.

Do not call `discover_official_split_manifest()` or construct
`CholecTrack20DatasetAdapter`.

- [ ] **Step 4: Run GT tests plus existing parser/mask tests**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_frame_ground_truth.py tests\unit\test_p1_data_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- src/surgical_agent/evaluation/frame_ground_truth.py src/surgical_agent/evaluation/__init__.py tests/unit/test_frame_ground_truth.py
git commit -m "feat: align split-local frame ground truth"
```

---

### Task 3: Formal evaluator and deterministic report writer

**Files:**
- Create: `src/surgical_agent/evaluation/offline_frame.py`
- Create: `tests/unit/test_offline_frame_evaluator.py`
- Modify: `src/surgical_agent/evaluation/__init__.py`

**Interfaces:**
- Consumes: `CompletedRun`, `EvaluationData`, `FrameMetricAccumulator`, `atomic_write_text`.
- Produces: `EvaluationArtifacts`, `resolve_evaluation_output(run_dir, dataset_root, output_dir) -> Path`, `align_scored_pairs(run, data) -> tuple[tuple[PredictionRecord, FrameSupervisionTarget], ...]`, `evaluate_and_write(run, data, destination, *, test_gt_authorized) -> EvaluationArtifacts`, and `evaluate_completed_run(run_dir, dataset_root, *, output_dir=None, authorize_test_gt_evaluation=False) -> EvaluationArtifacts`.

- [ ] **Step 1: Write alignment, eligibility, lifecycle, and output tests**

```python
def test_evaluator_scores_gt_and_reports_media_only_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = completed_validation_run_fixture(frame_ids=(1, 2))
    data = evaluation_data_fixture(runtime_ids=(1, 2), gt_ids=(1,))
    monkeypatch.setattr(offline_frame, "load_completed_run", lambda _: run)
    monkeypatch.setattr(offline_frame, "load_evaluation_data", lambda *a, **k: data)
    result = evaluate_completed_run(tmp_path / "run", tmp_path / "dataset")
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["scored_frame_counts"] == {"VID110": 1}
    assert report["unscored_prediction_counts"] == {"VID110": 1}
    assert report["paper_metric_eligible"] is False
    assert report["metrics"]["schema_version"] == "frame_recognition_metrics_v1"


def test_evaluator_rejects_missing_gt_bearing_prediction() -> None:
    with pytest.raises(OfflineEvaluationError, match="GT-bearing"):
        align_scored_pairs(run_with_ids((1,)), data_with_gt_ids((1, 2)))
```

Add tests for pairwise path overlap, fresh output, `INCOMPLETE` on failure,
canonical report/manifest SHA-256, exact schema fields, integer class-key
serialization, and `paper_mode_complete` remaining ineligible for v1/VID30.

- [ ] **Step 2: Run the tests and observe the missing-module failure**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_offline_frame_evaluator.py -q`

Expected: FAIL during import because `offline_frame.py` does not exist.

- [ ] **Step 3: Implement alignment, metric reuse, and report persistence**

```python
@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    status_path: Path
    report_path: Path
    manifest_path: Path


def evaluate_completed_run(
    run_dir: str | Path,
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    authorize_test_gt_evaluation: bool = False,
) -> EvaluationArtifacts:
    run = load_completed_run(run_dir)
    destination = resolve_evaluation_output(
        run.run_dir,
        dataset_root,
        output_dir,
    )
    data = load_evaluation_data(
        run,
        dataset_root,
        authorize_test_gt_evaluation=authorize_test_gt_evaluation,
    )
    return evaluate_and_write(
        run,
        data,
        destination,
        test_gt_authorized=authorize_test_gt_evaluation,
    )
```

Implement exact runtime/GT alignment, reject an empty scored set, count/hash
unscored identities, feed sorted scored pairs to `FrameMetricAccumulator`,
convert dataclasses/mapping keys to canonical JSON, derive scope and stable
eligibility reasons, write `INCOMPLETE -> report -> manifest -> COMPLETE`, and
persist only relative paths/hashes/provenance. Categorize raised failures into
the spec's closed failure vocabulary without printing source payloads.

- [ ] **Step 4: Run evaluator and formal metric tests**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\unit\test_offline_frame_evaluator.py tests\unit\test_frame_recognition_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/surgical_agent/evaluation/offline_frame.py src/surgical_agent/evaluation/__init__.py tests/unit/test_offline_frame_evaluator.py
git commit -m "feat: write formal offline frame evaluations"
```

---

### Task 4: CLI, usage documentation, and end-to-end verification

**Files:**
- Modify: `scripts/evaluate.py`
- Create: `tests/integration/test_offline_evaluate_cli.py`
- Modify: `docs/AUTODL_QUICKSTART.md`

**Interfaces:**
- Consumes: `evaluate_completed_run()`.
- Produces: `build_parser()`, `main(argv: Sequence[str] | None = None) -> int`, and the user-facing command.

- [ ] **Step 1: Write CLI contract and synthetic end-to-end tests**

```python
def test_cli_has_only_offline_evaluation_inputs() -> None:
    destinations = {action.dest for action in build_parser()._actions}
    assert {"run_dir", "dataset_root", "output_dir",
            "authorize_test_gt_evaluation"} <= destinations
    assert not ({"api_key", "provider", "model", "endpoint"} & destinations)


def test_cli_writes_complete_validation_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, dataset_root = write_end_to_end_validation_fixture(tmp_path)
    assert main(["--run-dir", str(run_dir), "--dataset-root", str(dataset_root)]) == 0
    assert "evaluation_report.json" in capsys.readouterr().out
```

The end-to-end fixture uses synthetic Validation-only paths and includes one
media-only frame; it does not create a Testing directory or API credential.

- [ ] **Step 2: Run the CLI tests and observe the blocked-stub failure**

Run: `.venv-p2\Scripts\python.exe -m pytest tests\integration\test_offline_evaluate_cli.py -q`

Expected: FAIL because `scripts/evaluate.py` still exits through `_phase_stub`.

- [ ] **Step 3: Replace the stub with a thin sanitized CLI**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorize-test-gt-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_completed_run(
        args.run_dir,
        args.dataset_root,
        output_dir=args.output_dir,
        authorize_test_gt_evaluation=args.authorize_test_gt_evaluation,
    )
    print(result.report_path)
    return 0
```

Catch only `OfflineEvaluationError`, print its sanitized message to stderr,
and return a nonzero code. Do not print parsed payloads, absolute GT paths, or
exception reprs.

- [ ] **Step 4: Document the Validation and authorized Test commands**

Add the following to `docs/AUTODL_QUICKSTART.md`:

```bash
python scripts/evaluate.py \
  --run-dir artifacts/api_dataset/<run-id> \
  --dataset-root "$CHOLECTRACK20_ROOT"
```

State that Test adds `--authorize-test-gt-evaluation`, engineering Test is
rejected, current v1/VID30 reports remain paper-ineligible, and the output is
written beside the run unless `--output-dir` is supplied.

- [ ] **Step 5: Run focused and full verification**

Run:

```powershell
.venv-p2\Scripts\python.exe -m pytest tests\unit\test_offline_artifacts.py tests\unit\test_frame_ground_truth.py tests\unit\test_offline_frame_evaluator.py tests\integration\test_offline_evaluate_cli.py tests\unit\test_frame_recognition_metrics.py -q
.venv-p2\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: focused tests PASS, full suite has zero failures, and diff check exits
zero.

- [ ] **Step 6: Run the local read-only Validation identity audit**

Run a small Python assertion through the new split-local loader against
`D:\cholec_dataset` without predictions or metrics, confirming VID110 runtime
1716/GT 1713 and VID30 runtime/GT 2717/2717. Do not access Testing and do not
call an API.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- scripts/evaluate.py tests/integration/test_offline_evaluate_cli.py docs/AUTODL_QUICKSTART.md
git commit -m "feat: expose offline frame evaluation CLI"
```

---

## Completion checklist

- [ ] `scripts/evaluate.py --help` succeeds without credentials.
- [ ] A synthetic Validation run produces COMPLETE status/report/manifest.
- [ ] Missing authorization blocks Test before dataset access.
- [ ] Existing formal metric semantics are unchanged.
- [ ] All focused and repository tests pass.
- [ ] Git diff contains no whitespace errors or unrelated changes.
