"""Cross-check current documentation, phase state, and canonical package ownership."""

from pathlib import Path

from surgical_agent.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACADEMIC_DOC = (
    PROJECT_ROOT
    / "docs/architecture/"
    "Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md"
)
IMPLEMENTATION_SPEC = (
    PROJECT_ROOT
    / "docs/architecture/"
    "Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md"
)


def test_docs_freeze_one_canonical_pipeline_and_package() -> None:
    academic = ACADEMIC_DOC.read_text(encoding="utf-8")
    implementation = IMPLEMENTATION_SPEC.read_text(encoding="utf-8")

    assert "单一运行主干与模块嵌入原则" in academic
    assert "单一 Canonical Pipeline" in implementation
    assert "src/surgical_agent/" in academic
    assert "src/surgical_agent/" in implementation
    assert "├── src/streaming_surgical_agent/" not in academic
    assert "├── src/streaming_surgical_agent/" not in implementation
    assert "P2-1 DatasetAdapter 与 TargetBuilder" in implementation
    assert "P2-6 运行命令与 PASS 门槛" in implementation
    assert "CholecTrack20 是多器械数据" in implementation
    assert "frame_multilabel" in academic
    assert "frame_multilabel" in implementation
    assert "video-identity 去重清单" in implementation


def test_p2_machine_state_matches_documented_checkpoint() -> None:
    base = load_yaml(PROJECT_ROOT / "configs/base.yaml")
    data = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")

    assert base["project"]["current_phase"] == "P2"
    assert base["project"]["phase_status"] == "PASS"
    assert base["project"]["next_phase"] == "P3"
    assert base["data"]["current_phase_blockers"] == []
    assert "p4_prediction_evaluation_granularity_and_matching" in base["data"][
        "deferred_phase_blockers"
    ]
    assert data["local_audit"]["p1_status"] == (
        "PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION"
    )
    assert data["media"]["identity_resolution"][
        "derived_supervision_manifest"
    ] == "repair_manifest.json"

    local_smoke = load_yaml(
        PROJECT_ROOT / "configs/experiments/local_smoke.yaml"
    )
    assert local_smoke["phase_required"] == "P2"
    assert local_smoke["pipeline"]["runner"] == "canonical_streaming"
    assert local_smoke["pipeline"]["gate_policy"] == "never_verify"
    assert local_smoke["smoke"]["forbid_test_for_training_or_selection"] is True


def test_main_documents_have_balanced_fences_and_ordered_phase_headings() -> None:
    academic = ACADEMIC_DOC.read_text(encoding="utf-8")
    implementation = IMPLEMENTATION_SPEC.read_text(encoding="utf-8")
    assert academic.count("```") % 2 == 0
    assert implementation.count("```") % 2 == 0

    headings = (
        "## P0 —",
        "## P1 —",
        "## P2 —",
        "## P3 —",
        "## P4 —",
        "## P5 —",
        "## P6 —",
        "## P7 —",
        "## P8 —",
        "## P9 —",
        "## P10-A —",
        "## P10-B —",
        "## P10-C —",
        "## P10-D —",
        "## P10-E —",
        "## P11 —",
        "## P12 —",
    )
    positions = [implementation.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_docs_freeze_sparse_specialist_routing_contract() -> None:
    academic = ACADEMIC_DOC.read_text(encoding="utf-8")
    implementation = IMPLEMENTATION_SPEC.read_text(encoding="utf-8")

    for document in (academic, implementation):
        assert "Benefit-Routed Sparse Specialist Verification" in document
        assert "VERIFY(scope)" in document
        assert "spatial_track" in document
        assert "interaction" in document
        assert "workflow" in document
        assert "route_observed_mask" in document
        assert "P8-A" in document
        assert "P8-B" in document

    assert "论文 Motivation Figure 冻结内容" in academic
    assert "Task-wise Benefit Gate" in academic
    assert "Deterministic Coordinator" in academic
    assert "class DeterministicCoordinator(Protocol)" in implementation
    assert "test_specialist_router_selects_at_most_one" in implementation
    assert "SPECIALIST ROUTING CLAIM = NOT SUPPORTED" in implementation
