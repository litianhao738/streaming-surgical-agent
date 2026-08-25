"""Executable P0 repository-layout smoke audit."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REQUIRED_PATHS = (
    PROJECT_ROOT
    / "docs"
    / "architecture"
    / "Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md",
    PROJECT_ROOT
    / "docs"
    / "architecture"
    / "Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md",
    PROJECT_ROOT / "docs" / "V3_1_API_IMPLEMENTATION_AUDIT.md",
    PROJECT_ROOT / "reports" / "P0_REPORT.md",
    PROJECT_ROOT / "src" / "surgical_agent" / "__init__.py",
    PROJECT_ROOT / "src" / "surgical_agent" / "api" / "contracts.py",
    PROJECT_ROOT / "src" / "surgical_agent" / "workflow" / "contracts.py",
    PROJECT_ROOT / "configs" / "api" / "default.yaml",
    PROJECT_ROOT / "configs" / "workflow" / "default.yaml",
    PROJECT_ROOT / "configs" / "memory" / "default.yaml",
    PROJECT_ROOT / "configs" / "gate" / "default.yaml",
    PROJECT_ROOT / "tests" / "unit",
    PROJECT_ROOT / "tests" / "integration",
    PROJECT_ROOT / "tests" / "fixtures",
)

REQUIRED_MODULES = (
    "surgical_agent",
    "surgical_agent.api",
    "surgical_agent.config",
    "surgical_agent.data",
    "surgical_agent.tracking",
    "surgical_agent.models",
    "surgical_agent.research.knowledge",
    "surgical_agent.research.gate",
    "surgical_agent.research.verification",
    "surgical_agent.research.reliability",
    "surgical_agent.research.memory",
    "surgical_agent.systems",
    "surgical_agent.training",
    "surgical_agent.inference",
    "surgical_agent.evaluation",
    "surgical_agent.runtime",
    "surgical_agent.artifacts",
    "surgical_agent.workflow",
    "streaming_surgical_agent",
)


def main() -> int:
    """Check P0 files and imports without opening the raw dataset."""

    missing_paths = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    import_errors: dict[str, str] = {}
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    git_initialized = (PROJECT_ROOT / ".git").is_dir()
    deferred_phase_blockers = [
        "p3_exact_provider_and_model_identifier",
        "p4_prediction_evaluation_granularity_and_matching",
        "p9_canonical_per_sample_task_error",
    ]
    nonblocking_provenance_caveats = [
        "official_png_extraction_wording_not_published",
        "official_negative_label_sentinel_wording_not_published",
        "vid30_candidate_reconstruction_requires_disclosure",
    ]
    result = {
        "phase": "P0-A/P0-C scaffold audit",
        "scope": "repository, imports, config boundaries; no method behavior",
        "project_root": str(PROJECT_ROOT),
        "missing_paths": missing_paths,
        "import_errors": import_errors,
        "git_initialized": git_initialized,
        "dataset_opened": False,
        "method_implementation": "NOT_STARTED",
        "p1_data_layer": "PASS_WITH_EXPLICIT_PARTIAL_SUPERVISION",
        "current_phase_blockers": [],
        "deferred_phase_blockers": deferred_phase_blockers,
        "nonblocking_provenance_caveats": nonblocking_provenance_caveats,
        "status": (
            "PASS"
            if not missing_paths and not import_errors and git_initialized
            else "FAIL"
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
