"""Thin V3 command wrappers must reuse the canonical dataset rollout."""

from __future__ import annotations

from pathlib import Path

import scripts.run_always_verify as always_verify
from scripts import infer_v3


def _required_args() -> list[str]:
    return [
        "--mode",
        "engineering",
        "--video-id",
        "VID30",
        "--max-frames",
        "1",
    ]


def test_infer_v3_defaults_to_the_no_training_rule_gate() -> None:
    args = infer_v3.parse_args(_required_args())

    assert args.pipeline_profile == "rule_gate"
    assert args.gate_artifact is None


def test_infer_v3_selects_learned_gate_when_artifact_is_explicit(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "gate.json"
    args = infer_v3.parse_args(
        [*_required_args(), "--gate-artifact", str(artifact)]
    )

    assert args.pipeline_profile == "learned_gate"
    assert args.gate_artifact == artifact


def test_always_verify_entrypoint_fixes_the_runtime_profile() -> None:
    args = always_verify.parse_args(_required_args())

    assert args.pipeline_profile == "always_verify"
