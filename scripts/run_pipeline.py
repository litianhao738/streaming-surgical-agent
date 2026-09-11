"""Dispatch the explicitly selected pipeline; no command only displays its manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "DEFAULT_PIPELINE_VERSION.json"


def write_review_diagnostics(output: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.write_review_diagnostics import write_report

    path, report = write_report(output)
    print(json.dumps({"review_diagnostics": str(path), "reason_counts": report["reason_counts"],
                      "predictions_changed": False, "additional_api_calls": 0}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="info",
                        choices=("info", "prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", type=Path,
                        help="New experiment directory for prepare; its saved directory for execute/score.")
    parser.add_argument("--dataset-root", type=Path,
                        help="Dataset root; otherwise use CHOLECTRACK20_ROOT or the existing runner default.")
    args = parser.parse_args(argv)
    selection = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8-sig"))
    if args.command == "info":
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for prepare, execute and score")
    entry = selection["scoring_entrypoint" if args.command == "score" else "implementation_entrypoint"]
    command = [sys.executable, "-X", "utf8", str(ROOT / entry)]
    if args.command != "score" or selection.get("scoring_requires_command", False):
        command.append(args.command)
    command.extend(("--output", str(args.output.resolve())))
    dataset_root = args.dataset_root or os.environ.get("CHOLECTRACK20_ROOT")
    if dataset_root:
        if selection.get("accepts_dataset_root", True):
            command.extend(("--dataset-root", str(Path(dataset_root).resolve())))
        elif Path(dataset_root).resolve() != Path(selection["dataset_root"]).resolve():
            parser.error("selected frozen runner uses the dataset_root recorded in its manifest")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode == 0 and args.command in selection.get("diagnostic_extension", {}).get("automatic_after", []):
        write_review_diagnostics(args.output.resolve())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
