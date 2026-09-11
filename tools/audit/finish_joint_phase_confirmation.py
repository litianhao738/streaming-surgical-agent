"""Finish the predeclared secondary arm, then score both closed confirmation arms."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/preflight"
PRIMARY = BASE / "joint_phase_confirm16_independent_20260910_v1"
SECONDARY = BASE / "joint_phase_confirm16_alternative_20260910_v1"


def run(script, arguments, name):
    print("START " + name, flush=True)
    with (BASE / f"{name}_stdout.log").open("w", encoding="utf-8") as out, \
         (BASE / f"{name}_stderr.log").open("w", encoding="utf-8") as err:
        subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=ROOT,
            stdout=out, stderr=err, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    print("DONE " + name, flush=True)


if __name__ == "__main__":
    if not (PRIMARY / "completion.json").exists() or SECONDARY.exists():
        raise ValueError("Requires completed primary and new secondary destination")
    run(ROOT / "scripts/run_joint_phase_parallel_trial.py", ["prepare", "--parent", PRIMARY,
        "--output", SECONDARY, "--variant", "v4"], "joint_confirm_secondary_prepare")
    run(ROOT / "scripts/run_joint_phase_parallel_trial.py", ["preflight", "--output", SECONDARY], "joint_confirm_secondary_preflight")
    run(SECONDARY / "frozen_source/scripts/run_joint_phase_parallel_trial.py", ["execute", "--output", SECONDARY], "joint_confirm_secondary_execute")
    def score(pair):
        name, path = pair
        run(path / "frozen_source/scripts/run_joint_phase_parallel_trial.py", ["score", "--output", path], f"joint_confirm_{name}_score")
        run(ROOT / "tools/audit/audit_joint_phase_feedback.py", [path], f"joint_confirm_{name}_audit")
        run(ROOT / "tools/audit/summarize_joint_phase_details.py", [path], f"joint_confirm_{name}_details")
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(score, [("primary", PRIMARY), ("secondary", SECONDARY)]))
    run(ROOT / "tools/audit/summarize_joint_phase_series.py", [], "joint_confirm_series")
