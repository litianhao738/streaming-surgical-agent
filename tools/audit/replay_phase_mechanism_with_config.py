"""Replay immutable trial sources with a hash-verified omitted config dependency.

No inference calls and no changes to the original snapshot. The configuration
is recovered from the predecessor snapshot whose plan hash is bound in ours.
"""
import importlib.util
import json
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
frozen = output / "frozen_source"
sys.path[:0] = [str(frozen), str(frozen / "src")]
from scripts import run_phase_mechanism_comparison as trial
from scripts import run_prior_panel_trial as prior

plan = trial.verify(output)
source_plan_paths = [Path(name) for name in plan["source_archive"] if Path(name).name == "plan.json"]
if len(source_plan_paths) != 1:
    raise ValueError("one hash-bound predecessor plan required")
source_plan_path = source_plan_paths[0]
source_plan = trial.old.read(source_plan_path)
relative = "configs/perception/joint_openrouter_h0.yaml"
config = source_plan_path.parent / "frozen_source" / relative
trial.old.same(trial.old.sha(config), source_plan["sources"][relative])
original_load = prior.load_api_config


def load_config(path):
    if Path(path) == frozen / relative:
        return original_load(config)
    return original_load(path)


prior.load_api_config = load_config
adapter = trial.old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
trial.score(output, adapter)
spec = importlib.util.spec_from_file_location("frozen_phase_audit", frozen / "tools/audit/audit_phase_mechanism_comparison.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.main(output)
supplement = {"profile": "verified_replay_config_supplement_v1", "api_calls": 0,
    "original_frozen_sources_modified": False, "predecessor_plan": str(source_plan_path),
    "predecessor_plan_sha256": trial.old.sha(source_plan_path), "config": str(config),
    "config_sha256": trial.old.sha(config), "raw_replayed": True,
    "metrics_sha256": trial.old.sha(output / "metrics.json")}
trial.old.save(output / "replay_dependency_supplement.json", supplement)
print(json.dumps(supplement))
