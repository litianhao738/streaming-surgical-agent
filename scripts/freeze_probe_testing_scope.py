"""Bind the existing Gemini input timeline to the new Gate without relabeling old results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import run_testing_half_pipeline as core

PARENT = ROOT/'artifacts/evaluation/testing_half_unified_gate_tracker_plan_20260916'


def freeze(parent, output):
    spec, _ = core.scope_rows(parent)
    _, gate, _ = core.app.load_gate()
    if gate.get('review_mode') != 'five_head_probe':
        raise ValueError('selected Gate is not the five-head probe Gate')
    result = deepcopy(spec)
    result.update(status='FIVE_HEAD_PROBE_GATE_SCOPE_FROZEN', base_model='google/gemini-3.8-flash',
                  parent_scope_sha256=core.sha(parent/'run_scope.json'), launch_ready=False,
                  data_provenance={'inputs': 'existing Gemini H0; reused without changing predictions',
                                   'existing_full_results': 'historical Gate version; NOT relabeled',
                                   'new_full_results': 'require fresh execution of this scope'})
    result['pipeline'].update(gate_version=gate['version'], gate_model_sha256=gate['model_sha256'],
                              phase_review_enabled=True, review_mode='five_head_probe')
    result['concurrency_contract']['per_target'] = (
        'proposal -> five-head Qwen probe including Phase -> 54-feature Gate -> '
        'conditional five-head panel reusing Qwen -> causal 60-second Phase smoothing')
    result['source_sha256'][gate['model_artifact']] = gate['model_sha256']
    result['remaining_before_launch'] = ['prepare a new execution plan with account caps and CUDA Tracker']
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent/'frame_inventory.jsonl', output/'frame_inventory.jsonl')
    shutil.copyfile(parent/'suggested_budget_limits.json', output/'suggested_budget_limits.json')
    core.write(output/'run_scope.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=core.SCOPE)
    args = parser.parse_args()
    result = freeze(args.parent.resolve(), args.output.resolve())
    print(json.dumps({'scope':str(args.output), 'gate':result['pipeline'], 'api_calls':0}))
