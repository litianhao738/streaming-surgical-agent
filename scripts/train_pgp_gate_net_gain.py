"""Independent offline net-gain trial; never promotes the selected PGP default."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import train_pgp_gate_no_tracker as util
from surgical_agent.research.gate import pgp_net_gain as net
from surgical_agent.research.gate import pgp_training as ref
from surgical_agent.research.gate.pgp_default import load_default
from tools.audit import gate_proposals_offline_20260913 as legacy

OLD = util.DEFAULT
OUT = ROOT / 'artifacts/training/gate/pgp_gate_net_gain_20260913_r2'


def run():
    OUT.mkdir(exist_ok=False); (OUT / 'models').mkdir()
    start = time.time()
    def emit(s):
        line = util.now() + ' ' + s
        print(line, flush=True)
        with (OUT / 'run.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    with legacy.offline():
        try:
            manifest, _ = load_default()
            receipt = json.loads((OLD / 'training_receipt.json').read_text('utf-8'))
            replay = json.loads((OLD / 'replay/pgp_replay_receipt.json').read_text('utf-8'))
            recipe = json.loads((OLD / 'recipe.json').read_text('utf-8'))
            bindings = {str(OLD / p): h for p, h in receipt['output_sha256'].items()}
            bindings.update(replay['source_sha256'])
            bindings.update(recipe['tracker_frozen_sha256'])
            for p in [Path(__file__), Path(net.__file__), ROOT / 'DEFAULT_PGP_GATE_VERSION.json',
                      ROOT / 'DEFAULT_PIPELINE_VERSION.json', OLD / 'training_receipt.json',
                      ROOT / 'docs/PGP_GATE_NET_GAIN_TRIAL_PROTOCOL_2026-09-13.md']:
                bindings[str(p.resolve())] = util.sha(p)
            emit(f'Verifying {len(bindings)} frozen source/default/Tracker/artifact bindings')
            for p, h in bindings.items():
                if util.sha(p) != h:
                    raise ValueError('source mismatch: ' + p)
            with np.load(OLD / 'replay/pgp_replay_arrays.npz', allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files if not k.startswith('cost_')}
                data['baselines_costs'] = {k[5:]: z[k].copy() for k in z.files if k.startswith('cost_')}
            data['feature_names'] = replay['feature_names']
            data['rows'] = json.loads((Path(legacy.SOURCE) / 'primary_rows.json').read_text('utf-8'))
            if isinstance(data['rows'], dict):
                data['rows'] = data['rows']['rows']
            assert len(data['ids']) == len(data['rows']) == 6059
            assert set(data['videos']) == {'VID103', 'VID23', 'VID31', 'VID96'}
            assert [r['sample_id'] for r in data['rows']] == data['ids'].tolist()
            assert all(r['source_split'] == 'Training' and r['origin'] == 'original_attempt_observed_behavior' for r in data['rows'])
            assert not any('tracker' in n.lower() for n in data['feature_names'])
            util.write(OUT / 'recipe.json', {'source_sha256': bindings, 'default_version': manifest['version'],
                'supervision': net.SUPERVISION, 'decision_rule': net.RULE, 'thresholds': net.THRESHOLDS,
                'candidate_count': len(net.POLICIES), 'tracker_enabled': False, 'warm_start': False,
                'api_calls': 0, 'Testing_access': False, 'deployable': False, 'automatic_promotion': False})
            result, scores, actions, targets = net.nested_fit(data, emit,
                lambda n, m: util.write(OUT / 'models' / (n + '.json'), m))
            emit('Computing fixed 300-draw block-random control')
            random_report, draws = ref.random_control(data, actions)
            result['random_control'] = random_report
            refs = json.loads((OLD / 'baselines.json').read_text('utf-8'))
            with np.load(OLD / 'semantic_benefit_original_harm/outer_predictions.npz', allow_pickle=False) as z:
                assert np.array_equal(z['ids'], data['ids'])
                refs['pinned_dual_pgp'] = ref.measure(data, z['actions'])
            util.save_arrays(OUT / 'outer_predictions.npz', ids=data['ids'], videos=data['videos'],
                scores=scores, net_scores=net.net_scores(scores), actions=actions, targets=targets,
                selected_counts=data['actions_counts'][np.arange(len(actions)), actions],
                selected_costs=data['action_costs'][np.arange(len(actions)), actions], random_draws=draws)
            state = 'CONDITIONAL_DEVELOPMENT_PASS' if result['conditional_development_pass'] else 'DEVELOPMENT_NOT_PASSED'
            util.write(OUT / 'report.json', result)
            util.write(OUT / 'baselines.json', refs)
            lines = ['# PGP net-gain trial', '', f'State: {state}. Default unchanged. Tracker off. No API or Testing.', '',
                     '| Scheme | Original F1 | FP+FN | Logical calls | Known USD |', '|---|---:|---:|---:|---:|']
            for name, r in [(k, refs[k]) for k in ('h0', 'cheap', 'full', 'old_whole_frame_qwen_gate', 'pinned_dual_pgp')] + [('net_gain_trial', result)]:
                q = r['original']
                lines.append(f"| {name} | {q['five_head_mean_f1']:.6f} | {q['total_errors']} | {r['logical_calls']:.0f} | {r['known_usd']:.6f} |")
            lines += ['', 'Known USD excludes unpriced GLM/DeepSeek. These are replay costs, not new expenditure.',
                      'Outer LOVO is conditional development evaluation, not independent validation. Final OOF threshold selection is tuning-set performance.', '',
                      '## Outer folds', '']
            for f in result['folds']:
                s = f['inner_selection']; q = f['held_result']['original']
                lines.append(f"- {f['held_video']}: {s['status']}, {s['feasible_candidates']}/289 feasible, thresholds={s['thresholds']}, held F1={q['five_head_mean_f1']:.6f}.")
            lines += ['', '## Final Training OOF calibration (not an independent result)', '', json.dumps(result['final_training_oof_selection']['selected'], ensure_ascii=False), '']
            (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
            emit('Rechecking preserved sources and default')
            for p, h in bindings.items():
                if util.sha(p) != h:
                    raise ValueError('frozen file changed: ' + p)
            emit('FINISHED: ' + state)
            util.write(OUT / 'training_receipt.json', {'state': 'COMPLETED', 'development_state': state,
                'elapsed_seconds': time.time() - start, 'verified_frozen_files': len(bindings),
                'default_unchanged': True, 'tracker_files_unchanged': len(recipe['tracker_frozen_sha256']),
                'api_calls': 0, 'Testing_access': False, 'VID110_access': False, 'deployable': False,
                'output_sha256': {str(p.relative_to(OUT)): util.sha(p) for p in OUT.rglob('*') if p.is_file()}})
        except Exception:
            emit(traceback.format_exc())
            util.write(OUT / 'failure_receipt.json', {'state': 'FAILED', 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    run()
