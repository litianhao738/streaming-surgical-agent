"""Fit and evaluate PGP-Gate from sealed Training caches; never load old weights.

Example: .venv-p2/Scripts/python.exe -B -X utf8 scripts/train_pgp_gate_no_tracker.py
"""
from __future__ import annotations

import os
for _pgp_thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_pgp_thread_var] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from surgical_agent.research.gate import pgp_gate as core
from surgical_agent.research.gate import pgp_training as training
from tools.audit import gate_proposals_offline_20260913 as legacy
from tools.audit import pgp_gate_replay_20260913 as replay

DEFAULT = ROOT / 'artifacts/training/gate/pgp_gate_no_tracker_20260913_r1'
TRACKER = ROOT / 'artifacts/training/tracker_clip_v2_oof5_20260906'
VARIANTS = ('semantic_benefit_original_harm', 'original', 'merged')
PROTOCOL = ROOT / 'docs/PGP_GATE_NO_TRACKER_TRAINING_PROTOCOL_2026-09-13.md'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def save_arrays(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def tracker_snapshot():
    # Byte hashes bind immutability only; no Tracker weights/predictions enter ML.
    files = sorted(p for p in TRACKER.rglob('*') if p.is_file())
    if len([p for p in files if p.name == 'checkpoint.pt']) != 6:
        raise ValueError('expected the corrected full and five OOF Tracker checkpoints')
    return {str(p): sha(p) for p in tqdm(files, desc='Freeze Tracker artifacts', mininterval=1)}


def baselines(data):
    result = {}
    n = len(data['ids'])
    for name in ('h0', 'cheap', 'full'):
        costs = data['baselines_costs'][name]
        result[name] = {'original': training.quality(data[name]), 'merged': training.quality(data[name + '_merged']),
                        **dict(zip(training.COST_FIELDS, costs.sum(axis=0).tolist())),
                        'by_video': {str(v): {'original': training.quality(data[name][data['videos'] == v]),
                                             'merged': training.quality(data[name + '_merged'][data['videos'] == v])}
                                     for v in sorted(set(data['videos']))}}
    for name, action in (('qwen_probe_only', 0), ('interaction_only_exact', 1), ('phase_only_exact', 2), ('always_qwen2_exact', 3)):
        result[name] = training.measure(data, np.full(n, action))
    previous = ROOT / 'artifacts/research/gate_proposals_followup_20260913_r1/claude3_qwen2_merged_help_hurt_routes.npz'
    with np.load(previous, allow_pickle=False) as r:
        if not np.array_equal(r['ids'], data['ids']):
            raise ValueError('old whole-frame Gate baseline cohort mismatch')
        result['old_whole_frame_qwen_gate'] = training.measure(data, np.where(r['route_balanced'], 3, 0))
    if not np.isclose(result['old_whole_frame_qwen_gate']['original']['five_head_mean_f1'], 61.959484015301236, atol=1e-9):
        raise ValueError('old whole-frame reference changed')
    return result


def report_markdown(summary):
    lines = ['# PGP-Gate 无 Tracker：离线训练结果', '',
             f"完成时间：{summary['finished_utc']}。主方案状态：**{summary['state']}**。", '',
             'Tracker 权重及预测保持冻结，未作为训练输入；Gate 从头拟合，未加载旧权重。',
             '本次为四个反复使用的 Training 视频、6,059 个首轮有观测结果的目标的条件化开发评估。',
             '上游先验没有在每个 Gate 外层折内完全重算；不构成独立泛化或稳定性保证。', '',
             '| 方案 | 原口径 F1 | FP+FN | 总逻辑调用 | 已知 USD |',
             '|---|---:|---:|---:|---:|']
    titles = {'h0': 'H0', 'cheap': 'H0 + 先验 cheap', 'full': '全审核（含先验）',
              'qwen_probe_only': '只初审后保留 cheap', 'interaction_only_exact': '固定只审交互',
              'phase_only_exact': '固定只审阶段', 'always_qwen2_exact': '同 Qwen 初审/停止，始终继续',
              'old_whole_frame_qwen_gate': '旧整帧 Qwen Gate',
              'semantic_benefit_original_harm': '新 PGP 主方案（归并收益 + 原口径伤害）',
              'original': '固定消融：原口径监督', 'merged': '固定消融：归并口径监督'}
    for name, r in {**summary['baselines'], **summary['variants']}.items():
        q = r['original']
        lines.append(f"| {titles[name]} | {q['five_head_mean_f1']:.4f} | {q['total_errors']:,} | {r['logical_calls']:,.0f} | {r['known_usd']:.2f} |")
    primary = summary['variants'][VARIANTS[0]]
    lines += ['', '已知 USD 不包含未计价 GLM/DeepSeek；缓存重放次数按真实推理逻辑调用计，不是本轮实际支出。',
              '没有新的审核 API 调用，也未运行 Testing、部署或 Tracker 训练。API 延迟无法从离线门控训练实测，本次只报告本地训练耗时。', '',
              '## 主方案逐视频稳定性', '', '| 视频 | cheap F1 | PGP F1 | F1 变化 | 错漏变化 |', '|---|---:|---:|---:|---:|']
    for v, d in primary['quality_cost_acceptance']['per_video_vs_cheap'].items():
        c = summary['baselines']['cheap']['by_video'][v]['original']
        r = primary['by_video'][v]['original']
        lines.append(f"| {v} | {c['five_head_mean_f1']:.4f} | {r['five_head_mean_f1']:.4f} | {d['f1_delta']:+.4f} | {d['errors_delta']:+,} |")
    lines += ['', '## 可行性与失败结果', '']
    for name, r in summary['variants'].items():
        lines.append(f"- {titles[name]}：开发验收={r['conditional_development_pass']}；动作分布={r['actions']}；四外层折内层均可行={r['all_inner_feasible']}。")
        lines.append(f"  - 最终研究模型门槛状态：{r['final_training_oof_selection']['status']}；可行候选={r['final_training_oof_selection']['feasible_candidates']}/625。")
        for f in r['folds']:
            s = f['inner_selection']
            lines.append(f"  - 留出 {f['held_video']}：{s['status']}，{s['feasible_candidates']}/{s['candidate_count']} 个策略可行，门槛={s['thresholds']}。")
    lines += ['', '不可行时的全跳过结果仍支付每帧五次初审前置调用，不能当成便宜档原成本或成功省钱方案。',
              '主监督在运行前固定；两个消融即使较好，也不自动替换主方案或当成独立验证。', '',
              '## 研究模型与重现', '',
              '每个变体保存四个外层模型和一个全 Training 研究模型，均为纯 JSON 权重、阈值及特征 schema。',
              'final_research_model 用 Training OOF 选阈值后在全部 Training 行拟合；表中成绩来自外层留出预测，不是该最终模型的训练集成绩。',
              '所有导出均 deployable=false。无可行门槛的模型明确标注 INFEASIBLE，不供上线使用。',
              '查看 recipe.json、replay/、各变体 report.json、outer_predictions.npz、models/ 和 training_receipt.json。',
              '原始总队列 7,372 条中无首轮最终结果的 1,313 条不被填成中性或负例；后续重试/语法修复回答没有混入本轮。', '']
    return '\n'.join(lines)


def run(output):
    output = Path(output).resolve()
    if output.parent != (ROOT / 'artifacts/training/gate').resolve() or not output.name.startswith('pgp_gate_no_tracker_'):
        raise ValueError('use a fresh direct artifacts/training/gate/pgp_gate_no_tracker_* output')
    output.mkdir(exist_ok=False)
    start = time.time()
    def emit(message):
        line = f'{now()} {message}'
        print(line, flush=True)
        with (output / 'run.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    with legacy.offline():
        try:
            frozen_tracker = tracker_snapshot()
            code = [Path(__file__), Path(core.__file__), Path(training.__file__), Path(replay.__file__), Path(legacy.__file__), PROTOCOL]
            frozen_source = {str(p.resolve()): sha(p) for p in code}
            old_baseline = ROOT / 'artifacts/research/gate_proposals_followup_20260913_r1/claude3_qwen2_merged_help_hurt_routes.npz'
            frozen_source[str(old_baseline)] = sha(old_baseline)
            recipe = {'version': 'pgp-branch-no-tracker-20260913-v1', 'created_utc': now(),
                      'primary': VARIANTS[0], 'fixed_supervision_ablations': list(VARIANTS[1:]),
                      'tracker_enabled': False, 'tracker_frozen_sha256': frozen_tracker,
                      'old_weights_loaded': False, 'warm_start': False, 'source_sha256': frozen_source,
                      'thresholds_per_branch': training.BRANCH_THRESHOLDS, 'candidate_count': len(training.POLICIES),
                      'threshold_selection': 'inner video OOF only, lowest logical calls subject to original pooled/per-video quality and actual positive gain; tie worst-video F1, pooled F1, errors, fixed index',
                      'outer_split': 'four Training videos LOVO', 'inner_split': 'other three videos LOVO',
                      'upstream_fully_nested': False, 'independent_validation': False,
                      'failure_policy': 'original single logical attempt; no retry/syntax substitution',
                      'infeasible': 'skip all with explicit failure; still count two-Qwen five-call prefix',
                      'random_control': {'draws': 300, 'seed': 3407, 'block': 30, 'per_video_call_cap': True},
                      'api_calls': 0, 'Testing_access': False, 'VID110_access': False, 'deployable': False,
                      'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}
            write(output / 'recipe.json', recipe)
            emit('Preparing and validating four cached actions; Tracker features excluded')
            replay_out = output / 'replay'; replay_out.mkdir()
            data = replay.prepare(replay_out)
            if any('tracker' in name.lower() for name in data['feature_names']):
                raise ValueError('Tracker feature entered the no-Tracker schema')
            refs = baselines(data)
            write(output / 'baselines.json', refs)
            summary = {'baselines': refs, 'variants': {}, 'primary': VARIANTS[0]}
            for variant in VARIANTS:
                folder = output / variant; folder.mkdir(); (folder / 'models').mkdir()
                emit('Starting fixed supervision: ' + variant)
                def save_model(name, model):
                    model.update({'source_sha256': frozen_source, 'prior_source': str(legacy.SOURCE),
                                  'upstream_fully_nested': False, 'independent_validation': False})
                    write(folder / 'models' / (name + '.json'), model)
                result, scores, actions = training.nested_fit(data, variant, emit, save_model)
                emit('Computing same-call-cap block-random branch control for ' + variant)
                random_report, random_draws = training.random_control(data, actions)
                result['random_control'] = random_report
                result['beats_random_both_original_metrics'] = bool(
                    result['original']['five_head_mean_f1'] > random_report['f1_p975'] and
                    result['original']['total_errors'] < random_report['errors_p025'])
                targets = core.branch_targets(data['cheap'], data['full'], data['cheap_merged'], data['full_merged'], supervision=variant)
                result['oof_auc_diagnostic'] = {f'{b}_{k}': float(roc_auc_score(targets[:, j, c], scores[:, j, c]))
                    if len(np.unique(targets[:, j, c])) > 1 else None
                    for j, b in enumerate(('interaction', 'phase')) for c, k in enumerate(('help', 'harm'))}
                save_arrays(folder / 'outer_predictions.npz', ids=data['ids'], videos=data['videos'], scores=scores,
                            actions=actions, selected_counts=data['actions_counts'][np.arange(len(actions)), actions],
                            selected_costs=data['action_costs'][np.arange(len(actions)), actions], targets=targets,
                            random_draws=random_draws)
                write(folder / 'report.json', result)
                summary['variants'][variant] = result
            summary.update({'state': 'CONDITIONAL_DEVELOPMENT_PASS' if summary['variants'][VARIANTS[0]]['conditional_development_pass'] else 'DEVELOPMENT_NOT_PASSED',
                            'finished_utc': now(), 'elapsed_seconds': time.time() - start,
                            'tracker_enabled': False, 'old_weights_loaded': False,
                            'api_calls': 0, 'Testing_access': False, 'VID110_access': False, 'deployable': False})
            write(output / 'summary.json', summary)
            (output / 'REPORT.md').write_text(report_markdown(summary), encoding='utf-8')
            all_bindings = {**frozen_source, **frozen_tracker, **data['replay_receipt']['source_sha256']}
            for path, digest in tqdm(all_bindings.items(), desc='Verify frozen sources and Tracker', mininterval=1):
                if sha(path) != digest:
                    raise ValueError('Frozen source changed during run: ' + path)
            emit('FINISHED: ' + summary['state'])
            products = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
            write(output / 'training_receipt.json', {'state': 'COMPLETED', 'development_state': summary['state'],
                  'completed_utc': now(), 'output_sha256': products, 'frozen_tracker_files_unchanged': len(frozen_tracker),
                  'new_api_calls': 0, 'Testing_access': False, 'VID110_access': False,
                  'weights_initialized_from_scratch': True, 'deleted_old_weights': False, 'deployable': False})
        except Exception:
            emit('FAILED: ' + traceback.format_exc())
            write(output / 'failure_receipt.json', {'state': 'FAILED', 'time': now(), 'api_calls': 0, 'Testing_access': False,
                                                  'VID110_access': False, 'deployable': False, 'error': traceback.format_exc()})
            raise
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.output)
