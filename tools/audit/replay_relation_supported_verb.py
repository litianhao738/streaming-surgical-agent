"""Offline exploratory admission policy, never changes IVT or default pipeline."""
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.run_prior_panel_trial import save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.prior_panel import COMPONENTS

SOURCES = [('verb_prompt_20260909_v1', ['control', 'verb_prompt']),
           ('new_training_verb_guard_20260909_v1', ['original'])]
OUTPUT = ROOT / 'artifacts/preflight/relation_supported_verb_offline_20260909_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def restrict(h0, final):
    out = deepcopy(final)
    permitted = set(h0['verb']) | {COMPONENTS[i]['verb'] for i in final['ivt']}
    out['verb'] = [v for v in final['verb'] if v in permitted]
    return out


def main():
    if OUTPUT.exists():
        raise ValueError('preserve previous exploration; do not overwrite')
    manifests = {}
    for name, _ in SOURCES:
        folder = ROOT / 'artifacts/preflight' / name
        done = read(folder / 'completion.json')
        assert not done['fatal_error']
        hashes = done.get('hashes', done.get('inference_artifact_sha256'))
        for n, h in hashes.items():
            assert hashlib.sha256((folder / n).read_bytes()).hexdigest() == h
        manifests[name] = {n:hashlib.sha256((folder / n).read_bytes()).hexdigest()
                           for n in ['predictions.json','scored_truth.json','completion.json']}
    save(OUTPUT / 'plan.json', {'sources':SOURCES, 'source_hashes':manifests,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rule':'Starting from original final output, remove only Verb labels absent from H0 and absent from verbs of all retained IVTs. Preserve all other labels and original deletions. No GT in restrict().',
        'scope':'Exploratory offline replay on previously inspected Training cohorts; no new API or independent validation. Report every arm; do not select thresholds.'})
    report = {}
    for name, arms in SOURCES:
        folder = ROOT / 'artifacts/preflight' / name
        rows = read(folder / 'predictions.json')
        if isinstance(rows,dict):
            rows = rows['targets']
        truth = {(t['video_id'],t['frame_id']):t for t in read(folder / 'scored_truth.json')}
        for arm in arms:
            modified, deltas, removed = [], [], []
            for r in rows:
                new = restrict(r['h0'],r[arm])
                assert all(new[t]==r[arm][t] for t in ('instrument','target','ivt','phase'))
                gt = truth[r['video_id'],r['frame_id']]
                modified.append({**gt,'h0':r[arm],'h1':None,'final':new})
                deltas.append({'key':r['key'],**frame_delta(r[arm],new,gt['gt'],gt['mask'])})
                for v in set(r[arm]['verb'])-set(new['verb']):
                    removed.append({'key':r['key'],'verb':v,'mask':gt['mask']['verb'],
                                    'was_correct':v in gt['gt']['verb']})
            report[name+'/'+arm] = {'metrics':compute_repair_comparison(modified),
                'changes':summarize_deltas(deltas),'removed':removed}
    save(OUTPUT / 'report.json',report)
    print(json.dumps({k:{'verb_before':v['metrics']['arms']['h0']['tasks']['verb'],
        'verb_after':v['metrics']['arms']['final']['tasks']['verb'],
        'changes':v['changes'],'removed':v['removed']} for k,v in report.items()},indent=2))


if __name__ == '__main__':
    main()
