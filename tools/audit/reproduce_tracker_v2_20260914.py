"""Run existing probes with only OUT relocated; deny network and preserve evidence."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import ast
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.audit.gate_proposals_offline_20260913 import offline


def main():
    out = ROOT / 'artifacts/research/tracker_scheme4_reproduction_20260914_r1'
    out.mkdir(exist_ok=False)
    results = {}
    for suffix in ('probe', 'ablation', 'rejected'):
        source = ROOT / f'tools/audit/tracker_pipeline_v2_{suffix}.py'
        expected = ROOT / f'artifacts/research/tracker_pipeline_v2_{suffix}_20260914.json'
        target = out / expected.name
        raw = source.read_bytes()
        tree = ast.parse(raw.decode('utf-8'))
        matched = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in node.targets):
                tree.body[i] = ast.parse(f'OUT = Path({str(target)!r})').body[0]
                matched += 1
        assert matched == 1
        print('Reproducing ' + suffix, flush=True)
        with offline(), (out / (suffix + '.log')).open('x', encoding='utf-8') as log, redirect_stdout(log):
            exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
        same = json.loads(target.read_bytes()) == json.loads(expected.read_bytes())
        results[suffix] = {'json_equal': same, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                           'reference_sha256': hashlib.sha256(expected.read_bytes()).hexdigest(),
                           'reproduced_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        print(suffix + ': JSON equal = ' + str(same), flush=True)
    (out / 'receipt.json').write_text(json.dumps({'api_calls': 0, 'only_source_change': 'OUT destination', 'results': results}, indent=2), encoding='utf-8')
    if not all(r['json_equal'] for r in results.values()):
        raise ValueError('Reproduction differs; inspect before interpreting new results')


if __name__ == '__main__':
    main()
