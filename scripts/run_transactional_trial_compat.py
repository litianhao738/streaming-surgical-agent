"""Qwen JSON-mode transport fix; preserve aborted v1 and share its total budget."""
import argparse
import json
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from scripts import run_transactional_trial as trial

SOURCE=trial.ROOT/'artifacts/preflight/transactional_trial_20260909_v1'


def add_json_declaration(body):
    result=deepcopy(body)
    packet=json.loads(result['messages'][0]['content'][0]['text'])
    packet['output_format']='Return exactly one JSON object matching response_schema.'
    result['messages'][0]['content'][0]['text']=json.dumps(packet,ensure_ascii=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','execute','score'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old=trial.read(SOURCE/'budget.json');done=trial.read(SOURCE/'completion.json')
    if not old['stopped'] or done['fatal_error']:
        raise ValueError('closed prior run required')
    for n,h in done['hashes'].items():
        trial.same(trial.sha(SOURCE/n),h,'aborted run preserved')
    original_limits=deepcopy(trial.LIMITS)
    trial.LIMITS={k:str(Decimal(v)-Decimal(old['occupied'][k])) for k,v in original_limits.items()}
    original_wire=trial.edit_wire
    def corrected(seat,*a,**kw):
        body=original_wire(seat,*a,**kw)
        return add_json_declaration(body) if seat=='qwen' and body.get('response_format',{}).get('type')=='json_object' else body
    trial.edit_wire=corrected
    adapter=trial.CholecTrack20DatasetAdapter(Path('D:/cholec_dataset'),causal_window_size=3)
    if args.command=='prepare':
        trial.prepare(args.output,adapter)
        plan=trial.read(args.output/'plan.json')
        extra=[Path(__file__).resolve(),trial.ROOT/'tests/unit/test_transactional_compat.py']
        for f in extra:
            plan['extra_sources'][f.relative_to(trial.ROOT).as_posix()]=trial.sha(f)
            dest=args.output/'frozen_source'/f.relative_to(trial.ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True);trial.shutil.copyfile(f,dest)
        plan['compatibility']={'change':'Only Qwen JSON output declaration restored; semantic template/selector unchanged.',
            'aborted_source':str(SOURCE),'aborted_completion_sha256':trial.sha(SOURCE/'completion.json'),
            'aborted_calls':done['post_calls'],'max_total_calls':done['post_calls']+trial.MAX_CALLS,
            'original_total_limits':original_limits,'prior_occupied':old['occupied'],
            'protocol':'Manual new version after deterministic format failure. Same fixed 24 unscored targets; all outputs re-requested, no cached H0 reuse. No automatic retry in either run. Old run not scored or used to tune semantics.'}
        trial.save(args.output/'plan.json',plan);trial.verify(args.output)
        print(json.dumps({'compat_ready':True,'max_total_calls':done['post_calls']+trial.MAX_CALLS,
                          'remaining_limits':trial.LIMITS,'plan_sha256':trial.sha(args.output/'plan.json')}),flush=True)
    else:
        plan=trial.verify(args.output)
        trial.same(plan['compatibility']['aborted_completion_sha256'],trial.sha(SOURCE/'completion.json'),'prior closure')
        getattr(trial,args.command)(args.output,adapter)


if __name__=='__main__':
    main()
