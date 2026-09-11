"""Add all recorded plan identities to exclusions, then time/mask-select new frames."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from scripts.prepare_new_training_verb_selection import VIDEOS, prepare
from scripts.run_candidate_panel_trial import sha
from scripts.run_contact_first_candidate_trial import metadata_preflight
from scripts.run_prior_panel_trial import read, save
from scripts.run_recent_mean_panel_trial import RATES_V2


def main():
    output=ROOT/'artifacts/preflight/transactional_inputs_20260909_v1'
    inventory=output.with_name(output.name+'_history.json')
    if output.exists() or inventory.exists():
        raise ValueError('preserve existing selection')
    parent=ROOT/'artifacts/preflight/verb_addition_confirmation_20260909_inventory/historical_inventory.json'
    old=read(parent)
    excluded={v:set(fs) for v,fs in old['excluded_targets_by_video'].items()}
    hashes={str(parent):sha(parent)}
    def visit(value,video=None):
        if isinstance(value,dict):
            video=value.get('video_id',video)
            if video in VIDEOS:
                for k in ('frame_id','target_frame_id'):
                    if type(value.get(k)) is int:
                        excluded.setdefault(video,set()).add(value[k])
                excluded.setdefault(video,set()).update(x for x in value.get('causal_frame_ids',[]) if type(x) is int)
            for child in value.values():
                visit(child,video)
        elif isinstance(value,list):
            for child in value:
                visit(child,video)
    for p in sorted((ROOT/'artifacts/preflight').rglob('plan.json')):
        if 'frozen_source' in p.parts or 'clean_checkout' in p.parts:
            continue
        visit(read(p));hashes[str(p.relative_to(ROOT))]=sha(p)
    save(inventory,{'scope':'Parent comprehensive history plus all preflight plan target/causal image identities. No labels used.',
        'source_sha256':hashes,'excluded_targets_by_video':{v:sorted(fs) for v,fs in excluded.items()}})
    prepare(inventory,output,Path('D:/cholec_dataset'))
    save(output/'model_metadata.json',{**metadata_preflight(),'paid_calls':0})
    save(output/'budget.json',{'limits':{'openrouter_usd':'6','xai_usd':'3','aliyun_cny':'3'},'rates':RATES_V2})


if __name__=='__main__':
    main()
