"""Regression: reviewed evidence wins over prior frequency, cheap prior stays enabled."""
from surgical_agent.research.gate import pgp_runtime as runtime
from surgical_agent.research.verification.prior_gated_repair import select_prior_gated
from tests.unit.test_prior_gated_repair import state, pool_with, prior, neutral_means


def test_post_review_does_not_veto_rare_or_add_frequent_relation():
    pool=pool_with(60)
    table=prior({59:0.001,60:0.9})
    out,log=select_prior_gated(state(),pool,neutral_means(pool),table,phase=3,**runtime.POST_REVIEW_GATE)
    assert out['ivt']==[59]
    assert out['target']==[1]
    assert log['vetoed']==log['prior_added']==[]
    cheap,_=select_prior_gated(state(),pool,None,table,phase=3,**runtime.GATE)
    assert cheap['ivt']==[60]


def test_panel_can_still_add_rare_relation_without_prior():
    pool=pool_with(60);means=neutral_means(pool)
    means.update({'ivt_60':4.4,'instrument_2':4.4,'verb_2':4.4,'target_0':4.4})
    out,log=select_prior_gated(state(),pool,means,prior({59:0.5,60:0.001}),phase=3,**runtime.POST_REVIEW_GATE)
    assert 60 in out['ivt']
    assert log['vetoed']==log['prior_added']==[]


def test_default_manifest_declares_the_active_repair_policy():
    import json
    from pathlib import Path
    config=json.loads((Path(__file__).resolve().parents[2]/'DEFAULT_PIPELINE_VERSION.json').read_text(encoding='utf-8'))
    assert config['post_review_prior_enabled'] is False
    assert config['post_review_prior']==runtime.POST_REVIEW_GATE
    assert config['repair_policy_version']==runtime.REPAIR_POLICY_VERSION
