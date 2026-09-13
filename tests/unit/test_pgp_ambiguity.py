import hashlib
import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from surgical_agent.research.gate import pgp_ambiguity as core


def test_rollback_additions_deletions_preserves_other_heads():
    g,r=sorted(core.AMB_VERBS); i,j=sorted(core.AMB_IVTS)[:2]
    cheap={'verb':[g], 'ivt':[i], 'instrument':[0], 'target':[0], 'phase':[0]}
    nonverb=next(k for k in range(10) if k not in core.AMB_VERBS)
    nonivt=next(k for k in range(100) if k not in core.AMB_IVTS)
    full={'verb':[r,nonverb], 'ivt':[j,nonivt], 'instrument':[1], 'target':[1], 'phase':[1]}
    out=core.repair(cheap,full)
    assert out['verb']==sorted([g,nonverb]) and out['ivt']==sorted([i,nonivt])
    assert all(out[k]==full[k] for k in ('instrument','target','phase'))
    assert cheap['verb']==[g] and full['verb']==[r,nonverb]


def test_prediction_roundtrip_and_tamper(tmp_path):
    x=np.arange(80).reshape(40,2); y=np.arange(40)%2
    estimator=HistGradientBoostingClassifier(max_iter=2,random_state=3407).fit(x,y)
    p=tmp_path/'artifacts/training/gate/test/model.joblib'; p.parent.mkdir(parents=True); joblib.dump(estimator,p)
    model={'schema':core.SCHEMA,'tracker_enabled':False,'deployable':False,'feature_names':['a','b'],
           'threshold':.5,'estimator_artifact':str(p.relative_to(tmp_path)), 'estimator_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    s,a=core.predict(model,x,['a','b'],tmp_path)
    np.testing.assert_array_equal(s,estimator.predict_proba(x)[:,1])
    assert set(a)<= {0,3}
    with pytest.raises(ValueError,match='schema'): core.predict(model,x,['b','a'],tmp_path)
    with pytest.raises(ValueError,match='matrix'): core.predict(model,[[float('nan'),1]],['a','b'],tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): core.predict(model,x,['a','b'],tmp_path)
