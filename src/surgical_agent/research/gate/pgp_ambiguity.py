"""Versioned ambiguity repair and single-probe whole-frame research Gate."""
from pathlib import Path
import hashlib
import joblib
import numpy as np
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS

SCHEMA = 'pgp_ambiguity_hgb_v1'
AMB_VERBS = {_TASK_NAMES['verb'].index(v) for v in ('grasp', 'retract')}
AMB_IVTS = {i for i,c in COMPONENTS.items() if c['verb'] in AMB_VERBS}


def repair(cheap, full):
    result = {k: sorted(set(v)) for k,v in full.items()}
    for task, ambiguous in (('verb', AMB_VERBS), ('ivt', AMB_IVTS)):
        result[task] = sorted((set(full[task])-ambiguous) | (set(cheap[task]) & ambiguous))
    return result


def predict(model, X, feature_names, root):
    if model.get('schema') != SCHEMA or model.get('tracker_enabled') is not False or model.get('deployable') is not False:
        raise ValueError('unsupported ambiguity research model')
    names=list(feature_names)
    if names!=model['feature_names'] or any('tracker' in n.lower() for n in names):
        raise ValueError('wrong feature schema or Tracker input')
    x=np.asarray(X,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(names) or not np.isfinite(x).all():
        raise ValueError('invalid feature matrix')
    threshold=float(model['threshold'])
    if not np.isfinite(threshold): raise ValueError('invalid threshold')
    root=Path(root).resolve(); path=(root/model['estimator_artifact']).resolve()
    if not path.is_relative_to(root/'artifacts/training/gate'): raise ValueError('estimator escaped Gate artifacts')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model['estimator_sha256']:
        raise ValueError('estimator hash changed')
    # Only deserialize the locally generated estimator bound by the selected manifest.
    estimator=joblib.load(path)
    scores=estimator.predict_proba(x)[:,1]
    if not np.isfinite(scores).all(): raise ValueError('nonfinite prediction')
    return scores, 3*(scores>=threshold).astype(np.uint8)
