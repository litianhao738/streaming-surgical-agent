"""Tracker + Gemini 3.8 probe profile. Old Qwen estimators are incompatible."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from surgical_agent.research.gate.pgp_runtime import run_target, PROBE_NAMES
from surgical_agent.research.gate.escalation_rule_training import FEATURE_ORDER
from surgical_agent.data.schemas import InferenceSample, DatasetSplit
from surgical_agent.tracking.oof_index import load_tracker_oof_index

PROFILE='pgp-tracker-gemini38-probe-v1'
MODEL='google/gemini-3.8-flash'
SCHEMA='pgp_tracker_gemini38_hgb_v1'
FEATURE_NAMES=tuple(sorted(FEATURE_ORDER))+tuple('gemini38'+k[len('qwen'):] for k in PROBE_NAMES)


class FrozenTracker:
    """Load held-out detector/tracker predictions once; serve exact causal windows."""
    def __init__(self,index_path):
        self.index=load_tracker_oof_index(index_path); self.providers={}
    def snapshot(self,selected):
        video=selected['video_id']
        if video not in self.providers: self.providers[video]=self.index.provider_for(video)
        sample=InferenceSample(video_id=video,target_frame_id=selected['frame_id'],
            causal_frame_ids=tuple(selected['causal_frame_ids']),
            media_refs=tuple(i['path'] for i in selected['images']),source_split=DatasetSplit.TRAINING,
            alignment_version=selected['alignment_version'])
        return self.providers[video].snapshot(sample)


def load_predictor(path,tracker_enabled):
    path=Path(path).resolve(); model=json.loads(path.read_text('utf-8'))
    if model.get('schema')!=SCHEMA or model.get('probe_model')!=MODEL or model.get('feature_names')!=list(FEATURE_NAMES):
        raise ValueError('a retrained Tracker/Gemini-3.8 Gate is required; old Qwen weights cannot be reused')
    if tracker_enabled not in model.get('supported_tracker_modes',[]):
        raise ValueError('Gate was not trained/evaluated for this Tracker mode')
    if model.get('selection_status')!='FEASIBLE': raise ValueError('Gate has no feasible calibrated threshold')
    estimator_path=(path.parent/model['estimator_file']).resolve()
    if not estimator_path.is_relative_to(path.parent): raise ValueError('estimator escaped model directory')
    if hashlib.sha256(estimator_path.read_bytes()).hexdigest()!=model['estimator_sha256']: raise ValueError('estimator changed')
    threshold=float(model['threshold'])
    if not np.isfinite(threshold): raise ValueError('invalid threshold')
    estimator=joblib.load(estimator_path)
    def predict(features):
        if set(features)!=set(FEATURE_NAMES): raise ValueError('feature schema mismatch')
        x=np.array([[features[k] for k in FEATURE_NAMES]])
        if not np.isfinite(x).all(): raise ValueError('invalid features')
        score=float(estimator.predict_proba(x)[0,1])
        return score,3*int(score>=threshold)
    return predict


def run(backend,selected,prior,predict_gate,*,tracker=None):
    if getattr(backend,'probe_model',None)!=MODEL:
        raise ValueError('backend must supply Gemini-3.8 review evidence, not old Gemini/Qwen or H0 answers')
    snapshot=tracker.snapshot(selected) if tracker is not None else None
    if tracker is not None and snapshot.get('status')!='AVAILABLE': raise ValueError('enabled Tracker has no causal prediction')
    result=run_target(backend,selected,prior,predict_gate,tracker_snapshot=snapshot,
        include_tracker_features=True,probe_seat='gemini',probe_prefix='gemini38')
    result.update(profile=PROFILE,probe_model=MODEL)
    return result
