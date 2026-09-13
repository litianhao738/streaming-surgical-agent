"""Pinned user-selected PGP research default. No live pipeline or API access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from surgical_agent.research.gate import pgp_gate

ROOT = Path(__file__).resolve().parents[4]


def load_default(root=None):
    root = ROOT if root is None else Path(root).resolve()
    manifest = json.loads((root / 'DEFAULT_PGP_GATE_VERSION.json').read_text('utf-8'))
    if manifest.get('schema_version') != 'pgp_research_default_v1' or manifest.get('decision_rule') not in ('dual_threshold_v1', 'whole_frame_change_threshold_v1'):
        raise ValueError('unsupported PGP research default')
    if manifest.get('tracker_enabled') is not False or manifest.get('deployable') is not False:
        raise ValueError('this entry supports only the frozen Tracker-free research default')
    path = (root / manifest['model_artifact']).resolve()
    if not path.is_relative_to(root / 'artifacts/training/gate'):
        raise ValueError('default model must be under the Gate artifact directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['model_sha256']:
        raise ValueError('selected PGP weight hash changed')
    model = json.loads(raw)
    expected_schema = pgp_gate.MODEL_SCHEMA if manifest['decision_rule'] == 'dual_threshold_v1' else 'pgp_ambiguity_hgb_v1'
    if model.get('schema') != expected_schema or model.get('tracker_enabled') is not False or model.get('deployable') is not False:
        raise ValueError('selected model is not a Tracker-free PGP research artifact')
    return manifest, model


def predict_default(X, feature_names, root=None):
    manifest, model = load_default(root)
    if manifest['decision_rule'] == 'whole_frame_change_threshold_v1':
        from surgical_agent.research.gate.pgp_ambiguity import predict
        scores, actions = predict(model, X, feature_names, ROOT if root is None else root)
    else:
        scores = pgp_gate.predict_scores(model, X, feature_names)
        actions = pgp_gate.select_actions(scores, model['thresholds'])
    return {'version': manifest['version'], 'scores': scores, 'actions': actions}
