"""Provider-native minimum/off thinking contract; preserve frozen v2 routes."""
from copy import deepcopy
from pathlib import Path
import json
from scripts import reviewer_routes_v2 as previous

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/reviewer_routes_v3.json'
VERSION='prior-gated-joint-reviewers-v3-min-thinking-20260912'
credentials=previous.credentials


def read_config():return json.loads(CONFIG.read_text(encoding='utf-8'))


def route_body(seat,body,config):
    if seat=='base':return deepcopy(body)
    out=previous.route_body(seat,body,config)
    policy=config['reasoning_policy'][seat]
    # Drop inherited gateway-specific knobs before applying the native protocol.
    for key in ('reasoning','reasoning_effort','thinking','enable_thinking','thinking_budget','include_reasoning'):
        out.pop(key,None)
    for key in ('thinking','reasoning_effort','reasoning','enable_thinking'):
        if key in policy:out[key]=deepcopy(policy[key])
    return out


def observation(raw):
    usage=raw.get('usage') or {}
    message=((raw.get('choices') or [{}])[0].get('message') or {})
    details=usage.get('completion_tokens_details') or {}
    token_fields=[details.get('reasoning_tokens'),usage.get('reasoning_tokens')]
    reported=[v for v in token_fields if isinstance(v,(int,float))]
    chars=sum(len(message.get(k) or '') for k in ('reasoning_content','reasoning') if isinstance(message.get(k),(str,type(None))))
    evidence=bool(chars or any(v>0 for v in reported) or message.get('reasoning_details'))
    return {'reasoning_tokens':max(reported) if reported else None,'reasoning_text_characters':chars,
        'reasoning_observed':evidence,'completion_tokens':usage.get('completion_tokens'),
        'absence_is_not_proof_of_internal_computation':True}


def check_response(seat,raw,config):
    evidence=observation(raw)
    if config['reasoning_policy'][seat]['mode']=='off' and evidence['reasoning_observed']:
        raise RuntimeError('provider returned reasoning despite disabled policy: '+seat)
    return evidence
