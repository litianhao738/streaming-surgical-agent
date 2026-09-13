"""Official GLM/DeepSeek candidate contract; no implicit production activation."""
from copy import deepcopy
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/reviewer_routes_official_v1.json'

def read_config():return json.loads(CONFIG.read_text(encoding='utf-8'))

def credentials(seat,config):
    if seat not in ('grok','deepseek'):
        from scripts import reviewer_routes_v3 as old
        return old.credentials(seat,config)
    spec=config['routes'][seat]
    value=(ROOT/spec['secret_file']).read_text(encoding='utf-8-sig').strip()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{20,256}',value):
        raise ValueError('expected a single credential in '+spec['secret_file'])
    return spec['base_url'],value

def route_body(seat,body,config):
    if seat not in ('grok','deepseek'):
        from scripts import reviewer_routes_v3 as old
        return old.route_body(seat,body,config)
    spec=config['routes'][seat];wire=deepcopy(body)
    for field in ('provider','reasoning','reasoning_effort','thinking','enable_thinking','thinking_budget','include_reasoning'):
        wire.pop(field,None)
    wire['model']=spec['model'];wire['thinking']=deepcopy(spec['thinking'])
    if 'reasoning_effort' in spec:wire['reasoning_effort']=spec['reasoning_effort']
    # Preserve images, ontology and task. Clarify the actual research context.
    found=False
    for message in wire.get('messages',[]):
        content=message.get('content')
        if not isinstance(content,list):continue
        for part in content:
            if part.get('type')!='text':continue
            try:payload=json.loads(part['text'])
            except (ValueError,KeyError):continue
            if isinstance(payload,dict) and 'academic_context' in payload:
                payload['academic_context']=config['prompt_context']+' Return only a JSON object matching response_schema.'
                part['text']=json.dumps(payload,ensure_ascii=False);found=True
    if not found:raise ValueError('expected explicit academic context in review payload')
    return wire

VERSION='prior-gated-joint-official-reviewers-v1-20260912'

def check_response(seat,raw,config):
    from scripts.reviewer_routes_v3 import check_response as check
    return check(seat,raw,config)
