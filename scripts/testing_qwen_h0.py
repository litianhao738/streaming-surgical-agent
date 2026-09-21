"""Durable Qwen H0 and candidate proposals, with distinct request journals."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
import time
import requests

MODEL = 'qwen3.8-max'
ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
CONFIG = dict(model=MODEL, endpoint=ENDPOINT, prompt_cny_per_million='12',
              completion_cny_per_million='36', input_token_ceiling=32768, max_tokens=4096,
              price_source='https://help.aliyun.com/zh/model-studio/qwen3-8-max', price_checked='2026-09-16')


def prepare_config(plan):
    from scripts import run_testing_half_pipeline as core
    from scripts import reviewer_routes_official as routes
    from urllib.parse import urlparse
    with core.app.frozen.joint.credential_context(plan):
        url, key = routes.credentials('qwen',plan['reviewer_config'])
        host = urlparse(url).hostname or ''
        if urlparse(url).scheme != 'https' or not (
                host == 'dashscope.aliyuncs.com' or host.endswith('.cn-beijing.maas.aliyuncs.com')):
            raise ValueError('Qwen H0 requires the configured Beijing Alibaba endpoint')
        response = requests.get(url.rstrip('/')+'/models',headers={'Authorization':'Bearer '+key},timeout=30)
        response.raise_for_status()
        if not any(row.get('id')==MODEL for row in response.json()['data']):
            raise ValueError('configured Qwen endpoint does not list qwen3.8-max')
    return {**CONFIG,'endpoint':url.rstrip('/')+'/chat/completions'}


def wire_body(body):
    wire = deepcopy(body)
    for key in ('provider','reasoning','reasoning_effort','include_reasoning','thinking'):
        wire.pop(key,None)
    wire.update(model=MODEL,enable_thinking=False,temperature=0,max_tokens=4096,stream=False)
    return wire


def charge(usage, config):
    return (Decimal(usage['prompt_tokens'])*Decimal(config['prompt_cny_per_million']) +
            Decimal(usage['completion_tokens'])*Decimal(config['completion_cny_per_million']))/Decimal(1000000)


def call_h0(out, plan, selected, budget, stop, body):
    return call_base(out, plan, selected, budget, stop, body, stage='h0')


def reservation_input_tokens(wire, config):
    """Conservative budget estimate, not a tokenizer or model context limit.

    The historical input_token_ceiling is the minimum reservation envelope.
    Longer proposal prompts reserve more funds instead of failing at 32K bytes.
    Actual usage still determines settlement; request content is never truncated.
    """
    import base64, io, math
    from PIL import Image
    text_bytes, image_tokens = 0, 0
    for message in wire['messages']:
        if isinstance(message['content'],str):
            text_bytes += len(message['content'].encode('utf-8'))
        else:
            for block in message['content']:
                if block['type']=='text':
                    text_bytes += len(block['text'].encode('utf-8'))
                elif block['type']=='image_url':
                    with Image.open(io.BytesIO(base64.b64decode(block['image_url']['url'].split(',',1)[1]))) as im:
                        image_tokens += math.ceil(im.width/14)*math.ceil(im.height/14)+128
    estimate = text_bytes + image_tokens + 1024
    return max(config['input_token_ceiling'], math.ceil(estimate / 1024) * 1024)


def call_base(out, plan, selected, budget, stop, body, *, stage):
    from scripts import run_testing_half_pipeline as core
    from scripts import reviewer_routes_official as routes
    from scripts.full_official_reviewer_transport import parse_changed
    from surgical_agent.research.gate.collection_budget import Budget, BudgetStop, AmbiguousDispatch
    if stage not in ('h0','proposal'):
        raise ValueError('unsupported Qwen base stage')
    wire = wire_body(body)
    config = plan['qwen_h0']
    if config != {**CONFIG,'endpoint':config['endpoint']}:
        raise ValueError('Qwen H0 route/pricing changed since preparation')
    safe = core.app.frozen.joint.roster.transport.redact_images(wire)
    base_seat = 'qwen_h0' if stage == 'h0' else 'qwen_proposal'
    attempt = 0
    while True:
        seat = base_seat if attempt == 0 else f'{base_seat}_retry_{attempt}'
        folder = out/'targets'/selected['key']/seat
        if not (folder/'record.json').exists():
            break
        previous = core.read(folder/'record.json')
        authorization = folder/'retry_authorized.json'
        authorized = (authorization.exists() and previous.get('status') == 'FAILED'
                      and previous.get('finished_utc')
                      and core.read(authorization).get('record_sha256') == core.sha(folder/'record.json'))
        if not retryable_record(previous) and not authorized:
            break
        if (core.read(folder/'request.json') != safe
                or core.sha(folder/'response.json') != previous['response_sha256']):
            raise AmbiguousDispatch('Qwen retry evidence changed')
        budget.settle(Budget.key(selected['key'],stage,seat),Decimal(previous['charge']))
        attempt += 1
    identity = Budget.key(selected['key'],stage,seat)
    if (folder/'record.json').exists():
        record = core.read(folder/'record.json')
        if core.read(folder/'request.json') != safe:
            raise ValueError('Qwen cached request changed')
        if (record['status']!='JSON_PARSED' or not record.get('finished_utc')
                or core.sha(folder/'response.json')!=record['response_sha256']):
            raise AmbiguousDispatch('Qwen '+stage+' failed or unfinished; no automatic resend: '+selected['key'])
        raw = core.read(folder/'response.json')
        budget.settle(identity,Decimal(record['charge']))
        return parse_changed(raw,wire)
    if stop.is_set():
        raise BudgetStop('run stopped before Qwen H0 dispatch')
    url, key = routes.credentials('qwen',plan['reviewer_config'])
    if url.rstrip('/')+'/chat/completions' != config['endpoint']:
        raise ValueError('Qwen H0 endpoint differs from the prepared plan')
    reserved_input = reservation_input_tokens(wire, config)
    reserve = charge(dict(prompt_tokens=reserved_input,completion_tokens=wire['max_tokens']),config)
    budget.reserve(identity,'aliyun_cny',reserve)
    folder.mkdir(parents=True,exist_ok=False)
    core.write(folder/'request.json',safe)
    now = lambda:datetime.now(timezone.utc).isoformat()
    record = dict(target=selected['key'],stage=stage,seat=seat,model=MODEL,endpoint=config['endpoint'],
                  account='aliyun_cny',reserve=str(reserve),charge=str(reserve),
                  reserved_input_tokens=reserved_input, reservation_policy='dynamic_text_vision_envelope_v1',
                  charge_kind='unknown_reserved',status='DISPATCHED',started_utc=now())
    core.write(folder/'record.json',record)
    started = time.perf_counter()
    try:
        response = requests.post(config['endpoint'],json=wire,headers={'Authorization':'Bearer '+key},
                                 timeout=(15,120),allow_redirects=False)
        try:
            raw = response.json()
        except ValueError:
            raw = {'non_json_response':response.text}
        raw = json.loads(json.dumps(raw).replace(key,'[REDACTED]'))
        core.write(folder/'response.json',raw)
        usage = raw.get('usage') or {}
        record.update(http_status=response.status_code,returned_model=raw.get('model'),usage=usage,
                      provider_error=raw.get('error'),
                      response_sha256=core.sha(folder/'response.json'))
        if 'prompt_tokens' in usage and 'completion_tokens' in usage:
            record.update(charge=str(charge(usage,config)),charge_kind='published_rate_estimate_no_cache_discount')
        if not response.ok or raw.get('error'):
            code = (raw.get('error') or {}).get('code', 'unknown')
            raise ValueError(f"Qwen {stage} HTTP {response.status_code}, code={code}, target={selected['key']}; inspect saved response")
        result = parse_changed(raw,wire)
        if stage == 'h0':
            core.app.frozen.gated.h0_from_raw(result)
        record['status']='JSON_PARSED'
        return result
    except BaseException as exc:
        record.update(status='FAILED',error_type=type(exc).__name__)
        if not isinstance(exc, Exception) or not retryable_record({**record, 'finished_utc': now()}):
            stop.set()
        raise
    finally:
        record.update(finished_utc=now(),seconds=time.perf_counter()-started)
        core.write(folder/'record.json',record)
        budget.settle(identity,Decimal(record['charge']))


def retryable_record(record):
    """Retry terminal content/request errors, never unavailable APIs or uncertain calls."""
    from scripts.testing_half_transport import billing_failure
    status = record.get('http_status')
    error = record.get('provider_error')
    content_failure = (status == 200 and record.get('returned_model') == MODEL and not error)
    request_failure = status in (400, 413, 422) and not billing_failure(status, error)
    return (record.get('status') == 'FAILED' and bool(record.get('finished_utc'))
            and (content_failure or request_failure)
            and record.get('error_type') in {'ApiSchemaError', 'JSONDecodeError', 'ValueError',
                                             'KeyError', 'TypeError', 'IndexError'}
            and bool(record.get('response_sha256')))


class ProposalCalls:
    """Only new candidate proposals move to Qwen; reviewers keep their existing routes."""
    def __init__(self, delegate, out, plan, selected, budget, stop):
        self.delegate=delegate
        self.args=(out,plan,selected,budget,stop)

    def call(self,target,stage,seat,body):
        if target != self.args[2]['key']:
            raise ValueError('proposal target mismatch')
        if stage=='proposal' and seat=='base':
            return call_base(*self.args,body,stage='proposal')
        return self.delegate.call(target,stage,seat,body)

    def __getattr__(self,name):
        return getattr(self.delegate,name)
