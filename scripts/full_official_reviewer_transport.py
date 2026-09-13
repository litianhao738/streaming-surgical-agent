"""Durable v2 changed-route calls plus exact reuse of unchanged old requests."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json
import threading
import requests
from scripts import collect_gate_escalation_v1 as old
from scripts import reviewer_routes_official as routes
from surgical_agent.research.gate.collection_budget import Budget,BudgetStop,AmbiguousDispatch

CHANGED=frozenset(('grok','deepseek','qwen'))


def parse_changed(raw,wire):
    if raw.get('model')!=wire['model']:raise ValueError('returned model identity mismatch')
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop' or choice['message'].get('refusal'):raise ValueError('incomplete/refused response')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON field')
            d[k]=v
        return d
    content=choice['message']['content'].strip()
    if content.startswith('```') and content.endswith('```'):
        content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(content,object_pairs_hook=pairs)


def qwen_charge(usage,reserve):
    if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:return reserve,'unknown_reserved'
    n=usage['prompt_tokens']
    inp,out=('0.2','0.8') if n<=32768 else ('0.6','2.4') if n<=262144 else ('1.2','4.8')
    return (Decimal(n)*Decimal(inp)+Decimal(usage['completion_tokens'])*Decimal(out))/Decimal(1000000),'published_rate_estimate'


class Calls:
    def __init__(self,run,plan,selected,budget,stop,*,replay_only=False):
        self.run,self.plan,self.s,self.budget,self.stop=Path(run),plan,selected,budget,stop
        self.folder=self.run/'targets'/selected['key'];self.replay_only=replay_only
        self.seen=set();self.lock=threading.RLock();self.rows=[];self.reused=[]
        self.cache=old.old.read(self.run/selected['reuse_cache']) if selected.get('reuse_cache') else {}
        self.delegate=None

    def reuse(self,target,stage,seat,body):
        item=self.cache.get(stage+'_'+seat)
        if item is None:return False,None
        wire=old.trial.base_body(body,'gemini') if seat=='base' else body
        if old.old.joint.roster.transport.redact_images(wire)!=item['request']:
            raise RuntimeError('cached request differs from new request: '+target+'/'+stage+'/'+seat)
        if item['record']['status'] in old.trial.VALID:
            raw=item['response']['body']
            if seat!='base':routes.check_response(seat,raw,self.plan['reviewer_config'])
            if raw['model']!=wire['model']:raise RuntimeError('cached model identity mismatch')
            content=raw['choices'][0]['message']['content']
            result=json.loads(content) if seat=='base' else old.trial.parse_review_json(content)[0]
        else:result=None
        self.reused.append(stage+'_'+seat)
        return True,result

    def call(self,target,stage,seat,body):
        if target!=self.s['key']:raise RuntimeError('target identity mismatch')
        identity=(target,stage,seat)
        with self.lock:
            if identity in self.seen:raise RuntimeError('duplicate request in target')
            self.seen.add(identity)
        if seat not in CHANGED:
            found,result=self.reuse(target,stage,seat,body)
            if found:return result
            with self.lock:
                if self.delegate is None:
                    if self.replay_only:self.delegate=old.trial.Replay(self.folder/'run','gemini')
                    else:self.delegate=old.ResumeCalls(self.folder/'run',self.plan,self.budget,self.stop)
            if self.replay_only and identity not in self.delegate.rows:
                # A previous explicit global stop may have produced no dispatch.
                return None
            return self.delegate.call(target,stage,seat,body)
        wire=routes.route_body(seat,body,self.plan['reviewer_config'])
        if seat=='qwen':
            found,result=self.reuse(target,stage,seat,wire)
            if found:return result
        safe=old.old.joint.roster.transport.redact_images(wire)
        folder=self.folder/'changed'/f'{stage}_{seat}'
        if (folder/'record.json').exists():
            record=old.old.read(folder/'record.json')
            if old.old.read(folder/'request.json')!=safe:raise RuntimeError('resumed changed request differs')
            if record['status']=='DISPATCHED' or 'finished_utc' not in record:
                raise AmbiguousDispatch('changed request has uncertain outcome: '+str(identity))
            if record.get('response_sha256') and old.digest(folder/'response.json')!=record['response_sha256']:
                raise RuntimeError('changed response drift')
            self.rows.append(record)
            if record['status']=='JSON_PARSED':
                raw=old.old.read(folder/'response.json');routes.check_response(seat,raw,self.plan['reviewer_config'])
                return parse_changed(raw,wire)
            return None
        if self.replay_only:raise RuntimeError('replay missing changed dispatch')
        if self.stop.is_set():raise BudgetStop('global stop before dispatch')
        url,key=routes.credentials(seat,self.plan['reviewer_config'])
        if url!=self.plan['endpoints'][seat]:raise RuntimeError('endpoint drift')
        account={'grok':'glm_requests','deepseek':'deepseek_requests','qwen':'aliyun_cny'}[seat]
        reserve=(old.old.joint.roster.transport.envelope(seat,wire,self.plan['call_rates']) if seat=='qwen' else Decimal(1))
        key_id=Budget.key(*identity)
        try:self.budget.reserve(key_id,account,reserve)
        except BudgetStop:self.stop.set();raise
        record={'target':target,'stage':stage,'seat':seat,'model':wire['model'],'endpoint':url,
            'account':account,'reserve':str(reserve),'charge':str(reserve),
            'charge_kind':'unknown_reserved' if seat=='qwen' else 'request_count_not_money',
            'status':'DISPATCHED','started_utc':old.old.now()}
        old.write(folder/'request.json',safe);old.write(folder/'record.json',record)
        parsed=None
        try:
            response=requests.post(url.rstrip('/')+'/chat/completions',json=wire,
                headers={'Authorization':'Bearer '+key},timeout=(15,120),allow_redirects=False)
            try:raw=response.json()
            except ValueError:raw={'non_json_response':response.text}
            raw=json.loads(json.dumps(raw).replace(key,'[REDACTED]'))
            old.write(folder/'response.json',raw)
            usage=raw.get('usage') or {};record.update(http_status=response.status_code,usage=usage,
                response_sha256=old.digest(folder/'response.json'),returned_model=raw.get('model'))
            if seat=='qwen':
                charge,kind=qwen_charge(usage,reserve);record.update(charge=str(charge),charge_kind=kind)
            if response.ok and not raw.get('error'):
                parsed=parse_changed(raw,wire);routes.check_response(seat,raw,self.plan['reviewer_config']);record['status']='JSON_PARSED'
            else:
                record.update(status='API_FAILED',provider_error=raw.get('error'))
                err=raw.get('error',{});refusal=isinstance(err,dict) and err.get('code')=='data_inspection_failed'
                if response.status_code in (400,401,402,403,404) and not refusal:
                    self.stop.set();record['global_stop']=True
        except (requests.RequestException,ValueError,KeyError,TypeError,IndexError,RuntimeError) as exc:
            record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc).replace(key,'[REDACTED]')[:200])
            if 'identity mismatch' in str(exc) or 'disabled policy' in str(exc):self.stop.set();record['global_stop']=True
        finally:
            record['finished_utc']=old.old.now();old.write(folder/'record.json',record)
            self.budget.settle(key_id,Decimal(record['charge']))
            with self.lock:self.rows.append(record)
        return parsed

    def close(self):
        if self.delegate is not None and not self.replay_only:
            self.delegate.close_ledger();self.rows+=self.delegate.rows


def reconcile(run,budget):
    for target,stage,seat in budget.pending():
        if seat in CHANGED:
            folder=Path(run)/'targets'/target/'changed'/f'{stage}_{seat}'
            if not (folder/'record.json').exists():raise AmbiguousDispatch('reservation without durable outcome')
            row=old.old.read(folder/'record.json')
            if row['status']=='DISPATCHED' or 'finished_utc' not in row:raise AmbiguousDispatch('uncertain changed request')
            if row.get('response_sha256') and old.digest(folder/'response.json')!=row['response_sha256']:raise RuntimeError('response drift')
            budget.settle(Budget.key(target,stage,seat),Decimal(row['charge']))
    old.reconcile(run,budget)
