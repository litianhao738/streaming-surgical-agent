"""Explicit new-route adapter. Does not mutate frozen historical runtimes.

Affected routes have independent call ledgers; their unknown prices must not
be charged using historical OpenRouter/Alibaba rate cards. A hard request cap
is mandatory. New paid dataset plans require their own verified price bounds.
"""
import hashlib
import json
from pathlib import Path
from copy import deepcopy
from threading import RLock
import datetime
import requests

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/reviewer_routes_v2.json'


def read_config():return json.loads(CONFIG.read_text(encoding='utf-8'))


def route_body(seat,body,config):
    out=deepcopy(body)
    if seat not in config['routes']:return out
    out['model']=config['routes'][seat]['model']
    out.pop('provider',None)  # OpenRouter provider routing has no meaning here.
    if seat=='qwen':out['enable_thinking']=False
    return out


def credentials(seat,config):
    from scripts import collect_full_gate_training as original
    spec=config['routes'][seat]
    if seat=='qwen':
        t=original.old.joint.roster.transport
        return t.MODELS['qwen'][0],original.old.joint.roster._KEY_FOR('qwen').reveal()
    secrets=dict(line.split('=',1) for line in (ROOT/config['secret_file']).read_text(encoding='utf-8').splitlines() if '=' in line)
    return spec['base_url'],secrets[spec['secret_name']]


class RoutedCalls:
    def __init__(self,unchanged_calls,output,*,max_changed_calls):
        if type(max_changed_calls) is not int or max_changed_calls<1:raise ValueError('explicit positive request cap required')
        self.config=read_config();self.delegate=unchanged_calls;self.output=Path(output)
        self.cap=max_changed_calls;self.lock=RLock();self.count=0;self.records=[]

    def call(self,target,stage,seat,body):
        if seat not in self.config['routes']:return self.delegate.call(target,stage,seat,body)
        from scripts import collect_full_gate_training as original
        wire=route_body(seat,body,self.config);base_url,key=credentials(seat,self.config)
        folder=self.output/target/(stage+'_'+seat)
        with self.lock:
            if self.count>=self.cap:raise RuntimeError('changed-route request cap reached')
            folder.mkdir(parents=True,exist_ok=False)  # Never silently repeat a dispatched identity.
            self.count+=1
            safe=original.old.joint.roster.transport.redact_images(wire)
            original.write(folder/'request.json',safe)
            record={'target':target,'stage':stage,'seat':seat,'requested_model':wire['model'],
                'base_url':base_url,'status':'DISPATCHED','price_verified':False,
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            original.write(folder/'record.json',record)
        parsed=None
        try:
            response=requests.post(base_url.rstrip('/')+'/chat/completions',json=wire,
                headers={'Authorization':'Bearer '+key},timeout=(15,120),allow_redirects=False)
            try:raw=response.json()
            except ValueError:raw={'non_json_response':response.text}
            raw=json.loads(json.dumps(raw).replace(key,'[REDACTED]'))
            original.write(folder/'response.json',raw)
            record.update(http_status=response.status_code,returned_model=raw.get('model'),usage=raw.get('usage',{}))
            if not response.ok or raw.get('error'):
                record.update(status='API_FAILED',error=raw.get('error'))
            else:
                choice=raw['choices'][0]
                if raw.get('model')!=wire['model']:raise ValueError('returned model identity mismatch')
                if choice.get('finish_reason')!='stop' or choice['message'].get('refusal'):raise ValueError('incomplete/refused response')
                # Reject duplicate keys, including conflicting semantic fields.
                def pairs(items):
                    d={}
                    for k,v in items:
                        if k in d:raise ValueError('duplicate JSON field')
                        d[k]=v
                    return d
                content=choice['message']['content'].strip()
                if content.startswith('```') and content.endswith('```'):
                    content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
                parsed=json.loads(content,object_pairs_hook=pairs)
                record['status']='JSON_PARSED'
        except (requests.RequestException,ValueError,KeyError,IndexError,TypeError) as exc:
            record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc).replace(key,'[REDACTED]')[:300])
        finally:
            record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            original.write(folder/'record.json',record)
            with self.lock:self.records.append(record)
        return parsed
