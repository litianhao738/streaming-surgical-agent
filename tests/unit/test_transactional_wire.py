import json
from copy import deepcopy
from types import SimpleNamespace

from scripts import run_transactional_trial as trial
from surgical_agent.research.verification.candidate_coordinator import make_pool


def test_new_wire_keeps_images_models_and_limits_and_replaces_entire_schema(monkeypatch):
    original={'model':'example','temperature':0,'max_tokens':4096,'reasoning':{'effort':'none'},
        'messages':[{'role':'user','content':[{'type':'text','text':'old'},
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,AA==','detail':'high'}}]}],
        'response_format':{'type':'json_schema','json_schema':{'old':True}}}
    monkeypatch.setattr(trial,'review_wire',lambda *a:deepcopy(original))
    h0={'instrument':[0],'verb':[1],'target':[0],'ivt':[17],'phase':[1]}
    new=trial.edit_wire('gpt',SimpleNamespace(images=[1,2,3]),{},h0,make_pool(h0))
    packet=json.loads(new['messages'][0]['content'][0]['text'])
    assert next(iter(packet))=='academic_context'
    assert all(p['operation']=='REMOVE' for p in packet['candidate_changes'])
    assert new['messages'][0]['content'][1:]==original['messages'][0]['content'][1:]
    assert new['response_format']['json_schema']['schema']==packet['response_schema']
    assert all(new[k]==original[k] for k in ('model','temperature','max_tokens','reasoning'))
    assert 'rating' not in str(packet['response_schema'])
    assert 'SUPPORTED' in packet['decisions']
