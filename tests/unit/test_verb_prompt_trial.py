import json
from copy import deepcopy

import pytest

from scripts import run_verb_prompt_trial as trial
from scripts.run_verb_prompt_trial import append_audit


def test_only_one_field_changes_and_input_is_immutable():
    packet = {"academic_context": "Academic surgical video research", "propositions": [{"task":"verb", "id":0}]}
    body = {"model":"test", "messages":[{"role":"user","content":[
        {"type":"text","text":json.dumps(packet,ensure_ascii=False)},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AA==","detail":"high"}}]}],
        "response_format":{"unchanged":True},"max_tokens":4096}
    frozen = deepcopy(body)
    result = append_audit(body)
    assert body == frozen
    parsed = json.loads(result['messages'][0]['content'][0]['text'])
    assert next(iter(parsed)) == 'academic_context'
    extra = parsed.pop('verb_visual_audit')
    assert 'only' in extra['scope'] and 'verb' in extra['scope']
    assert parsed == packet
    result['messages'][0]['content'][0]['text'] = json.dumps(parsed,ensure_ascii=False)
    assert result == body


def test_duplicate_append_is_rejected():
    body = {"messages":[{"content":[{"text":json.dumps({"verb_visual_audit":{}})}]}]}
    with pytest.raises(ValueError,match='duplicate'):
        append_audit(body)


def test_control_preserves_wire_and_unknown_arm_fails(monkeypatch):
    body = {"original": True}
    monkeypatch.setattr(trial, 'review_wire', lambda *args: deepcopy(body))
    assert trial.wire('gpt', None, {}, {}, 'control') == body
    with pytest.raises(ValueError, match='unknown'):
        trial.wire('gpt', None, {}, {}, 'other')


def test_empty_pool_retains_all_heads_without_call():
    h0 = {t:[] for t in trial.TASKS}
    h0['phase'] = [1]
    result = trial.run_arm(None, None, {}, {'pool':{'propositions':[]}, 'h0':h0}, 'control')
    assert result['prediction'] == h0
    assert result['prediction'] is not h0
    assert result['status'] == 'EMPTY_POOL_UNVERIFIED'
