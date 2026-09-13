from copy import deepcopy
import pytest
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts.trial_tracker_schemes56 import compact_wire, proposal_wire, failure_requires_stop
from surgical_agent.research.verification.candidate_coordinator import make_pool


@pytest.mark.parametrize('http,code,seat,stop', [
    (400, '1301', 'grok', False), (400, 1301, 'grok', False),
    (400, '1301', 'deepseek', True), (401, '1301', 'grok', True),
    (400, 'data_inspection_failed', 'qwen', False),
    (400, 'data_inspection_failed', 'gemini', True),
    (401, 'invalid_key', 'qwen', True), (402, 'balance', 'gpt', True),
    (200, None, 'qwen', False), (None, None, 'qwen', True),
])
def test_stop_contract_has_narrow_refusal_exception(http, code, seat, stop):
    row = {'status': 'API_FAILED' if code else 'JSON_PARSED', 'finished_utc': 'now',
           'http_status': http, 'provider_error': {'code': code}}
    assert failure_requires_stop(row, seat) == stop


def test_read_timeout_stops_even_with_partial_http_metadata():
    assert failure_requires_stop({'status': 'FAILED', 'finished_utc': 'now', 'http_status': 200, 'error_type': 'ReadTimeout'}, 'qwen')


def test_six_changes_only_qwen_prompt_fields(inputs):
    base, selected, prior, _ = inputs
    h0 = {'instrument': [0, 2], 'verb': [], 'target': [], 'ivt': [], 'phase': [3]}
    pool = make_pool(h0)
    packet = {'version': 'test', 'frames': []}
    with ready.old.joint.roster.lightweight_protocol():
        for seat in ('qwen', 'gpt', 'gemini', 'grok', 'deepseek'):
            control = compact_wire(base, selected, h0, pool, packet, 'control', seat)
            treatment = compact_wire(base, selected, h0, pool, packet, 'qwen', seat)
            if seat != 'qwen':
                assert treatment == control
            else:
                import json
                a = deepcopy(control); b = deepcopy(treatment)
                pa = json.loads(a['messages'][0]['content'][0]['text'])
                pb = json.loads(b['messages'][0]['content'][0]['text'])
                assert set(pb) - set(pa) == {'tracker_evidence', 'tracker_evidence_instructions'}
                del pb['tracker_evidence']; del pb['tracker_evidence_instructions']
                a['messages'][0]['content'][0]['text'] = pa
                b['messages'][0]['content'][0]['text'] = pb
                assert a == b


def test_five_does_not_change_wire_when_tracker_adds_no_class(inputs):
    base, selected, prior, _ = inputs
    h0 = {'instrument': [0, 2], 'verb': [], 'target': [], 'ivt': [], 'phase': [3]}
    pool = make_pool(h0)
    with ready.old.joint.roster.lightweight_protocol():
        a = proposal_wire(base, selected, h0, pool, prior, {0}, 'control')
        b = proposal_wire(base, selected, h0, pool, prior, {0}, 'prior')
    assert a == b
