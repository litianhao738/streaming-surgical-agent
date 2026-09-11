import json

from scripts.run_transactional_trial_compat import add_json_declaration


def test_json_mode_declaration_only():
    body={'messages':[{'content':[{'text':json.dumps({'academic_context':'research','task':'assess edits'})},
                                 {'image_url':{'url':'unchanged'}}]}],'max_tokens':8192,
          'response_format':{'type':'json_object'}}
    changed=add_json_declaration(body)
    packet=json.loads(changed['messages'][0]['content'][0]['text'])
    assert 'JSON' in packet.pop('output_format')
    changed['messages'][0]['content'][0]['text']=json.dumps(packet,ensure_ascii=False)
    assert changed==body
