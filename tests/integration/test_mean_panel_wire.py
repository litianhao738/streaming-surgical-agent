from types import SimpleNamespace

from scripts import run_mean_panel_trial as runner


def test_qwen_vl_omits_unsupported_reasoning_but_repair_keeps_low():
    base = SimpleNamespace(images=[], payload={"image_details": []})
    packet = {"academic_context": runner.ACADEMIC_CONTEXT}
    schema = runner.response_schema()
    for model, supports in (("qwen/qwen3-vl-8b-instruct", False), ("qwen/qwen3.8-max-0902", True)):
        endpoint = {"model": model, "tag": "alibaba", "max_output": 4096,
                    "endpoint": {"supported_parameters": ["reasoning"] if supports else []}}
        body = runner.request_body(base, endpoint, schema, runner.ACADEMIC_CONTEXT, packet)
        assert ("reasoning" in body) == supports
        if supports:
            assert body["reasoning"] == {"effort": "low"}
        assert body["provider"]["only"] == ["alibaba"]
        assert not body["provider"]["allow_fallbacks"]
        assert body["messages"][1]["content"][0]["text"].startswith('{"academic_context":')
