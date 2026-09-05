import pytest

from scripts.run_five_expert_ablation import build_variant, closure, BOUNDARY
from scripts.run_pure_h0_smoke import SchemaExplicitRequestBuilder
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.config.loader import load_api_config
from surgical_agent.perception.expert_ablation import TASK_COUNTS, expert_version
from tests.unit.test_joint_api_vlm import _context


def test_variants_preserve_visual_input_and_isolate_experts():
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    base = SchemaExplicitRequestBuilder(config=config).build(_context())
    assert build_variant(base, "A") == base
    b = build_variant(base, "B")
    assert b.payload["system_text"] == base.payload["system_text"] + BOUNDARY
    for task in TASK_COUNTS:
        c = build_variant(base, "C", task)
        assert c.images == base.images
        assert c.generation_parameters == base.generation_parameters
        assert c.payload["input_text"] == base.payload["input_text"]
        assert set(schema_for(c.response_schema_version)["properties"]) == {"schema_version", task}
        assert BOUNDARY in c.payload["system_text"]
        assert "no predictions from other experts" in c.payload["system_text"]


@pytest.mark.parametrize("task", list(TASK_COUNTS))
def test_expert_strict_validation(task):
    version = expert_version(task)
    field = {"selected_id": 0} if task == "phase" else {"selected_ids": [0]}
    field["topk"] = [{"id": i, "score": 1 - i / 10} for i in range(TASK_COUNTS[task])]
    payload = {"schema_version": version, task: field}
    validator_for(version)(payload)
    with pytest.raises(ApiSchemaError):
        validator_for(version)({**payload, "other": {}})
    field["topk"][0]["id"] = 999
    with pytest.raises(ApiSchemaError):
        validator_for(version)(payload)


def test_closure_diagnoses_without_repairing():
    labels = {"instrument": [0], "verb": [9], "target": [14], "ivt": [94]}
    assert closure(labels) == {"ivt_components_in_selected": True, "exact_projection": True}
    labels["verb"] = [0]
    assert not closure(labels)["ivt_components_in_selected"]
    assert labels["verb"] == [0]


def test_actual_openrouter_wire_keeps_joint_and_expert_schemas_separate():
    import json
    from surgical_agent.api.providers.openrouter import OpenRouterTransport
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    base = SchemaExplicitRequestBuilder(config=config).build(_context())
    # Body construction is credential-free and makes no network call.
    transport = object.__new__(OpenRouterTransport)
    for arm, task in (("C", "instrument"), ("A", None), ("C", "ivt"), ("B", None)):
        request = build_variant(base, arm, task)
        wire = json.loads(transport._request_body(request))
        expected = {"schema_version", task} if arm == "C" else {"schema_version", *TASK_COUNTS}
        assert set(wire["response_format"]["json_schema"]["schema"]["required"]) == expected
        assert wire["model"] == "qwen/qwen3.8-max-0902"
        assert len(wire["messages"][1]["content"]) == 4
