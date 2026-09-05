"""Missing task GT is not an empty negative or an API success."""

import json
from types import SimpleNamespace

import pytest

from scripts.run_pure_h0_smoke import (
    RawResponseAuditTransport,
    FinalOnlyRequestBuilder,
    SchemaExplicitRequestBuilder,
    evaluate_offline,
)
from surgical_agent.api.openrouter_routing import provider_preferences
from surgical_agent.api.request_hash import canonical_request_hash
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from tests.unit.test_joint_api_vlm import _config, _context


def test_final_only_ablation_preserves_visual_input_and_removes_only_topk_schema():
    from surgical_agent.api.schema import schema_for
    from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    original = SchemaExplicitRequestBuilder(config=config).build(_context())
    final = FinalOnlyRequestBuilder(config=config).build(_context())
    assert final.images == original.images
    assert final.payload["input_text"] == original.payload["input_text"]
    assert final.generation_parameters == original.generation_parameters
    assert final.payload["image_details"] == original.payload["image_details"]
    assert final.payload["openrouter_routing_profile"] == original.payload["openrouter_routing_profile"]
    assert canonical_request_hash(final) != canonical_request_hash(original)
    schema = schema_for(FINAL_ONLY_SCHEMA_VERSION)
    for task in ("instrument", "verb", "target", "ivt", "phase"):
        assert "topk" not in schema["properties"][task]["properties"]
    assert "Operational CholecTrack20 ontology" in final.payload["system_text"]
    assert "IVT 94-99" in final.payload["system_text"]


def test_final_only_validation_does_not_fabricate_candidates_or_correct_labels():
    from copy import deepcopy
    from surgical_agent.api.errors import ApiSchemaError
    from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION, validate_final_only
    payload = {"schema_version": FINAL_ONLY_SCHEMA_VERSION,
               **{task: {"selected_ids": []} for task in ("instrument", "verb", "target", "ivt")},
               "phase": {"selected_id": 0}}
    before = deepcopy(payload)
    validate_final_only(payload)
    assert payload == before
    for invalid in ([True], [0, 0], [100], "0", [[0]]):
        bad = deepcopy(payload)
        bad["instrument"]["selected_ids"] = invalid
        with pytest.raises(ApiSchemaError):
            validate_final_only(bad)
    bad = deepcopy(payload)
    bad["instrument"]["topk"] = []
    with pytest.raises(ApiSchemaError):
        validate_final_only(bad)


def test_schema_reminder_changes_hash_but_not_images_or_input_labels():
    config = load_api_config(
        "configs/perception/joint_openrouter_gemini38flash_pure_h0_fixed3.yaml"
    )
    context = _context()
    original = JointPerceptionRequestBuilder(config=config).build(context)
    clarified = SchemaExplicitRequestBuilder(config=config).build(context)
    assert clarified.images == original.images
    assert clarified.payload["input_text"] == original.payload["input_text"]
    assert clarified.generation_parameters == original.generation_parameters
    assert canonical_request_hash(clarified) != canonical_request_hash(original)
    assert f'"{original.response_schema_version}"' in clarified.payload["system_text"]


def test_raw_audit_preserves_even_invalid_json_version_before_validation(tmp_path):
    response = SimpleNamespace(
        returned_model_identifier="fixture", parsed_payload={"schema_version": "wrong"}
    )
    transport = SimpleNamespace(
        provider="mock",
        endpoint_identifier="mock://local/p3",
        send=lambda request: response,
    )
    request = JointPerceptionRequestBuilder(config=_config()).build(_context())
    assert RawResponseAuditTransport(transport, tmp_path).send(request) is response
    persisted = json.loads(
        next((tmp_path / "raw_responses").glob("*.json")).read_text()
    )
    assert persisted["parsed_payload"] == {"schema_version": "wrong"}


@pytest.mark.parametrize(
    "filename,model,profile,provider",
    [
        ("grok46", "x-ai/grok-4.6", "strict_xai", "xai"),
        ("qwen38max0902", "qwen/qwen3.8-max-0902", "strict_alibaba", "alibaba"),
        (
            "gemini38flash",
            "google/gemini-3.8-flash",
            "strict_google_ai_studio",
            "google-ai-studio",
        ),
    ],
)
def test_pure_h0_multimodel_configs_build_same_visual_contract(
    filename, model, profile, provider
):
    config = load_api_config(
        f"configs/perception/joint_openrouter_{filename}_pure_h0_fixed3.yaml"
    )
    builder = JointPerceptionRequestBuilder(config=config)
    request = builder.build(_context())
    assert request.model_identifier == model
    assert request.payload["openrouter_routing_profile"] == profile
    assert provider_preferences(profile) == {
        "only": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert "gpt56sol" not in builder.backend_name
    assert len(request.images) == 3


def test_pure_h0_scoring_distinguishes_missing_gt_empty_gt_and_api_failure():
    targets = [
        FrameSupervisionTarget(
            "VID01",
            1,
            (0,),
            (),
            (),
            (),
            0,
            FrameTaskMask(True, False, False, False, True),
            "frame_multilabel",
            "test",
        ),
        FrameSupervisionTarget(
            "VID01",
            26,
            (),
            (),
            (),
            (),
            1,
            FrameTaskMask(True, True, True, True, True),
            "frame_multilabel",
            "test",
        ),
        FrameSupervisionTarget(
            "VID01",
            51,
            (0,),
            (9,),
            (14,),
            (94,),
            1,
            FrameTaskMask(True, True, True, True, True),
            "frame_multilabel",
            "test",
        ),
    ]
    adapter = SimpleNamespace(
        iter_video=lambda video, frame_ids: iter(
            SimpleNamespace(
                inference=SimpleNamespace(target_frame_id=target.frame_id),
                frame_supervision=target,
                evaluation=None,
            )
            for target in targets
        )
    )
    predictions = [
        {
            "frame_id": 1,
            "status": "OK",
            "selected_ids": {
                "instrument": [0],
                "verb": [8],
                "target": [0],
                "ivt": [13],
                "phase": [0],
            },
        },
        {
            "frame_id": 26,
            "status": "OK",
            "selected_ids": {
                "instrument": [],
                "verb": [],
                "target": [],
                "ivt": [],
                "phase": [1],
            },
        },
        {"frame_id": 51, "status": "API_FAILURE"},
    ]
    report = evaluate_offline(adapter, "VID01", predictions)
    assert report["tasks"]["instrument"]["correct"] == 2
    assert report["tasks"]["instrument"]["valid_gt"] == 3
    assert report["tasks"]["instrument"]["scored"] == 2
    assert report["tasks"]["instrument"]["failed_with_gt"] == 1
    assert report["tasks"]["verb"]["correct"] == 1
    assert report["tasks"]["verb"]["valid_gt"] == 2
    assert report["tasks"]["verb"]["scored"] == 1
    assert report["frames"][0]["exact"]["verb"] is None
    assert report["frames"][1]["exact"]["verb"] is True
    assert report["frames"][2]["exact"]["verb"] is None
