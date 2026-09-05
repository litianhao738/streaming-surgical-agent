import io
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from PIL import Image

from scripts.run_grounded_contact_smoke import HttpAuditSender, make_request
from scripts.run_pure_h0_smoke import FinalOnlyRequestBuilder
from surgical_agent.api.contracts import ApiImageInput, thaw_json
from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.providers.openrouter import OpenRouterTransport, HttpResponse
from surgical_agent.api.request_hash import canonical_request_hash
from surgical_agent.api.schema import validator_for
from surgical_agent.config.loader import load_api_config
from surgical_agent.research.verification.grounded_repair import (
    LOCATOR_VERSION, PROPOSAL_VERSION, REVIEW_VERSION, make_contact_crops, proposed_labels, admit,
)
from tests.unit.test_joint_api_vlm import _context
from tests.unit.test_p3_openrouter import sample_request_value, _fixture_secret


def fixtures():
    h0 = {"instrument": [2], "verb": [2], "target": [0], "ivt": [60], "phase": [1]}
    locator = {"schema_version": LOCATOR_VERSION, "instances": [{"instance_id": 1, "tip_box": [.2,.2,.4,.4]}], "all_visible_tools_covered": True}
    proposal = {"schema_version": PROPOSAL_VERSION, "instances": [{"instance_id": 1, "instrument_id": 2,
        "ivt_ids": [96], "support": "SUPPORTED", "contact_observation": "Visible tool contact differs from the defined dissection interaction."}]}
    review = {"schema_version": REVIEW_VERSION, "preferred": "FIRST", "all_visible_tools_covered": True,
        "instances": [{"instance_id": 1, "crop_relevant": True, "instrument_identity_supported": True,
            "target_identity_or_oov_supported": True, "action_or_oov_supported": True,
            "distinguishing_observation": "The localized contact does not show the claimed tissue dissection."}]}
    return h0, locator, proposal, review


def test_detail_wire_revision_is_explicit_cache_isolated_and_validated():
    transport = object.__new__(OpenRouterTransport)
    original = sample_request_value()
    explicit = replace(original, payload={**thaw_json(original.payload), "image_details": ["high"], "openrouter_image_detail_mode": "explicit_v1"})
    assert canonical_request_hash(original) != canonical_request_hash(explicit)
    assert "detail" not in json.loads(transport._request_body(original))["messages"][1]["content"][1]["image_url"]
    assert json.loads(transport._request_body(explicit))["messages"][1]["content"][1]["image_url"]["detail"] == "high"
    for details in ([], ["invalid"], ["low", "high"]):
        with pytest.raises(ApiContractError):
            transport._request_body(replace(explicit, payload={**thaw_json(explicit.payload), "image_details": details}))


def test_grounded_proposal_is_closed_without_overwriting_phase_or_h0():
    h0, locator, proposal, review = fixtures()
    original = deepcopy(h0)
    h1 = proposed_labels(h0, locator, proposal)
    assert h1 == {"instrument": [2], "verb": [9], "target": [14], "ivt": [96], "phase": [1]}
    assert h0 == original
    assert admit(h0, h1, locator, proposal, review, proposal_slot="FIRST")["decision"] == "ACCEPT"
    assert admit(h0, h1, locator, proposal, review, proposal_slot="SECOND")["decision"] == "KEEP"
    assert admit(h0, h1, locator, proposal, None, proposal_slot="FIRST")["decision"] == "KEEP"


@pytest.mark.parametrize("failure", ["crop_relevant", "instrument_identity_supported", "target_identity_or_oov_supported", "action_or_oov_supported"])
def test_legality_or_preference_alone_never_suffices(failure):
    h0, locator, proposal, review = fixtures()
    review["instances"][0][failure] = False
    h1 = proposed_labels(h0, locator, proposal)
    assert admit(h0, h1, locator, proposal, review, proposal_slot="FIRST")["decision"] == "KEEP"


def test_unknown_duplicate_and_mismatched_instances_fail_closed():
    h0, locator, proposal, review = fixtures()
    proposal["instances"][0]["instance_id"] = 2
    assert proposed_labels(h0, locator, proposal) is None
    proposal["instances"][0]["ivt_ids"] = [94]
    with pytest.raises(ApiSchemaError):
        validator_for(PROPOSAL_VERSION)(proposal)
    locator["instances"].append(deepcopy(locator["instances"][0]))
    with pytest.raises(ApiSchemaError):
        validator_for(LOCATOR_VERSION)(locator)


def test_crops_preserve_source_pixels_and_bind_target_identity():
    _, locator, _, _ = fixtures()
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(buffer, format="PNG")
    source = ApiImageInput("target:100", "image/png", buffer.getvalue())
    crops, manifest = make_contact_crops(source, locator, target_frame_id=100)
    assert manifest[0]["source_sha256"] == source.sha256
    assert manifest[0]["target_frame_id"] == 100
    with Image.open(io.BytesIO(crops[0].content)) as im:
        assert im.getpixel((0, 0)) == (255, 0, 0)
    locator["instances"][0]["tip_box"] = [.4,.2,.2,.4]
    with pytest.raises(ApiSchemaError):
        make_contact_crops(source, locator, target_frame_id=100)


def test_locator_request_is_blind_and_crop_order_is_explicit():
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    base = FinalOnlyRequestBuilder(config=config).build(_context())
    request = make_request(base, LOCATOR_VERSION, "Locate visible tips.", {})
    data = json.loads(request.payload["input_text"])
    assert "hypotheses" not in data and "gt" not in data
    assert request.images == base.images
    assert request.payload["openrouter_image_detail_mode"] == "explicit_v1"


def test_http_audit_captures_invalid_content_before_parser_and_redacts_key(tmp_path):
    secret = _fixture_secret()
    def sender(*args):
        return HttpResponse(200, {"authorization": secret.reveal()}, b"not JSON " + secret.reveal().encode())
    wire = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}]}]}
    audit = HttpAuditSender(tmp_path, secret, sender)
    response = audit("url", {"Authorization": secret.reveal()}, json.dumps(wire).encode(), 1)
    assert response.body.startswith(b"not JSON")
    saved = (tmp_path / "http_response.json").read_text()
    assert "not JSON" in saved and secret.reveal() not in saved
    assert "authorization" not in saved
    assert "base64,test" not in (tmp_path / "wire_request.json").read_text()
