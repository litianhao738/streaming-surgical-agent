"""Leaf review modules must remain usable in a fresh Python process."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["grounded_repair", "diff_review"])
def test_review_cold_import_preserves_public_api(module):
    code = (
        f"import surgical_agent.research.verification.{module}; "
        "from surgical_agent.api import CachedMultimodalApiClient, "
        "build_transport, build_validator, determine_p3_status; "
        "from surgical_agent.api.schema import schema_for, validator_for; "
        "version = 'frame_label_diff_review_v1'; "
        "assert schema_for(version)['properties']['schema_version']['const'] == version; "
        "validator_for(version)({'schema_version': version, 'assessments': []})"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True,
        text=True, timeout=60,
    )
