"""Replay the five real Astra responses through today's pipeline without API.

This tests engineering behavior, not model quality. Original refusals are
replayed as refusals; a cache miss is a test failure, never a network request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.errors import ApiCallFailure, ApiTransportError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.config.final_experiment import load_tracker_gate_cell
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.signals.phase_graph import (
    build_phase_ivt_compatibility_from_training_adapter,
    build_phase_transition_graph_from_training_adapter,
)
from surgical_agent.systems.final_pipeline_factory import build_final_api_pipeline
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(r"D:\cholec_dataset")
RUN = ROOT / "artifacts/preflight/smoke_astra_six_issues_after_v9_20260905"
CACHE = ROOT / "artifacts/final_pipeline_cache/astra_six_issues_20260905"


@pytest.mark.local_data
@pytest.mark.integration
@pytest.mark.skipif(
    not DATASET.is_dir() or not (RUN / "manifest.json").is_file(),
    reason="requires the local five-frame Astra smoke artifacts",
)
def test_current_v9_replays_real_frames_with_no_network(tmp_path):
    usage = [
        json.loads(line) for line in (RUN / "api_usage.jsonl").read_text().splitlines()
    ]
    errors = {row["request_hash"]: row["error"] for row in usage if row.get("error")}

    class ReplayOnlyClient:
        def __init__(self):
            self.cache = FileApiCache(CACHE)
            self.misses = []
            self.cache_hits = 0
            self.refusals = 0

        def call(self, request):
            metadata = canonical_request_metadata(request)
            error = errors.get(metadata.request_hash)
            if error is not None:
                self.refusals += 1
                raise ApiCallFailure(
                    ApiTransportError(
                        "",
                        code=error["code"],
                        retryable=False,
                        status_code=error["status_code"],
                    ),
                    attempt_count=1,
                    retry_count=0,
                )
            cached = self.cache.get(metadata)
            if cached is None:
                self.misses.append(metadata.request_hash)
                raise AssertionError("offline replay cache miss")
            self.cache_hits += 1
            return cached

    adapter = CholecTrack20DatasetAdapter(DATASET, causal_window_size=3)
    support = build_phase_ivt_compatibility_from_training_adapter(adapter)
    # The complete local Training routes contain 87 observed phase-IVT pairs,
    # versus 46 in the old frame-only route. Unseen is still not forbidden.
    assert sum(map(len, support.values())) == 87
    client = ReplayOnlyClient()
    pipeline = build_final_api_pipeline(
        client=client,
        api_config=load_api_config(
            ROOT / "configs/perception/joint_openrouter_astra_fixed3.yaml"
        ),
        verification_api_config=load_api_config(
            ROOT / "configs/perception/targeted_openrouter_astra_v9_fixed3.yaml"
        ),
        cell=load_tracker_gate_cell(
            ROOT / "configs/ablations/b_tracker_astra_coverage_smoke.yaml"
        ),
        state_dir=tmp_path,
        track_provider=PrecomputedPredictedTrackProvider.from_json(
            ROOT / "artifacts/training/tracker/predicted_tracks.json"
        ),
        phase_transition_graph=build_phase_transition_graph_from_training_adapter(
            adapter
        ),
        phase_ivt_support=support,
    )
    pipeline.reset("VID110", recover=False)
    media = CausalApiMediaLoader()
    frame_ids = (4301, 4326, 4351, 4376, 4401)
    results = []
    for sample in adapter.iter_inference_video("VID110"):
        if sample.target_frame_id > frame_ids[-1]:
            break
        if sample.target_frame_id not in frame_ids:
            continue
        loaded = media.load(sample)
        results.append(pipeline.run(loaded.runtime_sample, loaded.frames))
    assert not client.misses
    assert len(results) == 5
    # These are regression expectations from real cached predictions, not GT.
    assert [result.outcome.hypothesis.triplet_ids for result in results] == [
        (13,),
        (13,),
        (13,),
        (12, 13),
        (12,),
    ]
    assert all(result.outcome.state == "Accepted" for result in results)
    assert all(result.outcome.lower_reliability for result in results)
    assert [len(result.proposal.attempts) for result in results] == [3, 2, 3, 3, 1]
    assert [
        result.finalization.audit["phase_ivt_check"]["final_unobserved_ivt_ids"]
        for result in results
    ] == [[13], [13], [13], [13], []]
    print(
        f"offline Astra replay: {len(results)} frames, {client.cache_hits} cache hits, "
        f"{client.refusals} original refusals, 0 network calls"
    )
