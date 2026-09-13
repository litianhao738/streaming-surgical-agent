from copy import deepcopy
import pytest
from tests.unit.test_gate_ready_mainline import inputs
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.retrieval.tracker_prior_candidates import retrieve_tracker_hints
from surgical_agent.research.verification.prior_panel import COMPONENTS, fit_prior, video_counts


def test_equal_eligibility_keeps_existing_packet(inputs):
    _, _, prior, _ = inputs
    h0 = {'instrument': [0, 2], 'verb': [], 'target': [], 'ivt': [], 'phase': [3]}
    assert retrieve_tracker_hints(h0, prior, video_id='query', tracker_classes={0}) == retrieve_candidate_hints(h0, prior, video_id='query')


def test_tracker_can_retrieve_for_missing_instrument_without_editing_h0(inputs):
    _, _, prior, _ = inputs
    h0 = {'instrument': [], 'verb': [], 'target': [], 'ivt': [], 'phase': [3]}
    row = {'frame_id': 100, 'mask': dict.fromkeys(h0, True),
           'gt': {'instrument': [0, 2], 'verb': [1, 2], 'target': [0, 1], 'ivt': [17, 59], 'phase': [3]}}
    counts = video_counts([{**row, 'frame_id': i} for i in range(40)])
    prior = fit_prior({v: counts for v in ('other_a', 'other_b', 'other_c')}, 'query')
    original = deepcopy(h0)
    assert retrieve_candidate_hints(h0, prior, video_id='query')['packet'] is None
    new = retrieve_tracker_hints(h0, prior, video_id='query', tracker_classes={0, 2})
    assert new['packet'] and 0 < len(new['packet']['relations']) <= 2
    assert all(COMPONENTS[r['ivt_id']]['instrument'] in {0, 2} for r in new['packet']['relations'])
    assert h0 == original


def test_tracker_retrieval_preserves_leakage_checks(inputs):
    _, _, prior, _ = inputs
    h0 = {'instrument': [], 'verb': [], 'target': [], 'ivt': [], 'phase': [3]}
    with pytest.raises(ValueError):
        retrieve_tracker_hints(h0, prior, video_id='other_a', tracker_classes={0})
