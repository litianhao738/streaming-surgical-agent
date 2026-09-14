from copy import deepcopy
from types import MappingProxyType
import pytest

from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts.run_pgp_pipeline import WireBackend
from surgical_agent.research.gate.pgp_runtime import run_target as original
from surgical_agent.research.gate.tracker_pipeline_v2 import (
    run_interaction, output_modules, CausalPhaseFilter, current_classes, CAPS, VERB_RULES,
)
from surgical_agent.research.verification.prior_panel import COMPONENTS


@pytest.mark.parametrize('action', [0, 1])
@pytest.mark.parametrize('replacement', [{}, {('proposal', 'base'): None}, {('control_graph', 'qwen'): None}, {('control_graph', 'gpt'): None}])
def test_interaction_runtime_keeps_prefix_and_predictions(inputs, action, replacement):
    base, selected, prior, _ = inputs
    selected = {**selected, 'source_split': 'Training'}
    a, b = RecordingMock(replacement), RecordingMock(replacement)
    with ready.old.joint.roster.lightweight_protocol():
        old = original(WireBackend(a, base, selected), selected, prior, lambda _: (.5, 3 * action))
        new = run_interaction(WireBackend(b, base, selected), selected, prior, lambda _: (.5, action))
    assert new['features'] == old['features']
    assert new['cheap'] == old['cheap']
    assert new['prediction'] == {**old['prediction'], 'phase': old['cheap']['phase']}
    assert new['call_keys'] == [k for k in old['call_keys'] if not k.startswith(('joint_r1|', 'phase_recommendation|'))]
    for key, wire in b.wires.items():
        assert wire == a.wires[key]


def test_phase_filter_raw_votes_boundaries_ties_and_isolation():
    filt = CausalPhaseFilter(10, fps=1)
    # A and B tie above current C: documented rule must keep C.
    for f, p in enumerate([0, 0, 1, 1]):
        filt.apply('v', f, p)
    assert filt.apply('v', 4, 2) == 2
    assert filt.apply('other', 0, 6) == 6
    assert filt.apply('v', 20, 3) == 3  # expired observations are not carried across gaps
    with pytest.raises(ValueError):
        filt.apply('v', 19, 3)
    # Raw B observations must eventually overturn A despite earlier filtered A outputs.
    filt = CausalPhaseFilter(3, fps=1)
    assert [filt.apply('v', f, p) for f, p in enumerate([0, 0, 1, 1])] == [0, 0, 0, 1]


def test_missing_is_different_from_valid_empty_and_m1_capacity():
    c = next(iter(COMPONENTS))
    pred = {'instrument': [COMPONENTS[c]['instrument']], 'verb': [COMPONENTS[c]['verb']],
            'target': [COMPONENTS[c]['target']], 'ivt': [c], 'phase': [0]}
    assert output_modules(pred, None)[0] == pred
    empty, log = output_modules(pred, set())
    assert empty['instrument'] == empty['ivt'] == [] and log['deleted_ivts'] == [c]
    assert output_modules(pred, set(range(CAPS['instrument'] + 1)))[0] == pred


def test_m2_capacity_falls_back_without_undoing_m1():
    inst, verb = VERB_RULES[0]
    oldverbs = [v for v in range(10) if v != verb][:CAPS['verb']]
    pred = {'instrument': [], 'verb': oldverbs, 'target': [], 'ivt': [], 'phase': [0]}
    out, log = output_modules(pred, {inst})
    assert out['instrument'] == [inst] and out['verb'] == oldverbs
    assert log['m2_capacity_fallback']


def test_snapshot_rejects_misalignment_and_bad_scores():
    selected = {'video_id': 'v', 'frame_id': 25}
    packet = {'status': 'AVAILABLE', 'video_id': 'v', 'source_max_frame_id': 25,
              'frames': [{'frame_id': 25, 'tracks': []}]}
    assert current_classes(packet, selected) == set()
    frozen = MappingProxyType({**packet, 'frames': (MappingProxyType({'frame_id': 25, 'tracks': ()}),)})
    assert current_classes(frozen, selected) == set()
    wrong = deepcopy(packet); wrong['source_max_frame_id'] = 50
    assert current_classes(wrong, selected) is None
    packet['frames'][0]['tracks'] = [{'instrument_id': 0, 'score': float('nan')}]
    assert current_classes(packet, selected) is None


def test_ontology_filter_drops_instruments_outside_every_triplet():
    from surgical_agent.research.gate.tracker_pipeline_v2 import TRIPLET_INSTRUMENTS
    bag = next(i for i in range(7) if i not in TRIPLET_INSTRUMENTS)
    pred = {'instrument': [0, bag], 'verb': [], 'target': [], 'ivt': [], 'phase': [0]}
    assert output_modules(pred, None)[0]['instrument'] == [0, bag]  # v2.1 behaviour is unchanged
    fallback, log = output_modules(pred, None, ontology_filter=True)
    assert fallback['instrument'] == [0] and log['excluded_instruments'] == [bag]
    fused, log = output_modules(pred, {2, bag}, ontology_filter=True)
    assert fused['instrument'] == [2] and log['excluded_tracker_classes'] == [bag]
    assert output_modules(pred, {2, bag})[0]['instrument'] == [2, bag]


def test_null_label_cleanup_uses_only_valid_probe_rating_one_and_keeps_consistency():
    from surgical_agent.research.gate.tracker_pipeline_v2 import (
        NULL_IVTS,
        NULL_TARGET,
        NULL_VERB,
        null_label_cleanup,
    )
    null = min(NULL_IVTS)
    act = next(c for c, comp in COMPONENTS.items() if comp['instrument'] == COMPONENTS[null]['instrument'] and c not in NULL_IVTS)
    pred = {'instrument': [COMPONENTS[null]['instrument']], 'verb': sorted({COMPONENTS[act]['verb'], NULL_VERB}),
            'target': sorted({COMPONENTS[act]['target'], NULL_TARGET}), 'ivt': sorted({act, null}), 'phase': [0]}
    out, log = null_label_cleanup(pred, {f'ivt:{null}': 1, f'ivt:{act}': 1})
    assert out['ivt'] == [act] and NULL_VERB not in out['verb'] and NULL_TARGET not in out['target']
    assert log['removed_null_ivts'] == [null] and log['removed_null_verb'] and log['removed_null_target']
    for ratings in ({f'ivt:{null}': 2}, {f'ivt:{null}': None}, {}):
        assert null_label_cleanup(pred, ratings)[0] == pred
    orphan = {**pred, 'ivt': [act]}
    assert null_label_cleanup(orphan, {})[0]['verb'] == [COMPONENTS[act]['verb']]
