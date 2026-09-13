"""Scheme 5: expand retrieval eligibility, never modify H0 or candidate capacity."""
from copy import deepcopy
from surgical_agent.research.retrieval import prior_candidates as original
from surgical_agent.research.verification.prior_panel import COMPONENTS, digest, labels
from surgical_agent.perception.ontology_prompt import _TASK_NAMES


def retrieve_tracker_hints(h0, prior, *, video_id, tracker_classes):
    h0 = labels(h0)
    allowed = set(h0['instrument']) | set(tracker_classes or ())
    if any(type(i) is not int or not 0 <= i < 7 for i in allowed):
        raise ValueError('invalid Tracker class')
    if allowed == set(h0['instrument']):
        return original.retrieve_candidate_hints(h0, prior, video_id=video_id)
    check = deepcopy(prior)
    claimed = check.pop('table_sha256')
    if digest(check) != claimed:
        raise ValueError('prior checksum differs')
    if prior['excluded_video'] != video_id or video_id in prior['fit_videos']:
        raise ValueError('query video must be excluded from prior')
    table, phase = prior['tasks']['ivt'], str(h0['phase'][0])
    selected, details = [], []
    for slot in range(original.MAX_HINTS):
        order = [('phase', table['phase'][phase]), ('global', table['global'])] if slot == 0 else [('global', table['global'])]
        choices = []
        for source, rows in order:
            for row in rows:
                ivt = row['id']; comp = COMPONENTS[ivt]
                if not row['eligible'] or ivt >= 94 or ivt in h0['ivt'] or ivt in selected or comp['instrument'] not in allowed:
                    continue
                neighbors = [c for c in h0['ivt'] if c < 94 and COMPONENTS[c]['instrument'] == comp['instrument']
                             and sum(COMPONENTS[c][t] != comp[t] for t in ('verb', 'target')) == 1]
                rank = (not bool(neighbors), -row['rate'], -row['positive_videos'], ivt)
                choices.append((rank, ivt, source, row, neighbors))
            if choices:
                break
        if not choices:
            continue
        _, ivt, source, stats, neighbors = min(choices, key=lambda x: x[0])
        selected.append(ivt)
        details.append({'ivt': ivt, 'ranking_source': source, 'statistics': deepcopy(stats), 'neighbor_ivts': neighbors})
    packet = None
    if selected:
        # Identical prose to the existing retriever: only eligible relations change.
        packet = {
            'use': 'Possible omitted relations from OTHER Training videos, not visual evidence or confidence for this frame. '
                   'The phase hint may be wrong. Inspect the supplied images and propose only supported relations. '
                   'Rare or unlisted relations remain allowed. Suggestions consume the existing four-IVT allowance; do not copy them automatically.',
            'source': 'Leave-query-video-out GT statistics linked to the existing IVT ontology.',
            'relations': [{'ivt_id': c, **{t: _TASK_NAMES[t][v] for t, v in COMPONENTS[c].items()}} for c in selected],
        }
    return {'packet': packet, 'audit': {'version': 'tracker_prior_eligibility_v1', 'excluded_video': video_id,
            'fit_videos': prior['fit_videos'], 'prior_sha256': claimed, 'selected': details,
            'allowed_instruments': sorted(allowed), 'phase_is_h0_prediction': True,
            'prior_is_not_visual_evidence': True, 'extra_api_calls': 0}}
