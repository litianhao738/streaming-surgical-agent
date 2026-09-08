"""Training-prior graph hints for the EXISTING visual proposer, never labels.

The caller supplies a leave-video-out prior built offline with video_counts /
fit_prior. Inference has no dataset access. Full IVT nodes retain their exact
components; phase and global prevalence rank at most two visual hypotheses.
"""
from copy import deepcopy

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.prior_panel import COMPONENTS, digest, labels

VERSION = "lovo_prior_candidate_hints_v1"
MAX_HINTS = 2
MAX_CHARS = 1800


def retrieve_candidate_hints(h0, prior, *, video_id):
    """One phase-informed and one global slot; global fallback, no GT query."""
    h0 = labels(h0)
    check = deepcopy(prior)
    claimed_hash = check.pop("table_sha256")
    if digest(check) != claimed_hash:
        raise ValueError("prior checksum differs")
    if prior["excluded_video"] != video_id or video_id in prior["fit_videos"]:
        raise ValueError("query video must be excluded from the full prior")
    table = prior["tasks"]["ivt"]
    phase = str(h0["phase"][0])
    selected, details = [], []
    for slot in range(MAX_HINTS):
        order = [("phase", table["phase"][phase]), ("global", table["global"])] if slot == 0 else [("global", table["global"])]
        choices = []
        for source, rows in order:
            for row in rows:
                ivt = row["id"]
                comp = COMPONENTS[ivt]
                if (not row["eligible"] or ivt >= 94 or ivt in h0["ivt"] or ivt in selected
                        or comp["instrument"] not in h0["instrument"]):
                    continue
                neighbors = [base for base in h0["ivt"] if base < 94
                             and COMPONENTS[base]["instrument"] == comp["instrument"]
                             and sum(COMPONENTS[base][t] != comp[t] for t in ("verb", "target")) == 1]
                # Prefer a precise one-component alternative, but allow a
                # same-instrument prior when H0 has only null or no relation.
                rank = (not bool(neighbors), -row["rate"], -row["positive_videos"], ivt)
                choices.append((rank, ivt, source, row, neighbors))
            if choices:
                break
        if not choices:
            continue
        _, ivt, source, stats, neighbors = min(choices, key=lambda item: item[0])
        selected.append(ivt)
        details.append({"ivt": ivt, "ranking_source": source, "statistics": deepcopy(stats),
                        "neighbor_ivts": neighbors, "graph_path": [
                            {"instrument": COMPONENTS[ivt]["instrument"]}, {"ivt": ivt},
                            {"training_prevalence": source, "phase_hint": phase if source == "phase" else None}]})
    if not selected:
        packet = None
    else:
        packet = {
            "use": "Possible omitted relations from OTHER Training videos, not visual evidence or confidence for this frame. "
                   "The phase hint may be wrong. Inspect the supplied images and propose only supported relations. "
                   "Rare or unlisted relations remain allowed. Suggestions consume the existing four-IVT allowance; do not copy them automatically.",
            "source": "Leave-query-video-out GT statistics linked to the existing IVT ontology.",
            "relations": [{"ivt_id": ivt, **{t: _TASK_NAMES[t][v] for t, v in COMPONENTS[ivt].items()}}
                          for ivt in selected],
        }
    # All candidates remain hypothetical: do not expose a modified prediction.
    make_pool(h0)
    return {"packet": packet, "audit": {"version": VERSION, "excluded_video": video_id,
            "fit_videos": prior["fit_videos"], "prior_sha256": claimed_hash,
            "selected": details, "phase_is_h0_prediction": True,
            "prior_is_not_visual_evidence": True, "extra_api_calls": 0}}
