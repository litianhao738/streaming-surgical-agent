"""Phase-only extension: original four-head predictions cannot be rewritten.

Selection is independent of ground truth, model transport and IVT priors.
Sparse history is drawn only from the target's available contiguous segment.
"""
import math
from copy import deepcopy
from itertools import pairwise

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import TASKS, labels


def phase_pool():
    return {"propositions": [{"id": f"phase_{i}", "task": "phase", "label_id": i,
        "name": _TASK_NAMES["phase"][i], "components": None} for i in range(7)]}


def choose_history(frame_ids, target_frame_id, offsets_seconds=(30, 10, 0)):
    """Actual frames at/before each anchor; no padding or crossing data gaps."""
    frames = list(frame_ids)
    if (not frames or any(type(f) is not int or f < 0 for f in frames)
            or any(b <= a for a, b in pairwise(frames))
            or type(target_frame_id) is not int or target_frame_id not in frames):
        raise ValueError("ordered unique real frame identities and target required")
    offsets = list(offsets_seconds)
    if (not offsets or offsets[-1] != 0 or any(type(s) is not int or s < 0 for s in offsets)
            or any(b >= a for a, b in pairwise(offsets))):
        raise ValueError("decreasing nonnegative integer offsets ending at zero required")
    end = frames.index(target_frame_id)
    start = end
    while start and frames[start] - frames[start - 1] == 25:
        start -= 1
    segment = frames[start:end + 1]
    chosen = set()
    for seconds in offsets:
        anchor = target_frame_id - 25 * seconds
        eligible = [f for f in segment if f <= anchor]
        if eligible:
            chosen.add(eligible[-1])
    return sorted(chosen)


def phase_apply(current, means, threshold=4.0):
    """Attach an atomic stage decision to a completed original repair result.

All original four-head lists are retained byte-for-byte in list order. Five-seat
validity is represented by each phase mean (None means incomplete evidence).
"""
    labels(current)  # Validate without sorting the user's existing lists.
    if threshold != 4.0 or set(means) != {f"phase_{i}" for i in range(7)}:
        raise ValueError("frozen threshold and all seven phase scores required")
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 1 <= v <= 5)
           for v in means.values()):
        raise ValueError("finite phase means in 1..5 required")
    out = deepcopy(current)
    old = current["phase"][0]
    old_score = means[f"phase_{old}"]
    candidates = [(means[f"phase_{p}"], p) for p in range(7)
                  if means[f"phase_{p}"] is not None and means[f"phase_{p}"] >= threshold]
    decision = {"before": old, "after": old, "reason": "NO_SUPPORTED_ALTERNATIVE"}
    if old_score is None:
        decision["reason"] = "INVALID_CURRENT_PHASE_EVIDENCE"
    elif candidates:
        best = max(s for s, _ in candidates)
        winners = [p for s, p in candidates if s == best]
        if len(winners) > 1:
            decision["reason"] = "TIED_PHASE_SUPPORT"
        elif winners[0] == old or best <= old_score:
            decision["reason"] = "CURRENT_PHASE_BEST"
        else:
            out["phase"] = [winners[0]]
            decision.update(after=winners[0], reason="UNIQUE_BETTER_SUPPORTED_PHASE")
    if any(out[t] != current[t] for t in TASKS):
        raise AssertionError("Phase repair must not change interaction heads")
    return out, decision


def phase_choice_error(raw, image_count):
    """A single mutually exclusive phase choice, or an explicit abstention."""
    if type(image_count) is not int or not 1 <= image_count <= 3:
        return "INVALID_IMAGE_COUNT"
    if not isinstance(raw, dict) or set(raw) != {"phase_id", "image_indices", "observation"}:
        return "INVALID_FIELDS"
    phase = raw["phase_id"]
    if phase is not None and (type(phase) is not int or not 0 <= phase < 7):
        return "INVALID_PHASE_ID"
    refs = raw["image_indices"]
    if (not isinstance(refs, list) or any(type(i) is not int or not 0 <= i < image_count for i in refs)
            or len(set(refs)) != len(refs)):
        return "INVALID_IMAGE_REFERENCE"
    if phase is not None and image_count - 1 not in refs:
        return "CURRENT_IMAGE_REQUIRED"
    observation = raw["observation"]
    if not isinstance(observation, str) or not observation.strip() or len(observation) > 1000:
        return "INVALID_OBSERVATION"
    return None


def apply_phase_choices(current, raw_reviews, image_count):
    """Strict majority of five independent single-label choices; never edit IVT heads."""
    labels(current)
    out = deepcopy(current)
    old = current["phase"][0]
    decision = {"before": old, "after": old, "reason": "INVALID_PANEL", "votes": {str(p): 0 for p in range(7)},
                "abstentions": 0}
    if (not isinstance(raw_reviews, dict) or set(raw_reviews) != set(SEATS)
            or any(phase_choice_error(raw_reviews[s], image_count) for s in SEATS)):
        return out, decision
    for raw in raw_reviews.values():
        if raw["phase_id"] is None:
            decision["abstentions"] += 1
        else:
            decision["votes"][str(raw["phase_id"])] += 1
    winners = [int(p) for p, count in decision["votes"].items() if count >= 3]
    decision["reason"] = "NO_MAJORITY"
    if winners:
        winner = winners[0]
        out["phase"] = [winner]
        decision.update(after=winner, reason="CURRENT_PHASE_MAJORITY" if winner == old else "MAJORITY_PHASE_SWITCH")
    if any(out[t] != current[t] for t in TASKS):
        raise AssertionError("Phase repair changed an interaction head")
    return out, decision
