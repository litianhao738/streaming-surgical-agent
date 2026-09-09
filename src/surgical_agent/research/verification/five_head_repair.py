"""SurgReflect-inspired joint revision and optional five-family verification.

An isolated adaptation to the fixed full ontology, not an author-code replica.
Phase is a single-label task and has its own atomic selection rule.
"""
import math
from copy import deepcopy

from jsonschema import Draft202012Validator

from surgical_agent.perception.final_only import final_only_schema
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import (
    BOUNDS,
    COMPONENTS,
    TASKS,
    digest,
    labels,
)
from surgical_agent.research.verification.review_normalization import (
    CORE_FIELDS,
    _extra_error,
)
from surgical_agent.research.verification.semantic_coordinator import item_error

FIVE_TASKS = (*TASKS, "phase")


def joint_schema():
    contract = final_only_schema()["properties"]
    heads = {task: deepcopy(contract[task]["properties"]["selected_ids"]) for task in TASKS}
    heads["phase"] = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 6},
                      "minItems": 1, "maxItems": 1}
    return {"type": "object", "properties": {
        "prediction": {"type": "object", "properties": heads, "required": list(FIVE_TASKS), "additionalProperties": False},
        "observations": {"type": "object", "properties": {t: {"type": "string", "minLength": 1, "maxLength": 1000} for t in FIVE_TASKS},
                         "required": list(FIVE_TASKS), "additionalProperties": False}},
        "required": ["prediction", "observations"], "additionalProperties": False}


def consistency_issues(prediction):
    current = labels(prediction)
    return [{"type": "MISSING_IVT_COMPONENT", "ivt_id": ivt, "task": task, "label_id": value}
            for ivt in current["ivt"] for task, value in COMPONENTS[ivt].items() if value not in current[task]]


def phase_hints(current, prior, *, video_id):
    """Rank phases with cached P(IVT|phase), not posterior confidence or GT.

Every current IVT must have eligible statistics for a phase to be ranked.
Insufficient support yields no forced suggestion. No dataset access occurs here.
"""
    current = labels(current)
    check = deepcopy(prior)
    claimed = check.pop("table_sha256")
    if digest(check) != claimed or prior["excluded_video"] != video_id or video_id in prior["fit_videos"]:
        raise ValueError("invalid or query-contaminated prior")
    ivts = [i for i in current["ivt"] if i < 94]
    ranks = []
    for phase in range(7):
        rows = {r["id"]: r for r in prior["tasks"]["ivt"]["phase"][str(phase)]}
        if not ivts or any(i not in rows or not rows[i]["eligible"] for i in ivts):
            continue
        value = sum(math.log(max(rows[i]["rate"], 1e-12)) for i in ivts) / len(ivts)
        ranks.append({"phase_id": phase, "phase_name": _TASK_NAMES["phase"][phase],
                      "mean_log_conditional_prevalence": value, "relations_used": ivts})
    ranks.sort(key=lambda r: (-r["mean_log_conditional_prevalence"], r["phase_id"]))
    return {"source": "Other Training videos only; query video excluded.", "excluded_video": video_id,
            "prior_sha256": claimed, "suggestions": ranks[:2],
            "current_phase_outside_supported_top_two": bool(ranks and current["phase"][0] not in [r["phase_id"] for r in ranks[:2]]),
            "use": "Soft compatibility hint only. P(IVT|phase) is not P(phase|image). Current IVTs may be wrong; reinspect the images. Never auto-overwrite Phase from prevalence."}


def compile_joint(current, previous_pool, raw):
    current = labels(current)
    if not Draft202012Validator(joint_schema()).is_valid(raw):
        raise ValueError("five-head repair schema invalid")
    proposed = labels(raw["prediction"])
    repaired = deepcopy(proposed)
    # Author-style guard: retain four old heads only if all four revised heads
    # became empty. A valid revised Phase can still be accepted separately.
    empty_guard = not any(proposed[t] for t in TASKS)
    if empty_guard:
        for task in TASKS:
            repaired[task] = deepcopy(current[task])
    added = []
    for ivt in repaired["ivt"]:
        for task, value in COMPONENTS[ivt].items():
            if value not in repaired[task]:
                repaired[task].append(value)
                added.append({"task": task, "label_id": value, "implied_by_ivt": ivt})
    repaired = labels(repaired)
    expanded = deepcopy(previous_pool)
    expanded["propositions"].extend({"task": t, "label_id": v} for t in TASKS for v in repaired[t])
    pool = deepcopy(make_pool(current, previous=expanded))
    pool["propositions"].extend({"id": f"phase_{v}", "task": "phase", "label_id": v,
                                 "name": _TASK_NAMES["phase"][v], "components": None} for v in range(7))
    return {"raw_prediction": proposed, "paper_style": repaired, "pool": pool,
            "empty_four_head_guard": empty_guard, "derived_components": added,
            "issues_before": consistency_issues(current), "issues_after": consistency_issues(repaired)}


def normalize_five_heads(raw_reviews, pool, *, image_count=3):
    """Keep all five tasks and source-bound row diagnostics; no score coercion."""
    if type(image_count) is not int or not 1 <= image_count <= 3:
        raise ValueError("one to three causal images required")
    if not isinstance(raw_reviews, dict) or set(raw_reviews) != set(SEATS):
        raise ValueError("five reviewer seats required")
    props = {p["id"]: p for p in pool["propositions"]}
    if len(props) != len(pool["propositions"]):
        raise ValueError("duplicate pool IDs")
    for pid, p in props.items():
        bound = 7 if p["task"] == "phase" else BOUNDS.get(p["task"], 0)
        if type(p["label_id"]) is not int or not 0 <= p["label_id"] < bound or pid != f"{p['task']}_{p['label_id']}":
            raise ValueError("invalid five-head pool binding")
    reviews, formats = {}, {}
    for seat in SEATS:
        raw = raw_reviews[seat]
        accepted, errors, ignored = {}, {}, []
        if isinstance(raw, dict) and isinstance(raw.get("judgments"), dict) and "rows" not in raw:
            entries = list(raw["judgments"].items())
        elif isinstance(raw, dict) and isinstance(raw.get("rows"), list) and "judgments" not in raw:
            entries = [(r.get("candidate_id") if isinstance(r, dict) else None, r) for r in raw["rows"]]
        else:
            entries = []
            errors = {pid: ["INVALID_REVIEW_ENVELOPE"] for pid in props}
        seen = set()
        for pid, item in entries:
            if not isinstance(pid, str) or pid not in props:
                ignored.append("UNKNOWN_OR_MISSING_CANDIDATE_ID")
                continue
            if pid in seen:
                errors[pid] = ["DUPLICATE_CANDIDATE_ID"]
                accepted.pop(pid, None)
                continue
            seen.add(pid)
            if not isinstance(item, dict):
                errors[pid] = ["SCHEMA_INVALID"]
                continue
            core = {k: deepcopy(v) for k, v in item.items() if k in CORE_FIELDS}
            reasons = []
            error = item_error(core, props[pid]["task"], image_count)
            if error:
                reasons.append(error)
            for key in item.keys() - CORE_FIELDS:
                error = _extra_error(key, item[key], core, props[pid])
                if error:
                    reasons.append(error)
            if reasons:
                errors[pid] = reasons
            else:
                accepted[pid] = core
        for pid in props.keys() - seen:
            errors.setdefault(pid, ["MISSING_CANDIDATE_ID"])
        reviews[seat] = {"judgments": accepted}
        formats[seat] = {"errors": errors, "unbound_rows": ignored}
    return reviews, formats


def aggregate_five_heads(reviews, pool, *, image_count=3):
    if set(reviews) != set(SEATS):
        raise ValueError("five seats required")
    means, diagnostics = {}, {}
    for p in pool["propositions"]:
        scores, invalid = [], {}
        for seat in SEATS:
            item = reviews[seat]["judgments"].get(p["id"])
            reason = item_error(item, p["task"], image_count)
            if reason:
                invalid[seat] = reason
                scores.append(None)
            else:
                scores.append(item["rating"])
        means[p["id"]] = None if invalid else sum(scores) / 5
        diagnostics[p["id"]] = {"scores": scores, "invalid": invalid}
    return means, diagnostics


def select_five_heads(current, pool, means):
    """Four-head original rule + single-label Phase with a unique supported best.

Require five valid judgments for the old phase and a competing phase. Switch
only if the unique best supported phase (>=4) beats the valid old-phase score.
Ties, missing old-phase evidence or no supported alternative preserve Phase.
"""
    current = labels(current)
    if set(means) != {p["id"] for p in pool["propositions"]}:
        raise ValueError("means must cover full five-head pool")
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 1 <= v <= 5) for v in means.values()):
        raise ValueError("invalid means")
    four_pool = {"propositions": [p for p in pool["propositions"] if p["task"] in TASKS]}
    four_means = {p["id"]: means[p["id"]] for p in four_pool["propositions"]}
    out = panel.select(current, four_pool, four_means, threshold=4)
    if {p["label_id"] for p in pool["propositions"] if p["task"] == "phase"} != set(range(7)):
        raise ValueError("all seven phase alternatives required")
    old = current["phase"][0]
    old_score = means[f"phase_{old}"]
    candidates = [(means[f"phase_{p}"], p) for p in range(7) if means[f"phase_{p}"] is not None and means[f"phase_{p}"] >= 4]
    decision = {"before": old, "after": old, "reason": "NO_SUPPORTED_ALTERNATIVE"}
    if old_score is None:
        decision["reason"] = "INVALID_CURRENT_PHASE_EVIDENCE"
    elif candidates:
        best = max(s for s, _ in candidates)
        winners = [p for s, p in candidates if s == best]
        if len(winners) != 1:
            decision["reason"] = "TIED_PHASE_SUPPORT"
        elif winners[0] == old or best <= old_score:
            decision["reason"] = "CURRENT_PHASE_BEST"
        else:
            out["phase"] = [winners[0]]
            decision.update(after=winners[0], reason="UNIQUE_BETTER_SUPPORTED_PHASE")
    return labels(out), decision
