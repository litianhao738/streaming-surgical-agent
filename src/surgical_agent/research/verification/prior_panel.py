"""Frozen-universe visual panel and local patch admission. No network or GT access.

Priors accept explicitly isolated Training sufficient statistics. Inference
functions only receive H0, the resulting table, images' identities and responses.
"""
import hashlib
import json
from collections import Counter
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _TASK_NAMES, _ivt_rows
from surgical_agent.research.verification.final_only_grounded import final_labels

TASKS = ("instrument", "verb", "target", "ivt")
COMPONENTS = {row[0]: dict(zip(TASKS[:3], row[1:], strict=True)) for row in _ivt_rows()}
BOUNDS = {"instrument": 7, "verb": 10, "target": 15, "ivt": 100}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def labels(value):
    return {k: sorted(v) for k, v in final_labels(value).items()}


def video_counts(rows):
    """Each row is one canonical frame, with independent per-task masks."""
    result = {q: {"n": 0, "x": [0] * BOUNDS[q], "phase": {}} for q in TASKS}
    seen = set()
    for row in rows:
        if row["frame_id"] in seen:
            raise ValueError("duplicate statistical frame")
        seen.add(row["frame_id"])
        for q in TASKS:
            if not row["mask"][q]:
                continue
            ids = set(row["gt"][q])
            if any(type(c) is not int or not 0 <= c < BOUNDS[q] for c in ids):
                raise ValueError("statistical label outside ontology")
            buckets = [result[q]]
            if row["mask"]["phase"]:
                p = str(row["gt"]["phase"][0])
                buckets.append(result[q]["phase"].setdefault(p, {"n": 0, "x": [0] * BOUNDS[q]}))
            for bucket in buckets:
                bucket["n"] += 1
                for c in ids:
                    bucket["x"][c] += 1
    return result


def fit_prior(training_counts, excluded_video):
    """LOVO at table construction, never subtract a probability fitted on all videos."""
    allowed = {v: s for v, s in training_counts.items() if v != excluded_video}
    if excluded_video in allowed or not allowed:
        raise ValueError("empty or contaminated prior sources")
    result = {"excluded_video": excluded_video, "fit_videos": sorted(allowed), "tasks": {}}
    for q in TASKS:
        global_buckets = [s[q] for s in allowed.values() if s[q]["n"]]
        global_rates = [sum(b["x"][c] / b["n"] for b in global_buckets) / len(global_buckets)
                        if global_buckets else 0 for c in range(BOUNDS[q])]

        def summarize(buckets, phase=False, global_rates=global_rates):
            n = sum(b["n"] for b in buckets)
            out = []
            for c, g in enumerate(global_rates):
                positives = sum(b["x"][c] for b in buckets)
                positive_videos = sum(b["x"][c] > 0 for b in buckets)
                rate = (sum((b["x"][c] + 20 * g) / (b["n"] + 20) for b in buckets)
                        / len(buckets)) if phase and buckets else g
                out.append({"id": c, "rate": rate, "positive_frames": positives,
                            "positive_videos": positive_videos, "valid_frames": n,
                            "valid_videos": len(buckets),
                            "eligible": n >= 100 and len(buckets) >= 3
                            and positives >= 5 and positive_videos >= 2})
            return out

        result["tasks"][q] = {"global": summarize(global_buckets), "phase": {
            str(p): summarize([s[q]["phase"][str(p)] for s in allowed.values()
                               if s[q]["phase"].get(str(p), {}).get("n", 0)], True)
            for p in range(7)}}
    result["table_sha256"] = digest(result)
    return result


def build_universe(h0, *, video_id, prior=None, global_only=False):
    h0 = labels(h0)
    if prior is not None and (prior["excluded_video"] != video_id or video_id in prior["fit_videos"]):
        raise ValueError("prior exclusion does not match the target video")
    origin = {}

    def add(q, c, source):
        origin.setdefault((q, c), set()).add(source)

    for q in TASKS:
        for c in h0[q]:
            add(q, c, "H0")
    for c in range(7):
        add("instrument", c, "INVENTORY")
    selected, selection_log = [], []
    if prior is not None:
        phase = str(h0["phase"][0])
        t = prior["tasks"]["ivt"]
        global_rank = sorted((r for r in t["global"] if r["eligible"]), key=lambda r: (-r["rate"], r["id"]))
        phase_rank = sorted((r for r in t["phase"][phase] if r["eligible"]), key=lambda r: (-r["rate"], r["id"]))
        same_global = [r for r in global_rank if COMPONENTS[r["id"]]["instrument"] in h0["instrument"]]
        same_phase = [r for r in phase_rank if COMPONENTS[r["id"]]["instrument"] in h0["instrument"]]
        for slot in range(4):
            lists = ([same_phase, same_global, global_rank] if slot < 2 and not global_only
                     else [same_global, global_rank] if slot < 3 else [global_rank])
            chosen = next(((i, r) for i, rows in enumerate(lists) for r in rows
                           if r["id"] not in h0["ivt"] and r["id"] not in selected), None)
            if chosen is None:
                selection_log.append({"slot": slot, "id": None, "reason": "INSUFFICIENT_SUPPORT"})
                continue
            i, r = chosen
            c = r["id"]
            selected.append(c)
            selection_log.append({"slot": slot, "id": c, "fallback_index": i})
            add("ivt", c, "PRIOR_IVT")
            for q, component in COMPONENTS[c].items():
                add(q, component, f"IVT:{c}")
        t = prior["tasks"]["target"]
        scores = []
        for g, p in zip(t["global"], t["phase"][phase], strict=True):
            if g["id"] in h0["target"]:
                continue
            if not global_only and p["eligible"]:
                scores.append((0.5 * (g["rate"] + p["rate"]), g["id"]))
            elif g["eligible"]:
                scores.append((g["rate"], g["id"]))
        if scores:
            add("target", min(scores, key=lambda x: (-x[0], x[1]))[1], "INDEPENDENT_PRIOR")
    ordered = sorted(origin, key=lambda item: (TASKS.index(item[0]), item[1]))
    if len(ordered) > 48:
        raise ValueError("universe exceeds 48; never truncate")
    public, private = [], {}
    for index, (q, c) in enumerate(ordered, 1):
        pid = f"p{index:03}"
        components = COMPONENTS[c] if q == "ivt" else None
        name = "/".join(_TASK_NAMES[k][v] for k, v in components.items()) if components else _TASK_NAMES[q][c]
        public.append({"proposition_id": pid, "task": q, "label_id": c, "name": name,
                       "components": components})
        private[pid] = {"origins": sorted(origin[q, c]), "independent": q == "instrument"
                        or bool(origin[q, c] & {"H0", "INDEPENDENT_PRIOR"})}
    return {"propositions": public, "eligibility": private, "selection_log": selection_log,
            "prior_sha256": prior["table_sha256"] if prior else None}


def normalize_review(response, universe, image_refs, schema):
    Draft202012Validator(schema).validate(response)
    props = {p["proposition_id"]: p for p in universe["propositions"]}
    ids = [a["proposition_id"] for a in response["assessments"]]
    if len(ids) != len(set(ids)) or set(ids) != set(props):
        raise ValueError("incomplete or duplicate proposition coverage")
    target = image_refs[-1]
    result = deepcopy(response)
    for a in result["assessments"]:
        p, w = props[a["proposition_id"]], a["witness"]
        errors = []
        refs, times = a["evidence_refs"], w["temporal_refs"]
        if len(refs) != len(set(refs)) or not set(refs) <= set(image_refs):
            errors.append("INVALID_IMAGE_REFERENCES")
        if w["target_frame_ref"] != target or len(times) != len(set(times)) or times != [r for r in image_refs if r in times]:
            errors.append("INVALID_WITNESS_TIMELINE")
        if "NONE" in a["issue_codes"] and len(a["issue_codes"]) != 1:
            errors.append("NONE_NOT_EXCLUSIVE")
        if a["presence"] in {"PRESENT", "ABSENT"} and target not in refs:
            errors.append("TARGET_FRAME_NOT_CITED")
        if a["presence"] == "ABSENT" and not (a["scope"] == "FRAME" and a["full_frame_reviewed"]
                                                and a["other_instances_accounted_for"]):
            errors.append("ABSENCE_NOT_FRAME_LEVEL")
        if a["presence"] == "PRESENT":
            q, c = p["task"], p["label_id"]
            comp = p["components"] or {}
            null_target = (q == "target" and c == 14) or (q == "ivt" and comp["target"] == 14)
            null_verb = (q == "verb" and c == 9) or (q == "ivt" and comp["verb"] == 9)
            if null_target or null_verb:
                allowed = {"JOINT_OUT_OF_VOCABULARY"}
                if null_target:
                    allowed.add("TARGET_OUT_OF_VOCABULARY")
                if null_verb:
                    allowed.add("VERB_OUT_OF_VOCABULARY")
                if w["ontology_boundary"] not in allowed or not w["local_instance_id"] or w["tool_point_xy"] is None:
                    errors.append("NULL_BOUNDARY_NOT_ESTABLISHED")
            exception = null_target and w["ontology_boundary"] in {"TARGET_OUT_OF_VOCABULARY", "JOINT_OUT_OF_VOCABULARY"}
            if q in {"target", "ivt"}:
                if not w["local_instance_id"]:
                    errors.append("NO_INSTANCE_WITNESS")
                if exception:
                    if w["target_role"] not in {"INTERACTION_TARGET", "NOT_APPLICABLE"}:
                        errors.append("NULL_TARGET_ROLE_CONFLICT")
                elif w["target_role"] != "INTERACTION_TARGET" or w["interaction_point_xy"] is None:
                    errors.append("NO_INTERACTION_TARGET_WITNESS")
            if q == "ivt" and (w["tool_point_xy"] is None or w["same_interaction"] not in
                               ({"YES", "NOT_APPLICABLE"} if exception else {"YES"})):
                errors.append("IVT_BINDING_NOT_ESTABLISHED")
        a["raw_presence"] = a["presence"]
        a["normalization_reasons"] = errors
        if errors:
            a["presence"] = "UNCLEAR"
    return result


def panel_votes(reviews):
    if len(reviews) != 3 or any(r is None for r in reviews):
        raise ValueError("all three complete judges required")
    ids = {a["proposition_id"] for a in reviews[0]["assessments"]}
    if any({a["proposition_id"] for a in r["assessments"]} != ids for r in reviews):
        raise ValueError("panel coverage differs")
    return {pid: dict(Counter(a["presence"] for r in reviews for a in r["assessments"]
                              if a["proposition_id"] == pid)) for pid in sorted(ids)}


def issues_for(current, universe, reviews):
    votes = panel_votes(reviews)
    issues = []
    for p in universe["propositions"]:
        pid = p["proposition_id"]
        op = "REMOVE" if p["label_id"] in current[p["task"]] else "ADD"
        vote = "ABSENT" if op == "REMOVE" else "PRESENT"
        if votes[pid].get(vote, 0) < 2:
            continue
        observations = [next(a for a in r["assessments"] if a["proposition_id"] == pid) for r in reviews]
        # New actionable feedback ignores prose, vote count, coordinate jitter,
        # and response-local instance names, which are not stable across calls.
        signature = digest({"pid": pid, "operation": op, "issues": sorted({code for a in observations
                            if a["presence"] == vote for code in a["issue_codes"]}),
                            "roles": sorted({(a["witness"]["target_role"], a["witness"]["same_interaction"],
                                               a["witness"]["ontology_boundary"]) for a in observations
                                              if a["presence"] == vote})})
        issues.append({"issue_id": f"issue:{pid}:{op}:{signature[:12]}", "proposition_id": pid,
                       "operation": op, "signature": signature, "observations": observations})
    return issues


def apply_patch_response(current, universe, issues, response, image_refs, schema):
    Draft202012Validator(schema).validate(response)
    props = {p["proposition_id"]: p for p in universe["propositions"]}
    issue_map = {i["issue_id"]: i for i in issues}
    draft, seen = labels(current), set()
    for edit in response["edits"]:
        pid = edit["proposition_id"]
        if pid in seen or pid not in props:
            raise ValueError("duplicate or pool-out patch")
        seen.add(pid)
        p = props[pid]
        q, c, op = p["task"], p["label_id"], edit["operation"]
        if (c in current[q]) != (op == "REMOVE"):
            raise ValueError("patch direction inconsistent with last accepted")
        references = edit["issue_refs"]
        if len(references) != len(set(references)) or any(r not in issue_map for r in references):
            raise ValueError("invalid issue reference")
        own = [issue_map[r] for r in references if issue_map[r]["proposition_id"] == pid
               and issue_map[r]["operation"] == op]
        if not own:
            raise ValueError("unrelated patch issue")
        if len(edit["evidence_refs"]) != len(set(edit["evidence_refs"])) or not set(edit["evidence_refs"]) <= set(image_refs):
            raise ValueError("unknown patch image")
        role, parents = edit["edit_role"], edit["dependency_ivt_proposition_ids"]
        if q == "ivt":
            if role != "IVT_EDIT" or parents:
                raise ValueError("invalid IVT edit role")
        elif universe["eligibility"][pid]["independent"]:
            if role != "INDEPENDENT_COMPONENT" or parents:
                raise ValueError("independent eligibility must use independent role")
        elif role != "IVT_COMPONENT_ONLY" or op != "ADD" or not parents:
            raise ValueError("dependency-only component cannot be independently edited")
        if len(parents) != len(set(parents)):
            raise ValueError("duplicate parents")
        for parent in parents:
            if parent not in props or props[parent]["task"] != "ivt" or props[parent]["components"][q] != c:
                raise ValueError("parent does not use component")
        if op == "ADD":
            draft[q].append(c)
        else:
            draft[q].remove(c)
    draft = labels(draft)
    for ivt in set(draft["ivt"]) - set(current["ivt"]):
        if any(c not in draft[q] for q, c in COMPONENTS[ivt].items()):
            raise ValueError("new IVT lacks necessary components in draft")
    for edit in response["edits"]:
        if any(props[parent]["label_id"] not in draft["ivt"] for parent in edit["dependency_ivt_proposition_ids"]):
            raise ValueError("parent not in draft")
    return draft


def admit(h0, draft, universe, reviews):
    """Reconstruct all cumulative edits relative to H0 using this round only."""
    h0, draft = labels(h0), labels(draft)
    if draft["phase"] != h0["phase"]:
        raise ValueError("Phase must stay H0")
    by_label = {(p["task"], p["label_id"]): p["proposition_id"] for p in universe["propositions"]}
    votes = panel_votes(reviews)

    def approved(q, c, op):
        pid = by_label.get((q, c))
        if pid is None:
            raise ValueError("draft contains an unreviewed edit")
        return votes[pid].get("PRESENT" if op == "ADD" else "ABSENT", 0) >= (2 if op == "ADD" else 3)

    out = deepcopy(h0)
    for c in set(h0["ivt"]) - set(draft["ivt"]):
        if approved("ivt", c, "REMOVE"):
            out["ivt"].remove(c)
    accepted_new = []
    for c in sorted(set(draft["ivt"]) - set(h0["ivt"])):
        if not approved("ivt", c, "ADD"):
            continue
        if all(v in draft[q] and (v in h0[q] or approved(q, v, "ADD"))
               for q, v in COMPONENTS[c].items()):
            out["ivt"].append(c)
            accepted_new.append(c)
    for q in TASKS[:3]:
        for c in sorted(set(draft[q]) - set(h0[q])):
            pid = by_label[q, c]
            used = any(COMPONENTS[parent][q] == c for parent in accepted_new)
            if approved(q, c, "ADD") and (universe["eligibility"][pid]["independent"] or used):
                out[q].append(c)
        for c in set(h0[q]) - set(draft[q]):
            if approved(q, c, "REMOVE") and not any(COMPONENTS[parent][q] == c for parent in out["ivt"]):
                out[q].remove(c)
    # A subset can exceed a cap when rejected deletions and approved additions
    # combine. Reject the round, rather than choose an arbitrary ranked subset.
    return labels(out)


def run_arm(h0, universe, panel_call, patch_call, judge_schema, patch_schema, image_refs, save):
    """Callbacks own accounting. No accepted state changes after a partial round."""
    h0 = labels(h0)
    state, draft = deepcopy(h0), deepcopy(h0)
    seen_drafts, seen_issues = {digest(h0)}, set()
    history, snapshots = [], []
    reason = "MAX_ROUNDS"
    for round_no in range(1, 4):
        raw = panel_call(round_no, draft)
        try:
            if raw is None or len(raw) != 3 or any(r is None for r in raw):
                raise ValueError("incomplete panel")
            normalized = [normalize_review(r, universe, image_refs, judge_schema) for r in raw]
            new_state = state if round_no == 1 else admit(h0, draft, universe, normalized)
        except (ValueError, KeyError, TypeError, ApiSchemaError, ValidationError) as exc:
            reason = "ROUND_INVALID:" + type(exc).__name__
            history.append({"round": round_no, "current": draft, "raw": raw, "status": reason})
            break
        record = {"round": round_no, "current": draft, "last_accepted": new_state,
                  "normalized": normalized, "votes": panel_votes(normalized), "status": "VALID"}
        # Durable publication before advancing in-memory state.
        save(f"round_{round_no}", record)
        state = deepcopy(new_state)
        snapshots.append(deepcopy(state))
        history.append(record)
        if round_no == 3:
            break
        issues = issues_for(state, universe, normalized)
        new_issues = [i for i in issues if i["signature"] not in seen_issues]
        if not new_issues:
            reason = "NO_NEW_ACTIONABLE_ISSUES"
            break
        seen_issues.update(i["signature"] for i in issues)
        patch = patch_call(round_no, state, issues)
        record["issues"], record["patch"] = issues, patch
        try:
            if patch is None:
                raise ValueError("missing patch")
            draft = apply_patch_response(state, universe, issues, patch, image_refs, patch_schema)
        except (ValueError, KeyError, TypeError, ApiSchemaError, ValidationError) as exc:
            reason = "PATCH_INVALID:" + type(exc).__name__
            break
        if digest(draft) in seen_drafts or draft == state:
            reason = "UNCHANGED_OR_REPEATED_DRAFT"
            break
        seen_drafts.add(digest(draft))
    while len(snapshots) < 3:
        snapshots.append(deepcopy(state))
    result = {"h0": h0, "final": state, "snapshots": snapshots, "history": history, "stop_reason": reason}
    save("result", result)
    return result
