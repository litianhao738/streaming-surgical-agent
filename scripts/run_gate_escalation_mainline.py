"""Split the frozen mainline at its one-H0 / twelve-repair-call boundary.

This is an execution seam, not a trained Gate.  The collector must persist its
H0-only features between generate_h0 and repair_from_h0.  Neither function reads
ground truth, tracker output, a dataset, or future frames.  The frozen runner is
left intact and supplies every prompt, model route, and admission rule.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_prior_gated_joint_mainline as frozen

from types import SimpleNamespace, FunctionType
from surgical_agent.research.verification import joint_empty_pool_patch
old = SimpleNamespace(**vars(frozen.old))
_decide_record = frozen.old.decide_record
old.decide_record = FunctionType(_decide_record.__code__,
    dict(_decide_record.__globals__, decide=joint_empty_pool_patch.decide),
    'decide_record_empty_pool', _decide_record.__defaults__)

VERSION = "gate-ready-postcheap-v1.0.1"
BASELINE_VERSION = joint_empty_pool_patch.VERSION
ARMS, PRIMARY = frozen.ARMS, frozen.PRIMARY
STAGES = dict(frozen.STAGES)
REPAIR_STAGES = {stage: count for stage, count in STAGES.items() if stage != "h0"}
CALLS_PER_TARGET = frozen.CALLS_PER_TARGET
REPAIR_CALLS_PER_TARGET = sum(REPAIR_STAGES.values())
MainlineCalls = frozen.MainlineCalls


def _check_prior(selected, prior):
    video = selected["video_id"]
    if prior["excluded_video"] != video or video in prior["fit_videos"]:
        raise ValueError("query video present in prior")


def generate_h0(calls, base, selected):
    """Make exactly one original H0 request, validate it, and return its raw JSON.

    Failed or malformed H0 raises the original schema error.  The transport's
    saved failure remains authoritative; this function never repeats a request.
    """
    raw = calls.call(selected["key"], "h0", "base", old.gemini_h0_wire(base))
    old.gated.h0_from_raw(raw)
    return raw


def repair_from_h0(calls, base, selected, prior, gate, h0_raw, *, h0_seconds=0.0, postcheap_callback=None):
    """Run only the original twelve dependent calls on a supplied valid H0.

    ``gate`` is the frozen statistical prior-admission configuration, not the
    new learned binary decision.  ``h0_seconds`` is measured by the caller; its
    default zero denotes that H0 was already available before this invocation.
    The returned record matches the original mainline record contract.
    """
    _check_prior(selected, prior)
    key, video = selected["key"], selected["video_id"]
    h0 = old.gated.h0_from_raw(h0_raw)
    timing = {"h0": h0_seconds}
    pool = old.make_pool(h0)
    hints = old.retrieve_candidate_hints(h0, prior, video_id=video)
    from surgical_agent.research.gate.escalation_rule_training import proposal_needed
    rule=proposal_needed(h0,prior,video_id=video,gate=gate)
    if postcheap_callback is not None and not rule:
        postcheap_callback(h0,pool)
    stamp = perf_counter()
    proposal = calls.call(key, "proposal", "base", old.proposal_wire(base, selected, h0, pool, hints["packet"]))
    timing["proposal"] = perf_counter() - stamp
    if proposal is not None:
        try:
            pool = old.make_pool(h0, proposal, pool)
        except ValueError:
            proposal = {"invalid": proposal}
    if postcheap_callback is not None and rule:
        postcheap_callback(h0, pool)
    compact = {s: old.joint.roster.review_wire(s, base, selected, pool) for s in old.SEATS}
    recommendation = old.phase_recommendation_wire(base, selected, h0, pool, hints)
    old.check_requests(h0, [*compact.values(), recommendation])
    record = {}

    def joint_branch():
        stamp = perf_counter()
        rec = calls.call(key, "phase_recommendation", "base", recommendation)
        error = old.joint.phase_choice_error(rec, 3)
        record["phase_recommendation"] = {"raw": rec, "error": error}
        wires = old.joint_review_wires(base, selected, h0, pool, None if error else rec["phase_id"])
        old.check_requests(h0, wires.values())
        record["joint_raw"] = old.joint.panel_call(calls, key, "joint_r1", wires)
        record["joint_branch_seconds"] = perf_counter() - stamp

    stamp = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as workers:
        graph = workers.submit(old.joint.panel_call, calls, key, "control_graph", compact)
        phase = workers.submit(joint_branch)
        review_raw = graph.result()
        phase.result()
    timing["panels_parallel"] = perf_counter() - stamp
    timing["joint_branch"] = record.pop("joint_branch_seconds")
    predictions, detail = old.decide_record(h0, pool, review_raw, None, record["joint_raw"], prior, gate)
    return {"key": key, "h0_raw": h0_raw, "h0": h0, "hints": hints, "proposal_raw": proposal,
            "pool": pool, "review_raw": review_raw, **record, **detail,
            "predictions": {a: predictions[a] for a in ARMS}, "timing_seconds": timing}


def run_target(calls, base, selected, prior, gate, *, verify=True):
    """Fixed-decision acceptance seam: 13 calls when on, one H0 call when off.

    ``verify`` must be an explicit boolean.  Training collection uses the split
    functions directly so its pre-repair features can be sealed between them.
    """
    if type(verify) is not bool:
        raise TypeError("verify must be a boolean decision")
    _check_prior(selected, prior)
    stamp = perf_counter()
    raw = generate_h0(calls, base, selected)
    h0_seconds = perf_counter() - stamp
    if verify:
        return repair_from_h0(calls, base, selected, prior, gate, raw, h0_seconds=h0_seconds)
    h0 = old.gated.h0_from_raw(raw)
    return {"key": selected["key"], "h0_raw": raw, "h0": h0,
            "predictions": {arm: deepcopy(h0) for arm in ARMS},
            "timing_seconds": {"h0": h0_seconds},
            "verification": {"decision": "SKIP", "reason": "explicit_boolean_decision",
                             "repair_calls": 0}}


def run_escalation_target(calls, base, selected, prior, gate, *, route_after_cheap):
    """Executable rule-aware seam. Caller supplies a frozen Gate decision function.

    The callback receives only (H0, available pool, cheap labels, rule flag).
    No learned weights are implicitly loaded or deployed.
    """
    from surgical_agent.research.gate import escalation_rule_training as contract
    _check_prior(selected,prior)
    raw=generate_h0(calls,base,selected);h0=old.gated.h0_from_raw(raw)
    pool=old.make_pool(h0)
    rule=contract.proposal_needed(h0,prior,video_id=selected['video_id'],gate=gate)
    proposal_raw=None;proposal_body=None
    if rule:
        hints=old.retrieve_candidate_hints(h0,prior,video_id=selected['video_id'])
        proposal_body=old.proposal_wire(base,selected,h0,pool,hints['packet'])
        proposal_raw=calls.call(selected['key'],'proposal','base',proposal_body)
        if proposal_raw is not None:
            try:pool=old.make_pool(h0,proposal_raw,pool)
            except ValueError:pass
    cheap,_=contract.v1.cheap_state(h0,pool,prior,video_id=selected['video_id'],gate=gate)
    decision=route_after_cheap(deepcopy(h0),deepcopy(pool) if rule else None,deepcopy(cheap),rule)
    if type(decision) is not bool:raise TypeError('Gate decision must be bool')
    if not decision:
        return {'key':selected['key'],'h0':h0,'cheap':cheap,'final':deepcopy(cheap),
                'proposal_rule':rule,'reviewed':False,'calls':1+int(rule)}
    class ReuseProposal:
        def call(self,target,stage,seat,body):
            if stage=='proposal' and rule:
                if target!=selected['key'] or seat!='base' or body!=proposal_body:
                    raise ValueError('cached proposal request changed')
                return deepcopy(proposal_raw)
            return calls.call(target,stage,seat,body)
    result=repair_from_h0(ReuseProposal(),base,selected,prior,gate,raw)
    return {**result,'cheap':cheap,'final':result['predictions'][PRIMARY],
            'proposal_rule':rule,'reviewed':True,'calls':13}
