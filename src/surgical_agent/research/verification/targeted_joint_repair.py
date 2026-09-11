"""Execute only changes proposed by Repair and admitted by the fresh panel."""
from copy import deepcopy

from surgical_agent.research.verification.prior_panel import COMPONENTS, TASKS


def apply_targeted_round(current, proposed, reviewed):
    result, rejected = deepcopy(current), []
    for task in TASKS:
        old, proposal, accepted = (set(x[task]) for x in (current, proposed, reviewed))
        additions = (accepted - old) & proposal
        removals = (old - accepted) - proposal
        result[task] = sorted((old | additions) - removals)
        for label in sorted((old ^ accepted) - (additions | removals)):
            rejected.append({"task": task, "label_id": label, "reason": "NOT_PROPOSED_BY_REPAIR"})
    if reviewed["phase"] == proposed["phase"]:
        result["phase"] = deepcopy(reviewed["phase"])
    elif reviewed["phase"] != current["phase"]:
        rejected.append({"task": "phase", "reason": "NOT_PROPOSED_BY_REPAIR"})
    for label in list(result["ivt"]):
        missing = [(t, c) for t, c in COMPONENTS[label].items() if c not in result[t]]
        if missing and label not in current["ivt"]:
            result["ivt"].remove(label)
            rejected.append({"task": "ivt", "label_id": label, "reason": "COMPONENT_NOT_ACCEPTED"})
        else:
            for task, component in missing:
                result[task] = sorted(set(result[task]) | {component})
                rejected.append({"task": task, "label_id": component, "reason": "RETAINED_IVT_COMPONENT"})
    return result, rejected
